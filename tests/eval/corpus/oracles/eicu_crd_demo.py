"""Ground truth for tests/eval/corpus/questions/eicu_crd_demo.yaml, in Polars only.

Written from the eICU-CRD documentation (PhysioNet project page, eicu-crd.mit.edu table
descriptions), not from the app's code. Tables are read as text from the published archive. A
stay's hospital attributes join patient to hospital on hospitalid (one hospital row each); the
APACHE IV score is the apacheversion 'IV' row, with -1 (no score) as missing.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from clinical_analytics.eval.corpus import oracle_kit as kit
from clinical_analytics.eval.corpus.score import OracleAnswer

ARCHIVE = "eicu-collaborative-research-database-demo-2.0.1.zip"
PREFIX = "eicu-collaborative-research-database-demo-2.0.1/"
TABLES = {name: f"{PREFIX}{name}.csv.gz" for name in ("patient", "hospital", "apachePatientResult")}
UNIT_DIED = pl.col("unitdischargestatus") == "Expired"
UNIT_RECORDED = pl.col("unitdischargestatus").is_in(["Alive", "Expired"])
HOSPITAL_DIED = pl.col("hospitaldischargestatus") == "Expired"
HOSPITAL_RECORDED = pl.col("hospitaldischargestatus").is_in(["Alive", "Expired"])


def _table(name: str) -> kit.FrameFn:
    return lambda tables: tables[name]


def _stays_with_hospital(tables: kit.Tables) -> pl.LazyFrame:
    return tables["patient"].join(tables["hospital"], on="hospitalid", how="left")


def _stays_with_apache(tables: kit.Tables) -> pl.LazyFrame:
    apache = (
        tables["apachePatientResult"].filter(pl.col("apacheversion") == "IV").select("patientunitstayid", "apachescore")
    )
    return tables["patient"].join(apache, on="patientunitstayid", how="left")


def _stays_per_patient(tables: kit.Tables) -> kit.Query:
    def finish(df: pl.DataFrame) -> OracleAnswer:
        return OracleAnswer({"mean": float(df.item())}, "stays per patient")

    frame = tables["patient"].select(pl.len() / pl.col("uniquepid").n_unique())
    return kit.Query("stays per patient", frame, finish)


_APACHE = kit.number("apachescore")
_SCORE = pl.when(_APACHE >= 0).then(_APACHE)
# Age as the question states it: '> 89' counted as 90, every other value a whole number of years.
_AGE = pl.when(pl.col("age") == "> 89").then(pl.lit(90.0)).otherwise(kit.number("age"))

ORACLES: dict[str, kit.Oracle] = {
    "e-stays-001": kit.count(_table("patient"), "visits", distinct="patientunitstayid"),
    "e-patients-002": kit.count(_table("patient"), "patients", distinct="uniquepid"),
    "e-region-mortality-003": kit.rates(_stays_with_hospital, "region", UNIT_DIED, UNIT_RECORDED),
    "e-sex-mortality-004": kit.rates(_table("patient"), "gender", HOSPITAL_DIED, HOSPITAL_RECORDED),
    "e-apache-005": kit.stat_by(_stays_with_apache, "median", _SCORE, "unittype", "apachescore"),
    "e-year-006": kit.rates(_table("patient"), "hospitaldischargeyear", HOSPITAL_DIED, HOSPITAL_RECORDED),
    "e-over89-007": kit.count(lambda t: t["patient"].filter(pl.col("age") == "> 89"), "visits"),
    "e-bedsize-008": kit.count_by(_table("hospital"), "numbedscategory", "hospitals"),
    "e-teaching-009": kit.rates(_stays_with_hospital, "teachingstatus", UNIT_DIED, UNIT_RECORDED),
    "e-stays-per-patient-010": _stays_per_patient,
    "e-before2020-011": kit.held("patient", "the data cover 2014-2015 only"),
    "e-listing-012": kit.held("patient", "a listing of patient IDs"),
    "e-or-013": kit.odds_ratio(
        _stays_with_hospital,
        pl.col("teachingstatus") == "t",
        HOSPITAL_DIED,
        HOSPITAL_RECORDED & pl.col("teachingstatus").is_in(["t", "f"]),
    ),
    "e-unittype-014": kit.count_by(_table("patient"), "unittype", "visits"),
    "e-mean-age-015": kit.stat_by(_table("patient"), "mean", _AGE, "gender", "age"),
}


def answers(files_dir: Path, oracles: dict[str, kit.Oracle] | None = None) -> dict[str, OracleAnswer]:
    return kit.answers(kit.read_archive(files_dir / ARCHIVE, TABLES), oracles or ORACLES)
