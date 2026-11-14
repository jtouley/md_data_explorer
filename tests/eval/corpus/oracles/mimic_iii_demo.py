"""Ground truth for tests/eval/corpus/questions/mimic_iii_demo.yaml, in Polars only.

Written from the MIMIC-III documentation (PhysioNet project page, mimic.mit.edu table
descriptions), not from the app's code. Tables are read as text from the published archive.
Admission outcomes join ADMISSIONS to PATIENTS on subject_id (one patient row per subject).
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from clinical_analytics.eval.corpus import oracle_kit as kit
from clinical_analytics.eval.corpus.score import OracleAnswer

ARCHIVE = "mimic-iii-clinical-database-demo-1.4.zip"
PREFIX = "mimic-iii-clinical-database-demo-1.4/"
TABLES = {name: f"{PREFIX}{name.upper()}.csv" for name in ("patients", "admissions", "icustays", "microbiologyevents")}
DIED = pl.col("hospital_expire_flag") == "1"
FLAG_RECORDED = pl.col("hospital_expire_flag").is_in(["0", "1"])


def _table(name: str) -> kit.FrameFn:
    return lambda tables: tables[name]


def _admissions_with_sex(tables: kit.Tables) -> pl.LazyFrame:
    return tables["admissions"].join(tables["patients"].select("subject_id", "gender"), on="subject_id", how="left")


def _micro(where: pl.Expr) -> kit.FrameFn:
    return lambda tables: tables["microbiologyevents"].filter(where)


def _median_admissions(tables: kit.Tables) -> kit.Query:
    def finish(df: pl.DataFrame) -> OracleAnswer:
        return OracleAnswer({"median": float(df.item())}, "admissions per patient")

    per_patient = tables["admissions"].group_by("subject_id").agg(pl.len().alias("n"))
    return kit.Query("admissions per patient", per_patient.select(pl.col("n").median()), finish)


_EMERGENCY_ELECTIVE = pl.col("admission_type").is_in(["EMERGENCY", "ELECTIVE"]) & FLAG_RECORDED
_CDIFF = pl.col("org_name").str.to_uppercase().str.contains("DIFFICILE").fill_null(False)
_POSITIVE_BLOOD = (pl.col("spec_type_desc") == "BLOOD CULTURE") & kit.present("org_name")

ORACLES: dict[str, kit.Oracle] = {
    "m3-patients-001": kit.count(_table("patients"), "patients", distinct="subject_id"),
    "m3-admissions-002": kit.count(_table("admissions"), "visits", distinct="hadm_id"),
    "m3-sex-003": kit.count_by(_table("patients"), "gender", "patients"),
    "m3-mortality-sex-004": kit.rates(_admissions_with_sex, "gender", DIED, FLAG_RECORDED),
    "m3-admtype-005": kit.count_by(_table("admissions"), "admission_type", "visits"),
    "m3-los-006": kit.stat_by(_table("icustays"), "median", kit.number("los"), "first_careunit", "los"),
    "m3-mortality-type-007": kit.rates(_table("admissions"), "admission_type", DIED, FLAG_RECORDED),
    "m3-age-008": kit.held(
        "patients", "DOB of patients over 89 is shifted: computed ages near 300 make a mean meaningless"
    ),
    "m3-trend-009": kit.held("admissions", "dates are shifted per patient into 2100-2200"),
    "m3-before2020-010": kit.held("admissions", "dates are shifted per patient into 2100-2200"),
    "m3-cdiff-011": kit.count(_micro(_CDIFF), "patients", distinct="subject_id"),
    "m3-blood-012": kit.count(_micro(_POSITIVE_BLOOD), "patients", distinct="subject_id"),
    "m3-or-013": kit.odds_ratio(
        _table("admissions"), pl.col("admission_type") == "EMERGENCY", DIED, _EMERGENCY_ELECTIVE
    ),
    "m3-listing-014": kit.held("admissions", "a listing of patient IDs"),
    "m3-adm-per-patient-015": _median_admissions,
}


def answers(files_dir: Path, oracles: dict[str, kit.Oracle] | None = None) -> dict[str, OracleAnswer]:
    return kit.answers(kit.read_archive(files_dir / ARCHIVE, TABLES), oracles or ORACLES)
