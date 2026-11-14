"""Ground truth for tests/eval/corpus/questions/mimic_iv_demo.yaml, in Polars only.

Written from the MIMIC-IV documentation (PhysioNet project page and README), not from the app's
code. Tables are read as text from the published archive. A C. diff test is a microbiology row
whose test_name names C. difficile; patients are counted once (distinct subject_id).
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from clinical_analytics.eval.corpus import oracle_kit as kit
from clinical_analytics.eval.corpus.score import OracleAnswer

ARCHIVE = "mimic-iv-clinical-database-demo-2.2.zip"
PREFIX = "mimic-iv-clinical-database-demo-2.2/"
TABLES = {
    "patients": f"{PREFIX}hosp/patients.csv.gz",
    "admissions": f"{PREFIX}hosp/admissions.csv.gz",
    "icustays": f"{PREFIX}icu/icustays.csv.gz",
    "microbiologyevents": f"{PREFIX}hosp/microbiologyevents.csv.gz",
}
DIED = pl.col("hospital_expire_flag") == "1"
FLAG_RECORDED = pl.col("hospital_expire_flag").is_in(["0", "1"])
AGE_65 = kit.number("anchor_age") >= 65
CDIFF_TEST = pl.col("test_name").str.contains("(?i)difficile").fill_null(False)


def _table(name: str) -> kit.FrameFn:
    return lambda tables: tables[name]


def _tested_patients(tables: kit.Tables) -> pl.LazyFrame:
    tested = tables["microbiologyevents"].filter(CDIFF_TEST).select("subject_id").unique()
    return tested.join(tables["patients"], on="subject_id", how="left")


ORACLES: dict[str, kit.Oracle] = {
    "m4-patients-001": kit.count(_table("patients"), "patients", distinct="subject_id"),
    "m4-sex-002": kit.count_by(_table("patients"), "gender", "patients"),
    "m4-anchor-003": kit.count_by(_table("patients"), "anchor_year_group", "patients"),
    "m4-age65-004": kit.count(lambda t: t["patients"].filter(AGE_65), "patients", distinct="subject_id"),
    "m4-mortality-type-005": kit.rates(_table("admissions"), "admission_type", DIED, FLAG_RECORDED),
    "m4-cdiff-tested-006": kit.count(_tested_patients, "patients", distinct="subject_id"),
    "m4-cdiff-65-007": kit.count(lambda t: _tested_patients(t).filter(AGE_65), "patients", distinct="subject_id"),
    "m4-los-008": kit.stat_by(_table("icustays"), "median", kit.number("los"), "first_careunit", "los"),
    "m4-insurance-009": kit.count_by(_table("admissions"), "insurance", "visits"),
    "m4-or-010": kit.odds_ratio(_table("patients"), AGE_65, kit.present("dod"), kit.number("anchor_age").is_not_null()),
    "m4-before2020-011": kit.held(
        "patients", "dates are shifted to 2110-2201; the only real-time anchor is anchor_year_group (2011-2016)"
    ),
    "m4-cdiff-anchor-012": kit.count_by(_tested_patients, "anchor_year_group", "patients"),
    "m4-listing-013": kit.held("patients", "a listing of patient IDs"),
    "m4-age-sex-014": kit.stat_by(_table("patients"), "mean", kit.number("anchor_age"), "gender", "anchor_age"),
    "m4-abx-or-015": kit.held("microbiologyevents", "inpatient orders only and cells of a handful of patients"),
}


def answers(files_dir: Path, oracles: dict[str, kit.Oracle] | None = None) -> dict[str, OracleAnswer]:
    return kit.answers(kit.read_archive(files_dir / ARCHIVE, TABLES), oracles or ORACLES)
