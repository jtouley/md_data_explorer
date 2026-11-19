"""Ground truth for tests/eval/corpus/questions/prostate_xls.yaml (Byar & Green prostate cancer
trial, hbiostat prostate.xls: one row per patient).

Written from the hbiostat description, not from the app's code. The file is a legacy BIFF .xls
read here with xlrd (the dependency the app's .xls reader brings; with no CODEPAGE record xlrd
reads text as iso-8859-1). rx is the arm as text ("placebo", "0.2 mg estrogen", ...); status is
"alive" or "dead - <cause>"; dtime is months of follow-up; stage is 3 or 4.
"""

from __future__ import annotations

import os
from pathlib import Path

import polars as pl

from clinical_analytics.eval.corpus import patient_kit as kit
from clinical_analytics.eval.corpus.score import OracleAnswer

DEAD = kit.text("status") != "alive"
PROSTATE_DEATH = kit.text("status") == "dead - prostatic ca"
VASCULAR_DEATH = kit.text("status") == "dead - heart or vascular"
HIGH_DOSE = kit.text("rx") == "5.0 mg estrogen"


def scan(path: Path) -> pl.LazyFrame:
    """The first sheet: a header row, then one row per patient, every column as text."""
    import xlrd  # noqa: PLC0415 - only this dataset needs it; run.py reports it when missing

    # xlrd writes its "no CODEPAGE record" notice to the log file it is given.
    with open(os.devnull, "w", encoding="utf-8") as quiet:
        sheet = xlrd.open_workbook(str(path), logfile=quiet).sheet_by_index(0)
    rows = [sheet.row_values(i) for i in range(sheet.nrows)]
    return kit.frame_from_rows(rows[0], rows[1:])


ORACLES: dict[str, kit.Oracle] = {
    "prostate_xls-total-001": kit.count(),
    "prostate_xls-arm-002": kit.count_by("rx"),
    "prostate_xls-deathrate-003": kit.rates(DEAD, "rx"),
    "prostate_xls-cause-004": kit.count_by("status"),
    "prostate_xls-followup-005": kit.stat_by("median", "dtime", "rx"),
    "prostate_xls-older-006": kit.count((kit.number("age") > 75) & PROSTATE_DEATH),
    "prostate_xls-stageage-007": kit.stat_by("mean", "age", "stage"),
    "prostate_xls-vascular-008": kit.odds_ratio(
        HIGH_DOSE, VASCULAR_DEATH, kit.is_value("rx", "placebo", "5.0 mg estrogen")
    ),
    "prostate_xls-after2020-009": kit.held(),
}


def answers(path: Path, oracles: dict[str, kit.Oracle] | None = None) -> dict[str, OracleAnswer]:
    return kit.answers(scan(path), oracles or ORACLES)
