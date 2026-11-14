"""Ground truth for tests/eval/corpus/questions/mesothelioma_xlsx.yaml (UCI Mesothelioma's
disease data set: sheet "Data", one row per patient, no ID column).

Written from the workbook's own Attributes and Description sheets and the UCI page, not from
the app's code. Read here with openpyxl. Every categorical is coded 0/1/2 with no labels in the
file; class of diagnosis is 1 healthy, 2 mesothelioma (Attributes sheet). From 'white blood' on,
the header row sits one column off the values, so no question uses those columns.
"""

from __future__ import annotations

from pathlib import Path

import openpyxl
import polars as pl

from clinical_analytics.eval.corpus import patient_kit as kit
from clinical_analytics.eval.corpus.score import OracleAnswer

SHEET = "Data"
DIAGNOSIS = "class of diagnosis"
MESOTHELIOMA = kit.text(DIAGNOSIS) == "2"


def scan(path: Path) -> pl.LazyFrame:
    """The Data sheet, every column as text."""
    book = openpyxl.load_workbook(path, read_only=True, data_only=True)
    try:
        rows = list(book[SHEET].iter_rows(values_only=True))
    finally:
        book.close()
    return kit.frame_from_rows(rows[0], rows[1:])


ORACLES: dict[str, kit.Oracle] = {
    "mesothelioma_xlsx-total-001": kit.count(),
    "mesothelioma_xlsx-diagnosis-002": kit.count_by(DIAGNOSIS),
    "mesothelioma_xlsx-asbestos-003": kit.count(kit.text("asbestos exposure") == "1"),
    "mesothelioma_xlsx-city-004": kit.count_by("city"),
    "mesothelioma_xlsx-duration-005": kit.stat_by("mean", "duration of asbestos exposure", DIAGNOSIS),
    "mesothelioma_xlsx-age-006": kit.stat_by("median", "age", "gender"),
    "mesothelioma_xlsx-dyspnoea-007": kit.rates(kit.text("dyspnoea") == "1", DIAGNOSIS),
    "mesothelioma_xlsx-older-008": kit.count((kit.number("age") > 65) & MESOTHELIOMA),
    "mesothelioma_xlsx-trend-009": kit.held(),
}


def answers(path: Path, oracles: dict[str, kit.Oracle] | None = None) -> dict[str, OracleAnswer]:
    return kit.answers(scan(path), oracles or ORACLES)
