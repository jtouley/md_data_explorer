"""Ground truth for tests/eval/corpus/questions/hcv_data.yaml (UCI HCV data, one row per patient).

Written from the UCI variable list, not from the app's code. Category is a code=label string
("3=Cirrhosis"); Sex is "m"/"f"; labs are numbers with "NA" for missing.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from clinical_analytics.eval.corpus import patient_kit as kit
from clinical_analytics.eval.corpus.score import OracleAnswer

CIRRHOSIS = kit.is_value("Category", "3=Cirrhosis")

ORACLES: dict[str, kit.Oracle] = {
    "hcv_data-total-001": kit.count(),
    "hcv_data-category-002": kit.count_by("Category"),
    "hcv_data-sex-003": kit.count_by("Sex"),
    "hcv_data-alt-004": kit.stat_by("median", "ALT", "Category"),
    "hcv_data-ast-005": kit.stat_by("mean", "AST", "Sex"),
    "hcv_data-cirrhosis-006": kit.rates(CIRRHOSIS, "Sex"),
    "hcv_data-older-007": kit.count((kit.number("Age") > 50) & CIRRHOSIS),
    "hcv_data-crea-008": kit.count(kit.number("CREA") > 150),
    "hcv_data-trend-009": kit.held(),
    "hcv_data-listing-010": kit.held("listing", CIRRHOSIS),
}


def answers(path: Path, oracles: dict[str, kit.Oracle] | None = None) -> dict[str, OracleAnswer]:
    return kit.answers(pl.scan_csv(path, infer_schema=False), oracles or ORACLES)
