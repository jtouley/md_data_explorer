# SPDX-License-Identifier: MIT
"""Golden samples of the model-written query path: plan replies replayed through the real
check-compile-execute layer on synthetic tables (tests/eval/model_query_golden.yaml).

No model runs. Every answered value is recomputed by an independent Polars oracle, and every
reading card passes the card linter.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import polars as pl
import pytest
import yaml

from clinical_analytics.model_query.card import card_problems
from tests.model_query import tables

GOLDEN = yaml.safe_load((Path(__file__).parent / "model_query_golden.yaml").read_text(encoding="utf-8"))["samples"]
TABLES: dict[str, tuple[Callable[[], pl.DataFrame], str]] = {
    "shots": (tables.shots, tables.SHOTS_DICTIONARY),
    "cohort": (tables.cohort, tables.COHORT_DICTIONARY),
}


def _latest() -> pl.DataFrame:
    return tables.shots().sort("shot_round").group_by("holder_ref").last()


ORACLES: dict[str, Callable[[], Any]] = {
    "cohort_total": lambda: tables.cohort().height,
    "cohort_under_65": lambda: tables.cohort().filter(pl.col("age") < 65).height,
    "cohort_stage3_alive_share": lambda: tables.cohort().filter(pl.col("stage") == "III")["died_1y"].eq("no").mean(),
    "cohort_over70_stage2": lambda: tables.cohort().filter((pl.col("age") > 70) & (pl.col("stage") == "II")).height,
    "shots_latest_extract": lambda: dict(_latest().group_by("extract").len().iter_rows()),
    "shots_never_wheezed": lambda: (
        tables.shots().group_by("holder_ref").agg(pl.col("wheeze").eq("N").all())["wheeze"].sum()
    ),
    # The first recorded value per patient (one rule for first and last, ADR016).
    "shots_baseline_ige": lambda: (
        tables.shots().drop_nulls("ige_kul").sort("shot_round").group_by("holder_ref").first()["ige_kul"].mean()
    ),
}
_EXTRACT_CODES = {"grass pollen": 1, "dust mite": 2, "bee venom": 3}


def _headline(analysis: dict[str, Any]) -> Any:
    if analysis.get("grouped_by"):
        return {
            _EXTRACT_CODES.get(g[analysis["grouped_by"]], g[analysis["grouped_by"]]): g["count"]
            for g in analysis["group_counts"]
        }
    if analysis["type"] == "proportion":
        return analysis["proportion"]
    if analysis["type"] == "descriptive":
        return analysis[analysis["statistic"]]
    return analysis["total_count"]


@pytest.mark.parametrize("sample", GOLDEN, ids=[s["id"] for s in GOLDEN])
def test_golden_sample_replayed_matchesOracle(make_query_table, make_model_ask, sample):
    # Arrange
    frame, dictionary = TABLES[sample["table"]]
    table = make_query_table(frame(), doc_context=dictionary)
    ask = make_model_ask(table)

    # Act
    answer, prompts = ask(sample["question"], *sample["replies"])

    # Assert
    assert answer.status == sample["status"], answer.reason
    assert len(prompts) == len(sample["replies"])
    lines = (answer.reading or "").splitlines()
    assert card_problems(lines, table[1]) == []
    for line in sample.get("reading", []):
        assert line in lines
    if sample["status"] == "answered":
        assert _headline(answer.analysis) == pytest.approx(ORACLES[sample["oracle"]]())
