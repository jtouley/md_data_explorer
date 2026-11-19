# SPDX-License-Identifier: MIT
"""Governed SQL and the log of shown counts (ADR013 §12): answers combined across questions, and
across the SQL path and the question box, on one dataset version's log. Synthetic data."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import polars as pl
import pytest

from clinical_analytics.core.query_plan import CohortStep, QueryPlan
from clinical_analytics.core.query_service import QueryService
from clinical_analytics.core.shown_counts import WITHHELD_REASON, MemoryShownCountLog
from clinical_analytics.sql.catalog import SqlCatalog
from clinical_analytics.sql.errors import SqlRefusedError
from clinical_analytics.sql.privacy import ShownLogTarget, SqlAnswer, answer_query
from clinical_analytics.sql.sandbox import SandboxConnection
from clinical_analytics.sql.validate import validate_sql


def _cohort() -> pl.DataFrame:
    """120 patients: ages 40-64 (60), 66-85 (55) and exactly 65 (5); half women."""
    return pl.DataFrame(
        {
            "patient_id": [f"P{i:03d}" for i in range(120)],
            "age": [(40 + i % 25) if i < 60 else (66 + i % 20) if i < 115 else 65 for i in range(120)],
            "sex": ["F" if i % 2 else "M" for i in range(120)],
        }
    )


def _sql(sql: str, catalog: SqlCatalog, log: MemoryShownCountLog) -> SqlAnswer:
    query = validate_sql(sql, catalog)
    with SandboxConnection(catalog) as sandbox:
        return answer_query(query, sandbox, catalog, shown_log=ShownLogTarget(log, "probe", "v1"))


class TestSqlAcrossQuestions:
    def test_answerQuery_overlappingWhereThenTheirUnion_withheld(
        self, make_sql_catalog: Callable[..., SqlCatalog]
    ) -> None:
        # Arrange: "65 or younger" (65) and "65 or older" (60) are shown; every patient is in one
        # of them, so with the total (120) they give the 5 aged exactly 65
        catalog, log = make_sql_catalog(_cohort()), MemoryShownCountLog()
        _sql("SELECT count(*) AS n FROM unified WHERE age <= 65", catalog, log)

        # Act / Assert
        with pytest.raises(SqlRefusedError) as refused:
            _sql("SELECT count(*) AS n FROM unified WHERE age >= 65", catalog, log)
        assert refused.value.reason == WITHHELD_REASON

    def test_answerQuery_groupThenFilterDifferingBy5_withheld(
        self, make_sql_catalog: Callable[..., SqlCatalog]
    ) -> None:
        # Arrange: women by sex (60), then women aged 65 or under minus... the women of 65+: the
        # grouped answer and a filtered one differing by the 5 aged exactly 65 among all
        catalog, log = make_sql_catalog(_cohort()), MemoryShownCountLog()
        _sql("SELECT count(*) AS n FROM unified WHERE age > 65", catalog, log)

        # Act / Assert
        with pytest.raises(SqlRefusedError):
            _sql("SELECT sex, count(*) AS n FROM unified WHERE age >= 65 GROUP BY sex", catalog, log)

    def test_answerQuery_ordinaryQuestions_allShown(self, make_sql_catalog: Callable[..., SqlCatalog]) -> None:
        # Arrange
        catalog, log = make_sql_catalog(_cohort()), MemoryShownCountLog()
        questions = [
            "SELECT count(*) AS n FROM unified",
            "SELECT sex, count(*) AS n FROM unified GROUP BY sex",
            "SELECT count(*) AS n FROM unified WHERE age > 65",
            "SELECT sex, count(*) AS n FROM unified WHERE age > 65 GROUP BY sex",
            "SELECT sex, avg(age) AS mean_age FROM unified GROUP BY sex",
        ]

        # Act
        answers = [_sql(sql, catalog, log) for sql in questions]

        # Assert
        assert all(answer.rows for answer in answers)


def _ask_box(service: QueryService, plan: QueryPlan) -> Any:
    intent = SimpleNamespace(
        confidence=0.99,
        intent_type=plan.intent,
        primary_variable=plan.metric,
        grouping_variable=plan.group_by,
        predictor_variables=[],
        time_variable=None,
        event_variable=None,
        filters=[],
    )
    service.nl_engine.parse_query = lambda *args, **kwargs: intent  # type: ignore[method-assign]
    service.nl_engine._intent_to_plan = lambda *args: plan  # type: ignore[method-assign]
    service._accounting_hold_reason = lambda *args: None  # type: ignore[method-assign]
    return service.ask("q", dataset_id="probe", dataset_version="v1")


class TestAcrossPaths:
    def test_questionBoxThenSql_sameLog_withheld(
        self, make_sql_catalog: Callable[..., SqlCatalog], make_semantic_layer: Any
    ) -> None:
        # Arrange: the question box and SQL read the same stored Parquet and share one log
        catalog, log = make_sql_catalog(_cohort()), MemoryShownCountLog()
        layer = make_semantic_layer(
            data=_cohort(), config_overrides={"init_params": {"source_path": str(Path(catalog.parquet_path))}}
        )
        shown = _ask_box(
            QueryService(layer, shown_log=log),
            QueryPlan(intent="COUNT", cohort_steps=[CohortStep("le", "age", 65)], confidence=0.99),
        )

        # Act / Assert
        with pytest.raises(SqlRefusedError) as refused:
            _sql("SELECT count(*) AS n FROM unified WHERE age >= 65", catalog, log)
        assert shown.claim is not None
        assert refused.value.reason == WITHHELD_REASON


class TestLimit:
    def test_answerQuery_groupCutByLimit_notLogged(
        self, make_sql_catalog: Callable[..., SqlCatalog], make_semantic_layer: Any
    ) -> None:
        # Arrange: wards A (70), B (30, 5 of them aged 20) and C (20); LIMIT 1 shows ward A only
        frame = pl.DataFrame(
            {
                "patient_id": [f"P{i:03d}" for i in range(120)],
                "ward": ["A"] * 70 + ["B"] * 30 + ["C"] * 20,
                "age": [50] * 70 + [20] * 5 + [50] * 45,
            }
        )
        catalog, log = make_sql_catalog(frame), MemoryShownCountLog()
        layer = make_semantic_layer(
            data=frame, config_overrides={"init_params": {"source_path": str(catalog.parquet_path)}}
        )
        top = _sql("SELECT ward, count(*) AS n FROM unified GROUP BY ward ORDER BY n DESC LIMIT 1", catalog, log)

        # Act: ward B over 30 (25) would differ by 5 from ward B, which was never shown
        answer = _ask_box(
            QueryService(layer, shown_log=log),
            QueryPlan(
                intent="COUNT",
                cohort_steps=[CohortStep("eq", "ward", "B"), CohortStep("gt", "age", 30)],
                confidence=0.99,
            ),
        )

        # Assert
        assert [row["ward"] for row in top.rows] == ["A"]
        assert answer.failure_reason is None
        assert answer.claim is not None
