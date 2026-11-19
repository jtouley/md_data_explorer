# SPDX-License-Identifier: MIT
"""Cross-question attacks on the question box, blocked by the log of shown counts (ADR013 §12).

PR #141's reviews worked out counts of 1-10 patients by combining answers. These are the attacks
main can already produce: overlapping filters, a group beside a filtered count, "ever" counts of
two values beside the total, and visit rows beside their patients. Each is asked through
QueryService with the plan fixed (the parser is not under test), on synthetic data.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import polars as pl
import pytest

from clinical_analytics.core.query_plan import CohortStep, QueryPlan
from clinical_analytics.core.query_service import QueryService
from clinical_analytics.core.shown_counts import (
    SHOWN_COUNTS_DIRNAME,
    UNAVAILABLE_REASON,
    WITHHELD_REASON,
    FileShownCountLog,
)


def _ask(service: QueryService, plan: QueryPlan, version: str = "v1") -> Any:
    """Ask with the plan fixed and word accounting passed; confirm a hold once (the grain)."""
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
    result = service.ask("q", dataset_id="probe", dataset_version=version)
    if result.requires_confirmation:
        result = service.ask(
            "q", dataset_id="probe", dataset_version=version, confirmed=True, confirm_plan_hash=result.plan_hash
        )
    return result


def _count(*steps: CohortStep, group_by: str | None = None) -> QueryPlan:
    return QueryPlan(intent="COUNT", cohort_steps=list(steps), group_by=group_by, confidence=0.99)


def _withheld(result: Any) -> bool:
    return result.analysis is None and result.failure_reason == WITHHELD_REASON


def _patients(n: int = 120) -> pl.DataFrame:
    """One row per patient: ages 40-64 (60 patients), 66-85 (55) and exactly 65 (5); half women,
    5 of the women over 90 instead of their age."""
    return pl.DataFrame(
        {
            "patient_id": [f"P{i:03d}" for i in range(n)],
            "age": [(40 + i % 25) if i < 60 else (66 + i % 20) if i < 115 else 65 for i in range(n)],
            "sex": ["F" if i % 2 else "M" for i in range(n)],
        }
    ).with_columns(
        pl.when((pl.col("sex") == "F") & (pl.int_range(pl.len()) < 10)).then(95).otherwise(pl.col("age")).alias("age")
    )


def _visits_ever(changed: int) -> pl.DataFrame:
    """100 patients with two visits: 40 positive at both, ``changed`` positive then negative, the
    rest negative at both."""
    rows = []
    for i in range(100):
        first, second = ("pos", "pos") if i < 40 else (("pos", "neg") if i < 40 + changed else ("neg", "neg"))
        rows += [{"patient_id": f"P{i:03d}", "screen": first}, {"patient_id": f"P{i:03d}", "screen": second}]
    return pl.DataFrame(rows)


class TestOverlappingFilters:
    def test_ask_ageUpTo65ThenFrom65_secondWithheld(self, make_semantic_layer: Any) -> None:
        # Arrange: "65 or younger" (65) and "65 or older" (60) overlap in the 5 aged exactly 65
        service = QueryService(make_semantic_layer(data=_patients()))
        first = _ask(service, _count(CohortStep("le", "age", 65)))

        # Act
        second = _ask(service, _count(CohortStep("ge", "age", 65)))

        # Assert
        assert first.claim is not None
        assert _withheld(second)
        assert second.result is None and second.claim is None

    def test_ask_thresholdMovedByOneYear_withheld(self, make_semantic_layer: Any) -> None:
        # Arrange: "over 65" then "65 and over" differ by the 5 aged 65
        service = QueryService(make_semantic_layer(data=_patients()))
        _ask(service, _count(CohortStep("gt", "age", 65)))

        # Act
        result = _ask(service, _count(CohortStep("ge", "age", 65)))

        # Assert
        assert _withheld(result)


class TestGroupBesideFilteredCount:
    def test_ask_womenBySexThenWomenUnder90_withheld(self, make_semantic_layer: Any) -> None:
        # Arrange: the by-sex table shows 60 women; "women aged 90 or under" would show 55
        service = QueryService(make_semantic_layer(data=_patients()))
        by_sex = _ask(service, _count(group_by="sex"))

        # Act
        result = _ask(service, _count(CohortStep("eq", "sex", "F"), CohortStep("le", "age", 90)))

        # Assert
        assert by_sex.claim is not None
        assert _withheld(result)

    def test_ask_filteredCountThenGroups_withheld(self, make_semantic_layer: Any) -> None:
        # Arrange: the same two answers, the filtered count first
        service = QueryService(make_semantic_layer(data=_patients()))
        _ask(service, _count(CohortStep("eq", "sex", "F"), CohortStep("le", "age", 90)))

        # Act
        result = _ask(service, _count(group_by="sex"))

        # Assert
        assert _withheld(result)


class TestEverCountsBesideTotal:
    @pytest.mark.parametrize(("changed", "withheld"), [(5, True), (10, True), (11, False), (0, False)])
    def test_ask_everPositiveThenEverNegative_withheldWhenChanged1To10(
        self, make_semantic_layer: Any, changed: int, withheld: bool
    ) -> None:
        # Arrange: patients (visit rows declared), then "ever positive"; ever pos + ever neg - total
        # is the patients whose result changed between visits
        service = QueryService(make_semantic_layer(data=_visits_ever(changed)))
        _ask(service, _count())
        _ask(service, _count(CohortStep("eq", "screen", "pos")))

        # Act
        result = _ask(service, _count(CohortStep("eq", "screen", "neg")))

        # Assert
        assert _withheld(result) is withheld


class TestVisitRowsBesidePatients:
    def test_ask_patientsThenMeanOverVisitRows_withheld(self, make_semantic_layer: Any) -> None:
        # Arrange: 100 patients, one of them readmitted: n of a statistic over visit rows is 101
        frame = pl.DataFrame(
            {"patient_id": [f"P{i:03d}" for i in range(100)] + ["P000"], "los": [float(i % 9) for i in range(101)]}
        )
        service = QueryService(make_semantic_layer(data=frame))
        patients = _ask(service, _count())

        # Act
        result = _ask(service, QueryPlan(intent="DESCRIBE", metric="los", confidence=0.99))

        # Assert
        assert patients.claim is not None
        assert _withheld(result)


class TestOrdinaryQuestions:
    def test_ask_typicalSequence_everyAnswerShown(
        self, make_cohort_with_categorical: Any, make_semantic_layer: Any
    ) -> None:
        # Arrange: a session of common questions on 400 patients
        cohort = make_cohort_with_categorical(
            patient_ids=[f"P{i:03d}" for i in range(400)],
            treatment=["Yes" if i % 3 else "No" for i in range(400)],
            status=["active" if i % 4 else "inactive" for i in range(400)],
            ages=[30 + (i * 7) % 50 for i in range(400)],
        )
        service = QueryService(make_semantic_layer(data=cohort))
        plans = [
            _count(),
            _count(group_by="treatment"),
            _count(group_by="status"),
            _count(CohortStep("gt", "age", 65)),
            _count(CohortStep("gt", "age", 65), group_by="treatment"),
            _count(CohortStep("eq", "treatment", "Yes")),
            QueryPlan(intent="DESCRIBE", metric="age", confidence=0.99),
            QueryPlan(intent="COUNT", proportion_column="status", proportion_value="active", confidence=0.99),
            QueryPlan(intent="COUNT", rate_column="status", rate_value="active", group_by="treatment", confidence=0.99),
        ]

        # Act
        results = [_ask(service, plan) for plan in plans]

        # Assert
        assert [r.failure_reason for r in results] == [None] * len(plans)


class TestAcrossRestartsAndVersions:
    def test_ask_afterRestart_stillWithheld(self, make_semantic_layer: Any, tmp_path: Path) -> None:
        # Arrange: one app run shows "65 or younger"; a new layer and service (a restart) on the
        # same data directory
        data_dir = tmp_path / "data"
        layer = make_semantic_layer(data=_patients(), workspace_name="first")
        _ask(QueryService(layer, data_dir=data_dir), _count(CohortStep("le", "age", 65)))
        restarted = QueryService(make_semantic_layer(data=_patients(), workspace_name="second"), data_dir=data_dir)

        # Act
        result = _ask(restarted, _count(CohortStep("ge", "age", 65)))

        # Assert
        assert _withheld(result)

    def test_ask_newVersion_startsFromAnEmptyLog(self, make_semantic_layer: Any, tmp_path: Path) -> None:
        # Arrange
        service = QueryService(make_semantic_layer(data=_patients()), data_dir=tmp_path)
        _ask(service, _count(CohortStep("le", "age", 65)), version="v1")

        # Act
        result = _ask(service, _count(CohortStep("ge", "age", 65)), version="v2")

        # Assert
        assert result.claim is not None

    def test_ask_unreadableLog_withholdsAnswers(self, make_semantic_layer: Any, tmp_path: Path) -> None:
        # Arrange: a log damaged on disk
        service = QueryService(make_semantic_layer(data=_patients()), data_dir=tmp_path)
        _ask(service, _count(CohortStep("le", "age", 65)))
        (path,) = (tmp_path / SHOWN_COUNTS_DIRNAME / "probe").glob("*.json")
        path.write_text(json.dumps({"format": 1}))

        # Act
        result = _ask(service, _count(group_by="sex"))

        # Assert
        assert result.analysis is None
        assert result.failure_reason == UNAVAILABLE_REASON

    def test_service_withDataDir_usesTheFileLog(self, make_semantic_layer: Any, tmp_path: Path) -> None:
        # Arrange / Act
        service = QueryService(make_semantic_layer(data=_patients()), data_dir=tmp_path)

        # Assert
        assert isinstance(service.shown_log, FileShownCountLog)
