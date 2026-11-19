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
    NOT_CHECKED_REASON,
    SHOWN_COUNTS_DIRNAME,
    UNAVAILABLE_REASON,
    WITHHELD_REASON,
    FileShownCountLog,
)


def _ask(service: QueryService, plan: QueryPlan, version: str = "v1", dataset_id: str = "probe") -> Any:
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
    result = service.ask("q", dataset_id=dataset_id, dataset_version=version)
    if result.requires_confirmation:
        result = service.ask(
            "q", dataset_id=dataset_id, dataset_version=version, confirmed=True, confirm_plan_hash=result.plan_hash
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

    def test_ask_sameFileUploadedTwice_sharesOneLog(self, make_semantic_layer: Any, tmp_path: Path) -> None:
        # Arrange: the review's case. The same rows under two upload ids and versions
        first = make_semantic_layer(data=_patients(), workspace_name="first")
        second = make_semantic_layer(data=_patients(), workspace_name="second")
        _ask(QueryService(first, data_dir=tmp_path), _count(CohortStep("le", "age", 65)), "v1", "upload-a")

        # Act
        result = _ask(QueryService(second, data_dir=tmp_path), _count(CohortStep("ge", "age", 65)), "v9", "upload-b")

        # Assert
        assert _withheld(result)

    def test_ask_columnsRenamed_sameRows_sharesOneLog(self, make_semantic_layer: Any, tmp_path: Path) -> None:
        # Arrange: the same rows with a column relabelled
        first = make_semantic_layer(data=_patients(), workspace_name="first")
        second = make_semantic_layer(data=_patients().rename({"age": "age_years"}), workspace_name="second")
        _ask(QueryService(first, data_dir=tmp_path), _count(CohortStep("le", "age", 65)))

        # Act
        result = _ask(QueryService(second, data_dir=tmp_path), _count(CohortStep("ge", "age_years", 65)))

        # Assert
        assert _withheld(result)

    def test_ask_otherRows_startFromAnEmptyLog(self, make_semantic_layer: Any, tmp_path: Path) -> None:
        # Arrange: a corrected upload (one value changed) is other rows
        corrected = _patients().with_columns(
            pl.when(pl.int_range(pl.len()) == 0).then(41).otherwise(pl.col("age")).alias("age")
        )
        _ask(
            QueryService(make_semantic_layer(data=_patients(), workspace_name="a"), data_dir=tmp_path),
            _count(CohortStep("le", "age", 65)),
        )

        # Act
        result = _ask(
            QueryService(make_semantic_layer(data=corrected, workspace_name="b"), data_dir=tmp_path),
            _count(CohortStep("ge", "age", 65)),
        )

        # Assert
        assert result.claim is not None

    def test_ask_unreadableLog_withholdsAnswers(self, make_semantic_layer: Any, tmp_path: Path) -> None:
        # Arrange: a log damaged on disk
        service = QueryService(make_semantic_layer(data=_patients()), data_dir=tmp_path)
        _ask(service, _count(CohortStep("le", "age", 65)))
        (path,) = (tmp_path / SHOWN_COUNTS_DIRNAME).glob("*.json")
        path.write_text(json.dumps({"format": 2}))

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


class TestReviewFollowUps:
    def test_ask_comparisonFilteredOnItsOwnGroup_answered(self, make_semantic_layer: Any) -> None:
        # Arrange: compute applies the filter on the grouping column again (the SQL cohort skips
        # it); the rebuilt cohort must do the same, or the answer is withheld as unchecked
        from clinical_analytics.core.query_plan import FilterSpec

        frame = pl.DataFrame(
            {
                "patient_id": [f"P{i:03d}" for i in range(150)],
                "arm": ["A", "B", "C"] * 50,
                "score": [float(10 + (i * 7) % 23) for i in range(150)],
            }
        )
        service = QueryService(make_semantic_layer(data=frame))
        plan = QueryPlan(
            intent="COMPARE_GROUPS",
            metric="score",
            group_by="arm",
            filters=[FilterSpec("arm", "IN", ["A", "B"])],
            scope="filtered",
            confidence=0.99,
        )

        # Act
        result = _ask(service, plan)

        # Assert
        assert result.failure_reason is None
        assert result.claim is not None

    @pytest.mark.parametrize(("missing", "withheld"), [(3, True), (30, False)])
    def test_ask_distinctIdsWithMissingIds_rebuiltWithoutThem(
        self, make_semantic_layer: Any, missing: int, withheld: bool
    ) -> None:
        # Arrange: one row per patient, some rows with no ID; a count of distinct IDs leaves them
        # out. Beside the row count the app shows, 3 missing IDs are a small count (withheld);
        # 30 are not, and the rebuilt set must match the count shown (no "can't be checked")
        frame = pl.DataFrame(
            {"patient_id": [f"P{i:03d}" for i in range(100 - missing)] + [None] * missing, "age": list(range(100))},
            schema={"patient_id": pl.String, "age": pl.Int64},
        )
        service = QueryService(make_semantic_layer(data=frame))

        # Act
        result = _ask(service, QueryPlan(intent="COUNT", entity_key="patient_id", confidence=0.99))

        # Assert
        assert _withheld(result) is withheld
        assert (result.claim is not None) is not withheld

    def test_ask_logRaises_nothingOfTheAnswerReturned(self, make_semantic_layer: Any) -> None:
        # Arrange: the check fails in a way no handler names
        service = QueryService(make_semantic_layer(data=_patients()))

        def broken(*_args: object) -> None:
            raise RuntimeError("unexpected")

        service.shown_log.admit = broken  # type: ignore[method-assign]

        # Act
        result = _ask(service, _count(group_by="sex"))

        # Assert
        assert (result.analysis, result.result, result.claim) == (None, None, None)

    def test_ask_presenterListsNoCounts_withheldAsNotChecked(
        self, make_semantic_layer: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange: a presenter that does not say what it shows
        from clinical_analytics.core import query_service, result_presentation

        def unlisted(analysis: Any) -> Any:
            presentation, _counts = result_presentation.present_with_counts(analysis)
            return presentation, None

        monkeypatch.setattr(query_service, "present_with_counts", unlisted)
        service = QueryService(make_semantic_layer(data=_patients()))

        # Act
        result = _ask(service, _count(group_by="sex"))

        # Assert
        assert result.analysis is None
        assert result.failure_reason == NOT_CHECKED_REASON


def _union_cohort(diabetes: bool = True) -> pl.DataFrame:
    """The review's case: 60 women over 65, 60 women 65 or under, 75 men over 65, 5 men 65 or under;
    diabetes splits each group (the 5 men into 3 and 2)."""
    rows: list[dict[str, Any]] = []
    for n, age, sex, with_dm in ((60, 70, "F", 25), (60, 50, "F", 30), (75, 72, "M", 40), (5, 55, "M", 3)):
        for k in range(n):
            rows.append(
                {"patient_id": f"P{len(rows):03d}", "age": age, "sex": sex, "diabetes": "yes" if k < with_dm else "no"}
            )
    return pl.DataFrame(rows)


def _ask_words(service: QueryService, question: str) -> Any:
    """Ask through the real parser, confirming a hold once (as a doctor pressing Confirm)."""
    result = service.ask(question, dataset_id="probe", dataset_version="v1")
    if result.requires_confirmation:
        result = service.ask(
            question, dataset_id="probe", dataset_version="v1", confirmed=True, confirm_plan_hash=result.plan_hash
        )
    return result


class TestUnionsOfAtoms:
    @pytest.mark.parametrize("diabetes_first", [True, False])
    def test_ask_totalMinusTwoGroupsPlusTheirOverlap_withheld(
        self, make_semantic_layer: Any, diabetes_first: bool
    ) -> None:
        # Arrange: 200 - 135 (over 65) - 120 (women) + 60 (women over 65) = 5 men of 65 or under.
        # Asked first, "diabetes" splits those 5 into atoms of 3 and 2
        service = QueryService(make_semantic_layer(data=_union_cohort()))
        earlier = [_count(), _count(CohortStep("gt", "age", 65)), _count(CohortStep("eq", "sex", "F"))]
        if diabetes_first:
            earlier.insert(1, _count(CohortStep("eq", "diabetes", "yes")))
        shown = [_ask(service, plan) for plan in earlier]

        # Act
        result = _ask(service, _count(CohortStep("eq", "sex", "F"), CohortStep("gt", "age", 65)))

        # Assert
        assert all(r.claim is not None for r in shown)
        assert _withheld(result)

    def test_ask_unionAttackThroughTheParser_withheld(self, make_semantic_layer: Any) -> None:
        # Arrange: the same questions in words, as the review asked them
        service = QueryService(make_semantic_layer(data=_union_cohort()))
        questions = [
            "how many patients?",
            "how many patients have diabetes yes?",
            "how many patients over 65?",
            "how many women?",
        ]
        shown = [_ask_words(service, question).claim for question in questions]

        # Act
        result = _ask_words(service, "how many women over 65?")

        # Assert
        assert None not in shown
        assert result.failure_reason == WITHHELD_REASON


class TestOtherIds:
    def test_ask_distinctOtherIdBesideALoggedCount_withheld(self, make_semantic_layer: Any) -> None:
        # Arrange: the review's case. "65 or younger" is logged; distinct mrn aged 65+ (60) would
        # give the 5 aged exactly 65
        frame = _patients().with_columns(("M" + pl.col("patient_id").str.slice(1)).alias("mrn"))
        service = QueryService(make_semantic_layer(data=frame))
        _ask(service, _count(CohortStep("le", "age", 65)))

        # Act
        result = _ask(
            service,
            QueryPlan(intent="COUNT", entity_key="mrn", cohort_steps=[CohortStep("ge", "age", 65)], confidence=0.99),
        )

        # Assert
        assert _withheld(result)

    def test_ask_admissionsBesidePatients_withheld(self, make_semantic_layer: Any) -> None:
        # Arrange: 100 patients, one readmitted: 101 distinct admission IDs over visit rows
        frame = pl.DataFrame(
            {
                "patient_id": [f"P{i:03d}" for i in range(100)] + ["P000"],
                "encounter_id": [f"A{i:03d}" for i in range(101)],
            }
        )
        service = QueryService(make_semantic_layer(data=frame))
        patients = _ask(service, _count())

        # Act
        result = _ask(service, QueryPlan(intent="COUNT", entity_key="encounter_id", confidence=0.99))

        # Assert
        assert patients.claim is not None
        assert _withheld(result)


class TestPositionColumns:
    @pytest.mark.parametrize("name", ["rowid", "file_row_number", "ROWID"])
    def test_ask_columnNamedLikeARowNumber_rowsStillNumberedByContent(
        self, make_semantic_layer: Any, name: str
    ) -> None:
        # Arrange: a dataset column that would shadow DuckDB's rowid or read_parquet's
        # file_row_number (here constant, so a shadowed number would put every row at one place)
        frame = _patients().with_columns(pl.lit(7).alias(name))
        service = QueryService(make_semantic_layer(data=frame))
        first = _ask(service, _count(CohortStep("le", "age", 65)))

        # Act
        result = _ask(service, _count(CohortStep("ge", "age", 65)))

        # Assert
        assert first.claim is not None
        assert _withheld(result)
