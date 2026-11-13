"""Regression terms never name a level held by 1-10 patients (PR #96 leftover).

A logistic regression on a categorical predictor fits one indicator per level ("site_West"),
and the presented table and claim named every indicator: a level of 3 patients was shown under
its own label, with an odds ratio that rests on those 3 patients. Folding the level into "Other"
would change the model, so a model with such a level (or such an outcome value) is refused.
"""

from __future__ import annotations

from typing import Any

import polars as pl
import pytest

from clinical_analytics.analysis.compute import compute_predictor_analysis, compute_survival_analysis
from clinical_analytics.core.nl_query_config import SMALL_CELL_THRESHOLD
from clinical_analytics.core.result_presentation import present_analysis
from clinical_analytics.eval.headless import predictor_label_leaks

_N = 60


def _cohort(make_cohort_with_categorical: Any, west: int, events: int | None = None) -> pl.DataFrame:
    """60 patients: ``west`` at site West, the rest East or North; status Yes for every third
    (or the first ``events``), so no level separates the outcome."""
    yes = [index % 3 == 0 if events is None else index < events for index in range(_N)]
    return make_cohort_with_categorical(
        patient_ids=[f"P{index:03d}" for index in range(_N)],
        treatment=["West" if index < west else ("East" if index % 2 else "North") for index in range(_N)],
        status=["Yes" if flag else "No" for flag in yes],
        ages=[30 + (index * 13) % 40 for index in range(_N)],
    )


@pytest.fixture
def site_context(sample_context_predictor: Any) -> Any:
    sample_context_predictor.primary_variable = "status"
    sample_context_predictor.predictor_variables = ["age", "treatment"]
    return sample_context_predictor


class TestComputePredictorSmallLevels:
    @pytest.mark.parametrize(
        ("west", "refused"),
        [
            pytest.param(3, True, id="three_patients"),
            pytest.param(1, True, id="one_patient"),
            pytest.param(SMALL_CELL_THRESHOLD - 1, True, id="just_under_threshold"),
            pytest.param(SMALL_CELL_THRESHOLD, False, id="at_threshold"),
        ],
    )
    def test_computePredictor_categoricalLevelSize_refusedOnlyWhenSmall(
        self, make_cohort_with_categorical: Any, site_context: Any, west: int, refused: bool
    ) -> None:
        # Arrange
        frame = _cohort(make_cohort_with_categorical, west)

        # Act
        result = compute_predictor_analysis(frame, site_context)

        # Assert
        if refused:
            assert "error" in result
            assert "West" not in result["error"]
            assert "summary" not in result
            assert f"fewer than {SMALL_CELL_THRESHOLD}" in result["error"]
        else:
            assert "error" not in result
            assert "treatment_West" in result["summary"]
            assert result["term_sizes"]["treatment_West"] == SMALL_CELL_THRESHOLD

    def test_computePredictor_smallReferenceLevel_refused(
        self, make_cohort_with_categorical: Any, site_context: Any
    ) -> None:
        # Arrange: "Alpha" sorts first, so it would be the reference every odds ratio compares to.
        frame = _cohort(make_cohort_with_categorical, 3).with_columns(pl.col("treatment").replace({"West": "Alpha"}))

        # Act
        result = compute_predictor_analysis(frame, site_context)

        # Assert
        assert "error" in result
        assert "Alpha" not in result["error"]

    def test_computePredictor_binaryPredictorWithFewOnes_refused(
        self, make_cohort_with_categorical: Any, site_context: Any
    ) -> None:
        # Arrange: a 0/1 indicator whose 1 is held by 3 patients; its term names that level.
        frame = _cohort(make_cohort_with_categorical, SMALL_CELL_THRESHOLD).with_columns(
            (pl.col("patient_id").str.slice(1).cast(pl.Int64) < 3).cast(pl.Int64).alias("smoker")
        )
        site_context.predictor_variables = ["age", "smoker"]

        # Act
        result = compute_predictor_analysis(frame, site_context)

        # Assert
        assert "error" in result
        assert "summary" not in result

    def test_computePredictor_outcomeWithFewEvents_refused(
        self, make_cohort_with_categorical: Any, site_context: Any
    ) -> None:
        # Arrange: 3 patients have status Yes.
        frame = _cohort(make_cohort_with_categorical, SMALL_CELL_THRESHOLD, events=3)
        site_context.predictor_variables = ["age"]

        # Act
        result = compute_predictor_analysis(frame, site_context)

        # Assert
        assert "error" in result
        assert "summary" not in result

    def test_computePredictor_noSmallLevel_reportsEveryTermSize(
        self, make_cohort_with_categorical: Any, site_context: Any
    ) -> None:
        # Arrange
        frame = _cohort(make_cohort_with_categorical, 20)

        # Act
        result = compute_predictor_analysis(frame, site_context)

        # Assert: a level's term rests on the smaller of its level and the reference (East, 20);
        # a slope names no level (None).
        terms = {term for term in result["summary"] if term != "Intercept"}
        assert set(result["term_sizes"]) == terms
        assert result["term_sizes"]["age"] is None
        assert result["term_sizes"]["treatment_West"] == 20


def test_ambiguousEventMessage_smallValue_notNamed(
    make_cohort_with_categorical: Any, sample_context_survival: Any
) -> None:
    # Arrange: bare codes 1 (3 patients) and 2, so the event cannot be told and the values are listed.
    frame = _cohort(make_cohort_with_categorical, 0).with_columns(
        pl.when(pl.col("patient_id").str.slice(1).cast(pl.Int64) < 3).then(1).otherwise(2).alias("event"),
        pl.col("age").alias("time"),
    )
    sample_context_survival.time_variable = "time"
    sample_context_survival.event_variable = "event"
    sample_context_survival.grouping_variable = None

    # Act
    result = compute_survival_analysis(frame, sample_context_survival)

    # Assert
    assert f"(values: 2, and values held by fewer than {SMALL_CELL_THRESHOLD} patients)" in result["error"]


def _predictor_payload(term_sizes: dict[str, Any] | None) -> dict[str, Any]:
    stats = {"Odds Ratio": 2.4, "CI Lower": 0.1, "CI Upper": 46.1, "P-Value": 0.01}
    payload: dict[str, Any] = {
        "type": "predictor",
        "outcome_col": "status",
        "event_level": "Yes",
        "n_used": _N,
        "n_excluded": 0,
        "summary": {"Intercept": stats, "age": stats, "site_West": stats},
        "significant_predictors": [
            {"variable": "site_West", "odds_ratio": 2.4, "p_value": 0.01, "ci_lower": 0.1, "ci_upper": 46.1}
        ],
    }
    if term_sizes is not None:
        payload["term_sizes"] = term_sizes
    return payload


class TestPresentPredictorSmallTerms:
    @pytest.mark.parametrize(
        ("term_sizes", "shown"),
        [
            pytest.param({"age": None, "site_West": 3}, False, id="small_term"),
            pytest.param({"age": None, "site_West": SMALL_CELL_THRESHOLD - 1}, False, id="just_under_threshold"),
            pytest.param({"age": None, "site_West": "3"}, False, id="size_not_a_count"),
            pytest.param({"age": None}, False, id="term_absent"),
            pytest.param(None, False, id="no_sizes"),
            pytest.param({"age": None, "site_West": SMALL_CELL_THRESHOLD}, True, id="at_threshold"),
        ],
    )
    def test_presentPredictor_termSize_namesTermOnlyWhenNotSmall(
        self, term_sizes: dict[str, Any] | None, shown: bool
    ) -> None:
        # Arrange
        analysis = _predictor_payload(term_sizes)

        # Act
        presentation = present_analysis(analysis)

        # Assert
        named = "West" in (presentation["claim"] or "") or "West" in str(presentation["table"])
        assert named is shown
        if not shown:
            assert presentation["table"] is None
            assert f"fewer than {SMALL_CELL_THRESHOLD}" in (presentation["claim"] or "")


class TestPredictorLabelLeaks:
    def test_predictorLabelLeaks_termOfSmallLevelShown_flagsIt(self, make_cohort_with_categorical: Any) -> None:
        # Arrange: the presentation main gave for a model with a 3-patient level.
        frame = _cohort(make_cohort_with_categorical, 3)
        presentation = {
            "claim": "age had the strongest evidence. Logistic regression of status = Yes on age, treatment_North, "
            "treatment_West (n=60 patients).",
            "table": {
                "columns": ["variable", "odds ratio"],
                "rows": [
                    {"variable": "treatment_North", "odds ratio": 0.9},
                    {"variable": "treatment_West", "odds ratio": 2.4},
                ],
            },
        }

        # Act
        leaks = predictor_label_leaks(presentation, frame, ["age", "treatment"], "status", SMALL_CELL_THRESHOLD)

        # Assert
        assert leaks == {"treatment_West"}

    @pytest.mark.parametrize(
        "presentation",
        [
            pytest.param({"claim": "The model is not shown.", "table": None}, id="refused"),
            pytest.param({"claim": None, "table": None}, id="nothing"),
        ],
    )
    def test_predictorLabelLeaks_noSmallLabelShown_returnsEmpty(
        self, make_cohort_with_categorical: Any, presentation: dict[str, Any]
    ) -> None:
        # Arrange
        frame = _cohort(make_cohort_with_categorical, 3)

        # Act
        leaks = predictor_label_leaks(presentation, frame, ["age", "treatment"], "status", SMALL_CELL_THRESHOLD)

        # Assert
        assert leaks == set()

    def test_predictorLabelLeaks_smallOutcomeValueInClaim_flagsIt(self, make_cohort_with_categorical: Any) -> None:
        # Arrange: 3 patients have status Yes, and the claim names that value.
        frame = _cohort(make_cohort_with_categorical, SMALL_CELL_THRESHOLD, events=3)
        presentation = {"claim": "Logistic regression of status = Yes on age (n=60 patients).", "table": None}

        # Act
        leaks = predictor_label_leaks(presentation, frame, ["age"], "status", SMALL_CELL_THRESHOLD)

        # Assert
        assert leaks == {"Yes"}
