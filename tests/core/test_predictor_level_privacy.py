"""Regression terms never name a level held by 1-10 patients (PR #96 leftover).

A logistic regression on a categorical predictor fits one indicator per level ("site_West"),
and the presented table and claim named every indicator: a level of 3 patients was shown under
its own label, with an odds ratio that rests on those 3 patients. Folding the level into "Other"
would change the model, so a model with such a level (or such an outcome value) is refused.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import polars as pl
import pytest

from clinical_analytics.analysis.compute import (
    _level_sizes,
    _term_sizes,
    compute_analysis_by_type,
    compute_predictor_analysis,
    compute_survival_analysis,
)
from clinical_analytics.analysis.grouping import (
    SMALL_CELL_MODEL_REFUSAL,
    SMALL_VALUE_MODEL_REFUSAL,
    UNRECORDED_SIZES_MODEL_REFUSAL,
    is_small_level,
)
from clinical_analytics.analysis.variable_types import classify_variable
from clinical_analytics.core.analysis_types import AnalysisContext, AnalysisIntent
from clinical_analytics.core.nl_query_config import SMALL_CELL_THRESHOLD
from clinical_analytics.core.query_plan import QueryPlan
from clinical_analytics.core.result_presentation import present_analysis
from clinical_analytics.eval.headless import predictor_label_leaks

_N = 60
CohortFactory = Callable[..., pl.DataFrame]


def _cohort(make_cohort_with_categorical: CohortFactory, west: int, events: int | None = None) -> pl.DataFrame:
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
def site_context(sample_context_predictor: AnalysisContext) -> AnalysisContext:
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
        ],
    )
    def test_computePredictor_categoricalLevelSize_refusedOnlyWhenSmall(
        self, make_cohort_with_categorical: CohortFactory, site_context: AnalysisContext, west: int, refused: bool
    ) -> None:
        # Arrange
        frame = _cohort(make_cohort_with_categorical, west)

        # Act
        result = compute_predictor_analysis(frame, site_context)

        # Assert
        assert refused
        assert "error" in result
        assert "West" not in result["error"]
        assert "summary" not in result
        assert f"fewer than {SMALL_CELL_THRESHOLD}" in result["error"]

    def test_computePredictor_largeLevelAndCells_notRefused(
        self, make_cohort_with_categorical: CohortFactory, site_context: AnalysisContext
    ) -> None:
        # Arrange: West 24 patients (12 Yes, 12 No); every other cell is larger
        frame = _two_by_two(
            make_cohort_with_categorical,
            {("West", "Yes"): 12, ("West", "No"): 12, ("East", "Yes"): 20, ("East", "No"): 20},
        )

        # Act
        result = compute_predictor_analysis(frame, site_context)

        # Assert
        assert "error" not in result
        assert "treatment_West" in result["summary"]
        assert result["term_sizes"]["treatment_West"] == 24

    def test_termSizes_levelAtThreshold_isNotSmall(self, make_cohort_with_categorical: CohortFactory) -> None:
        """The level rule's boundary on its own. A full model with a level of exactly the
        threshold is still refused: a yes/no outcome splits those patients into a cell of 1-10
        (``test_computePredictor_levelAtThreshold_refusedByOutcomeCell``)."""
        # Arrange
        frame = _cohort(make_cohort_with_categorical, SMALL_CELL_THRESHOLD)
        kinds = {"treatment": classify_variable(frame["treatment"])}

        # Act
        sizes = _term_sizes(frame, kinds, ["treatment"], None)

        # Assert
        assert sizes["treatment_West"] == SMALL_CELL_THRESHOLD
        assert not is_small_level(sizes["treatment_West"])

    def test_computePredictor_levelAtThreshold_refusedByOutcomeCell(
        self, make_cohort_with_categorical: CohortFactory, site_context: AnalysisContext
    ) -> None:
        # Arrange: 11 West patients, every third with status Yes, so West has 1-10 with each value
        frame = _cohort(make_cohort_with_categorical, SMALL_CELL_THRESHOLD)

        # Act
        result = compute_predictor_analysis(frame, site_context)

        # Assert
        assert result["error"] == SMALL_CELL_MODEL_REFUSAL
        assert "summary" not in result

    def test_computePredictor_smallReferenceLevel_refused(
        self, make_cohort_with_categorical: CohortFactory, site_context: AnalysisContext
    ) -> None:
        # Arrange: "Alpha" sorts first, so it would be the reference every odds ratio compares to.
        frame = _cohort(make_cohort_with_categorical, 3).with_columns(pl.col("treatment").replace({"West": "Alpha"}))

        # Act
        result = compute_predictor_analysis(frame, site_context)

        # Assert
        assert "error" in result
        assert "Alpha" not in result["error"]

    def test_computePredictor_binaryPredictorWithFewOnes_refused(
        self, make_cohort_with_categorical: CohortFactory, site_context: AnalysisContext
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
        self, make_cohort_with_categorical: CohortFactory, site_context: AnalysisContext
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
        self, make_cohort_with_categorical: CohortFactory, site_context: AnalysisContext
    ) -> None:
        # Arrange: every level-by-outcome cell has 25 patients (no small cell); West 50, East 40.
        frame = _two_by_two(
            make_cohort_with_categorical,
            {
                ("West", "Yes"): 25,
                ("West", "No"): 25,
                ("East", "Yes"): 20,
                ("East", "No"): 20,
                ("North", "Yes"): 30,
                ("North", "No"): 30,
            },
        )

        # Act
        result = compute_predictor_analysis(frame, site_context)

        # Assert: a level's term rests on the smaller of its level and the reference (East, 40);
        # a slope names no level (None).
        terms = {term for term in result["summary"] if term != "Intercept"}
        assert set(result["term_sizes"]) == terms
        assert result["term_sizes"]["age"] is None
        assert result["term_sizes"]["treatment_West"] == 40


@pytest.mark.parametrize(
    ("ones", "column"),
    [
        pytest.param(3, "event", id="one_value_small"),
        pytest.param(7, "event", id="both_values_small"),
    ],
)
def test_ambiguousEventMessage_smallValue_genericRefusalNamesNoColumn(
    make_cohort_with_categorical: CohortFactory, sample_context_survival: AnalysisContext, ones: int, column: str
) -> None:
    # Arrange: bare codes 1 and 2, so the event cannot be told; ``ones`` patients have code 1
    # (7 and 6 of 13 when both are small: n passes the floor, yet no value may be named).
    frame = _cohort(make_cohort_with_categorical, 0).with_columns(
        pl.when(pl.col("patient_id").str.slice(1).cast(pl.Int64) < ones).then(1).otherwise(2).alias(column),
        pl.col("age").alias("time"),
    )
    if ones == 7:
        frame = frame.head(13)
    sample_context_survival.time_variable = "time"
    sample_context_survival.event_variable = column
    sample_context_survival.grouping_variable = None

    # Act
    result = compute_survival_analysis(frame, sample_context_survival)

    # Assert
    assert result["error"] == SMALL_VALUE_MODEL_REFUSAL
    assert "values:" not in result["error"]


def test_ambiguousEventMessage_everyValueLarge_listsThem(
    make_cohort_with_categorical: CohortFactory, sample_context_survival: AnalysisContext
) -> None:
    # Arrange: bare codes 1 (20 patients) and 2 (40).
    frame = _cohort(make_cohort_with_categorical, 0).with_columns(
        pl.when(pl.col("patient_id").str.slice(1).cast(pl.Int64) < 20).then(1).otherwise(2).alias("event"),
        pl.col("age").alias("time"),
    )
    sample_context_survival.time_variable = "time"
    sample_context_survival.event_variable = "event"
    sample_context_survival.grouping_variable = None

    # Act
    result = compute_survival_analysis(frame, sample_context_survival)

    # Assert
    assert "(values: 1, 2)" in result["error"]


# Every outcome cell of site_West and its reference holds 40 patients; age is a slope.
_LARGE_CELLS: dict[str, Any] = {"age": None, "site_West": 40}


def _predictor_payload(term_sizes: dict[str, Any] | None, cell_sizes: Any = _LARGE_CELLS) -> dict[str, Any]:
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
    if cell_sizes is not None:
        payload["cell_sizes"] = cell_sizes
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

    @pytest.mark.parametrize(
        ("term_sizes", "expected"),
        [
            pytest.param({"age": None, "site_West": 3}, SMALL_VALUE_MODEL_REFUSAL, id="small_term"),
            pytest.param({"age": None, "site_West": "3"}, SMALL_VALUE_MODEL_REFUSAL, id="size_not_a_count"),
            pytest.param({"age": None}, UNRECORDED_SIZES_MODEL_REFUSAL, id="term_absent"),
            pytest.param(None, UNRECORDED_SIZES_MODEL_REFUSAL, id="no_sizes"),
            pytest.param([3], UNRECORDED_SIZES_MODEL_REFUSAL, id="sizes_not_a_mapping"),
        ],
    )
    def test_presentPredictor_withheld_saysWhy(self, term_sizes: Any, expected: str) -> None:
        # Arrange: a legacy payload has no sizes; that is not a small value, so it is not blamed on one.
        analysis = _predictor_payload(None)
        if term_sizes is not None:
            analysis["term_sizes"] = term_sizes

        # Act
        presentation = present_analysis(analysis)

        # Assert
        assert presentation["table"] is None
        assert expected in (presentation["claim"] or "")
        assert "status" not in (presentation["claim"] or "")

    @pytest.mark.parametrize(
        ("cell_sizes", "expected"),
        [
            pytest.param({"age": None, "site_West": 9}, SMALL_CELL_MODEL_REFUSAL, id="small_cell"),
            pytest.param({"age": None, "site_West": 1}, SMALL_CELL_MODEL_REFUSAL, id="one_patient_cell"),
            pytest.param(
                {"age": None, "site_West": SMALL_CELL_THRESHOLD - 1}, SMALL_CELL_MODEL_REFUSAL, id="just_under"
            ),
            pytest.param({"age": None, "site_West": -9}, SMALL_CELL_MODEL_REFUSAL, id="negative_not_a_count"),
            pytest.param({"age": None, "site_West": None}, UNRECORDED_SIZES_MODEL_REFUSAL, id="level_term_as_slope"),
            pytest.param({"age": 40, "site_West": 40}, UNRECORDED_SIZES_MODEL_REFUSAL, id="slope_with_cells"),
            pytest.param({"age": None}, UNRECORDED_SIZES_MODEL_REFUSAL, id="term_absent"),
            pytest.param(None, UNRECORDED_SIZES_MODEL_REFUSAL, id="no_cell_sizes"),
            pytest.param({"age": None, "site_West": SMALL_CELL_THRESHOLD}, None, id="at_threshold_shown"),
        ],
    )
    def test_presentPredictor_cellSizes_withheldWhenACellIsSmall(self, cell_sizes: Any, expected: str | None) -> None:
        """A payload whose levels are large but whose level-by-outcome cell is small (one that
        skipped compute's check) is withheld by the presenter, by the same rule."""
        # Arrange: every level is large, so term_sizes alone would show the model
        analysis = _predictor_payload({"age": None, "site_West": 40}, cell_sizes)

        # Act
        presentation = present_analysis(analysis)

        # Assert
        if expected is None:
            assert presentation["table"] is not None
            assert "West" in str(presentation["table"])
        else:
            assert presentation["table"] is None
            assert presentation["claim"] == expected
            assert "West" not in presentation["claim"]

    def test_presentPredictor_computedPayload_sameRuleAsCompute(
        self, make_cohort_with_categorical: CohortFactory, site_context: AnalysisContext
    ) -> None:
        # Arrange: a model compute shows; then its cell sizes as a payload that skipped the check
        frame = _two_by_two(
            make_cohort_with_categorical,
            {("West", "Yes"): 12, ("West", "No"): 12, ("East", "Yes"): 20, ("East", "No"): 20},
        )
        shown = compute_predictor_analysis(frame, site_context)
        forged = {**shown, "cell_sizes": {**shown["cell_sizes"], "treatment_West": 9}}

        # Act
        shown_presentation = present_analysis(shown)
        forged_presentation = present_analysis(forged)

        # Assert
        assert shown["cell_sizes"] == {"age": None, "treatment_West": 12}
        assert shown_presentation["table"] is not None
        assert forged_presentation == {"claim": SMALL_CELL_MODEL_REFUSAL, "table": None}


class TestPredictorLabelLeaks:
    def test_predictorLabelLeaks_termOfSmallLevelShown_flagsIt(
        self, make_cohort_with_categorical: CohortFactory
    ) -> None:
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

        # Assert: treatment_North is flagged too, as it is compared with East, whose 10 events
        # are a cell of 1-10 (the cell rule)
        assert leaks == {"treatment_West", "treatment_North"}

    @pytest.mark.parametrize(
        "presentation",
        [
            pytest.param({"claim": "The model is not shown.", "table": None}, id="refused"),
            pytest.param({"claim": None, "table": None}, id="nothing"),
        ],
    )
    def test_predictorLabelLeaks_noSmallLabelShown_returnsEmpty(
        self, make_cohort_with_categorical: CohortFactory, presentation: dict[str, Any]
    ) -> None:
        # Arrange
        frame = _cohort(make_cohort_with_categorical, 3)

        # Act
        leaks = predictor_label_leaks(presentation, frame, ["age", "treatment"], "status", SMALL_CELL_THRESHOLD)

        # Assert
        assert leaks == set()

    def test_predictorLabelLeaks_smallOutcomeValueInClaim_flagsIt(
        self, make_cohort_with_categorical: CohortFactory
    ) -> None:
        # Arrange: 3 patients have status Yes, and the claim names that value.
        frame = _cohort(make_cohort_with_categorical, SMALL_CELL_THRESHOLD, events=3)
        presentation = {"claim": "Logistic regression of status = Yes on age (n=60 patients).", "table": None}

        # Act
        leaks = predictor_label_leaks(presentation, frame, ["age"], "status", SMALL_CELL_THRESHOLD)

        # Assert
        assert leaks == {"Yes"}

    def test_predictorLabelLeaks_levelSmallOnlyInCompleteCases_flagsIt(
        self, make_cohort_with_categorical: CohortFactory
    ) -> None:
        # Arrange: West has 12 rows, 3 of them with no age, so the model rests on 9 West patients.
        frame = _cohort(make_cohort_with_categorical, 12).with_columns(
            pl.when(pl.col("patient_id").is_in(["P000", "P001", "P002"]))
            .then(None)
            .otherwise(pl.col("age"))
            .alias("age")
        )
        presentation = {
            "claim": "Logistic regression of status = Yes on age, treatment_North, treatment_West (n=57 patients).",
            "table": None,
        }

        # Act
        leaks = predictor_label_leaks(presentation, frame, ["age", "treatment"], "status", SMALL_CELL_THRESHOLD)

        # Assert: treatment_North is flagged too, as East (its reference) has a cell of 1-10
        assert leaks == {"treatment_West", "treatment_North"}

    @pytest.mark.parametrize(
        ("events", "shown_term", "expected"),
        [
            # Reference Comorbid (sorts first): its cell of 9 is behind treatment_None's odds ratio.
            pytest.param((9, 6), "treatment_None", {"treatment_None"}, id="both_cells_small"),
            pytest.param(
                (SMALL_CELL_THRESHOLD - 1, 40), "treatment_None", {"treatment_None"}, id="reference_cell_small"
            ),
            pytest.param((40, SMALL_CELL_THRESHOLD - 1), "treatment_None", {"treatment_None"}, id="level_cell_small"),
            pytest.param((40, 40), "treatment_None", set(), id="no_small_cell"),
            pytest.param((SMALL_CELL_THRESHOLD, SMALL_CELL_THRESHOLD), "treatment_None", set(), id="at_threshold"),
        ],
    )
    def test_predictorLabelLeaks_smallLevelByOutcomeCellShown_flagsIt(
        self,
        make_cohort_with_categorical: CohortFactory,
        events: tuple[int, int],
        shown_term: str,
        expected: set[str],
    ) -> None:
        # Arrange: both levels (328 and 813) and the event total are large; one cell may be 1-10
        frame = _two_by_two(
            make_cohort_with_categorical,
            {
                ("Comorbid", "Yes"): events[0],
                ("Comorbid", "No"): 328 - events[0],
                ("None", "Yes"): events[1],
                ("None", "No"): 813 - events[1],
            },
        )
        presentation = {
            "claim": f"{shown_term} had the strongest evidence. Logistic regression of status = Yes on {shown_term}.",
            "table": {"columns": ["variable", "odds ratio"], "rows": [{"variable": shown_term, "odds ratio": 2.4}]},
        }

        # Act
        leaks = predictor_label_leaks(presentation, frame, ["treatment"], "status", SMALL_CELL_THRESHOLD)

        # Assert
        assert leaks == expected

    def test_predictorLabelLeaks_twoValuedPredictorSmallCell_flagsColumn(
        self, make_cohort_with_categorical: CohortFactory
    ) -> None:
        # Arrange: dose 5 or 10 mg is fitted as one term named by its column; 9 of 328 at 5 mg had the event
        frame = _two_by_two(
            make_cohort_with_categorical,
            {("A", "Yes"): 9, ("A", "No"): 319, ("B", "Yes"): 40, ("B", "No"): 773},
        ).with_columns(pl.when(pl.col("treatment") == "A").then(5).otherwise(10).alias("dose"))
        presentation = {
            "claim": "dose had the strongest evidence. Logistic regression of status = Yes on dose.",
            "table": None,
        }

        # Act
        leaks = predictor_label_leaks(presentation, frame, ["dose"], "status", SMALL_CELL_THRESHOLD)

        # Assert
        assert leaks == {"dose"}

    def test_predictorLabelLeaks_slopeWithManyValues_notFlaggedForCells(
        self, make_cohort_with_categorical: CohortFactory
    ) -> None:
        # Arrange: age has many values, each a cell of a few rows, but a slope names no level
        frame = _two_by_two(
            make_cohort_with_categorical,
            {("West", "Yes"): 30, ("West", "No"): 30, ("East", "Yes"): 30, ("East", "No"): 30},
        )
        presentation = {
            "claim": "age had the strongest evidence. Logistic regression of status = Yes on age.",
            "table": None,
        }

        # Act
        leaks = predictor_label_leaks(presentation, frame, ["age"], "status", SMALL_CELL_THRESHOLD)

        # Assert
        assert leaks == set()


def _visits(frame: pl.DataFrame, patients: list[str], repeats: int) -> pl.DataFrame:
    """``frame`` with each of ``patients``' rows repeated as ``repeats`` visits in all."""
    extra = frame.filter(pl.col("patient_id").is_in(patients))
    return pl.concat([frame, *([extra] * (repeats - 1))])


class TestComputePredictorPatientsNotRows:
    @pytest.mark.parametrize(
        ("west", "visits", "refused"),
        [
            pytest.param(3, 4, True, id="three_patients_twelve_visits"),
            pytest.param(SMALL_CELL_THRESHOLD - 1, 2, True, id="just_under_threshold_patients"),
            # 11 West patients (level not small) as 22 visit rows: West's outcome cells hold 1-10
            # patients, so the model is withheld by the cell rule (rows would refuse here too; the
            # patients-not-rows path is TestComputePredictorOutcomeCells' visit case).
            pytest.param(SMALL_CELL_THRESHOLD, 2, True, id="at_threshold_level_small_cell"),
        ],
    )
    def test_computePredictor_visitRows_countsPatientsPerLevel(
        self,
        make_cohort_with_categorical: CohortFactory,
        site_context: AnalysisContext,
        west: int,
        visits: int,
        refused: bool,
    ) -> None:
        # Arrange: each West patient has ``visits`` rows.
        frame = _visits(_cohort(make_cohort_with_categorical, west), [f"P{i:03d}" for i in range(west)], visits)

        # Act
        result = compute_predictor_analysis(frame, site_context, entity_key="patient_id")

        # Assert
        if refused:
            # A small level is refused by the level rule; a level of 11 by its outcome cells.
            small_level = west < SMALL_CELL_THRESHOLD
            assert result["error"] == (SMALL_VALUE_MODEL_REFUSAL if small_level else SMALL_CELL_MODEL_REFUSAL)
        else:
            assert "error" not in result
            assert result["term_sizes"]["treatment_West"] == west

    def test_computePredictor_visitRowsOutcome_countsPatients(
        self, make_cohort_with_categorical: CohortFactory, site_context: AnalysisContext
    ) -> None:
        # Arrange: 3 patients have status Yes, over 12 visits.
        frame = _visits(_cohort(make_cohort_with_categorical, 20, events=3), ["P000", "P001", "P002"], 4)
        site_context.predictor_variables = ["age"]

        # Act
        result = compute_predictor_analysis(frame, site_context, entity_key="patient_id")

        # Assert
        assert result["error"] == SMALL_VALUE_MODEL_REFUSAL

    def test_computePredictor_entityKeyNotInFrame_refused(
        self, make_cohort_with_categorical: CohortFactory, site_context: AnalysisContext
    ) -> None:
        # Arrange: patients cannot be counted without their key, so no size can be trusted.
        frame = _cohort(make_cohort_with_categorical, 20).drop("patient_id")

        # Act
        result = compute_predictor_analysis(frame, site_context, entity_key="patient_id")

        # Assert
        assert "error" in result
        assert "summary" not in result

    def test_computeAnalysisByType_entityKey_reachesPredictor(
        self, make_cohort_with_categorical: CohortFactory, site_context: AnalysisContext
    ) -> None:
        # Arrange
        frame = _visits(_cohort(make_cohort_with_categorical, 3), ["P000", "P001", "P002"], 4)

        # Act
        result = compute_analysis_by_type(frame, site_context, entity_key="patient_id")

        # Assert
        assert result["error"] == SMALL_VALUE_MODEL_REFUSAL

    def test_formatExecutionResult_visitLayer_countsPatients(
        self, make_cohort_with_categorical: CohortFactory, make_semantic_layer: Callable[..., Any]
    ) -> None:
        # Arrange: the layer counts patients (rows are declared visits); West: 3 patients, 12 rows.
        frame = _visits(_cohort(make_cohort_with_categorical, 3), ["P000", "P001", "P002"], 4)
        layer = make_semantic_layer(dataset_name="visits", data=frame)
        layer.count_entity_key = "patient_id"
        plan = QueryPlan(intent="FIND_PREDICTORS", metric="status", confidence=0.99)
        context = AnalysisContext(primary_variable="status", predictor_variables=["age", "treatment"])
        context.inferred_intent = AnalysisIntent.FIND_PREDICTORS
        context.query_plan = plan

        # Act
        execution = layer.execute_query_plan(plan, confidence_threshold=0.0)
        analysis = layer.format_execution_result(execution, context)

        # Assert
        assert analysis["error"] == SMALL_VALUE_MODEL_REFUSAL


class TestPredictorRefusalNamesNothing:
    @pytest.mark.parametrize(
        ("predictors", "outcome_events", "hidden"),
        [
            pytest.param(["age", "hiv_positive"], None, "hiv_positive", id="binary_predictor_few_ones"),
            pytest.param(["age"], 3, "status", id="outcome_few_events"),
            pytest.param(["age", "treatment"], None, "treatment", id="categorical_small_level"),
        ],
    )
    def test_computePredictor_smallValue_genericRefusal(
        self,
        make_cohort_with_categorical: CohortFactory,
        site_context: AnalysisContext,
        predictors: list[str],
        outcome_events: int | None,
        hidden: str,
    ) -> None:
        # Arrange: hiv_positive 1 for 4 patients; West 3 patients; status Yes for 3 when asked.
        west = 3 if "treatment" in predictors else SMALL_CELL_THRESHOLD
        frame = _cohort(make_cohort_with_categorical, west, events=outcome_events).with_columns(
            (pl.col("patient_id").str.slice(1).cast(pl.Int64) < 4).cast(pl.Int64).alias("hiv_positive")
        )
        site_context.predictor_variables = predictors

        # Act
        result = compute_predictor_analysis(frame, site_context)

        # Assert
        assert result["error"] == SMALL_VALUE_MODEL_REFUSAL
        assert hidden not in result["error"]


def _two_by_two(make_cohort_with_categorical: CohortFactory, cells: dict[tuple[str, str], int]) -> pl.DataFrame:
    """One patient per row, with (treatment level, status) counts as given."""
    rows = [key for key, count in cells.items() for _ in range(count)]
    return make_cohort_with_categorical(
        patient_ids=[f"P{index:05d}" for index in range(len(rows))],
        treatment=[level for level, _status in rows],
        status=[status for _level, status in rows],
        ages=[30 + (index * 13) % 40 for index in range(len(rows))],
    )


class TestComputePredictorOutcomeCells:
    """Every level and the event total can be large while a level's events are few: 328 with a
    comorbidity (9 admitted) and 813 without (6 admitted). The odds ratio and its CI, with the
    level and event totals shown elsewhere, give back 9 and 6. The model is withheld then."""

    @pytest.mark.parametrize(
        ("events_a", "events_b", "refused"),
        [
            pytest.param(9, 6, True, id="both_cells_small"),
            pytest.param(SMALL_CELL_THRESHOLD - 1, 40, True, id="one_events_cell_small"),
            pytest.param(40, 40, False, id="no_small_cell"),
            pytest.param(SMALL_CELL_THRESHOLD, SMALL_CELL_THRESHOLD, False, id="at_threshold"),
        ],
    )
    def test_computePredictor_smallPredictorOutcomeCell_refused(
        self,
        make_cohort_with_categorical: CohortFactory,
        site_context: AnalysisContext,
        events_a: int,
        events_b: int,
        refused: bool,
    ) -> None:
        # Arrange: both levels and the event total are at least the threshold
        frame = _two_by_two(
            make_cohort_with_categorical,
            {
                ("Comorbid", "Yes"): events_a,
                ("Comorbid", "No"): 328 - events_a,
                ("None", "Yes"): events_b,
                ("None", "No"): 813 - events_b,
            },
        )
        site_context.predictor_variables = ["treatment"]

        # Act
        result = compute_predictor_analysis(frame, site_context)

        # Assert
        assert ("summary" not in result) is refused
        if refused:
            assert result["error"] == SMALL_CELL_MODEL_REFUSAL

    @pytest.mark.parametrize(
        ("levels", "events", "refused"),
        [
            pytest.param((5, 10), (9, 6), True, id="dose_5_10_mg"),
            pytest.param((0, 3), (4, 7), True, id="grade_0_3"),
            pytest.param((5, 10), (40, 40), False, id="dose_no_small_cell"),
        ],
    )
    def test_computePredictor_twoValuedNumericPredictor_cellsChecked(
        self,
        make_cohort_with_categorical: CohortFactory,
        site_context: AnalysisContext,
        levels: tuple[int, int],
        events: tuple[int, int],
        refused: bool,
    ) -> None:
        """A numeric predictor with two values is fitted as a slope, but slope and intercept
        give the two levels' cells back, as a 0/1 term does."""
        # Arrange
        frame = _two_by_two(
            make_cohort_with_categorical,
            {
                ("A", "Yes"): events[0],
                ("A", "No"): 328 - events[0],
                ("B", "Yes"): events[1],
                ("B", "No"): 813 - events[1],
            },
        ).with_columns(pl.when(pl.col("treatment") == "A").then(levels[0]).otherwise(levels[1]).alias("dose"))
        site_context.predictor_variables = ["dose"]

        # Act
        result = compute_predictor_analysis(frame, site_context)

        # Assert
        assert ("summary" not in result) is refused
        if refused:
            assert result["error"] == SMALL_CELL_MODEL_REFUSAL
        else:
            assert result["term_sizes"]["dose"] == 328

    def test_computePredictor_nullPredictorRows_cellsCountCompleteCasesOnly(
        self, make_cohort_with_categorical: CohortFactory, site_context: AnalysisContext
    ) -> None:
        # Arrange: Comorbid has 8 events among complete cases; 6 more events have no treatment
        frame = _two_by_two(
            make_cohort_with_categorical,
            {
                ("Comorbid", "Yes"): 8,
                ("Comorbid", "No"): 320,
                ("None", "Yes"): 40,
                ("None", "No"): 773,
                ("missing", "Yes"): 6,
            },
        ).with_columns(
            pl.when(pl.col("treatment") == "missing").then(None).otherwise(pl.col("treatment")).alias("treatment")
        )
        site_context.predictor_variables = ["treatment"]

        # Act
        result = compute_predictor_analysis(frame, site_context)

        # Assert: the 6 incomplete rows do not lift the 8 above the threshold
        assert result["error"] == SMALL_CELL_MODEL_REFUSAL

    def test_computePredictor_emptyCell_isNotACellRefusal(
        self, make_cohort_with_categorical: CohortFactory, site_context: AnalysisContext
    ) -> None:
        # Arrange: no Comorbid patient had the event (0 is a level nobody has, not a small one)
        frame = _two_by_two(
            make_cohort_with_categorical,
            {("Comorbid", "No"): 328, ("None", "Yes"): 40, ("None", "No"): 773},
        )
        site_context.predictor_variables = ["treatment"]

        # Act
        result = compute_predictor_analysis(frame, site_context)

        # Assert: whatever the fit does with an empty cell, the cell rule does not fire
        assert result.get("error") != SMALL_CELL_MODEL_REFUSAL

    def test_computePredictor_visitRows_cellCountsPatientsNotRows(
        self, make_cohort_with_categorical: CohortFactory, site_context: AnalysisContext
    ) -> None:
        # Arrange: 6 Comorbid patients with the event, 2 visits each (12 rows, 6 patients)
        frame = _two_by_two(
            make_cohort_with_categorical,
            {("Comorbid", "Yes"): 6, ("Comorbid", "No"): 322, ("None", "Yes"): 40, ("None", "No"): 773},
        )
        repeated = frame.filter((pl.col("treatment") == "Comorbid") & (pl.col("status") == "Yes"))
        visits = pl.concat([frame, repeated])
        site_context.predictor_variables = ["treatment"]

        # Act
        result = compute_predictor_analysis(visits, site_context, entity_key="patient_id")

        # Assert: counted as rows the cell is 12 (not small); as patients it is 6
        assert result["error"] == SMALL_CELL_MODEL_REFUSAL

    def test_computePredictor_visitRows_largeCells_termSizeCountsPatients(
        self, make_cohort_with_categorical: CohortFactory, site_context: AnalysisContext
    ) -> None:
        # Arrange: every cell large; each Comorbid patient has 2 visits
        frame = _two_by_two(
            make_cohort_with_categorical,
            {("Comorbid", "Yes"): 20, ("Comorbid", "No"): 30, ("None", "Yes"): 40, ("None", "No"): 60},
        )
        visits = pl.concat([frame, frame.filter(pl.col("treatment") == "Comorbid")])
        site_context.predictor_variables = ["treatment"]

        # Act
        result = compute_predictor_analysis(visits, site_context, entity_key="patient_id")

        # Assert: 50 Comorbid patients (100 rows); the reference None has 100
        assert "error" not in result
        assert result["term_sizes"]["treatment_None"] == 50


class TestComputePredictorUnkeyedRows:
    """A visit row with no patient key is still someone. Dropping those rows let a cell of 9
    unkeyed visits count 0 patients, which is not small, so the model was shown and its
    intercept gave the 9 events back."""

    @pytest.mark.parametrize(
        ("keys", "expected"),
        [
            pytest.param(["P1", "P2", "P3"], 3, id="all_keyed"),
            pytest.param([None, None, None], 1, id="all_unkeyed_counts_one"),
            pytest.param(["P1", None, None], 1, id="keyed_and_unkeyed_counts_keyed"),
            pytest.param(["P1", "P1", None], 1, id="repeat_key_and_unkeyed"),
        ],
    )
    def test_levelSizes_unkeyedRows_neverCountZero(self, keys: list[str | None], expected: int) -> None:
        # Arrange
        values = pl.Series("level", ["A", "A", "A"])

        # Act
        sizes = _level_sizes(values, pl.Series("key", keys, dtype=pl.Utf8))

        # Assert
        assert sizes == {"A": expected}

    @pytest.mark.parametrize(
        ("keyed_events", "unkeyed_events", "refused"),
        [
            pytest.param(0, 9, True, id="cell_all_unkeyed"),
            pytest.param(5, 10, True, id="cell_keyed_small_plus_unkeyed"),
            pytest.param(20, 3, False, id="cell_keyed_large_plus_unkeyed"),
        ],
    )
    def test_computePredictor_unkeyedVisitRowsInCell_failClosed(
        self,
        make_cohort_with_categorical: CohortFactory,
        site_context: AnalysisContext,
        keyed_events: int,
        unkeyed_events: int,
        refused: bool,
    ) -> None:
        # Arrange: Comorbid 328 rows; its event rows marked "unkeyed" lose their patient key
        frame = _two_by_two(
            make_cohort_with_categorical,
            {
                ("Comorbid", "Yes"): keyed_events,
                ("unkeyed", "Yes"): unkeyed_events,
                ("Comorbid", "No"): 328 - keyed_events - unkeyed_events,
                ("None", "Yes"): 40,
                ("None", "No"): 773,
            },
        ).with_columns(
            pl.when(pl.col("treatment") == "unkeyed").then(None).otherwise(pl.col("patient_id")).alias("patient_id"),
            pl.col("treatment").replace({"unkeyed": "Comorbid"}),
        )
        site_context.predictor_variables = ["treatment"]

        # Act
        result = compute_predictor_analysis(frame, site_context, entity_key="patient_id")

        # Assert
        assert ("summary" not in result) is refused
        if refused:
            assert result["error"] == SMALL_CELL_MODEL_REFUSAL
        else:
            assert result["cell_sizes"] == {"treatment_None": keyed_events}

    def test_computePredictor_unkeyedRowsOnlyLevel_refused(
        self, make_cohort_with_categorical: CohortFactory, site_context: AnalysisContext
    ) -> None:
        # Arrange: every West row (30 visits) has no patient key: West could be one patient
        frame = _two_by_two(
            make_cohort_with_categorical,
            {("West", "Yes"): 15, ("West", "No"): 15, ("East", "Yes"): 40, ("East", "No"): 40},
        ).with_columns(
            pl.when(pl.col("treatment") == "West").then(None).otherwise(pl.col("patient_id")).alias("patient_id")
        )

        # Act
        result = compute_predictor_analysis(frame, site_context, entity_key="patient_id")

        # Assert
        assert result["error"] == SMALL_VALUE_MODEL_REFUSAL
        assert "summary" not in result
