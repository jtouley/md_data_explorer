"""A column that behaves like categories is checked as a table, however many stray values it has.

PRs #133 (a model's term and cell sizes) and #138 (a test or correlation of a table of counts)
treated a numeric column as coded only when it had at most two values, so one stray value
switched their checks off:

* a dose of 0 (500 patients, 100 events) or 1 (500 patients, 5 events) was refused, but with one
  patient at 0.5 the model was shown (OR 0.04, CI 0.02-0.10), and the OR with the margins gives
  the 5 back;
* a 0/1 outcome compared by Welch's t-test with one value changed to 0.5 showed B's mean 0.075 of
  100 patients, which pins 7;
* two 0/1 columns with a cell of 5 and 21 stray exposure values gave r = -0.30, whose phi with
  the margins gives about 6.

One rule now decides ("behaves like categories", ``grouping.behaves_like_categories``): at most
``MAX_GROUP_LEVELS`` values, or all but ``MAX_STRAY_SHARE`` of the patients on that many, and
(with three or more values) not mostly one-patient values. Such a column's levels and cells are
counted in patients and checked by ``is_small_level``. The presenter withholds a comparison of
means whose cells were not recorded, and at visit grain group sizes and shown cells count
patients, not visits.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import date
from typing import Any

import numpy as np
import polars as pl
import pytest

from clinical_analytics.analysis.compute import compute_analysis_by_type
from clinical_analytics.analysis.grouping import (
    MAX_GROUP_LEVELS,
    NO_COUNT_TABLE,
    SMALL_CELL_MODEL_REFUSAL,
    SMALL_CELL_TEST_REFUSAL,
    SMALL_VALUE_MODEL_REFUSAL,
    UNRECORDED_CELLS_TEST_REFUSAL,
    behaves_like_categories,
)
from clinical_analytics.core.analysis_types import AnalysisContext, AnalysisIntent
from clinical_analytics.core.nl_query_config import SMALL_CELL_THRESHOLD
from clinical_analytics.core.result_presentation import SECONDARY_SUPPRESSED, present_analysis
from clinical_analytics.eval.headless import predictor_label_leaks

CohortFactory = Callable[..., pl.DataFrame]


def _frame(make_cohort_with_categorical: CohortFactory, columns: dict[str, list[Any]]) -> pl.DataFrame:
    """A cohort from the shared factory (one patient per row) with ``columns`` added."""
    n = len(next(iter(columns.values())))
    base = make_cohort_with_categorical(
        patient_ids=[f"P{i:05d}" for i in range(n)],
        treatment=["arm"] * n,
        status=["active"] * n,
        ages=[30 + i % 50 for i in range(n)],
    )
    return base.with_columns([pl.Series(name, values) for name, values in columns.items()])


def _dose_columns(
    stray: Sequence[tuple[float | None, int]] = (), low_events: int = 100, high_events: int = 5
) -> dict[str, list[Any]]:
    """Dose 0 for 500 patients (``low_events`` events) and 1 for 500 (``high_events``), plus a
    (dose, event) row for each ``stray``."""
    dose = [0.0] * 500 + [1.0] * 500 + [d for d, _ in stray]
    event = [int(i < low_events) for i in range(500)] + [int(i < high_events) for i in range(500)]
    return {"dose": dose, "event": event + [e for _, e in stray]}


def _context(intent: AnalysisIntent, **fields: Any) -> AnalysisContext:
    context = AnalysisContext(inferred_intent=intent)
    for name, value in fields.items():
        setattr(context, name, value)
    return context


_MODEL = {"primary_variable": "event", "predictor_variables": ["dose"]}


class TestBehavesLikeCategories:
    @pytest.mark.parametrize(
        ("sizes", "expected"),
        [
            pytest.param({0: 500, 1: 500}, True, id="two_values"),
            pytest.param({0: 500, 1: 500, 0.5: 1}, True, id="one_stray_value"),
            pytest.param({v: 50 for v in range(MAX_GROUP_LEVELS)}, True, id="twenty_levels"),
            pytest.param({v: 50 for v in range(MAX_GROUP_LEVELS + 1)}, True, id="twenty_one_levels_share_under_5pct"),
            pytest.param({v: 50 for v in range(MAX_GROUP_LEVELS + 2)}, False, id="twenty_two_levels_share_over_5pct"),
            pytest.param({0: 489, 1: 490, **{0.01 * k: 1 for k in range(1, 22)}}, True, id="21_strays_in_1000"),
            pytest.param({0: 400, 1: 400, **{0.001 * k: 1 for k in range(1, 201)}}, False, id="200_strays_in_1000"),
            pytest.param({v: 1 for v in range(12)}, False, id="every_value_one_patient"),
            pytest.param({0: 1, 1: 1}, True, id="two_values_always_categories"),
            pytest.param({0: 3, 1: 3, 2: 1, 3: 1}, True, id="few_singletons"),
            pytest.param({}, True, id="empty"),
            pytest.param({v: "1" for v in range(30)}, True, id="sizes_not_counts_fail_closed"),
            pytest.param({v: None for v in range(30)}, True, id="null_sizes_fail_closed"),
            pytest.param({**{v: 50 for v in range(30)}, 99: -5}, True, id="negative_size_fail_closed"),
        ],
    )
    def test_behavesLikeCategories_sizes_followShareRule(self, sizes: dict[Any, int], expected: bool) -> None:
        # Arrange / Act
        result = behaves_like_categories(sizes)

        # Assert
        assert result is expected


class TestPredictorStrayValue:
    def test_predictor_twoValuedDoseWithSmallCell_isRefused(self, make_cohort_with_categorical: CohortFactory) -> None:
        # Arrange: 5 events at dose 1 (the case #133 already refused)
        frame = _frame(make_cohort_with_categorical, _dose_columns())

        # Act
        result = compute_analysis_by_type(frame, _context(AnalysisIntent.FIND_PREDICTORS, **_MODEL))

        # Assert
        assert result["error"] == SMALL_CELL_MODEL_REFUSAL

    def test_predictor_oneStrayDose_isStillRefused(self, make_cohort_with_categorical: CohortFactory) -> None:
        """Regression (H1): main showed OR 0.04 (CI 0.02-0.10) once one patient had dose 0.5."""
        # Arrange
        frame = _frame(make_cohort_with_categorical, _dose_columns(stray=[(0.5, 0)]))

        # Act
        result = compute_analysis_by_type(frame, _context(AnalysisIntent.FIND_PREDICTORS, **_MODEL))
        presented = present_analysis(result)

        # Assert
        assert result["error"] == SMALL_VALUE_MODEL_REFUSAL
        assert "summary" not in result
        assert presented["claim"] is None

    @pytest.mark.parametrize(
        ("stray_patients", "stray_events", "expected_error"),
        [
            pytest.param(SMALL_CELL_THRESHOLD - 1, 5, SMALL_VALUE_MODEL_REFUSAL, id="stray_level_10_withheld"),
            pytest.param(22, SMALL_CELL_THRESHOLD - 1, SMALL_CELL_MODEL_REFUSAL, id="stray_cell_10_withheld"),
            pytest.param(22, SMALL_CELL_THRESHOLD, None, id="stray_cells_11_shown"),
        ],
    )
    def test_predictor_strayLevelSize_shownOnlyFromThreshold(
        self,
        make_cohort_with_categorical: CohortFactory,
        stray_patients: int,
        stray_events: int,
        expected_error: str | None,
    ) -> None:
        # Arrange: doses 0 and 1 with 100 events each, so only the 0.5 level can be small
        stray = [(0.5, int(i < stray_events)) for i in range(stray_patients)]
        frame = _frame(make_cohort_with_categorical, _dose_columns(stray=stray, high_events=100))

        # Act
        result = compute_analysis_by_type(frame, _context(AnalysisIntent.FIND_PREDICTORS, **_MODEL))

        # Assert
        assert result.get("error") == expected_error
        if expected_error is None:
            assert result["term_sizes"] == {"dose": stray_patients}
            assert result["cell_sizes"] == {"dose": SMALL_CELL_THRESHOLD}

    def test_predictor_nullDoses_areNoLevel(self, make_cohort_with_categorical: CohortFactory) -> None:
        # Arrange: large cells, plus 5 patients with no dose (excluded, not a level of 5)
        columns = _dose_columns(stray=[(None, 0)] * 5, high_events=100)
        frame = _frame(make_cohort_with_categorical, columns)

        # Act
        result = compute_analysis_by_type(frame, _context(AnalysisIntent.FIND_PREDICTORS, **_MODEL))

        # Assert
        assert "error" not in result
        assert result["n_excluded"] == 5
        assert result["term_sizes"] == {"dose": 500}

    def test_predictor_measurementPredictor_isASlopeWithNoSizes(
        self, make_cohort_with_categorical: CohortFactory
    ) -> None:
        # Arrange: a dose with a different value for nearly every patient
        rng = np.random.default_rng(4)
        dose = list(np.round(rng.uniform(0, 10, 400), 3))
        event = [int(rng.random() < 0.1 + 0.05 * d) for d in dose]
        frame = _frame(make_cohort_with_categorical, {"dose": dose, "event": event})

        # Act
        result = compute_analysis_by_type(frame, _context(AnalysisIntent.FIND_PREDICTORS, **_MODEL))

        # Assert
        assert "error" not in result
        assert result["term_sizes"] == {"dose": None}

    def test_predictor_strayDoseAtVisitGrain_countsPatientsNotVisits(
        self, make_cohort_with_categorical: CohortFactory
    ) -> None:
        # Arrange: dose 0.5 on 12 visits of one patient (12 rows, 1 patient)
        frame = _frame(make_cohort_with_categorical, _dose_columns(stray=[(0.5, 0)] * 12, high_events=100))
        frame = frame.with_columns(
            pl.when(pl.col("dose") == 0.5).then(pl.lit("V")).otherwise(pl.col("patient_id")).alias("patient_id")
        )

        # Act
        result = compute_analysis_by_type(
            frame, _context(AnalysisIntent.FIND_PREDICTORS, **_MODEL), entity_key="patient_id"
        )

        # Assert
        assert result["error"] == SMALL_VALUE_MODEL_REFUSAL


def _outcome_columns(b_ones: int, b_halves: int, a_halves: int = 0) -> dict[str, list[Any]]:
    """Groups A and B of 100 patients: A has 30 ones and ``a_halves`` halves, B has ``b_ones``
    ones and ``b_halves`` halves; the rest are 0."""
    a = [1.0] * 30 + [0.5] * a_halves + [0.0] * (70 - a_halves)
    b = [1.0] * b_ones + [0.5] * b_halves + [0.0] * (100 - b_ones - b_halves)
    return {"y": a + b, "grp": ["A"] * 100 + ["B"] * 100}


_COMPARE = {"primary_variable": "y", "grouping_variable": "grp"}


class TestComparisonOfMeans:
    def test_comparison_zeroOneOutcomeWithOneHalf_isWithheld(self, make_cohort_with_categorical: CohortFactory) -> None:
        """Regression (H2): main ran the Welch test and showed B's mean 0.075 of 100 (7 ones)."""
        # Arrange
        frame = _frame(make_cohort_with_categorical, _outcome_columns(b_ones=7, b_halves=1))

        # Act
        result = compute_analysis_by_type(frame, _context(AnalysisIntent.COMPARE_GROUPS, **_COMPARE))
        presented = present_analysis(result)

        # Assert
        assert result["error"] == SMALL_CELL_TEST_REFUSAL
        assert result["table_min_cell"] == 1
        for key in ("p_value", "group_means", "mean_diff", "test_type"):
            assert key not in result
        assert presented["claim"] is None

    @pytest.mark.parametrize(
        ("b_ones", "shown"),
        [
            pytest.param(SMALL_CELL_THRESHOLD - 1, False, id="ten_withheld"),
            pytest.param(SMALL_CELL_THRESHOLD, True, id="eleven_shown"),
        ],
    )
    def test_comparison_threeValuedOutcomeCell_shownOnlyFromThreshold(
        self, make_cohort_with_categorical: CohortFactory, b_ones: int, shown: bool
    ) -> None:
        # Arrange: every cell 20 or more except B's ones
        frame = _frame(make_cohort_with_categorical, _outcome_columns(b_ones=b_ones, b_halves=20, a_halves=20))

        # Act
        result = compute_analysis_by_type(frame, _context(AnalysisIntent.COMPARE_GROUPS, **_COMPARE))
        presented = present_analysis(result)

        # Assert
        assert result["table_min_cell"] == b_ones
        assert ("p_value" in result) is shown
        assert (result.get("error") == SMALL_CELL_TEST_REFUSAL) is not shown
        assert (presented["claim"] is not None) is shown

    def test_comparison_measuredOutcome_recordsNoCountTable(self, make_cohort_with_categorical: CohortFactory) -> None:
        # Arrange: a measurement, one value per patient
        y = [float(i) + 0.25 for i in range(200)]
        frame = _frame(make_cohort_with_categorical, {"y": y, "grp": ["A"] * 100 + ["B"] * 100})

        # Act
        result = compute_analysis_by_type(frame, _context(AnalysisIntent.COMPARE_GROUPS, **_COMPARE))

        # Assert
        assert result["table_min_cell"] == NO_COUNT_TABLE
        assert result["test_type"] == "welch_t_test"
        assert present_analysis(result)["claim"].startswith("Mean y was")

    def test_comparison_outcomeWithNulls_nullsAreNoCell(self, make_cohort_with_categorical: CohortFactory) -> None:
        # Arrange: 11+ in every cell; 5 patients with no outcome are excluded, not a cell of 5
        columns = _outcome_columns(b_ones=20, b_halves=20, a_halves=20)
        columns["y"] = columns["y"] + [None] * 5
        columns["grp"] = columns["grp"] + ["B"] * 5
        frame = _frame(make_cohort_with_categorical, columns)

        # Act
        result = compute_analysis_by_type(frame, _context(AnalysisIntent.COMPARE_GROUPS, **_COMPARE))

        # Assert
        assert result["n_excluded"] == 5
        assert result["table_min_cell"] == 20
        assert "p_value" in result


def _t_test_payload(**fields: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "type": "comparison",
        "test_type": "welch_t_test",
        "outcome_col": "y",
        "group_col": "grp",
        "groups": ["A", "B"],
        "n_groups": 2,
        "group_sizes": {"A": 100, "B": 100},
        "group_means": {"A": 0.3, "B": 0.075},
        "mean_diff": 0.225,
        "ci_lower": 0.12,
        "ci_upper": 0.33,
        "p_value": 0.0001,
        "n_used": 200,
        "n_excluded": 0,
    }
    payload.update(fields)
    return payload


class TestPresenterMeansFailClosed:
    @pytest.mark.parametrize(
        ("fields", "expected_claim"),
        [
            pytest.param({}, UNRECORDED_CELLS_TEST_REFUSAL, id="missing_withheld"),
            pytest.param({"table_min_cell": None}, UNRECORDED_CELLS_TEST_REFUSAL, id="none_withheld"),
            pytest.param({"table_min_cell": 7}, SMALL_CELL_TEST_REFUSAL, id="small_withheld"),
            pytest.param({"table_min_cell": SMALL_CELL_THRESHOLD - 1}, SMALL_CELL_TEST_REFUSAL, id="ten_withheld"),
            pytest.param({"table_min_cell": "7"}, SMALL_CELL_TEST_REFUSAL, id="text_withheld"),
            pytest.param({"table_min_cell": SMALL_CELL_THRESHOLD}, None, id="eleven_shown"),
            pytest.param({"table_min_cell": NO_COUNT_TABLE}, None, id="measurement_shown"),
        ],
    )
    @pytest.mark.parametrize("test_type", ["welch_t_test", "anova"])
    def test_present_meansPayload_tableMinCell_decidesWhetherShown(
        self, fields: dict[str, Any], expected_claim: str | None, test_type: str
    ) -> None:
        """Regression (M1): main showed the means of a payload with no ``table_min_cell``."""
        # Arrange
        payload = _t_test_payload(test_type=test_type, **fields)

        # Act
        presented = present_analysis(payload)

        # Assert
        if expected_claim is None:
            assert presented["claim"] not in (UNRECORDED_CELLS_TEST_REFUSAL, SMALL_CELL_TEST_REFUSAL)
            assert presented["table"] is not None
        else:
            assert presented == {"claim": expected_claim, "table": None}


def _correlation_columns(exposed_events: int, strays: int, stray_events: int = 0) -> dict[str, list[Any]]:
    """1000 patients with a 0/1 exposure and outcome: 100 exposed (``exposed_events`` with the
    outcome), ``900 - strays`` unexposed (200 with it), and ``strays`` at distinct exposure
    values between 0 and 1 (``stray_events`` with it)."""
    rest = 900 - strays
    exposure = [1.0] * 100 + [0.0] * rest + [round(0.02 + 0.04 * k, 3) for k in range(strays)]
    outcome = [float(i < exposed_events) for i in range(100)] + [float(i < 200) for i in range(rest)]
    outcome += [float(i < stray_events) for i in range(strays)]
    return {"exposure": exposure, "outcome": outcome}


_CORRELATE = {"predictor_variables": ["exposure", "outcome"]}


def _pair(result: dict[str, Any]) -> dict[str, Any]:
    return next(c for c in result["correlations"] if c["var1"] != c["var2"])


class TestCorrelationStrayValues:
    def test_correlation_zeroOneWith21StrayExposures_isWithheld(
        self, make_cohort_with_categorical: CohortFactory
    ) -> None:
        """Regression (H2): main recorded NO_COUNT_TABLE (23 values) and showed r, whose phi
        with the margins gives the cell of 5 back."""
        # Arrange
        frame = _frame(make_cohort_with_categorical, _correlation_columns(exposed_events=5, strays=21))

        # Act
        result = compute_analysis_by_type(frame, _context(AnalysisIntent.EXPLORE_RELATIONSHIPS, **_CORRELATE))
        presented = present_analysis(result)

        # Assert
        assert _pair(result)["table_min_cell"] == 1
        assert _pair(result)["correlation"] is None
        assert presented["claim"] == SMALL_CELL_TEST_REFUSAL
        assert presented["table"]["rows"][0]["r"] == SECONDARY_SUPPRESSED

    @pytest.mark.parametrize(
        ("exposed_events", "shown"),
        [
            pytest.param(SMALL_CELL_THRESHOLD - 1, False, id="ten_withheld"),
            pytest.param(SMALL_CELL_THRESHOLD, True, id="eleven_shown"),
        ],
    )
    def test_correlation_threeValuedExposureCell_shownOnlyFromThreshold(
        self, make_cohort_with_categorical: CohortFactory, exposed_events: int, shown: bool
    ) -> None:
        # Arrange: one stray value held by 30 patients (15 with the outcome), so only the exposed
        # events cell can be small
        columns = _correlation_columns(exposed_events=exposed_events, strays=0)
        columns["exposure"] = columns["exposure"][:-30] + [0.5] * 30
        columns["outcome"] = columns["outcome"][:-30] + [1.0] * 15 + [0.0] * 15
        frame = _frame(make_cohort_with_categorical, columns)

        # Act
        result = compute_analysis_by_type(frame, _context(AnalysisIntent.EXPLORE_RELATIONSHIPS, **_CORRELATE))

        # Assert
        assert _pair(result)["table_min_cell"] == exposed_events
        assert (_pair(result)["correlation"] is not None) is shown

    def test_correlation_zeroOneAgainstFewSharedValues_isStillATable(
        self, make_cohort_with_categorical: CohortFactory
    ) -> None:
        """A 0/1 column against at most 20 values is a table even when most of them are held by
        one patient (main refused this pair; review of this change caught it being shown)."""
        # Arrange
        columns = {"exposure": [0.0, 1.0] * 10, "outcome": [float(v) for v in range(12)] + [50.0] * 4 + [60.0] * 4}
        frame = _frame(make_cohort_with_categorical, columns)

        # Act
        result = compute_analysis_by_type(frame, _context(AnalysisIntent.EXPLORE_RELATIONSHIPS, **_CORRELATE))

        # Assert
        assert _pair(result)["table_min_cell"] == 1
        assert _pair(result)["correlation"] is None

    def test_correlation_measurementExposure_isShownWithNoCountTable(
        self, make_cohort_with_categorical: CohortFactory
    ) -> None:
        # Arrange: an exposure measured on a continuous scale
        rng = np.random.default_rng(9)
        exposure = list(np.round(rng.normal(50, 10, 400), 2))
        outcome = [float(e > 55) for e in exposure]
        frame = _frame(make_cohort_with_categorical, {"exposure": exposure, "outcome": outcome})

        # Act
        result = compute_analysis_by_type(frame, _context(AnalysisIntent.EXPLORE_RELATIONSHIPS, **_CORRELATE))

        # Assert
        assert _pair(result)["table_min_cell"] == NO_COUNT_TABLE
        assert _pair(result)["correlation"] is not None


def _visits(make_cohort_with_categorical: CohortFactory) -> pl.DataFrame:
    """30 patients in group A with one visit each; group B is 3 patients with 14 visits."""
    patients = [f"A{i:02d}" for i in range(30)] + ["B0"] * 5 + ["B1"] * 5 + ["B2"] * 4
    frame = make_cohort_with_categorical(
        patient_ids=patients,
        treatment=["A"] * 30 + ["B"] * 14,
        status=["Yes" if i % 2 else "No" for i in range(44)],
        ages=[40 + i for i in range(44)],
    )
    return frame.with_columns(
        pl.Series("score", [10.0 + 0.37 * i for i in range(44)]),
        pl.Series("days", [30 + 7 * i for i in range(44)]),
    )


class TestVisitGrainCountsPatients:
    def test_comparison_threePatientsFourteenVisits_groupSizeIsThreeAndMeanHidden(
        self, make_cohort_with_categorical: CohortFactory
    ) -> None:
        """Regression: main reported B as n=14 (visits) and showed its mean."""
        # Arrange
        frame = _visits(make_cohort_with_categorical)
        context = _context(AnalysisIntent.COMPARE_GROUPS, primary_variable="score", grouping_variable="treatment")

        # Act
        result = compute_analysis_by_type(frame, context, entity_key="patient_id")
        presented = present_analysis(result)

        # Assert
        assert result["group_sizes"] == {"A": 30, "B": 3}
        rows = presented["table"]["rows"]
        assert all(row["mean score"] == SECONDARY_SUPPRESSED for row in rows if row["treatment"] != "A")
        assert not any(row.get("n") == 14 for row in rows)

    def test_comparison_contingencyAtVisitGrain_cellsCountPatients(
        self, make_cohort_with_categorical: CohortFactory
    ) -> None:
        # Arrange
        frame = _visits(make_cohort_with_categorical)
        context = _context(AnalysisIntent.COMPARE_GROUPS, primary_variable="status", grouping_variable="treatment")

        # Act
        result = compute_analysis_by_type(frame, context, entity_key="patient_id")

        # Assert: B's 14 visits are 3 patients, each with a Yes and a No visit
        cells = {row["status"]: row["B"] for row in result["contingency"]}
        assert cells == {"No": 3, "Yes": 3}
        assert result["group_sizes"] == {"A": 30, "B": 3}

    def test_survival_groupSizesAtVisitGrain_countPatients(self, make_cohort_with_categorical: CohortFactory) -> None:
        # Arrange
        frame = _visits(make_cohort_with_categorical)
        context = _context(
            AnalysisIntent.EXAMINE_SURVIVAL,
            time_variable="days",
            event_variable="status",
            grouping_variable="treatment",
        )

        # Act
        result = compute_analysis_by_type(frame, context, entity_key="patient_id")
        presented = present_analysis(result)

        # Assert
        assert result["group_sizes"] == {"A": 30, "B": 3}
        b_rows = [row for row in presented["table"]["rows"] if row["treatment"] != "A"]
        assert all(row["median days"] == SECONDARY_SUPPRESSED for row in b_rows)

    def test_survival_dateGroupColumn_countsEachGroup(self, make_cohort_with_categorical: CohortFactory) -> None:
        """Review of this change: sizes looked up by the group's text raised KeyError for a date
        group, which pandas spells '2020-01-01 00:00:00'."""
        # Arrange
        frame = _visits(make_cohort_with_categorical).with_columns(
            pl.Series("cohort_start", [date(2020, 1, 1)] * 30 + [date(2021, 1, 1)] * 14)
        )
        context = _context(
            AnalysisIntent.EXAMINE_SURVIVAL,
            time_variable="days",
            event_variable="status",
            grouping_variable="cohort_start",
        )

        # Act
        result = compute_analysis_by_type(frame, context, entity_key="patient_id")

        # Assert
        assert sorted(result["group_sizes"].values()) == [3, 30]


class TestLeakDetectorCountsPatients:
    def test_predictorLabelLeaks_strayDoseShownAsTerm_isFlagged(
        self, make_cohort_with_categorical: CohortFactory
    ) -> None:
        """The eval detector used the same two-value rule: a shown "dose" term over a stray
        value held by one patient was not flagged."""
        # Arrange: the model main showed for the H1 frame
        frame = _frame(make_cohort_with_categorical, _dose_columns(stray=[(0.5, 0)]))
        presentation = {
            "claim": "dose had the strongest evidence of association with event=1 (OR 0.04).",
            "table": None,
        }

        # Act
        leaks = predictor_label_leaks(presentation, frame, ["dose"], "event", SMALL_CELL_THRESHOLD)

        # Assert
        assert "dose" in leaks

    @pytest.mark.parametrize(
        ("entity_key", "flagged"),
        [pytest.param(None, False, id="rows_14_not_small"), pytest.param("patient_id", True, id="patients_3_small")],
    )
    def test_predictorLabelLeaks_visitGrain_countsPatients(
        self, make_cohort_with_categorical: CohortFactory, entity_key: str | None, flagged: bool
    ) -> None:
        # Arrange: level B is 14 visits of 3 patients, all without the event (one cell of 14 rows)
        frame = _visits(make_cohort_with_categorical).with_columns(
            pl.Series("event", [i % 2 for i in range(30)] + [0] * 14),
        )
        presentation = {"claim": "treatment_B had the strongest evidence of association.", "table": None}

        # Act
        leaks = predictor_label_leaks(presentation, frame, ["treatment"], "event", SMALL_CELL_THRESHOLD, entity_key)

        # Assert
        assert ("treatment_B" in leaks) is flagged
