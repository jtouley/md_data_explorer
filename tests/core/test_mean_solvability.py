"""Statistics of category-like columns, end to end: compute records the exact test's level
(``STATISTICS_LEVEL``), and compute and the presenters decide by one function
(``grouping.value_statistic_refusal``).

#144 withheld a t-test or ANOVA of an integer measure with up to 20 values (length of stay by
arm, pain score by group) once a value x group cell held 1-10 patients; #146 withheld the mean,
SD, median, minimum and maximum of such a column (mean age over ages 51-69 or 79-84). Each is now
shown when ``analysis/cell_solvability.py`` finds that the statistics pin no hidden count, and
withheld, as before, when they do. All data is synthetic.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl
import pytest

from clinical_analytics.analysis.compute import _statistics_level, compute_analysis_by_type
from clinical_analytics.analysis.grouping import NO_COUNT_TABLE, SMALL_CELL_TEST_REFUSAL, STATISTICS_LEVEL
from clinical_analytics.core.analysis_types import AnalysisIntent
from clinical_analytics.core.query_plan import FilterSpec, QueryPlan
from clinical_analytics.core.result_presentation import SECONDARY_SUPPRESSED, VALUE_CELL_REASON, present_analysis
from tests.analysis.test_cell_solvability import AGES_51_TO_69, OLDEST_DECILE
from tests.core.test_answer_data_checks import _ask, _service
from tests.core.test_coded_column_cell_rules import CohortFactory, _context, _frame


def _length_of_stay() -> dict[str, list[Any]]:
    """300 patients, 150 per arm, length of stay 1-15 days (``length_of_stay_by_arm``)."""
    rng = np.random.default_rng(32)
    stay = np.clip(1 + rng.negative_binomial(2, 0.3, 300), 1, 15)
    return {"los": [int(v) for v in stay], "arm": ["A"] * 150 + ["B"] * 150}


def _pain() -> dict[str, list[Any]]:
    """120 patients in three groups, pain 0-10 (``pain_by_group``)."""
    rng = np.random.default_rng(1069)
    groups = rng.choice(["control", "drug A", "drug B"], 120)
    centre = {"control": 6.0, "drug A": 5.0, "drug B": 4.5}
    pain = [int(np.clip(np.rint(rng.normal(centre[g], 2.5)), 0, 10)) for g in groups]
    return {"pain": pain, "grp": [str(g) for g in groups]}


def _ages(counts: dict[int, int]) -> list[int]:
    return [age for age, n in counts.items() for _ in range(n)]


class TestComparisonOfMeans:
    def test_comparison_lengthOfStayByArm_tTestShown(self, make_cohort_with_categorical: CohortFactory) -> None:
        # Arrange
        frame = _frame(make_cohort_with_categorical, _length_of_stay())
        context = _context(AnalysisIntent.COMPARE_GROUPS, primary_variable="los", grouping_variable="arm")

        # Act
        result = compute_analysis_by_type(frame, context)
        claim = present_analysis(result)["claim"]

        # Assert: a cell of 1-10 patients, and the test shown
        assert 0 < result["table_min_cell"] < 11
        assert result[STATISTICS_LEVEL] == "median"
        assert result["test_type"] == "welch_t_test"
        assert claim is not None and claim != SMALL_CELL_TEST_REFUSAL

    def test_comparison_painByGroup_anovaShown(self, make_cohort_with_categorical: CohortFactory) -> None:
        # Arrange
        frame = _frame(make_cohort_with_categorical, _pain())
        context = _context(AnalysisIntent.COMPARE_GROUPS, primary_variable="pain", grouping_variable="grp")

        # Act
        result = compute_analysis_by_type(frame, context)

        # Assert
        assert 0 < result["table_min_cell"] < 11
        assert result["test_type"] == "anova"
        assert present_analysis(result)["table"] is not None

    def test_comparison_zeroOneOutcomeWithOneHalf_stillWithheld(
        self, make_cohort_with_categorical: CohortFactory
    ) -> None:
        """#144 H2 through compute: B's 7 ones are pinned, so the test stays withheld."""
        # Arrange
        y = [1.0] * 30 + [0.0] * 70 + [1.0] * 7 + [0.5] + [0.0] * 92
        frame = _frame(make_cohort_with_categorical, {"y": y, "grp": ["A"] * 100 + ["B"] * 100})
        context = _context(AnalysisIntent.COMPARE_GROUPS, primary_variable="y", grouping_variable="grp")

        # Act
        result = compute_analysis_by_type(frame, context)

        # Assert
        assert result[STATISTICS_LEVEL] == "none"
        assert result["error"] == SMALL_CELL_TEST_REFUSAL
        assert "p_value" not in result

    @pytest.mark.parametrize(
        ("level", "shown"),
        [
            pytest.param("median", True, id="median_level_shown"),
            pytest.param("sd", True, id="sd_level_shown"),
            pytest.param("mean", False, id="mean_level_withheld_test_shows_sd"),
            pytest.param("none", False, id="none_withheld"),
            pytest.param(None, False, id="not_recorded_withheld"),
        ],
    )
    def test_presentComparison_recordedLevel_decidesTest(self, level: str | None, shown: bool) -> None:
        # Arrange: a payload with a hidden cell; a test's CI and df give each group's SD
        payload: dict[str, Any] = {
            "type": "comparison",
            "test_type": "welch_t_test",
            "outcome_col": "los",
            "group_col": "arm",
            "groups": ["A", "B"],
            "group_sizes": {"A": 150, "B": 150},
            "group_means": {"A": 4.9, "B": 5.3},
            "mean_diff": -0.4,
            "ci_lower": -1.1,
            "ci_upper": 0.3,
            "p_value": 0.27,
            "n_used": 300,
            "n_excluded": 0,
            "table_min_cell": 2,
        }
        if level is not None:
            payload[STATISTICS_LEVEL] = level

        # Act
        presented = present_analysis(payload)

        # Assert
        assert (presented["claim"] != SMALL_CELL_TEST_REFUSAL) is shown


class TestDescribedColumn:
    def test_descriptive_meanOnlyLevel_sdAndMedianWithheld(self, make_cohort_with_categorical: CohortFactory) -> None:
        # Arrange: with the SD, age 66's count is known to +-1 (``cell_solvability`` tests)
        ages = _ages({60: 40, 66: 5, 67: 5, 68: 5, 69: 5})
        frame = _frame(make_cohort_with_categorical, {"years": ages})

        # Act
        result = compute_analysis_by_type(frame, _context(AnalysisIntent.DESCRIBE, primary_variable="years"))
        presented = present_analysis(result)

        # Assert
        assert result[STATISTICS_LEVEL] == "mean"
        row = presented["table"]["rows"][0]
        assert [row[k] for k in ("std", "median")] == [SECONDARY_SUPPRESSED] * 2
        assert row["mean"] == pytest.approx(62.5)
        assert (row["min"], row["max"]) == (60.0, 69.0)
        assert presented["claim"] == (
            f"Mean years was 62.50 (minimum 60.00; maximum 69.00; n=60). "
            f"The median and SD are not shown because {VALUE_CELL_REASON}."
        )

    @pytest.mark.parametrize(
        ("ages", "mean"),
        [
            pytest.param(AGES_51_TO_69, "60.07", id="ages_51_to_69"),
            pytest.param(OLDEST_DECILE, "81.70", id="ages_79_to_84"),
        ],
    )
    def test_ask_meanAgeOfNarrowCohort_shown(self, make_semantic_layer: Any, ages: dict[int, int], mean: str) -> None:
        """#146's trade-off: mean age within an age-limited cohort was withheld."""
        # Arrange: those ages plus 100 patients outside the filter
        values = _ages(ages) + [30 + i % 15 for i in range(100)]
        frame = pl.DataFrame({"patient_id": [f"P{i:04d}" for i in range(len(values))], "age": values})
        service = _service(make_semantic_layer, frame)
        lowest, highest = min(ages), max(ages)
        plan = QueryPlan(
            intent="DESCRIBE",
            metric="age",
            filters=[
                FilterSpec(column="age", operator=">=", value=lowest),
                FilterSpec(column="age", operator="<=", value=highest),
            ],
            confidence=0.99,
        )
        held = _ask(service, plan)

        # Act
        result = _ask(service, plan, confirm_hash=held.plan_hash if held.requires_confirmation else None)

        # Assert
        assert 0 < result.analysis["table_min_cell"] < 11
        assert result.analysis[STATISTICS_LEVEL] == "median"
        assert str(result.claim).startswith(f"Mean age was {mean} (SD ")

    def test_statisticsLevel_visitRows_nothingRelaxed(self) -> None:
        # Arrange: the 79-84 ages, one patient on two rows
        values = pl.Series("age", [float(a) for a in _ages(OLDEST_DECILE)])
        keys = pl.Series("pid", [f"P{i}" for i in range(values.len() - 1)] + ["P0"])

        # Act
        level = _statistics_level(values, None, keys)

        # Assert
        assert level == "none"


class TestGroupedStatistic:
    @pytest.mark.parametrize("statistic", ["mean", "median", "std"])
    def test_ask_lengthOfStayByArm_everyGroupShown(self, make_semantic_layer: Any, statistic: str) -> None:
        # Arrange
        columns = _length_of_stay()
        frame = pl.DataFrame({"patient_id": [f"P{i:03d}" for i in range(300)], **columns})
        service = _service(make_semantic_layer, frame)
        plan = QueryPlan(intent="DESCRIBE", metric="los", group_by="arm", statistic=statistic, confidence=0.99)
        held = _ask(service, plan)

        # Act
        result = _ask(service, plan, confirm_hash=held.plan_hash if held.requires_confirmation else None)

        # Assert
        groups = result.analysis["groups"]
        assert any(0 < g["table_min_cell"] < 11 for g in groups)
        assert {g.get(STATISTICS_LEVEL) for g in groups if g["table_min_cell"] < 11} == {"median"}
        rows = present_analysis(result.analysis)["table"]["rows"]
        assert all(row[f"{statistic} los"] != SECONDARY_SUPPRESSED for row in rows)

    def test_ask_visitRows_groupedMeanStillWithheld(self, make_semantic_layer: Any) -> None:
        # Arrange: the same stays, each patient on two rows
        columns = _length_of_stay()
        frame = pl.DataFrame(
            {"patient_id": [f"P{i:03d}" for i in range(300)] * 2, **{k: v * 2 for k, v in columns.items()}}
        )
        service = _service(make_semantic_layer, frame)
        plan = QueryPlan(intent="DESCRIBE", metric="los", group_by="arm", confidence=0.99)
        held = _ask(service, plan)

        # Act
        result = _ask(service, plan, confirm_hash=held.plan_hash if held.requires_confirmation else None)

        # Assert: nothing relaxed; any group with a hidden cell is withheld as before
        hidden = [
            g for g in result.analysis["groups"] if g["table_min_cell"] != NO_COUNT_TABLE and g["table_min_cell"] < 11
        ]
        assert hidden
        assert {g.get(STATISTICS_LEVEL) for g in hidden} == {"none"}
        rows = present_analysis(result.analysis)["table"]["rows"]
        assert any(row["mean los"] == SECONDARY_SUPPRESSED for row in rows)

    @pytest.mark.parametrize(
        ("level", "statistic", "shown"),
        [
            pytest.param("mean", "mean", True, id="mean_level_mean"),
            pytest.param("mean", "max", True, id="mean_level_max"),
            pytest.param("mean", "median", False, id="mean_level_median_withheld"),
            pytest.param("sd", "std", True, id="sd_level_sd"),
            pytest.param("none", "mean", False, id="none_withheld"),
        ],
    )
    def test_presentGroupedStatistic_recordedLevel_decidesStatistic(
        self, level: str, statistic: str, shown: bool
    ) -> None:
        # Arrange
        groups = [
            {"group": g, "value": 5.0, "n": 150, "n_rows": 150, "table_min_cell": 3, STATISTICS_LEVEL: level}
            for g in ("A", "B")
        ]
        analysis = {
            "type": "grouped_statistic",
            "statistic": statistic,
            "metric": "los",
            "group_col": "arm",
            "groups": groups,
            "n_used": 300,
            "n_excluded": 0,
            "n_filtered_out": 0,
        }

        # Act
        rows = present_analysis(analysis)["table"]["rows"]

        # Assert
        assert [row[f"{statistic} los"] != SECONDARY_SUPPRESSED for row in rows] == [shown, shown]
