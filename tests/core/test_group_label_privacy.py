"""Group and level labels never disclose patients (design review privacy blocker).

On main, "compare treatment between ldl groups" on the synthetic cohort showed a crosstab with
289 columns, one per exact LDL value; a column of "hidden" beside "0" pinned single patients'
LDL and arm. A many-level column is now refused as a grouping dimension, and any level with 1-10
patients is shown only inside one merged "Other" row or column, in every analysis type.
"""

from __future__ import annotations

from typing import Any

import polars as pl
import pytest

from clinical_analytics.analysis.compute import compute_comparison_analysis
from clinical_analytics.analysis.grouping import (
    FOLD_LABELS,
    MAX_GROUP_LEVELS,
    OTHER_LABEL,
    grouping_problem,
    is_small_level,
)
from clinical_analytics.core.analysis_types import AnalysisContext, AnalysisIntent
from clinical_analytics.core.query_plan import QueryPlan
from clinical_analytics.core.result_presentation import present_analysis


def _labels(presentation: dict[str, Any]) -> list[str]:
    """Every label a reader sees: table column headers and first-column row labels."""
    table = presentation.get("table") or {}
    columns = list(table.get("columns") or [])
    rows = list(table.get("rows") or [])
    first = columns[0] if columns else None
    return columns + [str(row.get(first)) for row in rows if first is not None]


class TestGroupingProblem:
    @pytest.mark.parametrize(
        ("levels", "refused"),
        [
            pytest.param(0, False, id="empty"),
            pytest.param(2, False, id="two_levels"),
            pytest.param(MAX_GROUP_LEVELS, False, id="at_cap"),
            pytest.param(MAX_GROUP_LEVELS + 1, True, id="above_cap"),
            pytest.param(289, True, id="exact_ldl_values"),
        ],
    )
    def test_groupingProblem_levelCount_refusesAboveCap(self, levels: int, refused: bool) -> None:
        # Arrange / Act
        problem = grouping_problem("ldl", levels)

        # Assert
        assert (problem is not None) == refused
        if refused:
            assert "cannot be used to split patients into groups" in str(problem)

    @pytest.mark.parametrize(
        ("count", "small"),
        [
            pytest.param(0, False, id="zero"),
            pytest.param(1, True, id="one"),
            pytest.param(10, True, id="ten"),
            pytest.param(11, False, id="eleven"),
            pytest.param(None, True, id="missing_fails_closed"),
            pytest.param("x", True, id="not_a_count_fails_closed"),
        ],
    )
    def test_isSmallLevel_boundaries_matchSmallCellRule(self, count: Any, small: bool) -> None:
        # Arrange / Act
        result = is_small_level(count)

        # Assert
        assert result is small


class TestManyLevelGroupingRefused:
    def test_compareGroups_cohortLdlAsGroup_refusedNotTabulated(self, load_synthetic_dataset: Any) -> None:
        # Arrange
        cohort = load_synthetic_dataset("cohort")
        context = AnalysisContext()
        context.inferred_intent = AnalysisIntent.COMPARE_GROUPS
        context.primary_variable = "treatment"
        context.grouping_variable = "ldl"

        # Act
        result = compute_comparison_analysis(cohort, context)

        # Assert
        assert "contingency" not in result
        assert "cannot be used to split patients into groups" in result["error"]

    def test_compareGroups_cohortLdlViaSemanticLayer_presentsNoLdlLabels(
        self, load_synthetic_dataset: Any, make_semantic_layer: Any
    ) -> None:
        # Arrange
        layer = make_semantic_layer(dataset_name="cohort", data=load_synthetic_dataset("cohort"))
        plan = QueryPlan(intent="COMPARE_GROUPS", metric="treatment", group_by="ldl", confidence=0.99)
        context = AnalysisContext()
        context.inferred_intent = AnalysisIntent.COMPARE_GROUPS
        context.primary_variable = "treatment"
        context.grouping_variable = "ldl"
        context.query_plan = plan

        # Act
        executed = layer.execute_query_plan(plan, confidence_threshold=0.0, query_text="q", confirmed=True)
        analysis = layer.format_execution_result(executed, context)
        shown = present_analysis(analysis)

        # Assert
        assert analysis.get("error")
        assert not any(label.startswith("ldl=") for label in _labels(dict(shown)))


def _contingency(group_sizes: dict[str, int]) -> dict[str, Any]:
    groups = list(group_sizes)
    return {
        "type": "comparison",
        "test_type": "chi_square",
        "outcome_col": "treatment",
        "group_col": "site",
        "outcome_kind": "binary",
        "groups": groups,
        "group_sizes": group_sizes,
        "contingency": [
            {"treatment": "drug", **{g: n // 2 + n % 2 for g, n in group_sizes.items()}},
            {"treatment": "placebo", **{g: n // 2 for g, n in group_sizes.items()}},
        ],
        "p_value": 0.3,
        "n_used": sum(group_sizes.values()),
        "n_excluded": 0,
    }


SMALL_LEVEL_CASES = [
    pytest.param(_contingency({"North": 60, "South": 50, "Rare": 3}), id="chi_square_column"),
    pytest.param(
        {
            "type": "comparison",
            "test_type": "anova",
            "outcome_col": "ldl",
            "group_col": "site",
            "groups": ["North", "South", "Rare"],
            "group_means": {"North": 120.0, "South": 130.0, "Rare": 90.0},
            "group_sizes": {"North": 60, "South": 50, "Rare": 3},
            "p_value": 0.02,
            "n_groups": 3,
            "n_used": 113,
            "n_excluded": 0,
        },
        id="compare_means",
    ),
    pytest.param(
        {
            "type": "count",
            "grouped_by": "site",
            "total_count": 113,
            "group_counts": [
                {"site": "North", "count": 60},
                {"site": "Rare", "count": 3},
                {"site": "South", "count": 50},
            ],
        },
        id="grouped_count",
    ),
    pytest.param(
        {
            "type": "rate",
            "column": "outcome",
            "value": 1,
            "group_col": "site",
            "groups": [
                {"group": "North", "n": 60, "events": 20, "rate": 20 / 60},
                {"group": "Rare", "n": 3, "events": 0, "rate": 0.0},
                {"group": "South", "n": 50, "events": 25, "rate": 0.5},
            ],
            "n_used": 113,
            "n_excluded": 0,
        },
        id="rate",
    ),
    pytest.param(
        {
            "type": "grouped_statistic",
            "statistic": "mean",
            "metric": "ldl",
            "group_col": "site",
            "groups": [
                {"group": "North", "n": 60, "value": 120.0},
                {"group": "Rare", "n": 3, "value": 90.0},
                {"group": "South", "n": 50, "value": 130.0},
            ],
            "n_used": 113,
            "n_excluded": 0,
        },
        id="grouped_statistic",
    ),
    pytest.param(
        {
            "type": "descriptive",
            "focused_variable": "site",
            "is_numeric": False,
            "row_count": 113,
            "value_counts": [
                {"site": "North", "count": 60},
                {"site": "South", "count": 50},
                {"site": "Rare", "count": 3},
            ],
        },
        id="describe_levels",
    ),
]


class TestSmallLevelLabelsMerged:
    @pytest.mark.parametrize("analysis", SMALL_LEVEL_CASES)
    def test_presentAnalysis_levelWithFewPatients_labelNeverShown(self, analysis: dict[str, Any]) -> None:
        # Arrange / Act
        shown = present_analysis(analysis)

        # Assert
        labels = _labels(dict(shown))
        assert not any("Rare" in label for label in labels)
        assert "Rare" not in (shown["claim"] or "")
        assert set(labels) & FOLD_LABELS

    @pytest.mark.parametrize("analysis", SMALL_LEVEL_CASES)
    def test_presentAnalysis_largestLevel_keepsItsOwnLabel(self, analysis: dict[str, Any]) -> None:
        # Arrange: South (50) is folded too, as the complement that keeps Rare's 3 from being
        # worked out from the total; North (60) is shown

        # Act
        shown = present_analysis(analysis)

        # Assert
        labels = _labels(dict(shown))
        assert any("North" in label for label in labels)

    def test_presentAnalysis_twoGroupTestWithSmallGroup_withholdsDifference(self) -> None:
        # Arrange
        analysis = {
            "type": "comparison",
            "test_type": "welch_t_test",
            "outcome_col": "ldl",
            "group_col": "treatment",
            "groups": ["drug", "rare"],
            "group_means": {"drug": 120.0, "rare": 90.0},
            "group_sizes": {"drug": 100, "rare": 4},
            "mean_diff": 30.0,
            "ci_lower": 20.0,
            "ci_upper": 40.0,
            "p_value": 0.01,
            "n_used": 104,
            "n_excluded": 0,
        }

        # Act
        shown = present_analysis(analysis)

        # Assert
        claim = shown["claim"] or ""
        assert "rare" not in claim
        assert "30.00" not in claim
        assert "rare" not in _labels(dict(shown))

    def test_presentAnalysis_moreLevelsThanCap_showsNoTable(self) -> None:
        # Arrange
        levels = [{"site": f"S{i}", "count": 20} for i in range(MAX_GROUP_LEVELS + 1)]
        analysis = {"type": "count", "grouped_by": "site", "total_count": 20 * len(levels), "group_counts": levels}

        # Act
        shown = present_analysis(analysis)

        # Assert
        assert shown["table"] is None
        assert "cannot be used to split patients into groups" in (shown["claim"] or "")


def test_presentAnalysis_mergedOtherCells_stillSuppressSmallCounts() -> None:
    # Arrange
    analysis = _contingency({"North": 60, "South": 50, "Rare": 3})

    # Act
    shown = present_analysis(analysis)

    # Assert
    table = shown["table"]
    assert table is not None
    other_cells = [cell for row in table["rows"] for key, cell in row.items() if key in FOLD_LABELS]
    assert other_cells
    assert all(not (isinstance(cell, int) and 0 < cell < 11) for cell in other_cells)


def test_presentAnalysis_frameRoundTrip_isPolarsSafe() -> None:
    # Arrange: group labels arrive from Polars frames as plain Python values
    frame = pl.DataFrame({"site": ["North", "Rare", "South"], "count": [60, 3, 50]})

    # Act
    shown = present_analysis(
        {"type": "count", "grouped_by": "site", "total_count": 113, "group_counts": frame.to_dicts()}
    )

    # Assert
    assert OTHER_LABEL in _labels(dict(shown))
