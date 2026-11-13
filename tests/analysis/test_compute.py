"""
Test Compute Functions - Phase 1

Unit tests for pure compute functions in analysis/compute.py.
These functions have no UI dependencies and can be tested independently.

Test name follows: test_unit_scenario_expectedBehavior
"""

import numpy as np
import polars as pl
import pytest

from clinical_analytics.analysis.compute import (
    compute_analysis_by_type,
    compute_comparison_analysis,
    compute_count_analysis,
    compute_descriptive_analysis,
    compute_predictor_analysis,
    compute_relationship_analysis,
    compute_survival_analysis,
)
from clinical_analytics.core.analysis_types import AnalysisContext, AnalysisIntent

UNTESTABLE_T_TEST = (
    "A t-test cannot be run on score by category. Every group needs at least 2 observations, and the "
    "values must differ within each group. Try groups with more data."
)
UNTESTABLE_ANOVA = (
    "An ANOVA cannot be run on score by category. Every group needs at least 2 observations, and the "
    "values must differ within at least one group. Try groups with more data."
)

# All fixtures moved to conftest.py - use shared fixtures
# sample_numeric_df, sample_categorical_df, sample_mixed_df
# sample_context_describe, sample_context_compare, sample_context_predictor
# sample_context_survival, sample_context_relationship, sample_context_count


class TestComputeDescriptiveAnalysis:
    """Test compute_descriptive_analysis function."""

    def test_compute_descriptive_analysis_returns_serializable_dict(self, sample_numeric_df, sample_context_describe):
        """Test that compute_descriptive_analysis returns serializable dict."""
        # Arrange: Already set up with fixtures

        # Act: Compute descriptive analysis
        result = compute_descriptive_analysis(sample_numeric_df, sample_context_describe)

        # Assert: Result is dict with expected keys
        assert isinstance(result, dict)
        assert result["type"] == "descriptive"
        assert "row_count" in result
        assert "column_count" in result
        assert "missing_pct" in result
        assert "summary_stats" in result
        assert "categorical_summary" in result

    def test_compute_descriptive_analysis_uses_polars_attributes(self, sample_numeric_df, sample_context_describe):
        """Test that compute_descriptive_analysis uses Polars attributes (height, width)."""
        # Arrange
        expected_rows = sample_numeric_df.height
        expected_cols = sample_numeric_df.width

        # Act
        result = compute_descriptive_analysis(sample_numeric_df, sample_context_describe)

        # Assert: Uses Polars attributes
        assert result["row_count"] == expected_rows
        assert result["column_count"] == expected_cols

    def test_compute_descriptive_analysis_returns_to_dicts_format(self, sample_numeric_df, sample_context_describe):
        """Test that summary_stats uses to_dicts() format (list of dicts)."""
        # Act
        result = compute_descriptive_analysis(sample_numeric_df, sample_context_describe)

        # Assert: summary_stats is list of dicts (Polars to_dicts format)
        assert isinstance(result["summary_stats"], list)
        if result["summary_stats"]:
            assert isinstance(result["summary_stats"][0], dict)

    def test_compute_descriptive_analysis_handles_empty_dataframe(self, sample_context_describe):
        """Test that compute_descriptive_analysis handles empty DataFrame."""
        # Arrange: Empty DataFrame with columns (to avoid null_count error)
        empty_df = pl.DataFrame({"col1": [], "col2": []})

        # Act
        result = compute_descriptive_analysis(empty_df, sample_context_describe)

        # Assert: Returns valid result with zeros
        assert result["row_count"] == 0
        assert result["column_count"] == 2
        assert result["missing_pct"] == 0.0

    def test_compute_descriptive_analysis_handles_null_values(self, sample_context_describe):
        """Test that compute_descriptive_analysis correctly calculates missing percentage."""
        # Arrange: DataFrame with nulls
        df_with_nulls = pl.DataFrame(
            {
                "age": [25, None, 35, None, 45],
                "score": [10, 20, None, 40, 50],
            }
        )

        # Act
        result = compute_descriptive_analysis(df_with_nulls, sample_context_describe)

        # Assert: Missing percentage calculated
        assert result["missing_pct"] > 0.0
        assert result["missing_pct"] <= 100.0

    def test_compute_descriptive_analysis_includes_breakdown_when_filters_applied(self):
        """Test that breakdown info is included when filters are applied."""
        # Arrange: DataFrame and context with filters
        from clinical_analytics.core.query_plan import FilterSpec, QueryPlan

        df = pl.DataFrame(
            {
                "age": [25, 30, 35, 40, 45, 50, 55, 60],
                "score": [10, 20, 30, 40, 50, 60, 70, 80],
            }
        )

        # Create QueryPlan with filters
        query_plan = QueryPlan(
            intent="DESCRIBE",
            metric="age",
            filters=[FilterSpec(column="age", operator=">=", value=35, exclude_nulls=True)],
            confidence=0.9,
        )

        context = AnalysisContext()
        context.primary_variable = "age"
        context.query_plan = query_plan

        # Act
        result = compute_descriptive_analysis(df, context)

        # Assert: Breakdown info included
        assert "original_count" in result
        assert "filtered_count" in result
        assert "filters_applied" in result
        assert "filter_description" in result
        assert result["original_count"] == 8  # All rows
        assert result["filtered_count"] == 6  # Rows with age >= 35: [35, 40, 45, 50, 55, 60]
        assert len(result["filters_applied"]) == 1
        assert "age" in result["filter_description"].lower()

    def test_compute_descriptive_analysis_no_breakdown_when_no_filters(
        self, sample_numeric_df, sample_context_describe
    ):
        """Test that breakdown info is empty when no filters are applied."""
        # Act
        result = compute_descriptive_analysis(sample_numeric_df, sample_context_describe)

        # Assert: Breakdown fields present but empty
        assert "original_count" in result
        assert "filtered_count" in result
        assert "filters_applied" in result
        assert "filter_description" in result
        assert result["original_count"] == result["filtered_count"]  # No filtering
        assert result["filters_applied"] == []
        assert result["filter_description"] == ""

    def test_compute_descriptive_analysis_breakdown_tracks_filtered_vs_unfiltered_counts(self):
        """Test that breakdown correctly tracks filtered vs unfiltered counts."""
        # Arrange: DataFrame with filters that exclude some rows
        from clinical_analytics.core.query_plan import FilterSpec, QueryPlan

        df = pl.DataFrame(
            {
                "age": [20, 25, 30, 35, 40, 45, 50],
                "category": ["A", "B", "A", "B", "A", "B", "A"],
            }
        )

        # Setup query plan with age >= 30 filter
        query_plan = QueryPlan(
            intent="DESCRIBE",
            metric="all",
            filters=[FilterSpec(column="age", operator=">=", value=30, exclude_nulls=True)],
            confidence=0.9,
        )

        context = AnalysisContext()
        context.primary_variable = "all"
        context.query_plan = query_plan

        # Act
        result = compute_descriptive_analysis(df, context)

        # Assert: Counts tracked correctly
        assert result["original_count"] == 7  # All rows
        assert result["filtered_count"] == 5  # Rows with age >= 30
        assert result["row_count"] == 5  # Analysis uses filtered count


class TestComputeComparisonAnalysis:
    """Test compute_comparison_analysis function."""

    def test_compute_comparison_analysis_returns_serializable_dict(self, sample_mixed_df, sample_context_compare):
        """Test that compute_comparison_analysis returns serializable dict."""
        # Arrange: Create DataFrame with groups
        df = pl.DataFrame(
            {
                "score": [10, 20, 30, 40, 50, 60, 70, 80],
                "category": ["A", "A", "A", "A", "B", "B", "B", "B"],
            }
        )
        context = AnalysisContext()
        context.inferred_intent = AnalysisIntent.COMPARE_GROUPS
        context.primary_variable = "score"
        context.grouping_variable = "category"

        # Act
        result = compute_comparison_analysis(df, context)

        # Assert: Result is dict with expected keys
        assert isinstance(result, dict)
        assert result["type"] == "comparison"
        assert "test_type" in result
        assert "p_value" in result
        assert "statistic" in result

    def test_compute_comparison_analysis_twoGroups_reportsGroupSizes(self, sample_mixed_df, sample_context_compare):
        """The claim reports n, so the t-test result carries per-group sizes."""
        # Act
        result = compute_comparison_analysis(sample_mixed_df, sample_context_compare)

        # Assert
        assert result["test_type"] == "welch_t_test"
        assert result["group_sizes"] == {"A": 3, "B": 2}

    def test_compute_comparison_analysis_t_test_for_two_groups(self):
        """Test that compute_comparison_analysis uses t-test for two numeric groups."""
        # Arrange: Two groups with clear difference
        df = pl.DataFrame(
            {
                "outcome": [10, 11, 12, 13, 14, 50, 51, 52, 53, 54],
                "group": ["A", "A", "A", "A", "A", "B", "B", "B", "B", "B"],
            }
        )
        context = AnalysisContext()
        context.inferred_intent = AnalysisIntent.COMPARE_GROUPS
        context.primary_variable = "outcome"
        context.grouping_variable = "group"

        # Act
        result = compute_comparison_analysis(df, context)

        # Assert: Welch t-test performed (unequal variances are not assumed away)
        assert result["test_type"] == "welch_t_test"
        assert "mean_diff" in result
        assert "ci_lower" in result
        assert "ci_upper" in result
        assert result["p_value"] < 0.05  # Should be significant

    def test_compute_comparison_analysis_anova_for_multiple_groups(self):
        """Test that compute_comparison_analysis uses ANOVA for multiple groups."""
        # Arrange: Three groups
        df = pl.DataFrame(
            {
                "outcome": [10, 11, 12, 20, 21, 22, 30, 31, 32],
                "group": ["A", "A", "A", "B", "B", "B", "C", "C", "C"],
            }
        )
        context = AnalysisContext()
        context.inferred_intent = AnalysisIntent.COMPARE_GROUPS
        context.primary_variable = "outcome"
        context.grouping_variable = "group"

        # Act
        result = compute_comparison_analysis(df, context)

        # Assert: ANOVA performed
        assert result["test_type"] == "anova"
        assert "n_groups" in result
        assert result["n_groups"] == 3
        assert "group_means" in result

    def test_compute_comparison_analysis_chi_square_for_categorical(self):
        """Chi-square for a categorical outcome when every expected count is at least 5."""
        # Arrange: 2x2 with cells 12/8 and 6/14 (all expected counts >= 9)
        df = pl.DataFrame(
            {
                "outcome": ["Yes"] * 12 + ["No"] * 8 + ["Yes"] * 6 + ["No"] * 14,
                "group": ["A"] * 20 + ["B"] * 20,
            }
        )
        context = AnalysisContext()
        context.inferred_intent = AnalysisIntent.COMPARE_GROUPS
        context.primary_variable = "outcome"
        context.grouping_variable = "group"

        # Act
        result = compute_comparison_analysis(df, context)

        # Assert: Chi-square performed
        assert result["test_type"] == "chi_square"
        assert "chi2" in result
        assert "dof" in result
        assert "contingency" in result

    def test_compute_comparison_analysis_handles_insufficient_data(self):
        """Test that compute_comparison_analysis handles insufficient data."""
        # Arrange: Not enough data
        df = pl.DataFrame({"outcome": [10], "group": ["A"]})
        context = AnalysisContext()
        context.inferred_intent = AnalysisIntent.COMPARE_GROUPS
        context.primary_variable = "outcome"
        context.grouping_variable = "group"

        # Act
        result = compute_comparison_analysis(df, context)

        # Assert: Error returned
        assert result["type"] == "comparison"
        assert "error" in result

    def test_compute_comparison_analysis_handles_single_group(self):
        """Test that compute_comparison_analysis handles single group."""
        # Arrange: Only one group
        df = pl.DataFrame(
            {
                "outcome": [10, 11, 12, 13],
                "group": ["A", "A", "A", "A"],
            }
        )
        context = AnalysisContext()
        context.inferred_intent = AnalysisIntent.COMPARE_GROUPS
        context.primary_variable = "outcome"
        context.grouping_variable = "group"

        # Act
        result = compute_comparison_analysis(df, context)

        # Assert: Error returned
        assert result["type"] == "comparison"
        assert "error" in result

    def test_comparison_analysis_with_string_numeric_outcome(self):
        """String numbers are read as numbers and their means compared."""
        # Arrange
        df = pl.DataFrame({"Treatment": ["A", "A", "B", "B"], "Viral Load": ["20", "120", "200", "150"]})
        context = AnalysisContext()
        context.inferred_intent = AnalysisIntent.COMPARE_GROUPS
        context.primary_variable = "Viral Load"
        context.grouping_variable = "Treatment"

        # Act
        result = compute_comparison_analysis(df, context)

        # Assert
        assert result["test_type"] == "welch_t_test"
        assert result["group_means"]["A"] == pytest.approx(70)
        assert result["group_means"]["B"] == pytest.approx(175)

    @pytest.mark.parametrize("limit", ["<20", "> 500"], ids=["below", "above"])
    def test_comparisonAnalysis_valueRecordedAsLimit_refusesNotSubstitutes(self, limit):
        """A value recorded as a limit is not its limit: main read "<20" as 20 (DQ review H4)."""
        # Arrange
        df = pl.DataFrame({"Treatment": ["A", "A", "B", "B"], "Viral Load": [limit, "120", "200", "150"]})
        context = AnalysisContext()
        context.inferred_intent = AnalysisIntent.COMPARE_GROUPS
        context.primary_variable = "Viral Load"
        context.grouping_variable = "Treatment"

        # Act
        result = compute_comparison_analysis(df, context)

        # Assert
        assert "group_means" not in result
        assert "recorded as a limit" in result["error"]

    def test_comparison_analysis_with_european_comma_format(self):
        """Test European comma format conversion (e.g., '1,234.5')."""
        # Arrange: European comma format, two varying values per group (a t-test needs n >= 2 and
        # within-group spread; constant groups are refused)
        df = pl.DataFrame(
            {
                "Treatment": ["A", "A", "B", "B"],
                "Score": ["1,233.5", "1,235.5", "2,344.6", "2,346.6"],
            }
        )

        context = AnalysisContext()
        context.inferred_intent = AnalysisIntent.COMPARE_GROUPS
        context.primary_variable = "Score"
        context.grouping_variable = "Treatment"

        # Act
        result = compute_comparison_analysis(df, context)

        # Assert: Verify conversion and mean computation
        assert result["test_type"] in ["welch_t_test", "anova"]
        if "group_means" in result:
            assert result["group_means"]["A"] == pytest.approx(1234.5, rel=0.01)
            assert result["group_means"]["B"] == pytest.approx(2345.6, rel=0.01)
        elif "group1_mean" in result and "group2_mean" in result:
            means = [result["group1_mean"], result["group2_mean"]]
            assert any(abs(m - 1234.5) < 1 for m in means)
            assert any(abs(m - 2345.6) < 1 for m in means)

    def test_comparison_analysis_works_identically_single_file_and_multi_table(self):
        """Test that comparison analysis works identically for both upload types."""
        # Arrange: Create identical test data for both upload types
        single_file_df = pl.DataFrame(
            {
                "Treatment": ["A", "A", "B", "B"],
                "Viral Load": ["20", "120", "200", "150"],
            }
        )

        multi_table_df = pl.DataFrame(
            {
                "Treatment": ["A", "A", "B", "B"],
                "Viral Load": ["20", "120", "200", "150"],
            }
        )

        context = AnalysisContext()
        context.inferred_intent = AnalysisIntent.COMPARE_GROUPS
        context.primary_variable = "Viral Load"
        context.grouping_variable = "Treatment"

        # Act: Run analysis on both
        single_result = compute_comparison_analysis(single_file_df, context)
        multi_result = compute_comparison_analysis(multi_table_df, context)

        # Assert: Results must be identical
        assert single_result["test_type"] == multi_result["test_type"]
        if "group_means" in single_result and "group_means" in multi_result:
            assert single_result["group_means"]["A"] == multi_result["group_means"]["A"]
            assert single_result["group_means"]["B"] == multi_result["group_means"]["B"]
        elif "group1_mean" in single_result and "group1_mean" in multi_result:
            assert single_result["group1_mean"] == multi_result["group1_mean"]
            assert single_result["group2_mean"] == multi_result["group2_mean"]


class TestComputePredictorAnalysis:
    """Test compute_predictor_analysis function."""

    def test_compute_predictor_analysis_returns_serializable_dict(self, sample_context_predictor):
        """Test that compute_predictor_analysis returns serializable dict."""
        # Arrange: Binary outcome with predictors
        df = pl.DataFrame(
            {
                "outcome": [0, 0, 0, 1, 1, 1, 0, 1, 0, 1],
                "age": [25, 30, 35, 40, 45, 50, 55, 60, 65, 70],
                # Not a linear function of age (collinear predictors cannot be estimated separately)
                "score": [40, 10, 70, 20, 90, 30, 60, 100, 50, 80],
            }
        )

        # Act
        result = compute_predictor_analysis(df, sample_context_predictor)

        # Assert: Result is dict with expected keys
        assert isinstance(result, dict)
        assert result["type"] == "predictor"
        assert "outcome_col" in result
        assert "summary" in result
        assert "significant_predictors" in result
        assert "schema" in result

    def test_compute_predictor_analysis_handles_insufficient_data(self, sample_context_predictor):
        """Test that compute_predictor_analysis handles insufficient data."""
        # Arrange: Not enough observations
        df = pl.DataFrame(
            {
                "outcome": [0, 1],
                "age": [25, 30],
                "score": [10, 20],
            }
        )

        # Act
        result = compute_predictor_analysis(df, sample_context_predictor)

        # Assert: Error returned
        assert result["type"] == "predictor"
        assert "error" in result

    def test_compute_predictor_analysis_handles_non_binary_outcome(self, sample_context_predictor):
        """Test that compute_predictor_analysis handles non-binary outcome."""
        # Arrange: Non-binary outcome
        df = pl.DataFrame(
            {
                "outcome": [0, 1, 2, 3, 4],
                "age": [25, 30, 35, 40, 45],
                "score": [10, 20, 30, 40, 50],
            }
        )

        # Act
        result = compute_predictor_analysis(df, sample_context_predictor)

        # Assert: Error returned
        assert result["type"] == "predictor"
        assert "error" in result

    def test_compute_predictor_analysis_noPredictors_returnsActionableError(
        self, sample_mixed_df, sample_context_predictor
    ):
        """An empty predictor list must not reach the regression as 'outcome ~'."""
        # Arrange
        sample_context_predictor.predictor_variables = []

        # Act
        result = compute_predictor_analysis(sample_mixed_df, sample_context_predictor)

        # Assert
        assert result["type"] == "predictor"
        assert result["error"].startswith(f"No predictors specified for {sample_context_predictor.primary_variable}")


class TestComputeSurvivalAnalysis:
    """Test compute_survival_analysis function."""

    def test_compute_survival_analysis_returns_serializable_dict(self, sample_context_survival):
        """Test that compute_survival_analysis returns serializable dict."""
        # Arrange: Survival data
        df = pl.DataFrame(
            {
                "time": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                "event": [1, 1, 0, 1, 0, 1, 1, 0, 1, 0],
            }
        )

        # Act
        result = compute_survival_analysis(df, sample_context_survival)

        # Assert: Result is dict with expected keys
        assert isinstance(result, dict)
        assert result["type"] == "survival"
        assert "time_col" in result
        assert "event_col" in result
        assert "summary" in result
        assert "unique_values" in result

    def test_compute_survival_analysis_handles_insufficient_data(self, sample_context_survival):
        """Test that compute_survival_analysis handles insufficient data."""
        # Arrange: Not enough observations
        df = pl.DataFrame({"time": [10, 20], "event": [1, 0]})

        # Act
        result = compute_survival_analysis(df, sample_context_survival)

        # Assert: Error returned
        assert result["type"] == "survival"
        assert "error" in result

    def test_compute_survival_analysis_handles_non_binary_event(self, sample_context_survival):
        """Test that compute_survival_analysis handles non-binary event variable."""
        # Arrange: Non-binary event
        df = pl.DataFrame(
            {
                "time": [10, 20, 30, 40, 50],
                "event": [0, 1, 2, 3, 4],
            }
        )

        # Act
        result = compute_survival_analysis(df, sample_context_survival)

        # Assert: Error returned
        assert result["type"] == "survival"
        assert "error" in result


class TestComputeRelationshipAnalysis:
    """Test compute_relationship_analysis function."""

    def test_compute_relationship_analysis_returns_serializable_dict(self, sample_context_relationship):
        """Test that compute_relationship_analysis returns serializable dict."""
        # Arrange: Multiple numeric variables
        df = pl.DataFrame(
            {
                "age": [25, 30, 35, 40, 45],
                "score": [10, 20, 30, 40, 50],
                "value": [1.5, 2.5, 3.5, 4.5, 5.5],
            }
        )

        # Act
        result = compute_relationship_analysis(df, sample_context_relationship)

        # Assert: Result is dict with expected keys
        assert isinstance(result, dict)
        assert result["type"] == "relationship"
        assert "variables" in result
        assert "correlations" in result
        assert "strong_correlations" in result

    def test_compute_relationship_analysis_handles_insufficient_variables(self):
        """Test that compute_relationship_analysis handles insufficient variables."""
        # Arrange: Only one variable
        df = pl.DataFrame({"age": [25, 30, 35]})
        context = AnalysisContext()
        context.inferred_intent = AnalysisIntent.EXPLORE_RELATIONSHIPS
        context.predictor_variables = ["age"]

        # Act
        result = compute_relationship_analysis(df, context)

        # Assert: Error returned
        assert result["type"] == "relationship"
        assert "error" in result

    def test_compute_relationship_analysis_handles_insufficient_observations(self, sample_context_relationship):
        """Test that compute_relationship_analysis handles insufficient observations."""
        # Arrange: Not enough observations
        df = pl.DataFrame(
            {
                "age": [25, 30],
                "score": [10, 20],
                "value": [1.5, 2.5],
            }
        )

        # Act
        result = compute_relationship_analysis(df, sample_context_relationship)

        # Assert: Error returned
        assert result["type"] == "relationship"
        assert "error" in result

    def test_compute_relationship_analysis_identifies_strong_correlations(self):
        """Test that compute_relationship_analysis identifies strong correlations."""
        # Arrange: Highly correlated variables
        df = pl.DataFrame(
            {
                "var1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                "var2": [2, 4, 6, 8, 10, 12, 14, 16, 18, 20],  # Perfect correlation
            }
        )
        context = AnalysisContext()
        context.inferred_intent = AnalysisIntent.EXPLORE_RELATIONSHIPS
        context.predictor_variables = ["var1", "var2"]

        # Act
        result = compute_relationship_analysis(df, context)

        # Assert: Strong correlation identified
        assert len(result["strong_correlations"]) > 0
        assert any(abs(item["correlation"]) >= 0.5 for item in result["strong_correlations"])


class TestComputeAnalysisByType:
    """Test compute_analysis_by_type router function."""

    def test_compute_analysis_by_type_routes_to_descriptive(self, sample_numeric_df, sample_context_describe):
        """Test that compute_analysis_by_type routes DESCRIBE intent correctly."""
        # Act
        result = compute_analysis_by_type(sample_numeric_df, sample_context_describe)

        # Assert: Descriptive analysis returned
        assert result["type"] == "descriptive"

    def test_compute_analysis_by_type_routes_to_comparison(self):
        """Test that compute_analysis_by_type routes COMPARE_GROUPS intent correctly."""
        # Arrange
        df = pl.DataFrame(
            {
                "score": [10, 20, 30, 40, 50, 60],
                "category": ["A", "A", "A", "B", "B", "B"],
            }
        )
        context = AnalysisContext()
        context.inferred_intent = AnalysisIntent.COMPARE_GROUPS
        context.primary_variable = "score"
        context.grouping_variable = "category"

        # Act
        result = compute_analysis_by_type(df, context)

        # Assert: Comparison analysis returned
        assert result["type"] == "comparison"

    def test_compute_analysis_by_type_routes_to_predictor(self, sample_context_predictor):
        """Test that compute_analysis_by_type routes FIND_PREDICTORS intent correctly."""
        # Arrange
        df = pl.DataFrame(
            {
                "outcome": [0, 0, 0, 1, 1, 1, 0, 1, 0, 1],
                "age": [25, 30, 35, 40, 45, 50, 55, 60, 65, 70],
                "score": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
            }
        )

        # Act
        result = compute_analysis_by_type(df, sample_context_predictor)

        # Assert: Predictor analysis returned
        assert result["type"] == "predictor"

    def test_compute_analysis_by_type_routes_to_survival(self, sample_context_survival):
        """Test that compute_analysis_by_type routes EXAMINE_SURVIVAL intent correctly."""
        # Arrange
        df = pl.DataFrame(
            {
                "time": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                "event": [1, 1, 0, 1, 0, 1, 1, 0, 1, 0],
            }
        )

        # Act
        result = compute_analysis_by_type(df, sample_context_survival)

        # Assert: Survival analysis returned
        assert result["type"] == "survival"

    def test_compute_analysis_by_type_routes_to_relationship(self, sample_context_relationship):
        """Test that compute_analysis_by_type routes EXPLORE_RELATIONSHIPS intent correctly."""
        # Arrange
        df = pl.DataFrame(
            {
                "age": [25, 30, 35, 40, 45],
                "score": [10, 20, 30, 40, 50],
                "value": [1.5, 2.5, 3.5, 4.5, 5.5],
            }
        )

        # Act
        result = compute_analysis_by_type(df, sample_context_relationship)

        # Assert: Relationship analysis returned
        assert result["type"] == "relationship"

    def test_compute_analysis_by_type_routes_to_count(self, sample_numeric_df, sample_context_count):
        """Test that compute_analysis_by_type routes COUNT intent correctly."""
        # Act
        result = compute_analysis_by_type(sample_numeric_df, sample_context_count)

        # Assert: Count analysis returned
        assert result["type"] == "count"
        assert "total_count" in result
        assert result["total_count"] == sample_numeric_df.height
        assert "headline" in result

    def test_compute_analysis_by_type_routes_to_count_with_grouping(self, sample_context_count):
        """Test that compute_analysis_by_type routes COUNT intent with grouping correctly."""
        # Arrange: COUNT intent with grouping variable
        df = pl.DataFrame(
            {
                "category": ["A", "A", "B", "B", "C"],
                "value": [10, 20, 30, 40, 50],
            }
        )
        context = sample_context_count
        context.grouping_variable = "category"

        # Act
        result = compute_analysis_by_type(df, context)

        # Assert: Count analysis with grouping returned
        assert result["type"] == "count"
        assert "total_count" in result
        assert result["total_count"] == df.height
        assert "grouped_by" in result
        assert result["grouped_by"] == "category"
        assert "group_counts" in result
        assert len(result["group_counts"]) == 3  # A, B, C

    def test_compute_count_analysis_applies_filters(self, sample_context_count):
        """Test that compute_count_analysis applies filters before counting."""
        # Arrange: DataFrame with filterable data
        from clinical_analytics.core.query_plan import FilterSpec

        df = pl.DataFrame(
            {
                "patient_id": [1, 2, 3, 4, 5],
                "status": ["active", "active", "inactive", "active", "inactive"],
                "value": [10, 20, 30, 40, 50],
            }
        )
        context = sample_context_count
        # Add filter: status == "active"
        context.filters = [FilterSpec(column="status", operator="==", value="active", exclude_nulls=True)]

        # Act
        result = compute_count_analysis(df, context)

        # Assert: Only filtered rows counted
        assert result["type"] == "count"
        assert result["total_count"] == 3  # Only 3 rows with status == "active"
        assert result["total_count"] < df.height  # Filtered count is less than total

    def test_compute_count_analysis_with_filters_and_grouping(self, sample_context_count):
        """Test that compute_count_analysis applies filters before grouping."""
        # Arrange: DataFrame with filterable and groupable data
        from clinical_analytics.core.query_plan import FilterSpec

        df = pl.DataFrame(
            {
                "patient_id": [1, 2, 3, 4, 5, 6],
                "status": ["active", "active", "active", "inactive", "active", "inactive"],
                "category": ["A", "A", "B", "A", "B", "B"],
            }
        )
        context = sample_context_count
        context.grouping_variable = "category"
        # Add filter: status == "active"
        context.filters = [FilterSpec(column="status", operator="==", value="active", exclude_nulls=True)]

        # Act
        result = compute_count_analysis(df, context)

        # Assert: Filtered and grouped counts
        assert result["type"] == "count"
        assert result["total_count"] == 4  # 4 rows with status == "active"
        assert result["grouped_by"] == "category"
        assert len(result["group_counts"]) == 2  # Categories A and B
        # Verify counts per category (after filtering)
        category_counts = {item["category"]: item["count"] for item in result["group_counts"]}
        assert category_counts["A"] == 2  # 2 active rows in category A
        assert category_counts["B"] == 2  # 2 active rows in category B

    def test_compute_count_analysis_without_filters_counts_all_rows(self, sample_context_count):
        """Test that compute_count_analysis counts all rows when no filters present."""
        # Arrange: DataFrame without filters
        df = pl.DataFrame(
            {
                "patient_id": [1, 2, 3, 4, 5],
                "status": ["active", "active", "inactive", "active", "inactive"],
            }
        )
        context = sample_context_count
        context.filters = []  # No filters

        # Act
        result = compute_count_analysis(df, context)

        # Assert: All rows counted
        assert result["type"] == "count"
        assert result["total_count"] == df.height  # All 5 rows

    def test_compute_analysis_by_type_handles_unknown_intent(self, sample_numeric_df):
        """Test that compute_analysis_by_type handles unknown intent."""
        # Arrange: Unknown intent
        context = AnalysisContext()
        context.inferred_intent = AnalysisIntent.UNKNOWN

        # Act
        result = compute_analysis_by_type(sample_numeric_df, context)

        # Assert: Error returned
        assert result["type"] == "unknown"
        assert "error" in result


class TestComputeMissingColumns:
    """Absent or unknown columns come back as a renderable error dict, never a Polars exception."""

    @pytest.mark.parametrize(
        ("compute_fn", "expected_type", "expected_error"),
        [
            (compute_comparison_analysis, "comparison", "Missing required columns for comparison"),
            (compute_predictor_analysis, "predictor", "No outcome variable specified"),
            (
                compute_survival_analysis,
                "survival",
                "Time and event variables required for survival analysis",
            ),
            (
                compute_relationship_analysis,
                "relationship",
                "Need at least 2 variables to examine relationships",
            ),
        ],
        ids=["comparison", "predictor", "survival", "relationship"],
    )
    def test_compute_noVariablesInContext_returnsError(
        self, sample_numeric_df, compute_fn, expected_type, expected_error
    ):
        """A context naming none of the columns the analysis needs returns that analysis' error."""
        # Arrange
        context = AnalysisContext()

        # Act
        result = compute_fn(sample_numeric_df, context)

        # Assert
        assert result["type"] == expected_type
        assert result["error"] == expected_error

    def test_compute_descriptive_analysis_unknownPrimaryVariable_returnsErrorWithColumns(self, sample_numeric_df):
        """An unknown primary variable names the variable and lists the real columns."""
        # Arrange
        context = AnalysisContext(inferred_intent=AnalysisIntent.DESCRIBE, primary_variable="not_a_column")

        # Act
        result = compute_descriptive_analysis(sample_numeric_df, context)

        # Assert
        assert result["error"].startswith("Column 'not_a_column' not found in dataset")
        assert result["available_columns"] == sample_numeric_df.columns

    def test_compute_relationship_analysis_unknownVariable_returnsErrorNamingIt(self, sample_numeric_df):
        """A relationship request naming an absent column says which one is missing."""
        # Arrange
        context = AnalysisContext(
            inferred_intent=AnalysisIntent.EXPLORE_RELATIONSHIPS,
            predictor_variables=["age", "not_a_column"],
        )

        # Act
        result = compute_relationship_analysis(sample_numeric_df, context)

        # Assert
        assert result["type"] == "relationship"
        assert result["error"].startswith("Variables not found in data: not_a_column")

    @pytest.mark.parametrize(
        ("intent", "context_fields", "expected_type"),
        [
            (
                AnalysisIntent.COMPARE_GROUPS,
                {"primary_variable": "age", "grouping_variable": "not_a_column"},
                "comparison",
            ),
            (
                AnalysisIntent.FIND_PREDICTORS,
                {"primary_variable": "not_a_column", "predictor_variables": ["age"]},
                "predictor",
            ),
            (AnalysisIntent.EXAMINE_SURVIVAL, {"time_variable": "age", "event_variable": "not_a_column"}, "survival"),
            (AnalysisIntent.EXPLORE_RELATIONSHIPS, {"predictor_variables": ["age", "not_a_column"]}, "relationship"),
        ],
        ids=["comparison", "predictor", "survival", "relationship"],
    )
    def test_compute_analysis_by_type_unknownColumn_returnsErrorBeforeSelect(
        self, sample_numeric_df, intent, context_fields, expected_type
    ):
        """The router reports unknown columns instead of letting df.select raise ColumnNotFoundError."""
        # Arrange
        context = AnalysisContext(inferred_intent=intent, **context_fields)

        # Act
        result = compute_analysis_by_type(sample_numeric_df, context)

        # Assert
        assert result["type"] == expected_type
        assert result["error"] == "Column(s) not found in dataset: ['not_a_column']"
        assert result["available_columns"] == sample_numeric_df.columns


class TestComputeMissingAndDegenerateInputs:
    """NaN rows, constant groups and Fisher orientation (PR #66 statistics review)."""

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")], ids=["nan", "inf", "neg_inf"])
    def test_compute_comparison_nonFiniteOutcome_excludesAndCountsRow(self, bad, sample_context_compare):
        # Arrange: 10 finite values per group plus one non-finite value in group A
        df = pl.DataFrame(
            {
                "score": [float(v) for v in range(1, 11)] + [float(v) for v in range(5, 15)] + [bad],
                "category": ["A"] * 10 + ["B"] * 10 + ["A"],
            }
        )

        # Act
        result = compute_comparison_analysis(df, sample_context_compare)

        # Assert
        assert (result["n_used"], result["n_excluded"]) == (20, 1)
        assert result["group_means"] == pytest.approx({"A": 5.5, "B": 9.5})
        assert np.isfinite(result["p_value"])

    @pytest.mark.parametrize("bad", [float("nan"), float("inf")], ids=["nan", "inf"])
    def test_compute_predictor_nonFinitePredictor_excludesAndCountsRow(self, bad, sample_context_predictor):
        # Arrange: 40 complete rows plus one with a non-finite age
        rng = np.random.default_rng(3)
        df = pl.DataFrame(
            {
                "outcome": [*rng.integers(0, 2, 40).tolist(), 1],
                "age": [*rng.normal(50.0, 10.0, 40).tolist(), bad],
                "score": rng.normal(0.0, 1.0, 41).tolist(),
            }
        )

        # Act
        result = compute_predictor_analysis(df, sample_context_predictor)

        # Assert
        assert (result["n_used"], result["n_excluded"]) == (40, 1)
        assert "error" not in result

    @pytest.mark.parametrize("bad", [float("nan"), float("inf")], ids=["nan", "inf"])
    def test_compute_survival_nonFiniteTime_excludesAndCountsRow(self, bad, sample_context_survival):
        # Arrange: 12 complete rows plus one with a non-finite time
        df = pl.DataFrame(
            {
                "time": [float(t) for t in range(10, 130, 10)] + [bad],
                "event": [1, 0] * 6 + [1],
            }
        )

        # Act
        result = compute_survival_analysis(df, sample_context_survival)

        # Assert
        assert (result["n_used"], result["n_excluded"]) == (12, 1)
        assert result["median_survival"] is not None

    def test_compute_relationship_nanValue_excludesRowFromCorrelation(self, sample_context_relationship):
        # Arrange: 5 complete rows plus one NaN
        df = pl.DataFrame(
            {
                "age": [30.0, 40.0, 50.0, 60.0, 70.0, 80.0],
                "score": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                "value": [2.0, 4.0, 6.0, 8.0, 10.0, float("nan")],
            }
        )

        # Act
        result = compute_relationship_analysis(df, sample_context_relationship)

        # Assert
        assert result["n_observations"] == 5
        assert all(np.isfinite(row["correlation"]) for row in result["correlations"])

    def test_compute_descriptive_nanValue_reportsFiniteMean(self):
        # Arrange
        df = pl.DataFrame({"ldl": [100.0, float("nan"), 140.0]})
        context = AnalysisContext()
        context.inferred_intent = AnalysisIntent.DESCRIBE
        context.primary_variable = "ldl"

        # Act
        result = compute_descriptive_analysis(df, context)

        # Assert
        assert result["mean"] == pytest.approx(120.0)

    @pytest.mark.parametrize(
        ("a_values", "b_values", "constant"),
        [
            ([5.0] * 10, [8.0] * 10, "A, B"),
            ([5.0] * 10, [float(v) for v in range(1, 11)], "A"),
            ([float(v) for v in range(1, 11)], [8.0, 8.0], "B"),
        ],
        ids=["both_constant", "one_constant", "small_constant_group"],
    )
    def test_compute_comparison_constantGroupWelch_refusesWithoutPValue(
        self, a_values, b_values, constant, sample_context_compare
    ):
        # Arrange
        df = pl.DataFrame({"score": a_values + b_values, "category": ["A"] * len(a_values) + ["B"] * len(b_values)})

        # Act
        result = compute_comparison_analysis(df, sample_context_compare)

        # Assert
        assert result["error"] == UNTESTABLE_T_TEST
        assert "p_value" not in result
        assert "ci_lower" not in result

    @pytest.mark.parametrize("other_level", ["alive", "survived"], ids=["alive", "survived"])
    def test_compute_comparison_fisherOtherLevelRenamed_oddsRatioUnchanged(self, other_level, sample_context_compare):
        # Arrange: A has 1 dead / 8 other, B has 8 dead / 1 other; odds of death B vs A = 8 / (1/8) = 64
        df = pl.DataFrame(
            {
                "score": ["dead"] + [other_level] * 8 + ["dead"] * 8 + [other_level],
                "category": ["A"] * 9 + ["B"] * 9,
            }
        )

        # Act
        result = compute_comparison_analysis(df, sample_context_compare)

        # Assert
        assert result["test_type"] == "fisher_exact"
        assert result["event_level"] == "dead"
        assert result["odds_ratio"] == pytest.approx(64.0)
        assert (result["odds_ratio_group"], result["odds_ratio_reference"]) == ("B", "A")

    def test_compute_comparison_fisherUnknownEventLevel_omitsOddsRatio(self, sample_context_compare):
        # Arrange: text codes 1/2 have no stated event, so the ratio has no orientation
        df = pl.DataFrame({"score": ["1"] + ["2"] * 8 + ["1"] * 8 + ["2"], "category": ["A"] * 9 + ["B"] * 9})

        # Act
        result = compute_comparison_analysis(df, sample_context_compare)

        # Assert
        assert result["test_type"] == "fisher_exact"
        assert "odds_ratio" not in result
        assert np.isfinite(result["p_value"])


class TestComputeDegenerateGroupsReview:
    """Float-noise constants, degenerate ANOVA, descriptive counts (second review of #66 fixes)."""

    @pytest.mark.parametrize(
        ("a_values", "b_values"),
        [
            ([0.1 + 0.2] * 5 + [0.3] * 5, [0.1 * 7] * 5 + [0.7] * 5),
            ([1e8 + 1e-7] * 5 + [1e8] * 5, [2e8] * 10),
        ],
        ids=["float_noise", "large_magnitude_noise"],
    )
    def test_compute_comparison_floatNoiseConstantWelch_refusesWithoutPValue(
        self, a_values, b_values, sample_context_compare
    ):
        # Arrange
        df = pl.DataFrame({"score": a_values + b_values, "category": ["A"] * 10 + ["B"] * 10})

        # Act
        result = compute_comparison_analysis(df, sample_context_compare)

        # Assert
        assert result["error"] == UNTESTABLE_T_TEST
        assert "p_value" not in result

    def test_compute_comparison_smallRealSpreadWelch_isNotRefused(self, sample_context_compare):
        # Arrange: spread of 0.01 on values near 5 is real variation, not float noise
        df = pl.DataFrame({"score": [5.0, 5.01] * 5 + [6.0, 6.01] * 5, "category": ["A"] * 10 + ["B"] * 10})

        # Act
        result = compute_comparison_analysis(df, sample_context_compare)

        # Assert
        assert result["test_type"] == "welch_t_test"
        assert np.isfinite(result["p_value"])

    @pytest.mark.parametrize(
        ("values", "groups"),
        [
            ([10.0] * 5 + [20.0] * 5 + [30.0] * 5, ["A"] * 5 + ["B"] * 5 + ["C"] * 5),
            ([10.0, 20.0, 30.0], ["A", "B", "C"]),
            ([10.0, 11.0, 20.0, 21.0, 30.0], ["A", "A", "B", "B", "C"]),
        ],
        ids=["all_constant", "one_per_group", "one_small_group"],
    )
    def test_compute_comparison_degenerateAnova_refusesWithoutPValue(self, values, groups, sample_context_compare):
        # Arrange
        df = pl.DataFrame({"score": values, "category": groups})

        # Act
        result = compute_comparison_analysis(df, sample_context_compare)

        # Assert: one message for every cause, so it does not say a group has one patient
        assert result["error"] == UNTESTABLE_ANOVA
        assert "p_value" not in result

    @pytest.mark.parametrize(
        ("values", "expected_counts"),
        [
            ([100.0, float("nan"), 140.0], (2, 1)),
            ([100.0, float("inf"), None, 140.0], (2, 2)),
        ],
        ids=["nan", "inf_and_null"],
    )
    def test_compute_descriptive_nonFiniteValue_countsMatchMean(self, values, expected_counts):
        # Arrange
        df = pl.DataFrame({"ldl": values}, schema={"ldl": pl.Float64})
        context = AnalysisContext()
        context.inferred_intent = AnalysisIntent.DESCRIBE
        context.primary_variable = "ldl"

        # Act
        result = compute_descriptive_analysis(df, context)

        # Assert
        assert (result["non_null_count"], result["null_count"]) == expected_counts
        assert result["mean"] == pytest.approx(120.0)

    def test_compute_comparison_textZeroOneOutcome_reportsEventLevelOne(self, sample_context_compare):
        # Arrange: 0/1 stored as text; the numbers parse to floats
        df = pl.DataFrame({"score": ["1"] + ["0"] * 8 + ["1"] * 8 + ["0"], "category": ["A"] * 9 + ["B"] * 9})

        # Act
        result = compute_comparison_analysis(df, sample_context_compare)

        # Assert
        assert result["event_level"] == "1"
        assert result["odds_ratio"] == pytest.approx(64.0)


class TestComputeNonFiniteEffectEstimates:
    """Effect estimates are a finite number or None in the payload; p-values are kept."""

    @pytest.mark.parametrize(
        ("dead_a", "dead_b", "expected_or"),
        [
            (0, 8, None),  # no event in the reference group A: OR is inf
            (1, 9, None),  # every patient in B has the event: OR is inf
            (8, 0, 0.0),  # no event in B: OR is 0, which is finite
        ],
        ids=["zero_event_reference", "zero_non_event_group", "zero_event_group"],
    )
    def test_compute_comparison_fisherZeroCell_reportsNoneNotInf(
        self, dead_a, dead_b, expected_or, sample_context_compare
    ):
        # Arrange: 9 patients per group
        outcome = ["dead"] * dead_a + ["alive"] * (9 - dead_a) + ["dead"] * dead_b + ["alive"] * (9 - dead_b)
        df = pl.DataFrame({"score": outcome, "category": ["A"] * 9 + ["B"] * 9})

        # Act
        result = compute_comparison_analysis(df, sample_context_compare)

        # Assert
        assert result["test_type"] == "fisher_exact"
        assert result["odds_ratio"] == expected_or
        assert np.isfinite(result["p_value"])

    @pytest.mark.parametrize("bad", [float("inf"), float("nan")], ids=["inf", "nan"])
    def test_compute_predictor_nonFiniteOddsRatioOrCi_reportsNone(self, bad, sample_context_predictor, monkeypatch):
        # Arrange: a fit whose CI overflowed. PANDAS EXCEPTION: run_logistic_regression returns pandas.
        import pandas as pd

        from clinical_analytics.analysis import stats as stats_module

        summary = pd.DataFrame(
            {"Odds Ratio": [1.0, bad], "CI Lower": [0.5, 2.0], "CI Upper": [2.0, bad], "P-Value": [0.5, 0.001]},
            index=["Intercept", "age"],
        )
        monkeypatch.setattr(stats_module, "run_logistic_regression", lambda *_args, **_kwargs: (None, summary))
        rng = np.random.default_rng(4)
        df = pl.DataFrame({"outcome": rng.integers(0, 2, 40), "age": rng.normal(50.0, 9.0, 40)})
        sample_context_predictor.predictor_variables = ["age"]

        # Act
        result = compute_predictor_analysis(df, sample_context_predictor)

        # Assert
        assert (result["summary"]["age"]["Odds Ratio"], result["summary"]["age"]["CI Upper"]) == (None, None)
        top = result["significant_predictors"][0]
        assert (top["odds_ratio"], top["ci_upper"], top["p_value"]) == (None, None, 0.001)


@pytest.mark.parametrize("token", ["inf", "-inf", "Infinity"], ids=["inf", "neg_inf", "infinity_word"])
def test_compute_comparison_infTextInNumericOutcome_excludedAsMissing(token, sample_context_compare):
    # Arrange: 20 numbers stored as text plus one "inf"
    values = [str(v) for v in range(100, 120)] + [token]
    df = pl.DataFrame({"score": values, "category": ["A", "B"] * 10 + ["A"]})

    # Act
    result = compute_comparison_analysis(df, sample_context_compare)

    # Assert
    assert result["test_type"] == "welch_t_test"
    assert (result["n_used"], result["n_excluded"]) == (20, 1)


@pytest.mark.parametrize(
    ("compute", "context_fixture", "floor", "data", "expected"),
    [
        (
            compute_predictor_analysis,
            "sample_context_predictor",
            "MIN_REGRESSION_N",
            {"outcome": [0, 1] * 2, "age": [25, 30, 35, 40], "score": [10, 20, 30, 40]},
            "Need at least 7 complete observations",
        ),
        (
            compute_survival_analysis,
            "sample_context_survival",
            "MIN_SURVIVAL_N",
            {"time": [10, 20, 30, 40], "event": [1, 0, 1, 0]},
            "Need at least 7 complete observations",
        ),
        (
            compute_relationship_analysis,
            "sample_context_relationship",
            "MIN_CORRELATION_N",
            {"age": [25, 30, 35, 40], "score": [10, 25, 30, 45], "value": [3, 1, 4, 1]},
            "Need at least 7 observations with complete data",
        ),
        (
            compute_comparison_analysis,
            "sample_context_compare",
            "MIN_GROUP_N",
            {"score": [5.0, 5.5, 6.0, 6.5], "category": ["A", "A", "B", "B"]},
            "A t-test cannot be run on score by category. Every group needs at least 7 observations, and the "
            "values must differ within each group. Try groups with more data.",
        ),
        (
            compute_comparison_analysis,
            "sample_context_compare",
            "MIN_GROUP_N",
            {"score": [5.0, 5.5, 6.0, 6.5, 7.0, 7.5], "category": ["A", "A", "B", "B", "C", "C"]},
            "An ANOVA cannot be run on score by category. Every group needs at least 7 observations, and the "
            "values must differ within at least one group. Try groups with more data.",
        ),
    ],
    ids=["regression", "survival", "correlation", "t_test", "anova"],
)
def test_compute_analysis_belowSampleFloor_errorNamesTheFloorConstant(
    compute, context_fixture, floor, data, expected, request, monkeypatch
):
    # Arrange: raise the floor to 7, above the four (or six) synthetic rows
    monkeypatch.setattr(f"clinical_analytics.analysis.compute.{floor}", 7)
    context = request.getfixturevalue(context_fixture)

    # Act
    result = compute(pl.DataFrame(data), context)

    # Assert: the message quotes the constant, not a number written into the string
    assert result["error"] == expected


@pytest.mark.parametrize(
    ("values", "groups", "expected"),
    [
        ([5.0, 5.5, 6.0, 6.5, 7.0], ["A", "A", "B", "B", "C"], UNTESTABLE_ANOVA),
        ([5.0, 6.0, 7.0], ["A", "B", "C"], UNTESTABLE_ANOVA),
        ([5.0, 5.5, 6.0, 9.0], ["A", "A", "A", "B"], UNTESTABLE_T_TEST),
        ([5.0, 9.0], ["A", "B"], UNTESTABLE_T_TEST),
    ],
    ids=["anova_one_patient_group", "anova_one_patient_each", "t_test_one_patient_group", "t_test_one_patient_each"],
)
def test_compute_comparison_onePatientGroup_errorNamesNoGroupAndNoCount(
    values, groups, expected, sample_context_compare
):
    # Arrange: a group below the floor of 2 has exactly one patient; the error reaches the user
    # as an issue, outside the presenter, so naming the group would state that count
    df = pl.DataFrame({"score": values, "category": groups})

    # Act
    error = compute_comparison_analysis(df, sample_context_compare)["error"]

    # Assert: the same words as for groups whose values do not vary, and no group label
    assert error == expected
    assert "(" not in error


@pytest.mark.parametrize(
    ("plan_filtered", "expected"), [(False, 4), (True, 0)], ids=["legacy_context_filters", "already_filtered_rows"]
)
def test_compute_analysis_by_type_contextFilters_reportRowsFilteredOut(plan_filtered, expected):
    # Arrange: 12 synthetic rows, 8 aged 50 or over; the filter rides on context.filters only
    from clinical_analytics.core.query_plan import FilterSpec

    df = pl.DataFrame({"age": [40, 45, 50, 55, 60, 65] * 2, "ldl": [float(v) for v in range(12)]})
    if plan_filtered:
        df = df.filter(pl.col("age") >= 50)
    context = AnalysisContext(
        primary_variable="ldl",
        inferred_intent=AnalysisIntent.DESCRIBE,
        filters=[FilterSpec(column="age", operator=">=", value=50)],
    )

    # Act
    result = compute_analysis_by_type(df, context)

    # Assert
    assert result["n_filtered_out"] == expected
    assert result["row_count"] == 8
