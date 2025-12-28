"""
Test Compute Functions - Phase 1

Unit tests for pure compute functions in analysis/compute.py.
These functions have no UI dependencies and can be tested independently.

Test name follows: test_unit_scenario_expectedBehavior
"""

import polars as pl
import pytest

from clinical_analytics.analysis.compute import (
    compute_analysis_by_type,
    compute_comparison_analysis,
    compute_descriptive_analysis,
    compute_predictor_analysis,
    compute_relationship_analysis,
    compute_survival_analysis,
)
from clinical_analytics.ui.components.question_engine import AnalysisContext, AnalysisIntent


@pytest.fixture
def sample_numeric_df():
    """Create sample Polars DataFrame with numeric columns."""
    return pl.DataFrame(
        {
            "age": [25, 30, 35, 40, 45, 50, 55, 60],
            "score": [10, 20, 30, 40, 50, 60, 70, 80],
            "value": [1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5],
        }
    )


@pytest.fixture
def sample_categorical_df():
    """Create sample Polars DataFrame with categorical columns."""
    return pl.DataFrame(
        {
            "category": ["A", "B", "A", "B", "A", "B", "A", "B"],
            "status": ["active", "inactive", "active", "inactive", "active", "inactive", "active", "inactive"],
        }
    )


@pytest.fixture
def sample_mixed_df():
    """Create sample Polars DataFrame with mixed column types."""
    return pl.DataFrame(
        {
            "age": [25, 30, 35, 40, 45],
            "category": ["A", "B", "A", "B", "A"],
            "score": [10, 20, 30, 40, 50],
        }
    )


@pytest.fixture
def sample_context_describe():
    """Create AnalysisContext for descriptive analysis."""
    context = AnalysisContext()
    context.inferred_intent = AnalysisIntent.DESCRIBE
    context.primary_variable = "all"
    return context


@pytest.fixture
def sample_context_compare():
    """Create AnalysisContext for comparison analysis."""
    context = AnalysisContext()
    context.inferred_intent = AnalysisIntent.COMPARE_GROUPS
    context.primary_variable = "score"
    context.grouping_variable = "category"
    return context


@pytest.fixture
def sample_context_predictor():
    """Create AnalysisContext for predictor analysis."""
    context = AnalysisContext()
    context.inferred_intent = AnalysisIntent.FIND_PREDICTORS
    context.primary_variable = "outcome"
    context.predictor_variables = ["age", "score"]
    return context


@pytest.fixture
def sample_context_survival():
    """Create AnalysisContext for survival analysis."""
    context = AnalysisContext()
    context.inferred_intent = AnalysisIntent.EXAMINE_SURVIVAL
    context.time_variable = "time"
    context.event_variable = "event"
    return context


@pytest.fixture
def sample_context_relationship():
    """Create AnalysisContext for relationship analysis."""
    context = AnalysisContext()
    context.inferred_intent = AnalysisIntent.EXPLORE_RELATIONSHIPS
    context.predictor_variables = ["age", "score", "value"]
    return context


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

        # Assert: T-test performed
        assert result["test_type"] == "t_test"
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
        """Test that compute_comparison_analysis uses chi-square for categorical outcome."""
        # Arrange: Categorical outcome
        df = pl.DataFrame(
            {
                "outcome": ["Yes", "Yes", "No", "No", "Yes", "No", "Yes", "No"],
                "group": ["A", "A", "A", "A", "B", "B", "B", "B"],
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


class TestComputePredictorAnalysis:
    """Test compute_predictor_analysis function."""

    def test_compute_predictor_analysis_returns_serializable_dict(self, sample_context_predictor):
        """Test that compute_predictor_analysis returns serializable dict."""
        # Arrange: Binary outcome with predictors
        df = pl.DataFrame(
            {
                "outcome": [0, 0, 0, 1, 1, 1, 0, 1, 0, 1],
                "age": [25, 30, 35, 40, 45, 50, 55, 60, 65, 70],
                "score": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
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
