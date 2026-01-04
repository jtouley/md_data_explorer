"""
Tests for variable type detector component.

The detector uses DATA-DRIVEN heuristics only (no hardcoded column name patterns).
Detection is based on:
- Data types (numeric, string, datetime)
- Cardinality (unique value counts)
- Value distributions
"""

from datetime import datetime

import polars as pl
from clinical_analytics.ui.components.variable_detector import VariableTypeDetector


class TestVariableTypeDetector:
    """Test suite for VariableTypeDetector."""

    def test_detect_binary_yes_no(self):
        """Test detecting binary variable with yes/no values."""
        series = pl.Series("outcome", ["yes", "no", "yes", "no", "yes"])
        var_type, metadata = VariableTypeDetector.detect_variable_type(series, "outcome")

        assert var_type == "binary"
        assert metadata["unique_count"] == 2

    def test_detect_binary_1_0(self):
        """Test detecting binary variable with 1/0 values."""
        series = pl.Series("status", [1, 0, 1, 0, 1])
        var_type, metadata = VariableTypeDetector.detect_variable_type(series, "status")

        assert var_type == "binary"

    def test_detect_binary_true_false(self):
        """Test detecting binary variable with True/False values."""
        series = pl.Series("active", [True, False, True, False])
        var_type, metadata = VariableTypeDetector.detect_variable_type(series, "active")

        assert var_type == "binary"

    def test_detect_categorical(self):
        """Test detecting categorical variable."""
        series = pl.Series("category", ["A", "B", "C", "A", "B", "C", "A"])
        var_type, metadata = VariableTypeDetector.detect_variable_type(series, "category")

        assert var_type == "categorical"
        assert metadata["unique_count"] == 3

    def test_detect_continuous(self):
        """Test detecting continuous variable (high cardinality numeric)."""
        # Needs >20 unique values to be continuous, and some nulls to not be identifier
        values = [1.5 + i * 0.1 for i in range(30)]
        values[5] = None  # Add null so not 100% unique
        series = pl.Series("measurement", values)
        var_type, metadata = VariableTypeDetector.detect_variable_type(series, "measurement")

        assert var_type == "continuous"

    def test_detect_datetime(self):
        """Test detecting datetime variable with Date type."""
        series = pl.Series("date", [datetime(2020, 1, 1), datetime(2020, 2, 1), datetime(2020, 3, 1)])
        var_type, metadata = VariableTypeDetector.detect_variable_type(series, "date")

        assert var_type == "datetime"

    def test_detect_datetime_parsed(self):
        """Test detecting datetime with Datetime type."""
        series = pl.Series("timestamp", [datetime(2020, 1, 1), datetime(2020, 2, 1), datetime(2020, 3, 1)])
        var_type, metadata = VariableTypeDetector.detect_variable_type(series, "timestamp")

        assert var_type == "datetime"

    def test_detect_id_column_by_uniqueness(self):
        """Test detecting ID column by high uniqueness (data-driven, not name pattern)."""
        # 100 unique values, no nulls = identifier
        series = pl.Series("secret_name", [f"P{i:05d}" for i in range(100)])
        var_type, metadata = VariableTypeDetector.detect_variable_type(series, "secret_name")

        assert var_type == "identifier"
        assert metadata["potential_id"] is True

    def test_detect_with_nulls(self):
        """Test detection with null values."""
        series = pl.Series("value", [1, 2, None, 3, None, 4])
        var_type, metadata = VariableTypeDetector.detect_variable_type(series, "value")

        # Should still detect correctly despite nulls (6 values, 4 unique = categorical)
        assert var_type == "categorical"

    def test_detect_all_nulls(self):
        """Test detection with all null values - should still return a type."""
        series = pl.Series("empty", [None, None, None])
        var_type, metadata = VariableTypeDetector.detect_variable_type(series, "empty")

        # Should handle gracefully
        assert var_type is not None

    def test_detect_categorical_threshold(self):
        """Test categorical threshold logic (<=20 unique = categorical)."""
        # 21 unique values = above threshold = not categorical
        series = pl.Series("high_cardinality", [f"cat_{i}" for i in range(25)])
        var_type, metadata = VariableTypeDetector.detect_variable_type(series, "high_cardinality")

        # Should be text (high cardinality string, no nulls but not numeric)
        assert var_type in ["text", "identifier"]


class TestSuggestSchemaMapping:
    """Tests for data-driven schema mapping suggestions."""

    def test_suggest_patient_id_by_uniqueness(self):
        """Test that highest-uniqueness column is suggested as patient_id."""
        df = pl.DataFrame(
            {
                "secret_code": [f"C_{i:04d}" for i in range(100)],
                "age": [45 + (i % 50) for i in range(100)],  # Some duplicates
                "outcome": [i % 2 for i in range(100)],
            }
        )

        suggestions = VariableTypeDetector.suggest_schema_mapping(df)

        # secret_code has 100% uniqueness, should be suggested as ID
        assert suggestions["patient_id"] == "secret_code"

    def test_suggest_outcome_by_binary_type(self):
        """Test that binary column is suggested as outcome."""
        df = pl.DataFrame(
            {
                "patient_id": [f"P{i:03d}" for i in range(10)],
                "hospitalized": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
                "age": [45, 55, 35, 65, 40, 50, 60, 30, 70, 45],
            }
        )

        suggestions = VariableTypeDetector.suggest_schema_mapping(df)

        # hospitalized is binary, should be suggested as outcome
        assert suggestions["outcome"] == "hospitalized"

    def test_suggest_time_zero_by_datetime_type(self):
        """Test that datetime column is suggested as time_zero."""
        df = pl.DataFrame(
            {
                "patient_id": [f"P{i:03d}" for i in range(10)],
                "admission_date": [datetime(2020, 1, i + 1) for i in range(10)],
                "outcome": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            }
        )

        suggestions = VariableTypeDetector.suggest_schema_mapping(df)

        # admission_date is datetime, should be suggested as time_zero
        assert suggestions["time_zero"] == "admission_date"

    def test_suggest_no_id_when_none_qualifies(self):
        """Test that no ID is suggested when no column qualifies."""
        # Use data with many duplicates and/or nulls so nothing qualifies as ID
        df = pl.DataFrame(
            {
                "age": [45, 45, 35, 35, 40, 40, 50, 50, 60, 60],  # 50% duplicates
                "bmi": [25.0, None, 22.0, None, 26.0, None, 28.0, None, 30.0, None],  # Has nulls
            }
        )

        suggestions = VariableTypeDetector.suggest_schema_mapping(df)

        # No column has >95% uniqueness with no nulls
        assert suggestions["patient_id"] is None


class TestDetectAllVariables:
    """Tests for detecting all variables in a DataFrame."""

    def test_detect_all_variables_returns_all_columns(self):
        """Test that all columns are detected."""
        df = pl.DataFrame(
            {
                "id": ["P001", "P002", "P003"],
                "age": [45, 55, 35],
                "outcome": [0, 1, 0],
            }
        )

        variable_info = VariableTypeDetector.detect_all_variables(df)

        assert len(variable_info) == 3
        assert "id" in variable_info
        assert "age" in variable_info
        assert "outcome" in variable_info

    def test_detect_all_variables_includes_missing_stats(self):
        """Test that missing data stats are included."""
        df = pl.DataFrame(
            {
                "id": ["P001", "P002", "P003"],
                "age": [45, None, 35],  # 1 missing
            }
        )

        variable_info = VariableTypeDetector.detect_all_variables(df)

        assert variable_info["age"]["missing_count"] == 1
        assert variable_info["age"]["missing_pct"] > 0
