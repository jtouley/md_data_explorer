"""
Tests for schema conversion (Phase 2 - ADR007).

Tests convert_schema() function that transforms variable_mapping to inferred_schema format.
"""

import polars as pl

from clinical_analytics.datasets.uploaded.schema_conversion import (
    convert_schema,
    infer_granularities,
    is_categorical,
)


class TestConvertSchema:
    """Test suite for convert_schema() function."""

    def test_convert_basic_variable_mapping_to_inferred_schema(self):
        """Test converting basic variable_mapping to inferred_schema format."""
        # Arrange
        df = pl.DataFrame(
            {
                "Patient ID": ["P001", "P002", "P003"],
                "Age": [25, 30, 35],
                "Outcome": [0, 1, 0],  # Binary outcome
            }
        )

        variable_mapping = {
            "patient_id": "Patient ID",
            "outcome": "Outcome",
        }

        # Act
        inferred = convert_schema(variable_mapping, df)

        # Assert
        assert "column_mapping" in inferred
        assert inferred["column_mapping"]["Patient ID"] == "patient_id"
        assert "outcomes" in inferred
        assert "Outcome" in inferred["outcomes"]
        assert inferred["outcomes"]["Outcome"]["type"] == "binary"  # 2 unique values

    def test_convert_schema_infers_binary_outcome_from_data(self):
        """Test that outcome type is inferred as binary when n_unique == 2."""
        # Arrange
        df = pl.DataFrame(
            {
                "patient_id": ["P001", "P002", "P003"],
                "mortality": [0, 1, 1],  # Binary: 2 unique values
            }
        )

        variable_mapping = {
            "patient_id": "patient_id",
            "outcome": "mortality",
        }

        # Act
        inferred = convert_schema(variable_mapping, df)

        # Assert
        assert inferred["outcomes"]["mortality"]["type"] == "binary"
        assert inferred["outcomes"]["mortality"]["source_column"] == "mortality"
        assert inferred["outcomes"]["mortality"]["confidence"] == 0.9

    def test_convert_schema_infers_continuous_outcome_from_data(self):
        """Test that outcome type is inferred as continuous when n_unique > 2."""
        # Arrange
        df = pl.DataFrame(
            {
                "patient_id": ["P001", "P002", "P003", "P004"],
                "viral_load": [100.5, 200.3, 150.7, 180.2],  # Continuous: >2 unique values
            }
        )

        variable_mapping = {
            "patient_id": "patient_id",
            "outcome": "viral_load",
        }

        # Act
        inferred = convert_schema(variable_mapping, df)

        # Assert
        assert inferred["outcomes"]["viral_load"]["type"] == "continuous"
        assert inferred["outcomes"]["viral_load"]["confidence"] == 0.7

    def test_convert_schema_maps_time_zero(self):
        """Test that time_zero is correctly mapped."""
        # Arrange
        df = pl.DataFrame(
            {
                "patient_id": ["P001", "P002"],
                "admission_date": ["2024-01-01", "2024-01-02"],
            }
        )

        variable_mapping = {
            "patient_id": "patient_id",
            "time_variables": {
                "time_zero": "admission_date",
            },
        }

        # Act
        inferred = convert_schema(variable_mapping, df)

        # Assert
        assert "time_zero" in inferred
        assert inferred["time_zero"]["source_column"] == "admission_date"

    def test_convert_schema_maps_predictors(self):
        """Test that predictors are correctly mapped to analysis section."""
        # Arrange
        df = pl.DataFrame(
            {
                "patient_id": ["P001", "P002"],
                "age": [25, 30],
                "gender": ["M", "F"],
                "bmi": [22.5, 24.3],
            }
        )

        variable_mapping = {
            "patient_id": "patient_id",
            "predictors": ["age", "gender", "bmi"],
        }

        # Act
        inferred = convert_schema(variable_mapping, df)

        # Assert
        assert "analysis" in inferred
        assert set(inferred["analysis"]["default_predictors"]) == {"age", "gender", "bmi"}

    def test_convert_schema_detects_categorical_variables(self):
        """Test improved categorical detection heuristic."""
        # Arrange
        df = pl.DataFrame(
            {
                "patient_id": ["P001", "P002", "P003", "P004", "P005"],
                "gender": ["M", "F", "M", "F", "M"],  # Low cardinality string
                "age": [25, 30, 35, 40, 45],  # Numeric: never categorical
                "lab_value": [100, 200, 100, 200, 100],  # Numeric: never categorical
            }
        )

        variable_mapping = {
            "patient_id": "patient_id",
            "predictors": ["gender", "age", "lab_value"],
        }

        # Act
        inferred = convert_schema(variable_mapping, df)

        # Assert
        categoricals = inferred["analysis"]["categorical_variables"]
        assert "gender" in categoricals  # String with low cardinality
        assert "age" not in categoricals  # Numeric columns never auto-categorical
        assert "lab_value" not in categoricals  # Numeric columns never auto-categorical

    def test_convert_schema_rejects_high_uniqueness_ratio_as_categorical(self):
        """Test that strings with high uniqueness ratio are not categorical."""
        # Arrange
        df = pl.DataFrame(
            {
                "patient_id": ["P001", "P002", "P003", "P004", "P005"],
                "diagnosis_code": ["D001", "D002", "D003", "D004", "D005"],  # 5 unique / 5 total = 1.0 ratio
            }
        )

        variable_mapping = {
            "patient_id": "patient_id",
            "predictors": ["diagnosis_code"],
        }

        # Act
        inferred = convert_schema(variable_mapping, df)

        # Assert
        categoricals = inferred["analysis"]["categorical_variables"]
        assert "diagnosis_code" not in categoricals  # High uniqueness ratio

    def test_convert_schema_infers_granularities_from_columns(self):
        """Test that granularities are inferred from column presence."""
        # Arrange
        df = pl.DataFrame(
            {
                "patient_id": ["P001", "P002"],
                "admission_id": ["A001", "A002"],
                "event_timestamp": ["2024-01-01 10:00:00", "2024-01-01 11:00:00"],
            }
        )

        variable_mapping = {
            "patient_id": "patient_id",
        }

        # Act
        inferred = convert_schema(variable_mapping, df)

        # Assert
        assert "granularities" in inferred
        assert "patient_level" in inferred["granularities"]  # Always present
        assert "admission_level" in inferred["granularities"]  # admission_id column exists
        assert "event_level" in inferred["granularities"]  # event_timestamp column exists

    def test_convert_schema_patient_level_always_supported(self):
        """Test that patient_level granularity is always included."""
        # Arrange
        df = pl.DataFrame(
            {
                "patient_id": ["P001", "P002"],
                "age": [25, 30],
            }
        )

        variable_mapping = {
            "patient_id": "patient_id",
        }

        # Act
        inferred = convert_schema(variable_mapping, df)

        # Assert
        assert "patient_level" in inferred["granularities"]

    def test_convert_schema_handles_missing_optional_fields(self):
        """Test that missing optional fields don't cause errors."""
        # Arrange
        df = pl.DataFrame(
            {
                "patient_id": ["P001", "P002"],
            }
        )

        variable_mapping = {
            "patient_id": "patient_id",
            # No outcome, time_variables, or predictors
        }

        # Act
        inferred = convert_schema(variable_mapping, df)

        # Assert
        assert "column_mapping" in inferred
        assert "outcomes" in inferred
        assert inferred["outcomes"] == {}  # Empty if no outcome specified
        assert inferred["analysis"]["default_outcome"] is None

    def test_convert_schema_sets_default_outcome(self):
        """Test that default_outcome is set when outcome is provided."""
        # Arrange
        df = pl.DataFrame(
            {
                "patient_id": ["P001", "P002"],
                "mortality": [0, 1],
            }
        )

        variable_mapping = {
            "patient_id": "patient_id",
            "outcome": "mortality",
        }

        # Act
        inferred = convert_schema(variable_mapping, df)

        # Assert
        assert inferred["analysis"]["default_outcome"] == "mortality"

    def test_convert_schema_preserves_polars_dtype_info(self):
        """Test that Polars dtypes are accessible via DataFrame schema."""
        # Arrange
        df = pl.DataFrame(
            {
                "patient_id": ["P001", "P002"],
                "age": [25, 30],
                "weight": [70.5, 80.2],
            }
        )

        variable_mapping = {
            "patient_id": "patient_id",
            "predictors": ["age", "weight"],
        }

        # Act
        inferred = convert_schema(variable_mapping, df)

        # Assert - verify schema is complete
        assert inferred["analysis"]["default_predictors"] == ["age", "weight"]
        # Type info comes from df.schema: df["age"].dtype == pl.Int64, df["weight"].dtype == pl.Float64


class TestCategoricalDetection:
    """Test suite for improved is_categorical() heuristic."""

    def test_categorical_detection_low_cardinality_string(self):
        """Test that strings with low cardinality are categorical."""
        # Arrange
        # 5 unique values out of 10 rows = 0.5 ratio (< 0.5 threshold)
        series = pl.Series("gender", ["M", "F", "M", "F", "M", "F", "M", "F", "M", "F"])

        # Act
        result = is_categorical(series)

        # Assert
        assert result is True

    def test_categorical_detection_high_cardinality_string(self):
        """Test that strings with high cardinality are not categorical."""
        # Arrange
        # 25 unique values out of 25 rows = 1.0 ratio (>= 0.5 threshold)
        series = pl.Series("icd_code", [f"ICD{i:03d}" for i in range(25)])

        # Act
        result = is_categorical(series)

        # Assert
        assert result is False

    def test_categorical_detection_numeric_never_categorical(self):
        """Test that numeric columns are never auto-categorical."""
        # Arrange
        # Even with low cardinality, numeric should NOT be categorical
        series = pl.Series("status", [0, 1, 0, 1, 0, 1, 0, 1])  # Only 2 unique values

        # Act
        result = is_categorical(series)

        # Assert
        assert result is False  # Numeric columns need explicit annotation


class TestGranularityInference:
    """Test suite for infer_granularities() helper."""

    def test_infer_granularities_patient_only(self):
        """Test that patient_level is always inferred."""
        # Arrange
        df = pl.DataFrame(
            {
                "patient_id": ["P001", "P002"],
                "age": [25, 30],
            }
        )

        # Act
        granularities = infer_granularities(df)

        # Assert
        assert granularities == ["patient_level"]

    def test_infer_granularities_with_admission_id(self):
        """Test that admission_level is inferred when admission_id exists."""
        # Arrange
        df = pl.DataFrame(
            {
                "patient_id": ["P001", "P002"],
                "admission_id": ["A001", "A002"],
            }
        )

        # Act
        granularities = infer_granularities(df)

        # Assert
        assert "patient_level" in granularities
        assert "admission_level" in granularities

    def test_infer_granularities_with_event_timestamp(self):
        """Test that event_level is inferred when event_timestamp exists."""
        # Arrange
        df = pl.DataFrame(
            {
                "patient_id": ["P001", "P002"],
                "event_timestamp": ["2024-01-01 10:00:00", "2024-01-01 11:00:00"],
            }
        )

        # Act
        granularities = infer_granularities(df)

        # Assert
        assert "patient_level" in granularities
        assert "event_level" in granularities

    def test_infer_granularities_with_all_columns(self):
        """Test that all granularities are inferred when all columns exist."""
        # Arrange
        df = pl.DataFrame(
            {
                "patient_id": ["P001", "P002"],
                "admission_id": ["A001", "A002"],
                "event_timestamp": ["2024-01-01 10:00:00", "2024-01-01 11:00:00"],
            }
        )

        # Act
        granularities = infer_granularities(df)

        # Assert
        assert set(granularities) == {"patient_level", "admission_level", "event_level"}
