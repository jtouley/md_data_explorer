"""
Tests for the column mapper module.
"""

import polars as pl
import pytest

from clinical_analytics.core.mapper import ColumnMapper, get_global_config, load_dataset_config
from clinical_analytics.core.schema import UnifiedCohort


class TestColumnMapper:
    """Test suite for ColumnMapper."""

    def test_mapper_initialization(self):
        """Test mapper can be initialized with config."""
        config = load_dataset_config("covid_ms")
        mapper = ColumnMapper(config)

        assert mapper.config is not None
        assert mapper.column_mapping is not None

    def test_get_default_predictors(self):
        """Test getting default predictors from config."""
        config = load_dataset_config("covid_ms")
        mapper = ColumnMapper(config)

        predictors = mapper.get_default_predictors()

        assert isinstance(predictors, list)
        assert len(predictors) > 0

    def test_get_categorical_variables(self):
        """Test getting categorical variables from config."""
        config = load_dataset_config("covid_ms")
        mapper = ColumnMapper(config)

        categoricals = mapper.get_categorical_variables()

        assert isinstance(categoricals, list)

    def test_get_default_outcome(self):
        """Test getting default outcome from config."""
        config = load_dataset_config("covid_ms")
        mapper = ColumnMapper(config)

        outcome = mapper.get_default_outcome()

        assert isinstance(outcome, str)
        assert len(outcome) > 0

    def test_get_default_filters(self):
        """Test getting default filters from config."""
        config = load_dataset_config("covid_ms")
        mapper = ColumnMapper(config)

        filters = mapper.get_default_filters()

        assert isinstance(filters, dict)

    def test_apply_outcome_transformations(self):
        """Test outcome transformations."""
        config = {
            "outcomes": {
                "test_outcome": {
                    "source_column": "source",
                    "type": "binary",
                    "mapping": {"yes": 1, "no": 0},
                }
            }
        }
        mapper = ColumnMapper(config)

        # Create test dataframe
        df = pl.DataFrame({"source": ["yes", "no", "yes", "No"]})

        result = mapper.apply_outcome_transformations(df)

        assert "test_outcome" in result.columns
        assert result["test_outcome"].to_list() == [1, 0, 1, 0]

    def test_map_to_unified_cohort(self):
        """Test mapping to UnifiedCohort schema."""
        config = {
            "column_mapping": {"id": "patient_id", "age": "age"},
            "analysis": {"default_outcome": "outcome"},
        }
        mapper = ColumnMapper(config)

        # Create test dataframe
        df = pl.DataFrame({"id": ["P001", "P002", "P003"], "age": [45, 62, 38], "outcome": [1, 0, 1]})

        result = mapper.map_to_unified_cohort(
            df, time_zero_value="2020-01-01", outcome_col="outcome", outcome_label="test_outcome"
        )

        assert UnifiedCohort.PATIENT_ID in result.columns
        assert UnifiedCohort.TIME_ZERO in result.columns
        assert UnifiedCohort.OUTCOME in result.columns
        assert UnifiedCohort.OUTCOME_LABEL in result.columns

    def test_apply_filters(self):
        """Test filter application engine."""
        config = {
            "filters": {
                "confirmed_only": {
                    "type": "equals",
                    "column": "status",
                    "description": "Filter confirmed cases",
                }
            }
        }
        mapper = ColumnMapper(config)

        # Create test dataframe
        df = pl.DataFrame({"status": ["yes", "no", "yes", "no"], "age": [25, 30, 35, 40]})

        # Test equals filter
        result = mapper.apply_filters(df, {"confirmed_only": True})
        assert len(result) == 2
        assert result["status"].to_list() == ["yes", "yes"]

        # Test with False
        result = mapper.apply_filters(df, {"confirmed_only": False})
        assert len(result) == 2
        assert result["status"].to_list() == ["no", "no"]

    def test_apply_filters_with_different_types(self):
        """Test boolean filters with different column types (regression test for type comparison error)."""
        # Test with integer column
        config_int = {"filters": {"active_int": {"type": "equals", "column": "active"}}}
        mapper_int = ColumnMapper(config_int)
        df_int = pl.DataFrame({"active": [1, 0, 1, 0], "id": ["A", "B", "C", "D"]})

        result = mapper_int.apply_filters(df_int, {"active_int": True})
        assert len(result) == 2
        assert result["active"].to_list() == [1, 1]

        result = mapper_int.apply_filters(df_int, {"active_int": False})
        assert len(result) == 2
        assert result["active"].to_list() == [0, 0]

        # Test with boolean column
        config_bool = {"filters": {"active_bool": {"type": "equals", "column": "active"}}}
        mapper_bool = ColumnMapper(config_bool)
        df_bool = pl.DataFrame({"active": [True, False, True, False], "id": ["A", "B", "C", "D"]})

        result = mapper_bool.apply_filters(df_bool, {"active_bool": True})
        assert len(result) == 2
        assert result["active"].to_list() == [True, True]

        # Test with float column
        config_float = {"filters": {"active_float": {"type": "equals", "column": "active"}}}
        mapper_float = ColumnMapper(config_float)
        df_float = pl.DataFrame({"active": [1.0, 0.0, 1.0, 0.0], "id": ["A", "B", "C", "D"]})

        result = mapper_float.apply_filters(df_float, {"active_float": True})
        assert len(result) == 2
        assert result["active"].to_list() == [1.0, 1.0]

    def test_apply_aggregations(self):
        """Test aggregation engine."""
        config = {
            "aggregation": {
                "static_features": [
                    {"column": "Age", "method": "first", "target": "age"},
                    {"column": "Gender", "method": "first", "target": "gender"},
                ],
                "outcome": {"column": "SepsisLabel", "method": "max", "target": "sepsis_label"},
            }
        }
        mapper = ColumnMapper(config)

        # Create time-series test dataframe
        df = pl.DataFrame(
            {
                "patient_id": ["P001", "P001", "P001", "P002", "P002"],
                "Age": [45, 45, 45, 62, 62],
                "Gender": ["M", "M", "M", "F", "F"],
                "SepsisLabel": [0, 0, 1, 0, 0],
            }
        )

        result = mapper.apply_aggregations(df, group_by="patient_id")

        assert len(result) == 2
        assert "age" in result.columns
        assert "gender" in result.columns
        assert "sepsis_label" in result.columns
        assert "num_hours" in result.columns

        # Check aggregation results
        p001 = result.filter(pl.col("patient_id") == "P001")
        assert p001["sepsis_label"].item() == 1  # max should be 1
        assert p001["age"].item() == 45  # first should be 45

    def test_get_time_zero_value(self):
        """Test getting time_zero value from config."""
        # Test with string value
        config = {"time_zero": "2020-01-01"}
        mapper = ColumnMapper(config)
        assert mapper.get_time_zero_value() == "2020-01-01"

        # Test with dict value
        config = {"time_zero": {"value": "2019-01-01"}}
        mapper = ColumnMapper(config)
        assert mapper.get_time_zero_value() == "2019-01-01"

        # Test with missing config
        config = {}
        mapper = ColumnMapper(config)
        assert mapper.get_time_zero_value() is None

    def test_get_default_outcome_label(self):
        """Test getting default outcome label from config."""
        config = {
            "outcome_labels": {
                "outcome_hospitalized": "hospitalization",
                "outcome_icu": "icu_admission",
            },
            "outcomes": {
                "outcome_hospitalized": {
                    "source_column": "source",
                    "type": "binary",
                    "label": "custom_label",
                }
            },
        }
        mapper = ColumnMapper(config)

        # Test with outcome_labels mapping
        label = mapper.get_default_outcome_label("outcome_hospitalized")
        assert label == "hospitalization"

        # Test with outcome definition label
        config2 = {"outcomes": {"test_outcome": {"source_column": "source", "type": "binary", "label": "test_label"}}}
        mapper2 = ColumnMapper(config2)
        label = mapper2.get_default_outcome_label("test_outcome")
        assert label == "test_label"

        # Test fallback to outcome column name
        config3 = {}
        mapper3 = ColumnMapper(config3)
        label = mapper3.get_default_outcome_label("unknown_outcome")
        assert label == "unknown_outcome"

    def test_idempotency(self):
        """Test that same config produces same result (idempotency)."""
        config = {
            "outcomes": {
                "test_outcome": {
                    "source_column": "source",
                    "type": "binary",
                    "mapping": {"yes": 1, "no": 0},
                }
            },
            "column_mapping": {"id": "patient_id", "source": "source"},
            "analysis": {"default_outcome": "test_outcome"},
        }
        mapper = ColumnMapper(config)

        # Create test dataframe
        df = pl.DataFrame({"id": ["P001", "P002", "P003"], "source": ["yes", "no", "yes"]})

        # Apply transformations multiple times
        result1 = mapper.apply_outcome_transformations(df)
        result2 = mapper.apply_outcome_transformations(df)

        # Results should be identical
        assert result1.equals(result2)

        # Map to unified cohort multiple times
        cohort1 = mapper.map_to_unified_cohort(
            result1, time_zero_value="2020-01-01", outcome_col="test_outcome", outcome_label="test"
        )
        cohort2 = mapper.map_to_unified_cohort(
            result2, time_zero_value="2020-01-01", outcome_col="test_outcome", outcome_label="test"
        )

        assert cohort1.equals(cohort2)


class TestConfigLoading:
    """Test suite for config loading functions."""

    def test_load_dataset_config(self):
        """Test loading dataset configuration."""
        config = load_dataset_config("covid_ms")

        assert isinstance(config, dict)
        assert "name" in config or "display_name" in config
        assert "init_params" in config

    def test_load_nonexistent_config(self):
        """Test loading nonexistent config raises error."""
        with pytest.raises(KeyError):
            load_dataset_config("nonexistent_dataset")

    def test_get_global_config(self):
        """Test loading global configuration."""
        global_config = get_global_config()

        assert isinstance(global_config, dict)
        # May be empty if no global config defined
