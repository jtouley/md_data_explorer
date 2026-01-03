"""
Tests for the column mapper module using config-driven approach.

Tests the ColumnMapper functionality without hardcoded dataset dependencies,
using dynamic config loading for tests that require real dataset configs.

Test name follows: test_unit_scenario_expectedBehavior
"""

import polars as pl
import pytest

from clinical_analytics.core.mapper import ColumnMapper, get_global_config, load_dataset_config
from clinical_analytics.core.schema import DataQualityError, UnifiedCohort


def get_first_available_dataset_config(discovered_datasets):
    """
    Get config from first available dataset using cached discovery.

    Performance optimization: Uses session-scoped fixture to avoid
    expensive dataset discovery on every test call.

    Args:
        discovered_datasets: Session-scoped fixture with cached datasets

    Returns:
        dict: Config for first available dataset, or None if none available
    """
    import logging

    logger = logging.getLogger(__name__)

    available = discovered_datasets["available"]
    configs = discovered_datasets["configs"]

    if not available:
        logger.warning(
            "test_mapper_no_datasets_available: all_datasets=%s, reason=%s",
            discovered_datasets["all_datasets"],
            "all datasets filtered out or none discovered",
        )
        return None

    selected = available[0]
    logger.info(
        "test_mapper_dataset_selected: selected_dataset=%s, available_options=%s",
        selected,
        available,
    )

    return configs.get(selected)


class TestColumnMapper:
    """Test suite for ColumnMapper."""

    @pytest.mark.slow
    @pytest.mark.integration
    def test_mapper_initialization_with_config(self, discovered_datasets):
        """Test mapper can be initialized with dataset config."""
        # Arrange: Get config from first available dataset
        config = get_first_available_dataset_config(discovered_datasets)
        if config is None:
            pytest.skip("No datasets available for testing - skipping integration test")

        # Act: Initialize mapper with config
        mapper = ColumnMapper(config)

        # Assert: Mapper initialized with config and column mapping
        assert mapper.config is not None
        assert mapper.column_mapping is not None

    @pytest.mark.slow
    @pytest.mark.integration
    def test_get_default_predictors_returns_list(self, discovered_datasets):
        """Test getting default predictors returns non-empty list."""
        # Arrange: Get config and create mapper
        config = get_first_available_dataset_config(discovered_datasets)
        if config is None:
            pytest.skip("No datasets available for testing - skipping integration test")
        mapper = ColumnMapper(config)

        # Act: Get default predictors
        predictors = mapper.get_default_predictors()

        # Assert: Returns non-empty list of predictor names
        assert isinstance(predictors, list)
        assert len(predictors) > 0

    @pytest.mark.slow
    @pytest.mark.integration
    def test_get_categorical_variables_returns_list(self, discovered_datasets):
        """Test getting categorical variables returns list."""
        # Arrange: Get config and create mapper
        config = get_first_available_dataset_config(discovered_datasets)
        if config is None:
            pytest.skip("No datasets available for testing - skipping integration test")
        mapper = ColumnMapper(config)

        # Act: Get categorical variables
        categoricals = mapper.get_categorical_variables()

        # Assert: Returns list (may be empty if no categoricals defined)
        assert isinstance(categoricals, list)

    @pytest.mark.slow
    @pytest.mark.integration
    def test_get_default_outcome_returns_non_empty_string(self, discovered_datasets):
        """Test getting default outcome returns non-empty string."""
        # Arrange: Get config and create mapper
        config = get_first_available_dataset_config(discovered_datasets)
        if config is None:
            pytest.skip("No datasets available for testing - skipping integration test")
        mapper = ColumnMapper(config)

        # Act: Get default outcome
        outcome = mapper.get_default_outcome()

        # Assert: Returns non-empty string
        assert isinstance(outcome, str)
        assert len(outcome) > 0

    @pytest.mark.slow
    @pytest.mark.integration
    def test_get_default_filters_returns_dict(self, discovered_datasets):
        """Test getting default filters returns dict."""
        # Arrange: Get config and create mapper
        config = get_first_available_dataset_config(discovered_datasets)
        if config is None:
            pytest.skip("No datasets available for testing - skipping integration test")
        mapper = ColumnMapper(config)

        # Act: Get default filters
        filters = mapper.get_default_filters()

        # Assert: Returns dict (may be empty if no filters defined)
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
        """Test boolean filters with different column types."""
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

    def test_apply_filters_range(self):
        """Test range filter."""
        config = {"filters": {"age_range": {"type": "range", "column": "age"}}}
        mapper = ColumnMapper(config)

        df = pl.DataFrame({"age": [20, 30, 40, 50, 60], "id": ["A", "B", "C", "D", "E"]})

        result = mapper.apply_filters(df, {"age_range": {"min": 25, "max": 45}})
        assert len(result) == 2
        assert set(result["age"].to_list()) == {30, 40}

    def test_apply_filters_in(self):
        """Test 'in' filter."""
        config = {"filters": {"status_filter": {"type": "in", "column": "status"}}}
        mapper = ColumnMapper(config)

        df = pl.DataFrame({"status": ["A", "B", "C", "A", "B"], "id": ["1", "2", "3", "4", "5"]})

        result = mapper.apply_filters(df, {"status_filter": ["A", "C"]})
        assert len(result) == 3
        assert set(result["status"].to_list()) == {"A", "C"}

    def test_apply_filters_exists(self):
        """Test 'exists' filter."""
        config = {"filters": {"has_value": {"type": "exists", "column": "value"}}}
        mapper = ColumnMapper(config)

        df = pl.DataFrame({"value": [1, None, 3, None, 5], "id": ["A", "B", "C", "D", "E"]})

        result = mapper.apply_filters(df, {"has_value": True})
        assert len(result) == 3
        assert result["value"].null_count() == 0

        result = mapper.apply_filters(df, {"has_value": False})
        assert len(result) == 2
        assert result["value"].null_count() == 2

    def test_apply_aggregations(self):
        """Test aggregation engine."""
        config = {
            "aggregation": {
                "static_features": [
                    {"column": "Age", "method": "first", "target": "age"},
                    {"column": "Gender", "method": "first", "target": "gender"},
                ],
                "outcome": {"column": "OutcomeLabel", "method": "max", "target": "outcome_label"},
            }
        }
        mapper = ColumnMapper(config)

        # Create time-series test dataframe
        df = pl.DataFrame(
            {
                "patient_id": ["P001", "P001", "P001", "P002", "P002"],
                "Age": [45, 45, 45, 62, 62],
                "Gender": ["M", "M", "M", "F", "F"],
                "OutcomeLabel": [0, 0, 1, 0, 0],
            }
        )

        result = mapper.apply_aggregations(df, group_by="patient_id")

        assert len(result) == 2
        assert "age" in result.columns
        assert "gender" in result.columns
        assert "outcome_label" in result.columns
        assert "num_hours" in result.columns

        # Check aggregation results
        p001 = result.filter(pl.col("patient_id") == "P001")
        assert p001["outcome_label"].item() == 1  # max should be 1
        assert p001["age"].item() == 45  # first should be 45

    def test_apply_aggregations_all_methods(self):
        """Test all aggregation methods."""
        config = {
            "aggregation": {
                "static_features": [
                    {"column": "val1", "method": "mean", "target": "mean_val"},
                    {"column": "val2", "method": "sum", "target": "sum_val"},
                    {"column": "val3", "method": "min", "target": "min_val"},
                    {"column": "val4", "method": "max", "target": "max_val"},
                    {"column": "val5", "method": "last", "target": "last_val"},
                    {"column": "val6", "method": "count", "target": "count_val"},
                ]
            }
        }
        mapper = ColumnMapper(config)

        df = pl.DataFrame(
            {
                "patient_id": ["P001", "P001", "P002", "P002"],
                "val1": [10, 20, 30, 40],
                "val2": [1, 2, 3, 4],
                "val3": [5, 3, 7, 2],
                "val4": [1, 5, 2, 8],
                "val5": [100, 200, 300, 400],
                "val6": [1, 1, 1, 1],
            }
        )

        result = mapper.apply_aggregations(df, group_by="patient_id")

        assert "mean_val" in result.columns
        assert "sum_val" in result.columns
        assert "min_val" in result.columns
        assert "max_val" in result.columns
        assert "last_val" in result.columns
        assert "count_val" in result.columns

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


class TestValueMappingDataQuality:
    """Test suite for Phase 0.1: value mapping data quality fixes."""

    def test_unmapped_values_raise_data_quality_error(self):
        """Unmapped values should raise DataQualityError with examples (Phase 0.1)."""
        # Arrange: DataFrame with unmapped values
        config = {
            "outcomes": {
                "outcome": {
                    "source_column": "status",
                    "type": "binary",
                    "mapping": {"yes": 1, "no": 0},
                }
            }
        }
        mapper = ColumnMapper(config)
        df = pl.DataFrame({"status": ["yes", "no", "unknown", "pending"]})

        # Act & Assert
        with pytest.raises(DataQualityError) as exc_info:
            mapper.apply_outcome_transformations(df)

        # Verify error message contains unmapped values
        assert "unknown" in str(exc_info.value) or "unknown" in exc_info.value.unmapped_values
        assert "pending" in str(exc_info.value) or "pending" in exc_info.value.unmapped_values

    def test_all_mapped_values_no_error(self):
        """All mapped values should not raise error (Phase 0.1)."""
        # Arrange: DataFrame with all mapped values
        config = {
            "outcomes": {
                "outcome": {
                    "source_column": "status",
                    "type": "binary",
                    "mapping": {"yes": 1, "no": 0},
                }
            }
        }
        mapper = ColumnMapper(config)
        df = pl.DataFrame({"status": ["yes", "no", "yes", "No", "YES"]})  # Case variations

        # Act - should not raise
        result = mapper.apply_outcome_transformations(df)

        # Assert: All values mapped correctly
        assert result["outcome"].null_count() == 0
        assert len(result) == 5

    def test_null_values_allowed_no_error(self):
        """NULL values in source should be allowed and map to NULL (Phase 0.1)."""
        # Arrange: DataFrame with NULL values
        config = {
            "outcomes": {
                "outcome": {
                    "source_column": "status",
                    "type": "binary",
                    "mapping": {"yes": 1, "no": 0},
                }
            }
        }
        mapper = ColumnMapper(config)
        df = pl.DataFrame({"status": ["yes", None, "no", None]})

        # Act - should not raise (NULL is allowed)
        result = mapper.apply_outcome_transformations(df)

        # Assert: NULLs preserved
        assert result["outcome"].null_count() == 2
        assert (result["outcome"] == 1).sum() == 1
        assert (result["outcome"] == 0).sum() == 1


class TestConfigLoading:
    """Test suite for config loading functions."""

    @pytest.mark.slow
    @pytest.mark.integration
    def test_load_dataset_config_returns_valid_dict(self, discovered_datasets):
        """Test loading dataset configuration returns valid dict."""
        # Arrange: Get first available dataset name from cached discovery
        available = discovered_datasets["available"]
        if len(available) == 0:
            pytest.skip("No datasets available for testing - skipping integration test")

        # Act: Load config for first available dataset (should be pre-cached)
        config = discovered_datasets["configs"].get(available[0])
        if config is None:
            # Fallback: load if not cached
            config = load_dataset_config(available[0])

        # Assert: Returns dict with required keys
        assert isinstance(config, dict)
        assert "name" in config or "display_name" in config
        assert "init_params" in config

    def test_load_nonexistent_config_raises_keyerror(self):
        """Test loading nonexistent config raises KeyError."""
        # Arrange: Use nonexistent dataset name
        # Act & Assert: Should raise KeyError
        with pytest.raises(KeyError):
            load_dataset_config("nonexistent_dataset")

    def test_get_global_config_returns_dict(self):
        """Test loading global configuration returns dict."""
        # Arrange: No setup needed
        # Act: Get global config
        global_config = get_global_config()

        # Assert: Returns dict (may be empty if no global config defined)
        assert isinstance(global_config, dict)
