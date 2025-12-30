"""
Generic dataset interface tests using registry discovery.

Tests the ClinicalDataset interface across all available datasets,
ensuring consistency and compliance with the semantic layer architecture.

Test name follows: test_unit_scenario_expectedBehavior
"""

import pandas as pd
import pytest

from clinical_analytics.core.registry import DatasetRegistry
from clinical_analytics.core.schema import UnifiedCohort


def get_available_datasets():
    """
    Helper to discover all available datasets from registry.

    Returns:
        List of dataset names to test against
    """
    DatasetRegistry.reset()
    DatasetRegistry.discover_datasets()
    DatasetRegistry.load_config()
    # Skip special cases like "uploaded" which require user data
    return [name for name in DatasetRegistry.list_datasets() if name != "uploaded"]


# ============================================================================
# Parametrized Dataset Interface Tests
# ============================================================================


@pytest.mark.parametrize("dataset_name", get_available_datasets())
class TestDatasetInterface:
    """Test ClinicalDataset interface across all datasets using registry."""

    def test_initialization_creates_dataset_instance(self, dataset_name):
        """Test that registry can create dataset instance."""
        # Act: Get dataset from registry
        dataset = DatasetRegistry.get_dataset(dataset_name)

        # Assert: Dataset instance created
        assert dataset is not None
        assert dataset.name == dataset_name
        assert hasattr(dataset, "config")
        # Note: not all datasets have mapper (some use different architecture)

    def test_validation_returns_boolean(self, dataset_name):
        """Test that validate() returns boolean for data availability."""
        # Arrange
        dataset = DatasetRegistry.get_dataset(dataset_name)

        # Act
        is_valid = dataset.validate()

        # Assert: Returns boolean indicating data availability
        assert isinstance(is_valid, bool)

    def test_load_populates_data_when_available(self, dataset_name):
        """Test that load() populates _data when dataset is available."""
        # Arrange
        dataset = DatasetRegistry.get_dataset(dataset_name)
        if not dataset.validate():
            pytest.skip(f"{dataset_name} data not available")

        # Act
        dataset.load()

        # Assert: Data loaded
        assert dataset._data is not None
        assert len(dataset._data) > 0

    def test_get_cohort_returns_unified_schema(self, dataset_name):
        """Test that get_cohort() returns UnifiedCohort-compliant DataFrame."""
        # Arrange
        dataset = DatasetRegistry.get_dataset(dataset_name)
        if not dataset.validate():
            pytest.skip(f"{dataset_name} data not available")

        # Act
        cohort = dataset.get_cohort()

        # Assert: Pandas DataFrame with required columns
        assert isinstance(cohort, pd.DataFrame)
        for col in UnifiedCohort.REQUIRED_COLUMNS:
            assert col in cohort.columns, f"{dataset_name} missing required column: {col}"

        # Assert: Correct data types
        assert pd.api.types.is_datetime64_any_dtype(cohort[UnifiedCohort.TIME_ZERO])
        assert pd.api.types.is_numeric_dtype(cohort[UnifiedCohort.OUTCOME])

    def test_get_cohort_auto_loads_if_not_loaded(self, dataset_name):
        """Test that get_cohort() auto-loads data if not already loaded."""
        # Arrange
        dataset = DatasetRegistry.get_dataset(dataset_name)
        if not dataset.validate():
            pytest.skip(f"{dataset_name} data not available")

        # Act: Call get_cohort without calling load() first
        cohort = dataset.get_cohort()

        # Assert: Still works - auto-loads
        assert isinstance(cohort, pd.DataFrame)
        assert len(cohort) > 0

    def test_idempotency_same_filters_produce_same_result(self, dataset_name):
        """Test that calling get_cohort() with same filters is idempotent."""
        # Arrange
        dataset = DatasetRegistry.get_dataset(dataset_name)
        if not dataset.validate():
            pytest.skip(f"{dataset_name} data not available")

        # Act: Get cohort multiple times with same parameters
        cohort1 = dataset.get_cohort()
        cohort2 = dataset.get_cohort()

        # Assert: Results are identical
        assert len(cohort1) == len(cohort2)
        assert set(cohort1.columns) == set(cohort2.columns)

        # Verify required columns present in both
        for col in UnifiedCohort.REQUIRED_COLUMNS:
            assert col in cohort1.columns
            assert col in cohort2.columns

    def test_config_driven_mapper_provides_defaults(self, dataset_name):
        """Test that mapper provides config-driven defaults."""
        # Arrange
        dataset = DatasetRegistry.get_dataset(dataset_name)

        # Skip if dataset doesn't use mapper architecture
        if not hasattr(dataset, "mapper") or dataset.mapper is None:
            pytest.skip(f"{dataset_name} does not use mapper architecture")

        # Assert: Mapper configured with defaults
        # Check mapper provides default predictors
        predictors = dataset.mapper.get_default_predictors()
        assert isinstance(predictors, list)

        # Check mapper provides default outcome
        outcome = dataset.mapper.get_default_outcome()
        assert isinstance(outcome, str)

    def test_config_driven_time_zero_value(self, dataset_name):
        """Test that time_zero value comes from config."""
        # Arrange
        dataset = DatasetRegistry.get_dataset(dataset_name)
        if not dataset.validate():
            pytest.skip(f"{dataset_name} data not available")
        if not hasattr(dataset, "mapper") or dataset.mapper is None:
            pytest.skip(f"{dataset_name} does not use mapper architecture")

        # Act
        time_zero_value = dataset.mapper.get_time_zero_value()
        cohort = dataset.get_cohort()

        # Assert: time_zero configured and present in cohort
        assert time_zero_value is not None
        assert isinstance(time_zero_value, str)
        assert UnifiedCohort.TIME_ZERO in cohort.columns

    def test_config_driven_outcome_label(self, dataset_name):
        """Test that outcome_label comes from config."""
        # Arrange
        dataset = DatasetRegistry.get_dataset(dataset_name)
        if not dataset.validate():
            pytest.skip(f"{dataset_name} data not available")
        if not hasattr(dataset, "mapper") or dataset.mapper is None:
            pytest.skip(f"{dataset_name} does not use mapper architecture")

        # Act
        outcome_col = dataset.mapper.get_default_outcome()
        outcome_label = dataset.mapper.get_default_outcome_label(outcome_col)
        cohort = dataset.get_cohort()

        # Assert: outcome_label configured and present
        assert outcome_label is not None
        assert isinstance(outcome_label, str)
        assert UnifiedCohort.OUTCOME_LABEL in cohort.columns

    def test_patient_level_granularity_supported(self, dataset_name):
        """Test that patient_level granularity works (M8 integration)."""
        # Arrange
        dataset = DatasetRegistry.get_dataset(dataset_name)
        if not dataset.validate():
            pytest.skip(f"{dataset_name} data not available")

        # Act
        cohort = dataset.get_cohort(granularity="patient_level")

        # Assert: Returns valid cohort
        assert isinstance(cohort, pd.DataFrame)
        for col in UnifiedCohort.REQUIRED_COLUMNS:
            assert col in cohort.columns


# ============================================================================
# Semantic Layer Integration Tests
# ============================================================================


@pytest.mark.parametrize("dataset_name", get_available_datasets())
class TestSemanticLayerIntegration:
    """Test that datasets properly integrate with semantic layer."""

    def test_semantic_layer_initialized_after_load(self, dataset_name):
        """Test that semantic layer is initialized after load()."""
        # Arrange
        dataset = DatasetRegistry.get_dataset(dataset_name)
        if not dataset.validate():
            pytest.skip(f"{dataset_name} data not available")

        # Act
        dataset.load()

        # Assert: Semantic layer available
        assert hasattr(dataset, "semantic")
        # Note: semantic might be None for single-table datasets without explicit semantic layer

    def test_config_loaded_correctly(self, dataset_name):
        """Test that dataset config is loaded from datasets.yaml."""
        # Arrange
        dataset = DatasetRegistry.get_dataset(dataset_name)

        # Assert: Config loaded (structure may vary by dataset)
        assert dataset.config is not None
        assert isinstance(dataset.config, dict)
        # Config should have some content
        assert len(dataset.config) > 0


# ============================================================================
# Schema Compliance Tests
# ============================================================================


@pytest.mark.parametrize("dataset_name", get_available_datasets())
class TestSchemaCompliance:
    """Test that all datasets comply with UnifiedCohort schema."""

    def test_required_columns_present(self, dataset_name):
        """Test that all required UnifiedCohort columns are present."""
        # Arrange
        dataset = DatasetRegistry.get_dataset(dataset_name)
        if not dataset.validate():
            pytest.skip(f"{dataset_name} data not available")

        # Act
        cohort = dataset.get_cohort()

        # Assert: All required columns present
        missing_cols = []
        for col in UnifiedCohort.REQUIRED_COLUMNS:
            if col not in cohort.columns:
                missing_cols.append(col)

        assert len(missing_cols) == 0, f"{dataset_name} missing columns: {missing_cols}"

    def test_patient_id_column_correct_type(self, dataset_name):
        """Test that patient_id column is correct type."""
        # Arrange
        dataset = DatasetRegistry.get_dataset(dataset_name)
        if not dataset.validate():
            pytest.skip(f"{dataset_name} data not available")

        # Act
        cohort = dataset.get_cohort()

        # Assert: patient_id is string or numeric (not null)
        assert UnifiedCohort.PATIENT_ID in cohort.columns
        assert cohort[UnifiedCohort.PATIENT_ID].notna().all(), f"{dataset_name} has null patient_ids"

    def test_time_zero_column_is_datetime(self, dataset_name):
        """Test that time_zero column is datetime type."""
        # Arrange
        dataset = DatasetRegistry.get_dataset(dataset_name)
        if not dataset.validate():
            pytest.skip(f"{dataset_name} data not available")

        # Act
        cohort = dataset.get_cohort()

        # Assert: time_zero is datetime
        assert pd.api.types.is_datetime64_any_dtype(
            cohort[UnifiedCohort.TIME_ZERO]
        ), f"{dataset_name} time_zero is not datetime type"

    def test_outcome_column_is_numeric(self, dataset_name):
        """Test that outcome column is numeric type."""
        # Arrange
        dataset = DatasetRegistry.get_dataset(dataset_name)
        if not dataset.validate():
            pytest.skip(f"{dataset_name} data not available")

        # Act
        cohort = dataset.get_cohort()

        # Assert: outcome is numeric
        assert pd.api.types.is_numeric_dtype(
            cohort[UnifiedCohort.OUTCOME]
        ), f"{dataset_name} outcome is not numeric type"

    def test_cohort_has_data(self, dataset_name):
        """Test that cohort returns non-empty DataFrame."""
        # Arrange
        dataset = DatasetRegistry.get_dataset(dataset_name)
        if not dataset.validate():
            pytest.skip(f"{dataset_name} data not available")

        # Act
        cohort = dataset.get_cohort()

        # Assert: Non-empty DataFrame
        assert len(cohort) > 0, f"{dataset_name} returned empty cohort"
        assert cohort.shape[1] > 0, f"{dataset_name} returned cohort with no columns"
