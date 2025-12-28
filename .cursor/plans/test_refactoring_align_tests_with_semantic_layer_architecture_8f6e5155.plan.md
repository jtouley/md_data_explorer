---
name: "Test Refactoring: Align Tests with Semantic Layer Architecture"
overview: ""
todos: []
---

# Test Refactoring: Align Tests with Semantic Layer Architecture

## Overview

Refactor all dataset-specific tests to align with the Ibis semantic layer approach. Tests should focus on:

1. **SemanticLayer interface** - Config-driven SQL generation
2. **ClinicalDataset interface** - Generic dataset behavior
3. **Registry patterns** - Auto-discovery and factory methods
4. **No hardcoded dataset names** - Use registry discovery and parametrization

## Architecture Alignment

The codebase uses a semantic layer architecture where:

- All datasets use `SemanticLayer` class (DRY principle)
- Everything is config-driven via `datasets.yaml`
- Datasets are thin wrappers around `SemanticLayer`
- Tests should verify the **interface**, not implementations

## Files to Refactor

### 1. Create Generic Dataset Interface Tests

**File:** `tests/core/test_dataset_interface.py` (NEW)

- Replace `test_covid_ms_dataset.py` with generic tests
- Test `ClinicalDataset` interface using registry discovery
- Parametrize tests to run against all available datasets
- Verify schema compliance, idempotency, granularity handling
- Test semantic layer integration

### 2. Refactor UI Integration Tests

**File:** `tests/test_ui_integration.py`

- Remove hardcoded `CovidMSDataset` import
- Remove dataset-specific test methods (test_covid_ms_*, test_sepsis_*, test_mimic3_*)
- Replace with parametrized tests using registry
- Test UI workflow with any available dataset

### 3. Refactor Registry Tests

**Files:**

- `tests/test_registry.py`
- `tests/core/test_registry.py`
- Remove hardcoded dataset name assertions (`"covid_ms" in datasets`)
- Use dynamic discovery and test with first available dataset
- Verify registry patterns work generically

### 4. Refactor Mapper Tests

**File:** `tests/test_mapper.py`

- Replace `load_dataset_config("covid_ms")` with parametrized tests
- Test mapper behavior with multiple dataset configs
- Focus on config-driven behavior, not specific datasets

### 5. Refactor Loader Tests

**Files:**

- `tests/loader/test_covid_ms_loader.py`
- `tests/loader/test_sepsis_loader.py`
- `tests/loader/test_mimic3_loader.py`
- Create generic loader test patterns
- Test loader interface/contract rather than specific implementations
- Use fixtures to create test data dynamically

### 6. Update Conftest

**File:** `tests/conftest.py`

- Remove dataset-specific fixtures (`sample_covid_ms_path`, `sample_sepsis_path`)
- Create generic dataset fixtures that work with registry
- Add helper functions for getting available datasets

### 7. Delete Legacy Test File

**File:** `tests/test_covid_ms_dataset.py`

- Delete after new generic tests are in place and passing

## Implementation Details

### Generic Dataset Test Pattern

```python
# tests/core/test_dataset_interface.py
def get_available_datasets():
    """Helper to discover all available datasets."""
    DatasetRegistry.reset()
    DatasetRegistry.discover_datasets()
    DatasetRegistry.load_config()
    return [name for name in DatasetRegistry.list_datasets() 
            if name != "uploaded"]  # Skip special cases

@pytest.mark.parametrize("dataset_name", get_available_datasets())
def test_get_cohort_schema_compliance(dataset_name):
    """Test that any dataset returns UnifiedCohort-compliant data."""
    dataset = DatasetRegistry.get_dataset(dataset_name)
    if not dataset.validate():
        pytest.skip(f"{dataset_name} data not available")
    
    cohort = dataset.get_cohort()
    is_valid, errors = validate_unified_cohort_schema(cohort)
    assert is_valid, f"Schema validation failed: {errors}"
```



### Semantic Layer Focus

Tests should verify:

- Config-driven SQL generation works
- Semantic layer properly registers data sources
- Filters, outcomes, metrics come from config
- No hardcoded transformations in dataset classes

### Registry Pattern Tests

```python
def test_registry_discovers_all_datasets():
    """Test registry auto-discovers without hardcoded names."""
    DatasetRegistry.reset()
    datasets = DatasetRegistry.discover_datasets()
    
    assert isinstance(datasets, dict)
    assert len(datasets) > 0
    # Don't assert specific names - just verify discovery works
```



## Testing Strategy

1. **Interface Tests** - Test `ClinicalDataset` ABC methods
2. **Semantic Layer Tests** - Test `SemanticLayer` config-driven behavior
3. **Integration Tests** - Test registry → dataset → semantic layer flow
4. **Schema Tests** - Verify UnifiedCohort compliance for all datasets

## Success Criteria

- ✅ No hardcoded dataset names in tests
- ✅ No direct imports of specific dataset classes (CovidMSDataset, etc.)
- ✅ All tests use registry discovery
- ✅ Tests are parametrized to run against all available datasets
- ✅ Tests verify semantic layer integration
- ✅ All existing tests pass after refactoring
- ✅ New tests are more maintainable and extensible

## Files Modified

1. `tests/core/test_dataset_interface.py` (NEW)
2. `tests/test_ui_integration.py` (MODIFY)
3. `tests/test_registry.py` (MODIFY)
4. `tests/core/test_registry.py` (MODIFY)
5. `tests/test_mapper.py` (MODIFY)