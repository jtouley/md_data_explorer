---
name: Remove Default Datasets
overview: Remove all built-in datasets (covid_ms, mimic3, sepsis) from the platform, keeping only user-uploaded datasets. This involves filtering registry discovery, updating all UI pages to only show uploads, and updating tests.
todos:
  - id: "1"
    content: Update DatasetRegistry.discover_datasets() to exclude built-in datasets (covid_ms, mimic3, sepsis)
    status: pending
  - id: "2"
    content: Update main app.py to only show uploaded datasets and remove built-in dataset logic
    status: pending
    dependencies:
      - "1"
  - id: "3"
    content: Update page 2_📊_Your_Dataset.py to remove built-in dataset selection
    status: pending
    dependencies:
      - "1"
  - id: "4"
    content: Update page 3_💬_Ask_Questions.py to remove built-in dataset selection
    status: pending
    dependencies:
      - "1"
  - id: "5"
    content: Update page 20_📊_Descriptive_Stats.py to remove built-in dataset selection
    status: pending
    dependencies:
      - "1"
  - id: "6"
    content: Update page 21_📈_Compare_Groups.py to remove built-in dataset selection
    status: pending
    dependencies:
      - "1"
  - id: "7"
    content: Update page 22_🎯_Risk_Factors.py to remove built-in dataset selection
    status: pending
    dependencies:
      - "1"
  - id: "8"
    content: Update page 23_⏱️_Survival_Analysis.py to remove built-in dataset selection
    status: pending
    dependencies:
      - "1"
  - id: "9"
    content: Update page 24_🔗_Correlations.py to remove built-in dataset selection
    status: pending
    dependencies:
      - "1"
  - id: "10"
    content: Update test helpers in test_dataset_interface.py, test_registry.py, test_mapper.py, and test_integration.py
    status: pending
    dependencies:
      - "1"
  - id: "11"
    content: Update error messages across all UI pages to guide users to upload page
    status: pending
    dependencies:
      - "2"
      - "3"
      - "4"
      - "5"
      - "6"
      - "7"
      - "8"
      - "9"
  - id: "12"
    content: Remove legacy test fixtures (sample_covid_ms_path, sample_sepsis_path) from conftest.py
    status: pending
  - id: "13"
    content: Update test_ask_questions_dataset_switching.py to remove built-in dataset examples
    status: pending
  - id: "14"
    content: Update test_mapper.py aggregation test to use generic column names instead of SepsisLabel
    status: pending
  - id: "15"
    content: Remove mimic3-specific test from test_integration.py
    status: pending
  - id: "16"
    content: Update tests/AGENTS.md to remove references to built-in dataset fixtures
    status: pending
    dependencies:
      - "12"
---

#Remove Default Datasets - Keep Only User Uploads

## Overview

Remove built-in datasets (`covid_ms`, `mimic3`, `sepsis`) from the platform. The tool will only support user-uploaded datasets going forward.

## Architecture Changes

### Current State

- `DatasetRegistry.discover_datasets()` auto-discovers all dataset classes in `datasets/` directory
- UI pages show both built-in datasets (from registry) and uploaded datasets (from `UploadedDatasetFactory`)
- Tests use `get_available_datasets()` which filters out `uploaded` but includes built-ins

### Target State

- Registry discovery excludes built-in datasets (`covid_ms`, `mimic3`, `sepsis`)
- UI pages only show datasets from `UploadedDatasetFactory.list_available_uploads()`
- All dataset loading uses `UploadedDatasetFactory.create_dataset()` only
- Tests updated to work with upload-only mode

## Implementation Plan

### 1. Update Dataset Registry Discovery

**File**: `src/clinical_analytics/core/registry.py`Modify `discover_datasets()` to exclude built-in datasets:

```python
@classmethod
def discover_datasets(cls) -> dict[str, type[ClinicalDataset]]:
    """
    Auto-discover all ClinicalDataset implementations in the datasets package.
    
    Excludes built-in datasets (covid_ms, mimic3, sepsis) - only user uploads are supported.
    """
    import clinical_analytics.datasets as datasets_pkg

    datasets_path = Path(datasets_pkg.__file__).parent
    
    # Built-in datasets to exclude (only user uploads are supported)
    BUILTIN_DATASETS = {"covid_ms", "mimic3", "sepsis"}

    # Iterate through all subdirectories in datasets/
    for module_info in pkgutil.iter_modules([str(datasets_path)]):
        if module_info.ispkg:
            module_name = module_info.name
            
            # Skip built-in datasets
            if module_name in BUILTIN_DATASETS:
                continue
            # ... rest of discovery logic
```

**Note**: Keep `uploaded` dataset class in discovery - it's the implementation class for user uploads, not a built-in dataset.

### 2. Update Main App Page

**File**: `src/clinical_analytics/ui/app.py`**Changes**:

- Remove `DatasetRegistry.list_datasets()` call (lines 215-223)
- Remove built-in dataset display logic (lines 246-251)
- Only use `UploadedDatasetFactory.list_available_uploads()` for dataset selection
- Update `load_dataset()` function to only use `UploadedDatasetFactory.create_dataset()`
- Simplify dataset info display (all datasets are uploads now)

**Key changes**:

```python
# Remove this:
available_datasets = DatasetRegistry.list_datasets()
dataset_info = DatasetRegistry.get_all_dataset_info()
# ... build display names from registry

# Replace with:
dataset_display_names = {}
uploaded_datasets = {}
try:
    uploads = UploadedDatasetFactory.list_available_uploads()
    for upload in uploads:
        upload_id = upload["upload_id"]
        dataset_name = upload.get("dataset_name", upload_id)
        display_name = f"📤 {dataset_name}"
        dataset_display_names[display_name] = upload_id
        uploaded_datasets[upload_id] = upload
except Exception as e:
    st.sidebar.warning(f"Could not load uploaded datasets: {e}")
```



### 3. Update All UI Pages

Update the following pages to remove built-in dataset logic:**Files to update**:

- `src/clinical_analytics/ui/pages/2_📊_Your_Dataset.py` (lines 34-80)
- `src/clinical_analytics/ui/pages/3_💬_Ask_Questions.py` (lines 1053-1110)
- `src/clinical_analytics/ui/pages/20_📊_Descriptive_Stats.py` (lines 152-222)
- `src/clinical_analytics/ui/pages/21_📈_Compare_Groups.py` (lines 173-218)
- `src/clinical_analytics/ui/pages/22_🎯_Risk_Factors.py` (similar pattern)
- `src/clinical_analytics/ui/pages/23_⏱️_Survival_Analysis.py` (similar pattern)
- `src/clinical_analytics/ui/pages/24_🔗_Correlations.py` (similar pattern)

**Pattern for each page**:

1. Remove `DatasetRegistry.list_datasets()` and `get_all_dataset_info()` calls
2. Remove loop that builds display names from registry datasets
3. Keep only `UploadedDatasetFactory.list_available_uploads()` logic
4. Remove `is_uploaded` checks (all datasets are uploads)
5. Remove `else` branch that calls `DatasetRegistry.get_dataset()`
6. Always use `UploadedDatasetFactory.create_dataset(dataset_choice)`

**Example transformation**:

```python
# BEFORE:
available_datasets = DatasetRegistry.list_datasets()
dataset_info = DatasetRegistry.get_all_dataset_info()
dataset_display_names = {}
for ds_name in available_datasets:
    # ... build from registry
# ... add uploads
is_uploaded = dataset_choice in uploaded_datasets or ...
if is_uploaded:
    dataset = UploadedDatasetFactory.create_dataset(dataset_choice)
else:
    dataset = DatasetRegistry.get_dataset(dataset_choice)

# AFTER:
dataset_display_names = {}
uploaded_datasets = {}
try:
    uploads = UploadedDatasetFactory.list_available_uploads()
    for upload in uploads:
        upload_id = upload["upload_id"]
        dataset_name = upload.get("dataset_name", upload_id)
        display_name = f"📤 {dataset_name}"
        dataset_display_names[display_name] = upload_id
        uploaded_datasets[upload_id] = upload
except Exception as e:
    st.sidebar.warning(f"Could not load uploaded datasets: {e}")

# Load dataset (always uploaded)
dataset = UploadedDatasetFactory.create_dataset(dataset_choice)
```



### 4. Update Test Helpers

**Files to update**:

- `tests/core/test_dataset_interface.py` - `get_available_datasets()` function
- `tests/core/test_registry.py` - `get_first_available_dataset()` function
- `tests/core/test_mapper.py` - `get_first_available_dataset_config()` function
- `tests/ui/test_integration.py` - `get_available_datasets()` function

**Changes**:

- Update `get_available_datasets()` to return empty list or skip tests when no uploads available
- Update filters to exclude all built-in datasets: `if name not in ["covid_ms", "mimic3", "sepsis", "uploaded"]`
- For tests that require datasets, mark as `@pytest.mark.skip` or use fixtures that create test uploads

**Example**:

```python
def get_available_datasets():
    """
    Helper to discover all available datasets from registry.
    
    Returns:
        List of dataset names to test against (only user uploads)
    """
    DatasetRegistry.reset()
    DatasetRegistry.discover_datasets()
    DatasetRegistry.load_config()
    # Only return uploaded datasets (if any exist in test environment)
    datasets = DatasetRegistry.list_datasets()
    # Filter out built-in datasets
    return [name for name in datasets if name not in ["covid_ms", "mimic3", "sepsis"]]
```



### 5. Update Error Messages

Update all "No datasets available" messages to be more specific:**Files**: All UI pages listed above**Change**: Update error messages to guide users to upload page:

```python
if not dataset_display_names:
    st.error("No datasets found! Please upload data using the 'Add Your Data' page.")
    st.info("👈 Go to **Add Your Data** to upload your first dataset")
    return
```



### 6. Remove Legacy Test Fixtures and References

**Files to update**:

- `tests/conftest.py` - Remove `sample_covid_ms_path` and `sample_sepsis_path` fixtures (lines 60-70)
- `tests/unit/ui/pages/test_ask_questions_dataset_switching.py` - Remove built-in dataset examples from test data (line 77: remove `("covid_ms", "mimic3")` from dataset_pairs)
- `tests/core/test_mapper.py` - Update `test_apply_aggregations()` to use generic column names instead of "SepsisLabel" (line 252)
- `tests/ui/test_integration.py` - Remove `test_dataset_without_data_can_be_created_but_validation_fails()` test that specifically tests mimic3 (lines 191-203)
- `tests/AGENTS.md` - Remove references to `sample_covid_ms_path` and `sample_sepsis_path` fixtures (lines 588-589)

**Changes**:

1. **Remove unused fixtures**:
```python
# Remove these fixtures from conftest.py:
@pytest.fixture
def sample_covid_ms_path(test_data_dir):
    """Return path to COVID-MS test data if available."""
    path = test_data_dir / "covid_ms" / "GDSI_OpenDataset_Final.csv"
    return path if path.exists() else None

@pytest.fixture
def sample_sepsis_path(test_data_dir):
    """Return path to Sepsis test data if available."""
    path = test_data_dir / "sepsis"
    return path if path.exists() else None
```




2. **Update test_dataset_switching.py**:
```python
# Remove built-in dataset examples, keep only generic/user upload examples
dataset_pairs = [
    ("user_upload_123", "user_upload_456"),
    ("dataset_a", "dataset_b"),
    ("📤 Statin use", "📤 DEXA results"),
]
```




3. **Update test_mapper.py aggregation test**:
```python
# Change from sepsis-specific:
"outcome": {"column": "SepsisLabel", "method": "max", "target": "sepsis_label"},

# To generic:
"outcome": {"column": "OutcomeLabel", "method": "max", "target": "outcome_label"},
# And update test data accordingly
```




4. **Remove mimic3-specific test**:

- Delete `test_dataset_without_data_can_be_created_but_validation_fails()` from `test_integration.py`
- If validation testing is needed, create a generic test using uploaded datasets

### 7. Update Documentation

**Files to review/update**:

- `docs/architecture/dataset-registry.md` - Remove references to built-in datasets
- `docs/development/setup.md` - Update dataset structure documentation
- `tests/AGENTS.md` - Remove references to built-in dataset fixtures
- Any other docs mentioning built-in datasets

## Testing Strategy

1. **Unit Tests**: Update test helpers to handle upload-only mode
2. **Integration Tests**: Ensure UI pages work with only uploaded datasets
3. **Manual Testing**: Verify all pages load correctly with no built-in datasets
4. **Edge Cases**: Test behavior when no uploads exist

## Rollout Considerations

- **Backward Compatibility**: None required - this is a breaking change
- **Data Migration**: No data migration needed (built-in datasets are not user data)
- **User Impact**: Users will only see their uploaded datasets

## Files Summary

**Core Changes**:

- `src/clinical_analytics/core/registry.py` - Filter discovery

**UI Changes** (8 files):

- `src/clinical_analytics/ui/app.py`
- `src/clinical_analytics/ui/pages/2_📊_Your_Dataset.py`
- `src/clinical_analytics/ui/pages/3_💬_Ask_Questions.py`
- `src/clinical_analytics/ui/pages/20_📊_Descriptive_Stats.py`
- `src/clinical_analytics/ui/pages/21_📈_Compare_Groups.py`
- `src/clinical_analytics/ui/pages/22_🎯_Risk_Factors.py`
- `src/clinical_analytics/ui/pages/23_⏱️_Survival_Analysis.py`
- `src/clinical_analytics/ui/pages/24_🔗_Correlations.py`

**Test Changes** (6 files):

- `tests/core/test_dataset_interface.py` - Update `get_available_datasets()` helper
- `tests/core/test_registry.py` - Update `get_first_available_dataset()` helper
- `tests/core/test_mapper.py` - Update `get_first_available_dataset_config()` helper and fix aggregation test
- `tests/ui/test_integration.py` - Update `get_available_datasets()` helper and remove mimic3-specific test
- `tests/conftest.py` - Remove legacy fixtures (`sample_covid_ms_path`, `sample_sepsis_path`)