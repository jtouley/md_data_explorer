# Phase 8.2: Dataset Loader Migration Guide

## Overview

This migration guide documents how to update remaining UI pages to use the new `dataset_loader` component introduced in Phase 8.2.

## Component Location

```python
from clinical_analytics.ui.components.dataset_loader import render_dataset_selector
```

## Migration Checklist

**Status**: 1/7 pages migrated

- [x] `03_💬_Ask_Questions.py` - **COMPLETED** (Phase 8.2)
- [ ] `02_📊_Your_Dataset.py`
- [ ] `20_📊_Descriptive_Stats.py`
- [ ] `21_📈_Compare_Groups.py`
- [ ] `22_🎯_Risk_Factors.py`
- [ ] `23_⏱️_Survival_Analysis.py`
- [ ] `24_🔗_Correlations.py`

## Migration Pattern

### Before (Old Pattern - 60+ lines)

```python
# Dataset selection
st.sidebar.header("Data Selection")

# Load datasets - only user uploads
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

if not dataset_display_names:
    st.error("No datasets found! Please upload data using the 'Add Your Data' page.")
    st.info("👈 Go to **Add Your Data** to upload your first dataset")
    return

dataset_choice_display = st.sidebar.selectbox("Choose Dataset", list(dataset_display_names.keys()))
dataset_choice = dataset_display_names[dataset_choice_display]

# Load dataset (always uploaded)
with st.spinner(f"Loading {dataset_choice_display}..."):
    try:
        dataset = UploadedDatasetFactory.create_dataset(dataset_choice)
        dataset.load()
        cohort = dataset.get_cohort()
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        st.stop()

# Get dataset version for lifecycle management
dataset_version = get_dataset_version(dataset, dataset_choice)

# Show Semantic Scope in sidebar
with st.sidebar.expander("🔍 Semantic Scope", expanded=False):
    st.markdown("**V1 Cohort-First Mode**")
    # ... semantic scope details (15+ lines)
```

### After (New Pattern - 5 lines)

```python
# Dataset selection (Phase 8.2: Use reusable component)
result = render_dataset_selector(show_semantic_scope=True)
if result is None:
    return  # No datasets available (error message already shown)

dataset, cohort, dataset_choice, dataset_version = result
```

## Line Count Reduction

**Before**: ~60 lines per page × 7 pages = **420 lines**
**After**: ~5 lines per page × 7 pages = **35 lines**
**Savings**: **385 lines eliminated** (92% reduction)

## Page-Specific Considerations

### Ask Questions Page (COMPLETED)

- Requires Polars conversion: `cohort = pl.from_pandas(cohort)`
- Has chat transcript state management (page-specific, preserved)
- Has dataset change detection (page-specific, preserved)

### Other Pages

Most pages use pandas DataFrames directly and don't need conversion:

```python
# Direct usage (most pages)
result = render_dataset_selector()
if result is None:
    return

dataset, cohort, dataset_choice, dataset_version = result
# cohort is already a pandas DataFrame - use directly
```

## Testing

After migration, verify each page:

1. **Manual Testing**:
   - Page loads without errors
   - Dataset selection works
   - Dataset loading works
   - Semantic scope displays correctly
   - Page-specific functionality still works

2. **UI Tests** (when available):
   - `make test-ui` should pass
   - No regressions in page functionality

## Benefits

1. **DRY Principle**: Eliminates 385 lines of duplication
2. **Maintainability**: Bug fixes and improvements in one place
3. **Consistency**: All pages use same loading pattern
4. **Testability**: Centralized logic easier to test
5. **Future-Proof**: Easy to enhance (caching, error handling, etc.)

## Next Steps

1. Migrate remaining 6 pages using the pattern above
2. Run `make test-ui` to verify no regressions
3. Update this checklist as pages are migrated
4. Remove `get_dataset_version()` function (replaced by component)
5. Commit migration with message:
   ```
   refactor: Migrate all UI pages to dataset_loader component

   Eliminates 385 lines of duplicated dataset loading code across 7 pages.
   All pages now use centralized dataset_loader component.

   Pages migrated:
   - 02_📊_Your_Dataset.py
   - 20_📊_Descriptive_Stats.py
   - 21_📈_Compare_Groups.py
   - 22_🎯_Risk_Factors.py
   - 23_⏱️_Survival_Analysis.py
   - 24_🔗_Correlations.py

   Tests: All UI tests passing
   ```
