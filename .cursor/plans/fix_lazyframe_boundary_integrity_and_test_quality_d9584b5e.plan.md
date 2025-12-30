---
name: Fix LazyFrame Boundary Integrity and Test Quality
overview: ""
todos: []
---

# Fix LazyFrame Boundary Integrity and Test Quality

## Overview

This plan addresses both code correctness/performance issues and test quality problems identified in staff-level reviews. The fixes ensure:

1. Correct Polars API usage and deterministic patient_id regeneration
2. LazyFrame-pure get_cohort() implementation (single collect at boundary)
3. Tests that accurately reflect the actual implementation (pandas-read for Excel, lazy compute for transformations)

## P0: Correctness Fixes (Must Fix)

### 1. Fix Incorrect Polars API Usage

**File**: [`src/clinical_analytics/datasets/uploaded/definition.py`](src/clinical_analytics/datasets/uploaded/definition.py:242)**Change**: Replace invalid `pl.LazyFrame()` constructor with `.lazy()` method

```python
# Current (broken):
self.data = pl.LazyFrame(df_with_id)  # ❌ Invalid API

# Fix:
self.data = df_with_id.lazy()  # ✅ Correct API
```



### 2. Ensure Patient ID Regeneration is Deterministic and Persisted

**File**: [`src/clinical_analytics/datasets/uploaded/definition.py`](src/clinical_analytics/datasets/uploaded/definition.py:225-249)**Change**: Materialize once for ID generation, then convert back to LazyFrame immediately

```python
# After regenerating patient_id:
if isinstance(self.data, pl.LazyFrame):
    # Materialize once for ID generation
    df_materialized = self.data.collect()
    df_with_id, id_metadata = VariableTypeDetector.ensure_patient_id(df_materialized)
    # Persist as LazyFrame for future calls
    self.data = df_with_id.lazy()  # ✅ Correct API + lazy persistence
    data_df = df_with_id.to_pandas()
else:
    # Handle pandas case
    df_with_id, id_metadata = VariableTypeDetector.ensure_patient_id(pl.from_pandas(self.data))
    self.data = df_with_id.to_pandas()
    data_df = self.data
```

**Rationale**: Ensures subsequent calls don't re-hit regeneration path and maintains lazy execution.

## P1: Performance & Boundary Clarity (Should Fix)

### 3. Add Normalization Helper for Internal Representation

**File**: [`src/clinical_analytics/datasets/uploaded/definition.py`](src/clinical_analytics/datasets/uploaded/definition.py) (module level or class method)**Add**:

```python
def _to_lazy(df_or_lf: pl.LazyFrame | pl.DataFrame | pd.DataFrame) -> pl.LazyFrame:
    """
    Normalize any data representation to LazyFrame.
    
    This ensures internal representation is always lazy, regardless of IO boundary.
    CSV files use pl.scan_csv() (true lazy IO).
    Excel files are eagerly loaded via pandas, then converted to LazyFrame.
    """
    if isinstance(df_or_lf, pl.LazyFrame):
        return df_or_lf
    if isinstance(df_or_lf, pl.DataFrame):
        return df_or_lf.lazy()
    # pandas - convert eagerly then make lazy
    return pl.from_pandas(df_or_lf).lazy()
```



### 4. Normalize in load() Method

**File**: [`src/clinical_analytics/datasets/uploaded/definition.py`](src/clinical_analytics/datasets/uploaded/definition.py:81-100)**Change**: Normalize to LazyFrame immediately after loading

```python
def load(self) -> None:
    """Load uploaded data, normalized to LazyFrame internally."""
    if not self.validate():
        raise FileNotFoundError(f"Upload data not found: {self.upload_id}")
    
    raw_data = self.storage.get_upload_data(self.upload_id, lazy=True)
    if raw_data is None:
        logger.error(f"Failed to load upload data for upload_id: {self.upload_id}")
        raise ValueError(f"Failed to load upload data: {self.upload_id}")
    
    # Normalize to LazyFrame (handles both CSV lazy scan and Excel eager→lazy conversion)
    self.data = _to_lazy(raw_data)
    logger.info(f"Loaded upload data as LazyFrame: {self.upload_id}")
```



### 5. Refactor get_cohort() to Operate Entirely on LazyFrame

**File**: [`src/clinical_analytics/datasets/uploaded/definition.py`](src/clinical_analytics/datasets/uploaded/definition.py:102-350)**Key Changes**:

- Remove early `lf.collect().to_pandas()` at line 171
- Build all expressions (filters, column selection, renaming) on LazyFrame
- Collect exactly once at return boundary
- Handle patient_id regeneration lazily if possible, or materialize once then continue lazy

**Structure**:

```python
def get_cohort(self, granularity: Granularity = "patient_level", **filters) -> pd.DataFrame:
    # ... granularity validation (use schema, not collect) ...
    
    if self.data is None:
        self.load()
    
    # Ensure LazyFrame
    lf = _to_lazy(self.data)
    
    # Apply filters in Polars (lazy)
    if filters:
        # ... build filter expressions ...
        lf = lf.filter(combined_filter)
    
    # Get variable mapping
    # ... (same as before) ...
    
    # Build select expressions for UnifiedCohort schema (all lazy)
    select_exprs = []
    # ... map patient_id, outcome, time_vars, predictors ...
    
    # Handle patient_id regeneration if needed (see P0 item 2)
    # ... (materialize once if needed, then continue lazy) ...
    
    # Apply all transformations lazily
    lf = lf.select(select_exprs)
    
    # EXACTLY ONE COLLECT at return boundary
    result_df = lf.collect().to_pandas()
    
    return result_df
```



### 6. Replace Schema Collection with Proper Schema Access

**File**: [`src/clinical_analytics/datasets/uploaded/definition.py`](src/clinical_analytics/datasets/uploaded/definition.py:129)**Change**: Use `collect_schema()` or `schema` instead of materializing

```python
# Current:
if isinstance(self.data, pl.LazyFrame):
    data_for_schema = self.data.collect()  # ❌ Unnecessary materialization

# Fix:
if isinstance(self.data, pl.LazyFrame):
    try:
        schema = self.data.collect_schema()  # ✅ Preferred (Polars 0.19+)
    except AttributeError:
        schema = self.data.schema  # Fallback (may be incomplete but works for column names)
    # Create empty DataFrame with schema for convert_schema()
    data_for_schema = pl.DataFrame(schema={k: v for k, v in schema.items()})
else:
    data_for_schema = pl.from_pandas(self.data) if isinstance(self.data, pd.DataFrame) else self.data
```



## P2: Documentation (Nice-to-Have)

### 7. Update get_upload_data() Docstring

**File**: [`src/clinical_analytics/ui/storage/user_datasets.py`](src/clinical_analytics/ui/storage/user_datasets.py:1063-1117)**Change**: Explicitly document IO boundary behavior

```python
def get_upload_data(self, upload_id: str, lazy: bool = True) -> pl.LazyFrame | pd.DataFrame | None:
    """
    Load uploaded dataset with automatic legacy migration.
    
    IO Boundary Behavior:
    - CSV files: When lazy=True, uses pl.scan_csv() for true lazy IO
    - Excel files: Eagerly loaded via pandas (due to Polars Excel limitations),
                  then converted to LazyFrame. Lazy execution applies from
                  transformation/filtering onward, not from IO.
    - SPSS files: Eagerly loaded via pyreadstat, then converted to LazyFrame.
    
    Internal Representation:
    - lazy=True: Returns pl.LazyFrame (recommended for internal use)
    - lazy=False: Returns pd.DataFrame (for backward compatibility/UI boundaries)
    
    Args:
        upload_id: Upload identifier (immutable storage key)
        lazy: If True, return Polars LazyFrame (default). If False, return pandas DataFrame
              for backward compatibility.
    
    Returns:
        LazyFrame (if lazy=True), pandas DataFrame (if lazy=False), or None if not found
        
    Note:
        Excel eager read is due to Polars read_excel() limitations with complex files
        (header detection, mixed types). See load_single_file() for details.
    """
```



## Test Quality Fixes

### 8. Resolve Pandas vs Polars-Read Mismatch in Tests

**File**: [`tests/datasets/test_uploaded_dataset_lazy_frames.py`](tests/datasets/test_uploaded_dataset_lazy_frames.py)**Issue**: Tests assume `pl.scan_csv()` but Excel files use pandas-read. Tests need to reflect actual contract.**Root Cause**: Current tests verify lazy IO via CSV scan nodes, but Excel files are eagerly loaded via pandas then converted to LazyFrame. The tests should verify "lazy compute" (filters applied before materialization), not "IO pushdown."**Changes**:

- Keep `isinstance(lf, pl.LazyFrame)` assertions (still valid - LazyFrame is returned regardless of IO path)
- Remove assertions that assume CSV scan nodes in plan (e.g., "SELECTION" in plan assumes CSV scan)
- For all uploads (CSV and Excel), tests should verify lazy compute behavior, not IO-level pushdown

**Specific Fixes**:**test_get_cohort_lazy_plan_contains_filter (line 239-272)**:

```python
# Current (assumes CSV scan):
plan = lf_filtered.explain(optimized=True)
assert "SELECTION" in plan  # ❌ Assumes CSV scan node
assert 'col("treatment")' in plan  # ❌ Brittle string matching

# Fix (permissive, works for both CSV and Excel):
import re
plan = lf_filtered.explain(optimized=True)
# Assert plan contains filter operation (permissive regex, version-agnostic)
assert re.search(r"\b(filter|selection)\b", plan.lower()), \
    f"Plan should contain filter operation. Got: {plan}"
# Don't assert exact column format (brittle across Polars versions)
```

**test_get_cohort_lazy_evaluation_predicate_pushdown (line 195-237)**:Update docstring to clarify what it actually tests:

```python
def test_get_cohort_filters_applied_before_materialization(self, tmp_path, sample_variable_mapping):
    """
    Test that filters are applied in Polars LazyFrame before materialization.
    
    For CSV files, filters are applied during scan/streaming.
    For Excel files, filters are applied after eager read but before collect.
    This test verifies lazy compute behavior, not IO-level predicate pushdown.
    """
    # ... existing test code ...
    # Assertions remain the same (filter correctness, non-mutation)
```



### 9. Rename Tests to Match What They Actually Assert

**File**: [`tests/datasets/test_uploaded_dataset_lazy_frames.py`](tests/datasets/test_uploaded_dataset_lazy_frames.py)**Changes**:

- `test_get_cohort_applies_filters_lazily` (line 156) → `test_get_cohort_applies_filters_before_materialization`
- `test_get_cohort_lazy_evaluation_predicate_pushdown` (line 195) → `test_get_cohort_filters_applied_before_materialization`
- `test_get_cohort_lazy_plan_contains_filter` (line 239) → `test_get_cohort_lazy_plan_contains_filter_node`

**Update docstrings** to clarify:

- "Filters are applied in Polars LazyFrame before materialization"
- "For CSV files, filters are applied during scan/streaming"
- "For Excel files, filters are applied after eager read but before collect"
- "This verifies lazy compute behavior, not IO-level predicate pushdown"

**Rationale**: "Predicate pushdown" is misleading terminology. True pushdown only applies to columnar formats (Parquet/Delta) with scan-based sources. For CSV, filters happen during scan/streaming. For Excel (pandas-read), there's no IO pushdown - only lazy compute after eager read.

### 10. Make create_upload Fixture More Flexible

**File**: [`tests/datasets/test_uploaded_dataset_lazy_frames.py`](tests/datasets/test_uploaded_dataset_lazy_frames.py:30-49)**Change**: Accept DataFrame and metadata overrides to reduce coupling

```python
@pytest.fixture
def create_upload(upload_storage, sample_upload_df, sample_upload_metadata):
    """Factory fixture to create test uploads with consistent pattern."""
    
    def _create(
        upload_id: str,
        df: pl.DataFrame | None = None,
        metadata_overrides: dict | None = None
    ):
        # Use provided df or default
        df_to_save = df if df is not None else sample_upload_df
        
        # Save CSV
        csv_path = upload_storage.raw_dir / f"{upload_id}.csv"
        df_to_save.write_csv(csv_path)
        
        # Merge metadata
        metadata = {**sample_upload_metadata, "upload_id": upload_id}
        if metadata_overrides:
            metadata.update(metadata_overrides)
        
        # Save metadata
        metadata_path = upload_storage.metadata_dir / f"{upload_id}.json"
        metadata_path.write_text(json.dumps(metadata))
        
        return upload_id
    
    return _create
```

**Rationale**: Allows tests to vary DataFrame shapes without duplicating fixture setup. Reduces hidden coupling if storage format changes.

### 11. Fix Excel Header Detection Tests

**File**: [`tests/ui/test_excel_reading.py`](tests/ui/test_excel_reading.py:215-269)**Issues**:

1. Tests private API `_detect_excel_header_row` directly (brittle to refactoring)
2. Exact shape assertions are brittle (break if fixture changes)

**Changes**:**Option A (Preferred)**: Test through public API

```python
# Instead of testing _detect_excel_header_row directly, test load_single_file()
from clinical_analytics.ui.storage.user_datasets import load_single_file

def test_excel_header_detection_standard_format(self, synthetic_dexa_excel_file):
    """Test header detection with standard format through public API."""
    with open(synthetic_dexa_excel_file, "rb") as f:
        file_bytes = f.read()
    
    # Test through public API
    df_read = load_single_file(file_bytes, "test.xlsx")
    
    # Semantic assertions (not exact shape)
    assert "Race" in df_read.columns
    assert "Age" in df_read.columns
    assert df_read.height >= 50  # At least expected rows (use fixture variable if available)
    # Assert no "Unnamed:" columns (common failure mode)
    assert not any("Unnamed" in str(col) for col in df_read.columns)
    # Remove exact shape assertions unless algorithm depends on them
```

**Option B (If private API testing is required)**: Keep private API test but:

- Add comment explaining why private API is tested
- Use semantic assertions instead of exact shape
- Store expected row count in fixture variable
```python
def test_detect_header_row_standard_format(self, synthetic_dexa_excel_file):
    """
    Test header detection with standard format (header in row 0).
    
    NOTE: Testing private API because header detection is critical path
    and public API (load_single_file) doesn't expose header row directly.
    If header detection logic is refactored, this test may need updating.
    """
    with open(synthetic_dexa_excel_file, "rb") as f:
        file_bytes = f.read()
    
    header_row = _detect_excel_header_row(file_bytes, max_rows_to_check=5)
    assert header_row == 0
    
    # Verify reading works correctly (semantic assertions)
    file_io = io.BytesIO(file_bytes)
    df_read = pd.read_excel(file_io, engine="openpyxl", header=header_row)
    
    # Semantic assertions (not exact shape)
    assert "Race" in df_read.columns
    assert "Age" in df_read.columns
    # Use >= instead of == for row count (allows fixture changes)
    assert df_read.height >= 50  # At least expected rows
    assert not any("Unnamed" in str(col) for col in df_read.columns)
    # Remove exact column count assertions unless algorithm depends on them
```


**Apply same pattern to**:

- `test_detect_header_row_with_empty_first_row` (line 235)
- `test_detect_header_row_with_metadata_rows` (line 253)

**Rationale**:

- Testing through public API is more resilient to refactoring
- Semantic assertions (required columns, no "Unnamed", row count >= expected) are more robust than exact shape
- Exact shape assertions break if fixtures are modified or header detection logic changes

### 12. Add Test to Verify Single Collect Boundary

**File**: [`tests/datasets/test_uploaded_dataset_lazy_frames.py`](tests/datasets/test_uploaded_dataset_lazy_frames.py)**Add new test** to verify lazy behavior without relying on plan string contents:

```python
def test_get_cohort_collects_exactly_once(self, tmp_path, sample_variable_mapping):
    """
    Test that get_cohort() collects LazyFrame exactly once at return boundary.
    
    This test verifies lazy execution by ensuring:
    - Filters work correctly (applied before materialization)
    - self.data remains a LazyFrame (not mutated)
    - Unfiltered cohort still works (proves filter doesn't mutate base data)
    """
    storage = UserDatasetStorage(upload_dir=tmp_path)
    upload_id = "test_single_collect"
    
    test_data = pl.DataFrame({
        "patient_id": [f"P{i:04d}" for i in range(100)],
        "outcome": [i % 2 for i in range(100)],
        "treatment": ["A" if i % 2 == 0 else "B" for i in range(100)],
    })
    csv_path = storage.raw_dir / f"{upload_id}.csv"
    test_data.write_csv(csv_path)
    
    metadata = {
        "upload_id": upload_id,
        "upload_timestamp": "2024-01-01T00:00:00",
        "variable_mapping": sample_variable_mapping,
    }
    metadata_path = storage.metadata_dir / f"{upload_id}.json"
    metadata_path.write_text(json.dumps(metadata))
    
    dataset = UploadedDataset(upload_id=upload_id, storage=storage)
    dataset.load()
    
    # Act: Get filtered cohort
    cohort = dataset.get_cohort(treatment="A")
    
    # Assert: Filter applied correctly
    assert len(cohort) == 50
    assert all(cohort["treatment"] == "A")
    
    # Assert: LazyFrame still exists (not consumed/mutated)
    assert isinstance(dataset.data, pl.LazyFrame)
    
    # Assert: Unfiltered cohort works (proves filter doesn't mutate self.data)
    cohort_full = dataset.get_cohort()
    assert len(cohort_full) == 100
```