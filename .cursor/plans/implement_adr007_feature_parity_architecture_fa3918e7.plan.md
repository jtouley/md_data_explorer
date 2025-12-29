---
name: Implement ADR007 Feature Parity Architecture
overview: Implement unified architecture where single-table uploads are treated as multi-table uploads with 1 table. This eliminates feature gaps, code duplication, and conditional logic. Both upload types will use identical code paths, persistence mechanisms, semantic layer registration, and data access patterns.
todos:
  - id: phase1-normalize-upload
    content: Create normalize_upload_to_table_list() function and unified save_table_list() function
    status: pending
  - id: phase1-update-save-methods
    content: Update save_upload() and save_zip_upload() to use normalization and unified save logic
    status: pending
    dependencies:
      - phase1-normalize-upload
  - id: phase2-unify-persistence
    content: Modify save_table_list() to save individual tables to {upload_id}_tables/ for both upload types
    status: pending
    dependencies:
      - phase1-update-save-methods
  - id: phase2-convert-schema
    content: Convert variable_mapping to inferred_schema format during save for single-table uploads
    status: pending
    dependencies:
      - phase2-unify-persistence
  - id: phase3-register-tables
    content: Modify _maybe_init_semantic() to register all tables (not just multi-table) in DuckDB
    status: pending
    dependencies:
      - phase2-convert-schema
  - id: phase3-remove-granularity-restriction
    content: Remove hardcoded patient_level restriction in get_cohort() for single-table uploads
    status: pending
    dependencies:
      - phase3-register-tables
  - id: phase4-lazy-frames
    content: Update get_upload_data() to return Polars lazy frames instead of pandas DataFrames
    status: pending
    dependencies:
      - phase3-remove-granularity-restriction
  - id: phase4-update-load
    content: Update load() and get_cohort() methods to use lazy Polars evaluation
    status: pending
    dependencies:
      - phase4-lazy-frames
  - id: phase5-audit-conditionals
    content: Audit codebase for upload-type conditionals and document all instances
    status: pending
    dependencies:
      - phase4-update-load
  - id: phase5-remove-conditionals
    content: Refactor to unified code paths and remove all upload-type conditionals
    status: pending
    dependencies:
      - phase5-audit-conditionals
  - id: phase5-update-tests
    content: Update tests to verify unified behavior and feature parity for both upload types
    status: pending
    dependencies:
      - phase5-remove-conditionals
---

# Implement ADR007 Feature Parity Architecture

## Overview

This plan implements [ADR007: Feature Parity Architecture](../../docs/implementation/ADR/ADR007.md), establishing the principle **"Single-Table = Multi-Table with 1 Table"**. All upload types will use unified code paths, eliminating feature gaps and conditional logic.

## Critical Fixes Required Before Merge

**These three fixes must be implemented before the plan can be merged:**

1. **Phase 2: Schema Conversion Circular Dependency** - Move schema conversion to happen AFTER table normalization in `save_table_list()`, so DataFrame access is available for type inference.

2. **Phase 3: DuckDB Table Collision Handling** - Change from `CREATE OR REPLACE TABLE` to `CREATE TABLE IF NOT EXISTS` to prevent data loss on semantic layer re-init within same session.

3. **Phase 4: Lazy Frame Rollback Strategy** - Add compatibility shim with `lazy` feature flag in `get_upload_data()` to allow gradual migration and rollback if needed.

**See detailed implementation in each phase below.**

## Current State Analysis

### Key Divergences

1. **Persistence**: 

- Single-table: Only saves `{upload_id}.csv` (unified cohort)
- Multi-table: Saves both `{upload_id}.csv` AND `{upload_id}_tables/table_X.csv` (individual tables)

2. **Semantic Layer Registration**:

- Single-table: Only unified cohort registered in DuckDB
- Multi-table: Both unified cohort AND individual tables registered

3. **Data Access**:

- Single-table: Returns pandas DataFrame (eager)
- Multi-table: Uses Polars (but not consistently lazy)
- Single-table: Hardcoded `patient_level` granularity restriction

4. **Metadata Schema**:

- Single-table: Uses `variable_mapping` format
- Multi-table: Uses `inferred_schema` format

5. **Conditional Logic**:

- `_maybe_init_semantic()` checks `inferred_schema` vs `variable_mapping`
- `get_cohort()` restricts single-table to `patient_level`
- Multiple places check `MULTI_TABLE_ENABLED` flag

## Implementation Phases

### Phase 1: Normalize Upload Handling

**Goal**: Create unified entry point that normalizes both upload types to same data structure.**Files to Modify**:

- [`src/clinical_analytics/ui/storage/user_datasets.py`](src/clinical_analytics/ui/storage/user_datasets.py)

**Tasks**:

1. **Create `normalize_to_table_list()` function** (consistent naming: `verb_object()`):
   ```python
      def normalize_upload_to_table_list(
          file_bytes: bytes,
          filename: str,
          metadata: dict[str, Any] | None = None,
      ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
          """
          Normalize any upload to unified table list.
          
          This is the ONLY function that detects upload type.
          Everything downstream uses unified table list format.
          
          Returns:
              (tables, metadata) where tables is list of {"name": str, "data": pl.DataFrame}
          """
          # Detect upload type from file extension
          if filename.endswith('.zip'):
              # Multi-table: extract from ZIP
              tables = extract_zip_tables(file_bytes)
          else:
              # Single-file: wrap in list (becomes multi-table with 1 table)
              df = load_single_file(file_bytes, filename)
              tables = [{"name": "table_0", "data": df}]
          
          return tables, {"table_count": len(tables)}
   ```




2. **Create helper functions**:

- `extract_zip_tables()`: Extract from ZIP (reuse logic from `save_zip_upload()`)
  - **Concrete error handling logic**:
    ```python
    def extract_zip_tables(file_bytes: bytes) -> list[dict[str, Any]]:
        try:
            with zipfile.ZipFile(io.BytesIO(file_bytes)) as z:
                # Validate: Check for path traversal
                for entry in z.namelist():
                    if ".." in entry or entry.startswith("/"):
                        raise SecurityError(f"Invalid path: {entry}")
                
                # Validate: Check for non-CSV
                csv_files = [e for e in z.namelist() 
                            if (e.endswith(".csv") or e.endswith(".csv.gz"))
                            and not e.startswith("__MACOSX")
                            and not e.endswith("/")]
                if not csv_files:
                    raise UploadError("No CSV files in ZIP")
                
                # Handle name collisions
                seen = set()
                tables = []
                for entry in csv_files:
                    name = Path(entry).stem
                    if name.endswith(".csv"):  # Handle .csv.gz case
                        name = Path(name).stem
                    if name in seen:
                        raise UploadError(f"Duplicate table name: {name}")
                    seen.add(name)
                    
                    # Load and add to tables list
                    # ...
        except zipfile.BadZipFile:
            raise UploadError("Corrupted ZIP file")
    ```
  - **Raises**: `UploadError` for malformed ZIPs, `SecurityError` for path traversal attempts
- `load_single_file()`: Load CSV/Excel/SPSS as Polars DataFrame
  - **Naming**: Use original filename stem for table name (e.g., `patient_outcomes.csv` → `"patient_outcomes"`)
  - **Single-table**: `{"name": Path(filename).stem, "data": df}` (not `"table_0"`)

3. **Create unified `save_table_list()` function**:

- Accepts normalized table list (with DataFrames)
- **CRITICAL: Schema conversion happens AFTER normalization** (fixes circular dependency):
  ```python
  def save_table_list(self, tables, upload_id, metadata, ...):
      # 1. Normalize tables (get DataFrames) - already done by caller
      # 2. Convert schema (now has df access)
      if "variable_mapping" in metadata:
          metadata["inferred_schema"] = convert_schema(
              metadata["variable_mapping"],
              tables[0]["data"]  # Access normalized df
          )
      # 3. Save individual tables to {upload_id}_tables/
      # 4. Build unified cohort
      # 5. Save unified cohort CSV
      # 6. Save metadata with inferred_schema format
  ```
- Saves individual tables to `{upload_id}_tables/`
- Builds unified cohort
- Saves unified cohort CSV
- Returns metadata in `inferred_schema` format

4. **Update `save_upload()` and `save_zip_upload()`**:

- Both call `normalize_upload_to_table_list()`
- Both call unified `save_table_list()`
- Remove duplicate logic

**Success Criteria**:

- Both upload types normalize to same table list structure
- Single-table → `[{"name": "patient_outcomes", "data": df}]` (uses filename stem, not "table_0")
- Multi-table → `[{"name": "patients", "data": df1}, {"name": "admissions", "data": df2}, ...]` (uses ZIP entry names)
- No upload-type detection after normalization
- ZIP extraction handles errors gracefully (corrupted files, nested dirs, name collisions)

### Phase 2: Unify Persistence

**Goal**: Both upload types use identical persistence mechanism.

**Scope Note**: This is file-based CSV persistence only. Persistent DuckDB database (ADR002 Phase 1) and Parquet export (ADR002 Phase 2) are deferred to future work.

**Files to Modify**:

- [`src/clinical_analytics/ui/storage/user_datasets.py`](src/clinical_analytics/ui/storage/user_datasets.py)

**Tasks**:

1. **Modify `save_table_list()` to save individual tables**:

- Create `{upload_id}_tables/` directory
- Save each table as `table_X.csv` (where X is index or name)
- Single-table: Save as `table_0.csv`
- Multi-table: Save as `table_0.csv`, `table_1.csv`, etc.

2. **Ensure unified cohort CSV is saved** (for backward compatibility):

- Both upload types save `{upload_id}.csv` (unified cohort)
- Reuse existing unified cohort building logic

3. **Convert `variable_mapping` to `inferred_schema` format**:

- Create `convert_schema()` helper (consistent naming: `verb_object()`)
- **Concrete mapping rules** (requires DataFrame access for type inference):
  
  ```python
  def convert_schema(
      variable_mapping: dict[str, Any],
      df: pl.DataFrame,
  ) -> dict[str, Any]:
      """
      Convert variable_mapping to inferred_schema format.
      
      Input (variable_mapping):
      {
          "patient_id": "Patient ID",
          "outcome": "Mortality",
          "time_variables": {"time_zero": "Admission Date"},
          "predictors": ["Age", "Gender"],
          "outcome_label": "Death"
      }
      
      Output (inferred_schema):
      {
          "column_mapping": {"Patient ID": "patient_id"},
          "outcomes": {
              "Mortality": {
                  "source_column": "Mortality",
                  "type": "binary",  # Infer from data: unique_count == 2
                  "confidence": 0.9
              }
          },
          "time_zero": {"source_column": "Admission Date"},
          "analysis": {
              "default_outcome": "Mortality",
              "default_predictors": ["Age", "Gender"],
              "categorical_variables": [...]  # Infer from df: unique_count <= 20
          }
      }
      """
      # Type inference from DataFrame:
      # - polars_dtype: df[col].dtype
      # - categorical: df[col].n_unique() <= CATEGORICAL_THRESHOLD
      # - outcome type: "binary" if unique_count == 2, else "continuous"
  ```
  
- **Field-by-field transformation**:
  - `variable_mapping["patient_id"]` → `column_mapping[col] = "patient_id"`
  - `variable_mapping["outcome"]` → `outcomes[col] = {"source_column": col, "type": inferred_from_df}`
  - `variable_mapping["time_variables"]["time_zero"]` → `time_zero = {"source_column": col}`
  - `variable_mapping["predictors"]` → `analysis["default_predictors"]`
  - **Better categorical detection heuristic** (not just `n_unique() <= 20`):
    ```python
    def is_categorical(col: pl.Series) -> bool:
        unique_count = col.n_unique()
        total_count = len(col)
        
        # String columns with low cardinality and low uniqueness ratio
        if col.dtype == pl.Utf8:
            return unique_count <= 20 and unique_count / total_count < 0.5
        
        # Numeric columns need explicit annotation (never auto-categorical)
        # Prevents patient IDs, lab values, dates from being misclassified
        return False
    ```
  - **Infer granularities from columns**:
    ```python
    def infer_granularities(df: pl.DataFrame) -> list[str]:
        granularities = ["patient_level"]  # Always supported
        if "admission_id" in df.columns:
            granularities.append("admission_level")
        if "event_timestamp" in df.columns or "event_date" in df.columns:
            granularities.append("event_level")
        return granularities
    ```
- **Call during save AFTER table normalization** (fixes circular dependency)
- Store `inferred_schema` in metadata (keep `variable_mapping` for backward compatibility during transition)

4. **Update metadata schema**:

- Both upload types use `inferred_schema` format
- Add `tables` list to metadata: `["table_0"]` for single-table, `["table_0", "table_1", ...]` for multi-table
- Add `table_counts` dict: `{"table_0": row_count}`

**Success Criteria**:

- Single-table uploads save to `{upload_id}_tables/{filename_stem}.csv` (e.g., `patient_outcomes.csv`)
- Both upload types have identical directory structure
- Both upload types use `inferred_schema` metadata format (with complete type inference)
- Unified cohort CSV exists for both (backward compatibility)
- Schema conversion includes all required fields (column_mapping, outcomes, time_zero, analysis, granularities)
- Schema conversion happens AFTER table normalization (fixes circular dependency)
- Categorical detection uses improved heuristic (prevents misclassification of patient IDs, lab values)

### Phase 3: Unify Semantic Layer Registration

**Goal**: Both upload types register tables in DuckDB identically.

**Scope Note**: This registers tables in in-memory DuckDB during semantic layer initialization. Persistent DuckDB database (ADR002 Phase 1) is deferred to future work.

**Files to Modify**:

- [`src/clinical_analytics/datasets/uploaded/definition.py`](src/clinical_analytics/datasets/uploaded/definition.py)

**Tasks**:

1. **Modify `_maybe_init_semantic()` to register all tables**:

- Remove conditional logic checking `inferred_schema` vs `variable_mapping`
- Always use `inferred_schema` format (convert `variable_mapping` if needed)
- Register unified cohort table (existing logic)
- Register ALL individual tables from `{upload_id}_tables/` directory (currently only multi-table)

2. **Update table registration logic**:

- Check for `{upload_id}_tables/` directory (both upload types now have this)
- Read `tables` list from metadata
- **DuckDB collision handling**: Use `CREATE TABLE IF NOT EXISTS` (idempotent, prevents data loss on re-init)
  - **CRITICAL FIX**: Changed from `CREATE OR REPLACE` to `IF NOT EXISTS` to prevent data loss
  - Rationale: `CREATE OR REPLACE` wipes data on semantic layer re-init within same session
  - Table names: `{safe_dataset_name}_{table_name}` (e.g., `upload_abc123_patient_outcomes`)
  - Idempotent: Safe to call multiple times without side effects
  - Register each table: `CREATE TABLE IF NOT EXISTS {safe_dataset_name}_{table_name} AS SELECT * FROM read_csv_auto(?)`

3. **Remove hardcoded `patient_level` restriction**:

- Remove granularity validation in `get_cohort()` (line 114-119)
- **Add runtime validation**: Check if requested granularity exists in schema
  ```python
  if granularity != "patient_level":
      # Infer granularities if not present (backward compatibility)
      supported = metadata["inferred_schema"].get("granularities")
      if not supported:
          # Infer from columns during schema conversion
          supported = infer_granularities(df)  # ["patient_level", "admission_level", ...]
          metadata["inferred_schema"]["granularities"] = supported
      
      if granularity not in supported:
          raise ValueError(
              f"Dataset does not support {granularity} granularity. "
              f"Supported: {supported}"
          )
  ```
- Both upload types support all granularity levels (with validation)
- Update docstring to reflect this

4. **Unify config building**:

- Remove `_build_config_from_variable_mapping()` vs `_build_config_from_inferred_schema()` conditional
- Always use `inferred_schema` format
- Convert `variable_mapping` to `inferred_schema` if needed (for backward compatibility)

**Success Criteria**:

- Single-table uploads register individual table in DuckDB semantic layer (e.g., `upload_abc123_patient_outcomes`)
- Both upload types have identical query capabilities
- Both upload types support all granularity levels (with runtime validation)
- No conditional logic based on upload type in semantic layer initialization
- DuckDB table collisions handled with `CREATE TABLE IF NOT EXISTS` (idempotent, prevents data loss)
- Granularities inferred from columns and stored in schema

### Phase 4: Unify Data Access

**Goal**: Both upload types use same data loading and query patterns.

**Scope Note**: This changes the interface to return lazy frames from CSV scanning. Full Parquet export and lazy scanning from Parquet (ADR002 Phase 2) are deferred to future work.

**Files to Modify**:

- [`src/clinical_analytics/ui/storage/user_datasets.py`](src/clinical_analytics/ui/storage/user_datasets.py)
- [`src/clinical_analytics/datasets/uploaded/definition.py`](src/clinical_analytics/datasets/uploaded/definition.py)

**Tasks**:

1. **Update `get_upload_data()` to return Polars lazy frame**:

- Change return type from `pd.DataFrame | None` to `pl.LazyFrame | pd.DataFrame | None` (with feature flag)
- **CRITICAL: Add compatibility shim for rollback strategy**:
  ```python
  def get_upload_data(
      self, 
      upload_id: str,
      lazy: bool = True  # Feature flag for gradual migration
  ) -> pl.LazyFrame | pd.DataFrame | None:
      if lazy:
          return pl.scan_csv(...)
      else:
          # Fallback for backward compatibility during transition
          return pl.read_csv(...).to_pandas()
  ```
- Use `pl.scan_csv()` instead of `pd.read_csv()` when `lazy=True`
- Support predicate pushdown and lazy evaluation
- **After all callers updated**: Remove `lazy=False` path and return type becomes `pl.LazyFrame | None`

2. **Update `load()` method in `UploadedDataset`**:

- Accept Polars lazy frame from `get_upload_data()`
- Store as lazy frame (not eager DataFrame)
- Update logging to reflect lazy frame

3. **Update `get_cohort()` to use lazy evaluation**:

- Work with Polars lazy frames throughout
- Apply filters using Polars expressions
- Collect only at the end (or return lazy frame if caller supports it)

4. **Remove eager pandas loading paths**:

- Audit codebase for `pd.read_csv()` calls on upload data
- Replace with `pl.scan_csv()` where appropriate
- Ensure predicate pushdown works for both upload types

5. **Explicit caller migration checklist** (breaking change):

**Phase 4 Callers to Update**:
- [ ] `src/clinical_analytics/analysis/compute.py`:
  - [ ] `compute_descriptive_analysis()` - convert pandas `.describe()` to Polars
  - [ ] `compute_comparison_analysis()` - convert pandas groupby to Polars
  - [ ] `_try_convert_to_numeric()` - ensure works with Polars Series
- [ ] `src/clinical_analytics/datasets/uploaded/definition.py`:
  - [ ] `load()` - accept `pl.LazyFrame`, store as lazy
  - [ ] `get_cohort()` - work with lazy frames, collect at end
- [ ] `src/clinical_analytics/ui/pages/3_💬_Ask_Questions.py`:
  - [ ] All `st.dataframe()` calls - add `.collect().to_pandas()` before rendering
  - [ ] `execute_analysis()` - handle lazy frames
- [ ] `src/clinical_analytics/ui/pages/2_📊_Your_Dataset.py`:
  - [ ] Data preview rendering - collect lazy frame before display
- [ ] `src/clinical_analytics/ui/pages/20_📊_Descriptive_Stats.py`:
  - [ ] All analysis functions - ensure Polars compatibility
- [ ] `src/clinical_analytics/ui/pages/21_📈_Compare_Groups.py`:
  - [ ] Comparison analysis - ensure Polars compatibility
- [ ] `src/clinical_analytics/ui/pages/22_🎯_Risk_Factors.py`:
  - [ ] Risk factor analysis - ensure Polars compatibility
- [ ] `src/clinical_analytics/ui/pages/23_⏱️_Survival_Analysis.py`:
  - [ ] Survival analysis - ensure Polars compatibility
- [ ] `src/clinical_analytics/ui/pages/24_🔗_Correlations.py`:
  - [ ] Correlation analysis - ensure Polars compatibility
- [ ] `src/clinical_analytics/ui/app.py`:
  - [ ] Any data loading - ensure lazy frame handling

6. **Add test to verify caller migration completion**:
   ```python
   def test_all_callers_handle_lazy_frames():
       """Verify no eager pandas DataFrame assumptions remain."""
       violations = []
       patterns = [
           r"\.to_pandas\(\)",  # Should only appear before st.dataframe()
           r"pd\.read_csv.*upload",  # Should be pl.scan_csv
           r"isinstance.*pd\.DataFrame",  # Should check pl.LazyFrame
       ]
       
       for pattern in patterns:
           matches = grep_codebase(pattern, include_dirs=["src/clinical_analytics"])
           # Filter out acceptable uses (e.g., right before st.dataframe)
           violations.extend(validate_usage(matches))
       
       assert len(violations) == 0, (
           f"Found {len(violations)} eager pandas anti-patterns:\n"
           + "\n".join(f"  {v}" for v in violations)
       )
   ```

**Success Criteria**:

- Both upload types return Polars lazy frames (with compatibility shim for gradual migration)
- Both upload types support lazy evaluation
- Both upload types have identical query performance characteristics
- No eager pandas loading for upload data (after migration complete)
- All callers updated to handle lazy frames (see migration checklist above)
- Streamlit rendering collects lazy frames before display
- Test verifies no eager pandas anti-patterns remain

### Phase 5: Remove Conditional Logic

**Goal**: Eliminate all upload-type conditionals.**Files to Audit**:

- [`src/clinical_analytics/datasets/uploaded/definition.py`](src/clinical_analytics/datasets/uploaded/definition.py)
- [`src/clinical_analytics/ui/storage/user_datasets.py`](src/clinical_analytics/ui/storage/user_datasets.py)
- [`src/clinical_analytics/ui/pages/1_📤_Add_Your_Data.py`](src/clinical_analytics/ui/pages/1_📤_Add_Your_Data.py)
- [`src/clinical_analytics/ui/pages/2_📊_Your_Dataset.py`](src/clinical_analytics/ui/pages/2_📊_Your_Dataset.py)
- [`src/clinical_analytics/ui/pages/3_💬_Ask_Questions.py`](src/clinical_analytics/ui/pages/3_💬_Ask_Questions.py)

**Tasks**:

1. **Audit codebase for upload-type conditionals**:

- Search for `if.*single|if.*multi|upload_type|MULTI_TABLE_ENABLED` patterns
- Document all conditionals found
- Categorize: necessary (feature flag) vs unnecessary (upload type check)
- **Automated test to prevent regressions**:
  ```python
  def test_no_upload_type_conditionals_in_codebase():
      """Verify no upload-type branching exists (except feature flags)."""
      patterns = [
          r"if.*upload_type\s*==\s*['\"]single",
          r"if.*upload_type\s*==\s*['\"]multi",
          r"if.*inferred_schema.*else.*variable_mapping",
          r"if.*granularity\s*!=\s*['\"]patient_level['\"].*single",
      ]
      violations = []
      for pattern in patterns:
          matches = grep_codebase(pattern, exclude_dirs=["tests", ".venv"])
          violations.extend(matches)
      
      assert len(violations) == 0, (
          f"Found {len(violations)} upload-type conditionals:\n"
          + "\n".join(f"  {v}" for v in violations)
      )
  ```

2. **Refactor to unified code paths**:

- Remove `if inferred_schema else variable_mapping` conditionals
- Remove `if granularity != "patient_level"` restrictions
- Remove upload-type-specific error messages
- Use unified table list structure everywhere

3. **Update tests**:

- Verify unified behavior for both upload types
- Test feature parity explicitly: same query produces identical results
- Remove upload-type-specific test branches

4. **Remove duplicate code**:

- Consolidate similar logic from `save_upload()` and `save_zip_upload()`
- Remove upload-type-specific helper functions
- Use unified functions for both types

**Success Criteria**:

- No conditional logic based on upload type (except feature flag checks)
- Single code path handles both upload types
- Tests verify identical behavior for both upload types
- No duplicate code between upload types
- Automated test prevents regression of upload-type conditionals

## Implementation Details

### Normalization Function Design

```python
def normalize_upload_to_table_list(
    file_bytes: bytes,
    filename: str,
    metadata: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Normalize any upload to unified table list.
    
    This is the ONLY function that detects upload type.
    Everything downstream uses unified table list format.
    
    Args:
        file_bytes: File content
        filename: Original filename (used to detect type)
        metadata: Optional metadata (for progress callbacks)
    
    Returns:
        (tables, table_metadata) where:
    - tables: list of {"name": str, "data": pl.DataFrame}
    - table_metadata: dict with table_count, table_names, etc.
    """
```



### Unified Save Function Design

```python
def save_table_list(
    self,
    tables: list[dict[str, Any]],
    upload_id: str,
    metadata: dict[str, Any],
    progress_cb: Callable[[int, str], None] | None = None,
) -> tuple[bool, str]:
    """
    Save normalized table list to disk.
    
    Unified save logic for both single-table and multi-table uploads.
    
    Args:
        tables: Normalized table list [{"name": str, "data": pl.DataFrame}]
        upload_id: Upload identifier
        metadata: Upload metadata
        progress_cb: Optional progress callback
    
    Returns:
        (success, message)
    """
    # 1. Convert schema (AFTER normalization, has df access)
    #    - If variable_mapping exists, convert using tables[0]["data"]
    # 2. Save individual tables to {upload_id}_tables/
    # 3. Build unified cohort
    # 4. Save unified cohort CSV
    # 5. Save metadata with inferred_schema format
```



### Schema Conversion

**Function naming**: Use `convert_schema()` (consistent `verb_object()` pattern)

**Detailed mapping specification**:

```python
def convert_schema(
    variable_mapping: dict[str, Any],
    df: pl.DataFrame,
) -> dict[str, Any]:
    """
    Convert variable_mapping format to inferred_schema format.
    
    Input structure (variable_mapping):
    {
        "patient_id": str | None,  # Column name
        "outcome": str | None,     # Column name
        "time_variables": {
            "time_zero": str | None  # Column name
        },
        "predictors": list[str],   # List of column names
        "outcome_label": str        # Optional label
    }
    
    Output structure (inferred_schema):
    {
        "column_mapping": {
            col_name: "patient_id"  # Maps to UnifiedCohort schema
        },
        "outcomes": {
            col_name: {
                "source_column": str,
                "type": "binary" | "continuous",  # Infer from df
                "confidence": float  # 0.0-1.0
            }
        },
        "time_zero": {
            "source_column": str  # Or {"value": str} if static
        },
        "analysis": {
            "default_outcome": str | None,
            "default_predictors": list[str],
            "categorical_variables": list[str]  # Infer from df
        }
    }
    
    Type inference rules:
    - outcome type: "binary" if df[col].n_unique() == 2, else "continuous"
    - categorical: Use improved `is_categorical()` heuristic (not just n_unique <= 20)
    - polars_dtype: df[col].dtype (Int64, Float64, Utf8, Boolean, Date, Datetime)
    - granularities: Infer from column presence (admission_id, event_timestamp)
    """
    
    # Helper functions for schema conversion:
    
    def is_categorical(col: pl.Series) -> bool:
        """Improved categorical detection heuristic."""
        unique_count = col.n_unique()
        total_count = len(col)
        
        # String columns with low cardinality and low uniqueness ratio
        if col.dtype == pl.Utf8:
            return unique_count <= 20 and unique_count / total_count < 0.5
        
        # Numeric columns need explicit annotation (never auto-categorical)
        # Prevents patient IDs, lab values, dates from being misclassified
        return False
    
    def infer_granularities(df: pl.DataFrame) -> list[str]:
        """Infer supported granularities from column presence."""
        granularities = ["patient_level"]  # Always supported
        if "admission_id" in df.columns:
            granularities.append("admission_level")
        if "event_timestamp" in df.columns or "event_date" in df.columns:
            granularities.append("event_level")
        return granularities
    inferred = {
        "column_mapping": {},
        "outcomes": {},
        "time_zero": {},
        "analysis": {
            "default_outcome": None,
            "default_predictors": [],
            "categorical_variables": [],
        }
    }
    
    # Map patient_id
    if patient_id_col := variable_mapping.get("patient_id"):
        inferred["column_mapping"][patient_id_col] = "patient_id"
    
    # Map outcome with type inference
    if outcome_col := variable_mapping.get("outcome"):
        unique_count = df[outcome_col].n_unique()
        outcome_type = "binary" if unique_count == 2 else "continuous"
        inferred["outcomes"][outcome_col] = {
            "source_column": outcome_col,
            "type": outcome_type,
            "confidence": 0.9 if outcome_type == "binary" else 0.7,
        }
        inferred["analysis"]["default_outcome"] = outcome_col
    
    # Map time_zero
    if time_zero_col := variable_mapping.get("time_variables", {}).get("time_zero"):
        inferred["time_zero"] = {"source_column": time_zero_col}
    
    # Map predictors and detect categoricals (better heuristic)
    predictors = variable_mapping.get("predictors", [])
    for col in predictors:
        if col in df.columns:
            inferred["analysis"]["default_predictors"].append(col)
            # Detect categorical using improved heuristic
            if is_categorical(df[col]):
                inferred["analysis"]["categorical_variables"].append(col)
    
    # Infer granularities from columns
    inferred["granularities"] = infer_granularities(df)
    
    return inferred
```



## Testing Strategy

### Feature Parity Tests

```python
def test_single_csv_and_multi_table_zip_produce_identical_query_results():
    """Verify single-table CSV and multi-table ZIP produce identical semantic layer queries."""
    # Create single-table upload
    single_upload = create_single_table_upload(test_data)
    
    # Create multi-table upload (with 1 table)
    multi_upload = create_multi_table_upload([test_data])
    
    # Verify identical persistence
    assert single_upload.persistence_structure == multi_upload.persistence_structure
    
    # Verify identical semantic layer registration
    assert single_upload.semantic_layer_tables == multi_upload.semantic_layer_tables
    
    # Verify identical query capabilities
    single_result = query_semantic_layer(single_upload, "SELECT * FROM table_0")
    multi_result = query_semantic_layer(multi_upload, "SELECT * FROM table_0")
    assert single_result == multi_result

def test_viral_load_descriptive_stats_identical_for_csv_and_zip_uploads():
    """Verify viral load descriptive statistics are identical for CSV vs ZIP uploads."""
    # Test with clinical scenario: descriptive stats on viral load
    # Single-table: CSV with Treatment, Viral Load columns
    # Multi-table: ZIP with same data
    # Both should produce identical mean, median, std, etc.
```



### Lazy Evaluation Tests

```python
def test_lazy_evaluation_works_for_both_upload_types():
    """Verify both upload types support lazy Polars evaluation."""
    # Test predicate pushdown
    # Test lazy frame operations
    # Test collect() only at end
```



## Migration Considerations

### Backward Compatibility

1. **Existing single-table uploads**:

- May not have `{upload_id}_tables/` directory
- May have `variable_mapping` instead of `inferred_schema`
- **Migration trigger point**: In `get_upload_data()` method (read path)
  ```python
  def get_upload_data(self, upload_id: str, lazy: bool = True) -> pl.LazyFrame | pd.DataFrame | None:
      # Check migration status in metadata (prevents re-running on every call)
      metadata = self.get_upload_metadata(upload_id)
      if not metadata.get("migrated_to_v2", False):
          # Legacy single-table upload - migrate on-the-fly
          _migrate_legacy_upload(upload_id)
          # Mark as migrated
          metadata["migrated_to_v2"] = True
          self.update_metadata(upload_id, {"migrated_to_v2": True})
      
      # Now proceed with lazy frame loading
      tables_dir = self.raw_dir / f"{upload_id}_tables"
      if lazy:
          return pl.scan_csv(tables_dir / "table_0.csv")
      else:
          return pl.read_csv(tables_dir / "table_0.csv").to_pandas()
  ```
- **Migration function**: `_migrate_legacy_upload(upload_id)`
  - Load existing `{upload_id}.csv` as DataFrame
  - Create `{upload_id}_tables/` directory
  - Save as `{filename_stem}.csv` (use original filename stem if available in metadata)
  - Convert `variable_mapping` to `inferred_schema` (requires DataFrame for type inference)
  - Update metadata JSON with `inferred_schema` and `migrated_to_v2: true`

2. **Metadata format**:

- Keep `variable_mapping` in metadata during transition
- Add `inferred_schema` alongside it
- Eventually deprecate `variable_mapping`

### Breaking Changes

1. **`get_upload_data()` return type**:

- Changes from `pd.DataFrame` to `pl.LazyFrame`
- Callers must be updated to use Polars API

2. **Granularity restrictions**:

- Single-table uploads now support all granularities
- May affect existing code that assumes `patient_level` only

## Success Criteria

### Must-Have (Go/No-Go)

- [ ] Single-table and multi-table uploads have identical persistence structure
- [ ] Both upload types register all tables in DuckDB semantic layer
- [ ] Both upload types support lazy Polars evaluation
- [ ] Both upload types use same metadata schema (`inferred_schema` format)
- [ ] Both upload types go through same validation pipeline
- [ ] Both upload types support all granularity levels
- [ ] No conditional logic based on upload type exists in codebase
- [ ] Same query produces identical results for both upload types
- [ ] All analysis features work identically for both upload types

### Nice-to-Have

- [ ] Migration script converts existing single-table uploads to new format
- [ ] Performance benchmarks show no regression
- [ ] Code coverage for unified code paths >90%
- [ ] Documentation explains unified architecture (not separate upload types)

## Implementation Order

Execute phases sequentially:

1. **Phase 1** (Normalize Upload Handling) - Foundation for all other phases
2. **Phase 2** (Unify Persistence) - Enables semantic layer registration
3. **Phase 3** (Unify Semantic Layer) - Enables query capabilities
4. **Phase 4** (Unify Data Access) - Enables lazy evaluation
5. **Phase 5** (Remove Conditionals) - Final cleanup

## Scope Boundaries

**This plan (ADR007) focuses on:**
- File-based CSV persistence (not DuckDB persistence - deferred to ADR002 Phase 1)
- In-memory DuckDB registration (not persistent DuckDB database - deferred to ADR002 Phase 1)
- Lazy frame interface from CSV scanning (not Parquet export - deferred to ADR002 Phase 2)
- Feature parity between upload types

**Deferred to ADR002:**
- Persistent DuckDB database at `data/analytics.duckdb` (Phase 1)
- Parquet export for columnar storage (Phase 2)
- Full lazy evaluation with Parquet scanning (Phase 2)
- Conversation history (JSONL) (Phase 3)

**Deferred to ADR003:**
- Trust layer (verification UI) (Phase 1)
- Enhanced LLM parsing (Phase 2)
- Adaptive dictionary persistence (Phase 3)

## Dependencies

- **Blocks**: ADR001 (Comparison Analysis & Conversational UI), ADR002 Phase 1+ (DuckDB persistence)
- **Prerequisites**: None (this is the foundation)
- **Related**: Multi-Table Handler Refactor Plan (advanced patterns)

## References

- [ADR007: Feature Parity Architecture](../../docs/implementation/ADR/ADR007.md) - **AUTHORITATIVE SOURCE**
- [ADR002: Persistent Storage Layer](../../docs/implementation/ADR/ADR002.md) - Defines persistent DuckDB storage