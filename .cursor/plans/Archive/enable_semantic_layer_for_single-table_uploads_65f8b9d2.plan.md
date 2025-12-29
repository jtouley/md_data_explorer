---
name: Enable semantic layer for single-table uploads
overview: Fix the gap where single-table uploads don't get semantic layers initialized, preventing NL queries. Update `_maybe_init_semantic()` to handle both `variable_mapping` (single-table) and `inferred_schema` (multi-table), and add a method to build semantic config from `variable_mapping`.
todos:
  - id: phase1-write-tests
    content: "Phase 1: Write failing tests for single-table semantic layer initialization (test-first workflow)"
    status: pending
  - id: phase1-implement-build-config
    content: "Phase 1: Implement `_build_config_from_variable_mapping()` method to pass tests"
    status: pending
    dependencies:
      - phase1-write-tests
  - id: phase1-quality-gates
    content: "Phase 1: Run quality gates (make format, make lint-fix, make type-check, make test-fast) and commit"
    status: pending
    dependencies:
      - phase1-implement-build-config
  - id: phase2-write-tests
    content: "Phase 2: Write failing tests for `_maybe_init_semantic()` single-table path"
    status: pending
    dependencies:
      - phase1-quality-gates
  - id: phase2-update-maybe-init
    content: "Phase 2: Update `_maybe_init_semantic()` to handle variable_mapping path"
    status: pending
    dependencies:
      - phase2-write-tests
  - id: phase2-quality-gates
    content: "Phase 2: Run quality gates and commit"
    status: pending
    dependencies:
      - phase2-update-maybe-init
  - id: phase3-write-tests
    content: "Phase 3: Write integration tests for end-to-end NL query flow with single-table uploads"
    status: pending
    dependencies:
      - phase2-quality-gates
  - id: phase3-update-docstrings
    content: "Phase 3: Update docstrings and add structured logging"
    status: pending
    dependencies:
      - phase3-write-tests
  - id: phase3-quality-gates
    content: "Phase 3: Run full quality gates (make check) and commit"
    status: pending
    dependencies:
      - phase3-update-docstrings
---

# Enable Semantic Layer for Single-Table Uploads

## Problem

The UI expects semantic layers for all uploads (line 666 in `3_💬_Ask_Questions.py`), but `_maybe_init_semantic()` only initializes semantic layers for multi-table uploads with `inferred_schema`. Single-table uploads have `variable_mapping` instead, so they return early and never get semantic layers, causing "Semantic layer not ready" errors.

## Solution

Update `UploadedDataset._maybe_init_semantic()` to handle both:

1. **Multi-table uploads**: Use existing `_build_config_from_inferred_schema()` (unchanged)
2. **Single-table uploads**: Add new `_build_config_from_variable_mapping()` method

## Development Workflow

**MANDATORY: Follow test-first workflow (Red-Green-Refactor)**

1. **Write failing test** (Red) - Use AAA pattern (Arrange-Act-Assert)
2. **Run test immediately** - `make test-fast` to verify it fails as expected
3. **Implement minimum code to pass** (Green)
4. **Run test again** - `make test-fast` to verify it passes
5. **Fix code quality issues immediately** - Run `make lint-fix` and `make format` if needed
6. **Refactor** (Refactor)
7. **Run full test suite** - `make test-fast` before commit

**MANDATORY: Phase commit discipline**Before starting the next phase, you MUST:

1. Write tests for the phase
2. Run tests immediately (`make test-fast`)
3. Fix any test failures
4. Run `make check` - Ensure all quality gates pass (format, lint, type-check, test)
5. Commit all changes - Include both implementation AND tests in the commit

**Never commit code without tests. Never commit tests without running them.**

## Phase 1: Build Config from Variable Mapping

### Step 1.1: Write Failing Tests (RED)

**File**: `tests/datasets/test_uploaded_dataset.py` (create if doesn't exist)**Test cases** (use AAA pattern):

1. `test_build_config_from_variable_mapping_with_all_fields_returns_complete_config`

- Arrange: Create `UploadedDataset` with `variable_mapping` containing patient_id, outcome, time_zero, predictors
- Act: Call `_build_config_from_variable_mapping()`
- Assert: Config has correct structure (column_mapping, outcomes, time_zero, analysis)

2. `test_build_config_from_variable_mapping_without_outcome_handles_gracefully`

- Arrange: Create `UploadedDataset` with `variable_mapping` without outcome
- Act: Call `_build_config_from_variable_mapping()`
- Assert: Config has empty outcomes dict, default_outcome is None

3. `test_build_config_from_variable_mapping_detects_categorical_variables`

- Arrange: Create `UploadedDataset` with data containing categorical predictors (string type, numeric with ≤20 unique values, boolean)
- Act: Call `_build_config_from_variable_mapping()`
- Assert: `categorical_variables` list contains correct columns (assert behavior, not implementation)

4. `test_build_config_from_variable_mapping_infers_outcome_type_from_data`

- Arrange: Create `UploadedDataset` with outcome column that is binary (2 unique values) vs continuous (>2 unique values)
- Act: Call `_build_config_from_variable_mapping()`
- Assert: Outcome type is "binary" for binary outcomes, "continuous" for continuous (or log warning if ambiguous)

5. `test_build_config_from_variable_mapping_time_zero_matches_multi_table_format`

- Arrange: Create `UploadedDataset` with time_zero in variable_mapping
- Act: Call `_build_config_from_variable_mapping()`
- Assert: `time_zero` config is exactly `{"source_column": str}` (matches multi-table format from `inferred_schema`)

**Run tests immediately**: `make test-fast` (should fail - method doesn't exist yet)

### Step 1.2: Implement Method (GREEN)

**File**: `src/clinical_analytics/datasets/uploaded/definition.py`

**Add constant** (at module level, after imports):
```python
# Centralize categorical threshold (matches VariableTypeDetector)
CATEGORICAL_THRESHOLD = 20  # If unique values <= this, likely categorical
```

**Add method** (after `_build_config_from_inferred_schema()`, around line 362):

```python
def _build_config_from_variable_mapping(self, variable_mapping: dict[str, Any]) -> dict[str, Any]:
    """
    Build semantic layer config from variable_mapping (single-table uploads).
    
    Converts variable_mapping format to semantic layer config format.
    Matches structure of _build_config_from_inferred_schema() for consistency.
    
    Args:
        variable_mapping: Dictionary with patient_id, outcome, time_variables, predictors
        
    Returns:
        Semantic layer config dictionary compatible with SemanticLayer
    """
    # Load data if needed for categorical detection and outcome type inference
    if self.data is None:
        self.load()
    
    # Convert to Polars for efficient processing
    import polars as pl
    df_polars = pl.from_pandas(self.data) if isinstance(self.data, pd.DataFrame) else self.data
    
    # Build column_mapping
    column_mapping = {}
    patient_id_col = variable_mapping.get("patient_id")
    if patient_id_col:
        column_mapping[patient_id_col] = "patient_id"
    
    # Build outcomes with type inference
    outcomes = {}
    outcome_col = variable_mapping.get("outcome")
    if outcome_col:
        # Infer outcome type from data (don't hardcode "binary")
        outcome_type = self._infer_outcome_type(df_polars, outcome_col)
        outcomes[outcome_col] = {
            "source_column": outcome_col,
            "type": outcome_type,
        }
    
    # Build time_zero (must match multi-table format exactly: {"source_column": str})
    time_zero = {}
    time_vars = variable_mapping.get("time_variables", {})
    time_zero_col = time_vars.get("time_zero")
    if time_zero_col:
        # Match multi-table format: {"source_column": str}
        time_zero["source_column"] = time_zero_col
    
    # Detect categorical variables from predictors
    # TODO: Consider sampling strategy for large columns (series.n_unique() can be expensive)
    predictors = variable_mapping.get("predictors", [])
    categorical_variables = []
    
    for col in predictors:
        if col not in df_polars.columns:
            continue
        
        series = df_polars[col]
        dtype = series.dtype
        
        # String type → categorical
        if dtype == pl.Utf8 or dtype == pl.Categorical:
            categorical_variables.append(col)
        # Boolean → categorical
        elif dtype == pl.Boolean:
            categorical_variables.append(col)
        # Numeric with ≤CATEGORICAL_THRESHOLD unique values → categorical
        elif dtype.is_numeric():
            # For large columns, consider sampling (TODO: implement if performance issues)
            unique_count = series.n_unique()
            if unique_count > 100_000:
                logger.warning(
                    f"Column '{col}' has {unique_count:,} unique values. "
                    "Categorical detection may be slow. Consider sampling strategy."
                )
            if unique_count <= CATEGORICAL_THRESHOLD:
                categorical_variables.append(col)
    
    config = {
        "name": self.name,
        "display_name": self.metadata.get("original_filename", self.name),
        "status": "available",
        "init_params": {},  # Will be set to absolute CSV path in _maybe_init_semantic()
        "column_mapping": column_mapping,
        "outcomes": outcomes,
        "time_zero": time_zero,
        "analysis": {
            "default_outcome": outcome_col,
            "default_predictors": predictors,
            "categorical_variables": categorical_variables,
        },
    }
    return config

def _infer_outcome_type(self, df_polars: pl.DataFrame, outcome_col: str) -> str:
    """
    Infer outcome type from data characteristics.
    
    Args:
        df_polars: Polars DataFrame
        outcome_col: Outcome column name
        
    Returns:
        Outcome type: "binary", "continuous", or "time_to_event"
        
    Note:
        Defaults to "binary" if ambiguous, but logs warning.
    """
    if outcome_col not in df_polars.columns:
        logger.warning(f"Outcome column '{outcome_col}' not found in data, defaulting to 'binary'")
        return "binary"
    
    series = df_polars[outcome_col]
    dtype = series.dtype
    unique_count = series.n_unique()
    
    # Binary: exactly 2 unique values
    if unique_count == 2:
        return "binary"
    
    # Time-to-event: datetime type or name suggests time
    if dtype in (pl.Date, pl.Datetime, pl.Time):
        return "time_to_event"
    
    # Continuous: numeric with >2 unique values
    if dtype.is_numeric() and unique_count > 2:
        logger.warning(
            f"Outcome '{outcome_col}' has {unique_count} unique values. "
            "Inferring 'continuous' type. If this is binary, ensure data is encoded as 0/1."
        )
        return "continuous"
    
    # Default to binary with warning
    logger.warning(
        f"Outcome '{outcome_col}' type ambiguous (dtype={dtype}, unique={unique_count}). "
        "Defaulting to 'binary'. Verify outcome type is correct."
    )
    return "binary"
```

**Run tests**: `make test-fast` (should pass)

### Step 1.3: Quality Gates

**Run quality gates**:

```bash
make format        # Auto-format code
make lint-fix      # Auto-fix linting issues
make type-check    # Verify type hints
make test-fast     # Run fast tests
```

**If any fail, fix immediately before proceeding.**

### Step 1.4: Commit Phase 1

**Commit message template**:

```javascript
feat: Phase 1 - Add _build_config_from_variable_mapping() method

- Add method to convert variable_mapping to semantic layer config
- Implement categorical variable detection using Polars (centralized threshold)
- Infer outcome type from data (binary/continuous/time_to_event)
- Ensure time_zero format matches multi-table exactly
- Add comprehensive test suite (5 tests passing)

All tests passing: 5/5
```

**Verify commit includes**:

- Implementation: `src/clinical_analytics/datasets/uploaded/definition.py`
- Tests: `tests/datasets/test_uploaded_dataset.py`

## Phase 2: Update _maybe_init_semantic() for Single-Table Path

### Step 2.1: Write Failing Tests (RED)

**File**: `tests/datasets/test_uploaded_dataset.py`**Test cases**:

1. `test_maybe_init_semantic_with_variable_mapping_initializes_semantic_layer`

- Arrange: Create `UploadedDataset` with `variable_mapping` (no `inferred_schema`), mock CSV file exists
- Act: Call `get_semantic_layer()`
- Assert: `self.semantic` is not None, semantic layer config matches variable_mapping

2. `test_maybe_init_semantic_with_inferred_schema_still_works_multi_table`

- Arrange: Create `UploadedDataset` with `inferred_schema` (regression test)
- Act: Call `get_semantic_layer()`
- Assert: `self.semantic` is not None, uses existing `_build_config_from_inferred_schema()` path

3. `test_maybe_init_semantic_with_variable_mapping_skips_table_registration`

- Arrange: Create `UploadedDataset` with `variable_mapping`, no `{upload_id}_tables` directory
- Act: Call `get_semantic_layer()`
- Assert: Semantic layer created, no table registration attempted (verify `{upload_id}_tables` directory not accessed)

4. `test_maybe_init_semantic_without_schema_or_mapping_raises_valueerror`

- Arrange: Create `UploadedDataset` with neither `inferred_schema` nor `variable_mapping`
- Act: Call `get_semantic_layer()`
- Assert: Raises `ValueError` with clear message (explicit error semantics, not silent failure)

5. `test_maybe_init_semantic_sets_init_params_with_absolute_csv_path`

- Arrange: Create `UploadedDataset` with `variable_mapping`, mock CSV file exists
- Act: Call `get_semantic_layer()`
- Assert: `config["init_params"]["source_path"]` is set to absolute path of CSV file (critical regression test)

**Run tests immediately**: `make test-fast` (should fail - single-table path not implemented)

### Step 2.2: Update Method (GREEN)

**File**: `src/clinical_analytics/datasets/uploaded/definition.py`**Update `_maybe_init_semantic()`** (lines 287-341):**Changes**:

1. Update docstring: "Lazy initialization of semantic layer for all uploads (single-table and multi-table)"
2. Remove early return when no `inferred_schema`
3. Add branch for `variable_mapping` path with explicit error semantics:
   ```python
      # Check for inferred_schema first (multi-table path)
      inferred_schema = self.metadata.get("inferred_schema")
      if inferred_schema:
          config = self._build_config_from_inferred_schema(inferred_schema)
          upload_type = "multi-table"
      else:
          # Check for variable_mapping (single-table path)
          variable_mapping = self.metadata.get("variable_mapping")
          if variable_mapping:
              config = self._build_config_from_variable_mapping(variable_mapping)
              upload_type = "single-table"
          else:
              # Explicit error: raise instead of silent return
              raise ValueError(
                  f"No schema or mapping found for upload {self.upload_id}. "
                  "Upload must have either 'inferred_schema' (multi-table) or 'variable_mapping' (single-table)."
              )
   ```




4. Add structured logging:
   ```python
      logger.info(f"Initializing semantic layer for {upload_type} upload: {self.upload_id}")
      logger.info(f"Built semantic layer config from {upload_type} schema")
   ```




5. Use same CSV path check and SemanticLayer initialization for both paths
6. Only register individual tables for multi-table uploads (when `{upload_id}_tables` directory exists)
7. Update success logging to include upload type

**Run tests**: `make test-fast` (should pass)

### Step 2.3: Quality Gates

**Run quality gates**:

```bash
make format
make lint-fix
make type-check
make test-fast
```



### Step 2.4: Commit Phase 2

**Commit message**:

```javascript
feat: Phase 2 - Enable semantic layer for single-table uploads

- Update _maybe_init_semantic() to handle variable_mapping path
- Add explicit error semantics (raise ValueError instead of silent return)
- Add structured logging for single-table vs multi-table paths
- Maintain backward compatibility with multi-table uploads
- Add comprehensive test suite (5 tests passing)

All tests passing: 10/10 (Phase 1 + Phase 2)
```



## Phase 3: Integration Tests and Documentation

### Step 3.1: Write Integration Tests (RED)

**File**: `tests/datasets/test_uploaded_dataset_integration.py` (create if doesn't exist)**Test cases**:

1. `test_single_table_upload_semantic_layer_enables_nl_queries`

- Arrange: Create complete single-table upload with variable_mapping, mock UI call
- Act: Call `dataset.get_semantic_layer()` from UI context
- Assert: Semantic layer available, no "Semantic layer not ready" error

2. `test_multi_table_upload_semantic_layer_still_works_regression`

- Arrange: Create complete multi-table upload with inferred_schema
- Act: Call `dataset.get_semantic_layer()`
- Assert: Semantic layer available, all tables registered

3. `test_semantic_layer_config_structure_matches_expected_format`

- Arrange: Create single-table upload
- Act: Get semantic layer config
- Assert: Config structure matches SemanticLayer expectations (column_mapping, outcomes, time_zero, analysis)

4. `test_single_table_semantic_layer_init_params_has_absolute_path`

- Arrange: Create single-table upload with CSV file
- Act: Get semantic layer and check config
- Assert: `config["init_params"]["source_path"]` is absolute path (integration-level verification)

**Run tests**: `make test-fast` (should pass if Phases 1-2 done correctly)

### Step 3.2: Update Docstrings and Logging

**File**: `src/clinical_analytics/datasets/uploaded/definition.py`**Update docstrings**:

1. `get_semantic_layer()` (line 273):

- Change: "Overrides base class to support multi-table uploads"
- To: "Overrides base class to support all uploads (single-table and multi-table)"

2. `_maybe_init_semantic()` (line 288):

- Already updated in Phase 2, verify it's correct

**Add structured logging** (if not already done in Phase 2):

- Log upload type in all relevant log statements
- Use consistent log levels (info for success, warning for failures)

### Step 3.3: Full Quality Gates

**Run full quality gate**:

```bash
make check  # Runs format-check, lint, type-check, test
```

**If any fail, fix immediately before committing.**

### Step 3.4: Commit Phase 3

**Commit message**:

```javascript
feat: Phase 3 - Integration tests and documentation updates

- Add integration tests for end-to-end NL query flow
- Update docstrings to reflect support for all upload types
- Verify semantic layer config structure matches expectations
- Add comprehensive test suite (4 integration tests passing)

All tests passing: 14/14 (Phases 1-3)
```



## Files to Modify

1. **`src/clinical_analytics/datasets/uploaded/definition.py`**:

- Add `_build_config_from_variable_mapping()` method (Phase 1)
- Update `_maybe_init_semantic()` (Phase 2)
- Update docstrings (Phase 3)

2. **`tests/datasets/test_uploaded_dataset.py`** (create if doesn't exist):

- Phase 1: Tests for `_build_config_from_variable_mapping()`
- Phase 2: Tests for `_maybe_init_semantic()` single-table path

3. **`tests/datasets/test_uploaded_dataset_integration.py`** (create if doesn't exist):

- Phase 3: Integration tests for end-to-end flow

## Test Structure Standards

**Follow AAA pattern (Arrange-Act-Assert)**:

```python
def test_build_config_from_variable_mapping_with_all_fields_returns_complete_config():
    # Arrange: Set up test data and dependencies
    variable_mapping = {
        "patient_id": "patient_id",
        "outcome": "mortality",
        "time_variables": {"time_zero": "admission_date"},
        "predictors": ["age", "sex", "treatment"],
    }
    dataset = UploadedDataset(upload_id="test_123")
    dataset.metadata = {"variable_mapping": variable_mapping, ...}
    
    # Act: Execute the unit under test
    config = dataset._build_config_from_variable_mapping(variable_mapping)
    
    # Assert: Verify expected outcomes
    assert config["column_mapping"]["patient_id"] == "patient_id"
    assert "mortality" in config["outcomes"]
    assert config["analysis"]["default_outcome"] == "mortality"
```

**Use shared fixtures from `conftest.py`**:

- Check `tests/conftest.py` for existing fixtures
- Create fixtures for common test data (e.g., `sample_variable_mapping`, `sample_uploaded_dataset`)
- Never duplicate imports across test files

## Makefile Commands (MANDATORY)

**Always use Makefile commands**:

- `make format` / `make format-check` (never `ruff format` directly)
- `make lint` / `make lint-fix` (never `ruff check` directly)
- `make type-check` (never `mypy` directly)
- `make test-fast` / `make test` (never `pytest` directly)
- `make check` (full quality gate before commit)

**Never run tools directly.**

## Polars-First Patterns

**Use Polars for categorical detection**:

- Convert pandas to Polars: `pl.from_pandas(self.data)`
- Use Polars methods: `df[col].n_unique()`, `df[col].dtype`
- Use Polars types: `pl.Utf8`, `pl.Boolean`, `dtype.is_numeric()`

**Pandas only at boundaries** (if needed for existing code):

```python
# PANDAS EXCEPTION: Required for existing load() method compatibility
# TODO: Migrate to Polars throughout
```



## Verification Checklist

After all phases complete:

- [ ] All test files created and passing (`make test-fast`)
- [ ] All quality gates passing (`make check`)
- [ ] Implementation files committed
- [ ] Test files committed
- [ ] Commit messages include test counts
- [ ] No duplicate imports
- [ ] All linting issues fixed
- [ ] Code formatted correctly
- [ ] Single-table uploads can call `get_semantic_layer()` without errors
- [ ] NL queries work for single-table uploads in Ask Questions page
- [ ] Multi-table uploads still work (regression test)
- [ ] Config structure matches what SemanticLayer expects
- [ ] Categorical variables are correctly identified

## Notes

- CSV file path is the same for both: `{upload_id}.csv` in `raw_dir`
- Single-table uploads don't have `{upload_id}_tables` directory, so skip table registration
- Categorical detection loads data if not already loaded (lazy loading pattern)
- Outcome type inference: Defaults to "binary" if ambiguous, but logs warning (don't silently lie)
- Time_zero format: Must match multi-table exactly: `{"source_column": str}` (not `{"value": str}`)
- Error semantics: Raise `ValueError` instead of silent return (explicit failures prevent UI bugs)
- Categorical threshold: Centralized constant `CATEGORICAL_THRESHOLD = 20` (matches `VariableTypeDetector`)
- Performance: Log warning for large columns (>100k unique values) during categorical detection