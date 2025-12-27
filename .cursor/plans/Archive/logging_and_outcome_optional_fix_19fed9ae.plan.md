---
name: Logging and Outcome Optional Fix
overview: Replace print statements with structured logging and make outcomes optional in UploadedDataset.get_cohort() to align with semantic layer architecture. Also add granularity parameter support to all dataset implementations to match base class contract.
todos:
  - id: logging_covid_ms
    content: "Add logging to covid_ms/definition.py: replace print with logger.info()"
    status: pending
  - id: logging_mimic3
    content: "Add logging to mimic3/definition.py: replace all print statements with appropriate log levels"
    status: pending
  - id: logging_sepsis_def
    content: "Add logging to sepsis/definition.py: replace print with logger.warning()"
    status: pending
  - id: logging_sepsis_loader
    content: "Add logging to sepsis/loader.py: replace print statements with logger.info() and logger.error()"
    status: pending
  - id: logging_uploaded
    content: "Add logging to uploaded/definition.py: replace print with logger.warning()"
    status: pending
  - id: outcome_optional
    content: Make outcomes optional in UploadedDataset.get_cohort() - remove ValueError when outcome_col is None
    status: pending
  - id: granularity_covid_ms
    content: Add granularity parameter to CovidMSDataset.get_cohort() signature
    status: pending
  - id: granularity_mimic3
    content: Add granularity parameter to Mimic3Dataset.get_cohort() signature
    status: pending
  - id: granularity_sepsis
    content: Add granularity parameter to SepsisDataset.get_cohort() signature
    status: pending
  - id: granularity_uploaded
    content: Add granularity parameter to UploadedDataset.get_cohort() signature
    status: pending
  - id: verify_no_prints
    content: "Repo-wide verification: Run 'rg \"print\\(\" src/' to verify all print statements removed"
    status: pending
    dependencies:
      - logging_covid_ms
      - logging_mimic3
      - logging_sepsis_def
      - logging_sepsis_loader
      - logging_uploaded
  - id: verify_type_imports
    content: Verify all datasets import Granularity from base class (no Literal redeclarations)
    status: pending
    dependencies:
      - granularity_covid_ms
      - granularity_mimic3
      - granularity_sepsis
      - granularity_uploaded
  - id: test_outcome_optional
    content: Test that UploadedDataset.get_cohort() works without outcome column
    status: pending
    dependencies:
      - outcome_optional
  - id: test_granularity_validation
    content: Test that single-table datasets raise ValueError for non-patient_level granularity
    status: pending
    dependencies:
      - granularity_covid_ms
      - granularity_mimic3
      - granularity_sepsis
      - granularity_uploaded
  - id: test_pages_no_outcome
    content: "Integration test: Verify Descriptive Stats and Correlations pages work with datasets without outcomes"
    status: pending
    dependencies:
      - outcome_optional
  - id: verify_outcome_checks
    content: Verify pages requiring outcomes (Risk Factors, Survival Analysis) explicitly check for OUTCOME column
    status: pending
    dependencies:
      - outcome_optional
---

# Logging and Outcome Optional Fix

## Overview

This plan addresses two immediate issues:

1. **Logging**: Replace `print()` statements with structured logging for better observability
2. **Outcome Requirement**: Make outcomes optional in `UploadedDataset.get_cohort()` to align with semantic layer pattern (some analyses like Descriptive Stats and Correlations don't require outcomes)

Additionally, update all dataset implementations to support the `granularity` parameter to match the base class contract (already defined in `ClinicalDataset`).

## Problems

### 1. Missing Structured Logging

Current state: Multiple `print()` statements scattered across dataset implementations:

- `covid_ms/definition.py`: Line 47 - "Semantic layer initialized"
- `mimic3/definition.py`: Lines 55, 65, 67 - Validation failures and initialization
- `sepsis/definition.py`: Line 54 - Warning messages
- `sepsis/loader.py`: Lines 39, 59 - Processing messages
- `uploaded/definition.py`: Line 266 - Warning messages

Issues:

- No timestamps or log levels
- Can't filter by severity
- Not integrated with Streamlit's logging system
- Hard to debug in production

### 2. Hardcoded Outcome Requirement

Current state: `UploadedDataset.get_cohort()` raises `ValueError` if no outcome column is specified (line 122).Problem: Some analyses don't require outcomes:

- Descriptive Statistics (just needs patient_id and variables)
- Correlations (just needs numeric variables)
- Only Risk Factors and Survival Analysis require outcomes

The semantic layer already handles optional outcomes gracefully (see `semantic.py` lines 265-274), but `UploadedDataset` enforces them.

### 3. Granularity Parameter Missing

Current state: Base class `ClinicalDataset.get_cohort()` already defines `granularity` parameter (line 63), but implementations don't support it yet.Problem: Type mismatch - implementations use `**filters` only, breaking the contract.

## Solution Design

### Logging Architecture

Use Python's `logging` module with module-level loggers:

```python
import logging

logger = logging.getLogger(__name__)
```

Log levels:

- `logger.info()`: Normal operations (semantic layer initialization)
- `logger.warning()`: Recoverable issues (missing data, validation failures)
- `logger.error()`: Errors that prevent operation
- `logger.debug()`: Detailed debugging (optional, for development)

### Outcome Optional Pattern

Follow semantic layer pattern: outcomes are optional, only include if specified:

```python
# Map outcome (optional - semantic layer pattern)
if outcome_col:
    cohort_data[UnifiedCohort.OUTCOME] = self.data[outcome_col]
    cohort_data[UnifiedCohort.OUTCOME_LABEL] = outcome_label
# If no outcome specified, skip it - some analyses don't need outcomes
```



### Granularity Parameter Support

Add `granularity` parameter to all `get_cohort()` implementations with default `"patient_level"` for backward compatibility.**Import Pattern**: Import `Granularity` from base class to avoid type drift:

```python
from clinical_analytics.core.dataset import Granularity
```

**Validation Strategy**: For single-table datasets that only support patient-level:

- **Option A**: Raise `ValueError` with clear message (fail fast, explicit)
- **Option B**: Log warning and coerce to `"patient_level"` (permissive, but may hide errors)

**Decision**: Use **Option A (raise ValueError)** for consistency and clarity. Single-table datasets explicitly document their limitation.**Implementation**:

```python
def get_cohort(
    self,
    granularity: Granularity = "patient_level",
    **filters
) -> pd.DataFrame:
    # Validate granularity for single-table datasets
    if granularity != "patient_level":
        raise ValueError(
            f"{self.__class__.__name__} only supports patient_level granularity. "
            f"Requested: {granularity}"
        )
    # ... rest of implementation
```



## Implementation

### File: `src/clinical_analytics/datasets/covid_ms/definition.py`

**Changes:**

1. Add logging import and logger
2. Replace print statement with logger.info()
```python
import logging

logger = logging.getLogger(__name__)

# In load() method, line 47:
# OLD: print(f"Semantic layer initialized for {self.name}")
# NEW: logger.info(f"Semantic layer initialized for {self.name}")
```




3. Update `get_cohort()` signature to include `granularity` parameter
```python
from clinical_analytics.core.dataset import Granularity

def get_cohort(
    self,
    granularity: Granularity = "patient_level",
    **filters
) -> pd.DataFrame:
    # Validate: single-table datasets only support patient_level
    if granularity != "patient_level":
        raise ValueError(
            f"{self.__class__.__name__} only supports patient_level granularity. "
            f"Requested: {granularity}"
        )
    # ... existing implementation
```




### File: `src/clinical_analytics/datasets/mimic3/definition.py`

**Changes:**

1. Add logging import and logger
2. Replace print statements:

- Line 55: `logger.warning(f"MIMIC-III validation failed: {e}")`
- Line 65: `logger.warning("MIMIC-III database not available. Dataset will be empty.")`
- Line 67: `logger.info(f"Semantic layer initialized for {self.name}")`

3. Update `get_cohort()` signature to include `granularity` parameter with validation
```python
from clinical_analytics.core.dataset import Granularity

def get_cohort(
    self,
    granularity: Granularity = "patient_level",
    **filters
) -> pd.DataFrame:
    # Validate: MIMIC-III is patient-level only (for now)
    if granularity != "patient_level":
        raise ValueError(
            f"Mimic3Dataset only supports patient_level granularity. "
            f"Requested: {granularity}"
        )
    # ... existing implementation
```




### File: `src/clinical_analytics/datasets/sepsis/definition.py`

**Changes:**

1. Add logging import and logger
2. Replace print statement (line 54): `logger.warning(f"No PSV files found in {self.source_path}. Sepsis dataset will be empty.")`
3. Update `get_cohort()` signature to include `granularity` parameter with validation
```python
from clinical_analytics.core.dataset import Granularity

def get_cohort(
    self,
    granularity: Granularity = "patient_level",
    **filters
) -> pd.DataFrame:
    # Validate: Sepsis dataset is patient-level only
    if granularity != "patient_level":
        raise ValueError(
            f"SepsisDataset only supports patient_level granularity. "
            f"Requested: {granularity}"
        )
    # ... existing implementation
```




### File: `src/clinical_analytics/datasets/sepsis/loader.py`

**Changes:**

1. Add logging import and logger
2. Replace print statements:

- Line 39: `logger.info(f"Found {len(psv_files)} patient files. Processing...")`
- Line 59: `logger.error(f"Error processing {psv}: {e}")`

### File: `src/clinical_analytics/datasets/uploaded/definition.py`

**Changes:**

1. Add logging import and logger
2. Replace print statement (line 266): `logger.warning(f"Failed to load upload {upload_id}: {e}")`
3. **Critical Fix**: Make outcomes optional in `get_cohort()` (lines 118-122)
```python
# Map outcome (optional - semantic layer pattern)
# Some analyses (Descriptive Stats, Correlations) don't require outcomes
if outcome_col:
    cohort_data[UnifiedCohort.OUTCOME] = self.data[outcome_col]
    # Add outcome_label if available
    outcome_label = variable_mapping.get('outcome_label', 'outcome')
    cohort_data[UnifiedCohort.OUTCOME_LABEL] = outcome_label
# If no outcome specified, skip it - semantic layer handles this gracefully
# Downstream code must check for OUTCOME/OUTCOME_LABEL existence before using
```


**Important**: Pages that require outcomes (Risk Factors, Survival Analysis) must explicitly check:

```python
if UnifiedCohort.OUTCOME not in cohort.columns:
    st.error("This analysis requires an outcome column. Please upload data with outcome mapping.")
    return
```



4. Update `get_cohort()` signature to include `granularity` parameter with validation
```python
from clinical_analytics.core.dataset import Granularity

def get_cohort(
    self,
    granularity: Granularity = "patient_level",
    **filters
) -> pd.DataFrame:
    # Validate: single-table uploads only support patient_level
    if granularity != "patient_level":
        raise ValueError(
            f"UploadedDataset (single-table) only supports patient_level granularity. "
            f"Requested: {granularity}. Multi-table ZIP uploads support all granularities."
        )
    # ... existing implementation
```




# Validate: single-table uploads only support patient_level

if granularity != "patient_level":raise ValueError(f"UploadedDataset (single-table) only supports patient_level granularity. "f"Requested: {granularity}. Multi-table ZIP uploads support all granularities.")

# ... existing implementation

````javascript

**Note**: For single-table uploads, granularity is always `"patient_level"` and this is enforced via ValueError. Multi-table ZIP uploads will use the semantic layer which handles granularity properly.

## Testing Strategy

### Unit Tests

1. **Outcome Optional Test**: Verify `UploadedDataset.get_cohort()` works without outcome column

2. **Granularity Validation Test**: Verify single-table datasets raise `ValueError` for non-patient_level granularity

3. **Granularity Parameter Test**: Verify all implementations accept `granularity="patient_level"` without errors

**Note**: Skip logging tests - they're brittle and don't add value. Logging is for observability, not behavior.

### Integration Tests

1. **Descriptive Stats Page**: Verify it works with uploaded dataset that has no outcome

2. **Correlations Page**: Verify it works with uploaded dataset that has no outcome

3. **Risk Factors Page**: Verify it still works with outcome (backward compatibility)

4. **Outcome Validation**: Verify pages that require outcomes (Risk Factors, Survival Analysis) explicitly check for OUTCOME column and show clear error if missing

## Dependencies

- Python `logging` module (standard library, no new dependencies)

- Base class already defines `granularity` parameter (no changes needed to `dataset.py`)

## Success Criteria

- [ ] **All `print()` statements removed**: Repo-wide check `rg "print\(" src/` returns zero matches
- [ ] `UploadedDataset.get_cohort()` works without outcome column
- [ ] Descriptive Stats and Correlations pages work with datasets without outcomes
- [ ] Pages that require outcomes (Risk Factors, Survival Analysis) explicitly check for OUTCOME column and show clear error if missing
- [ ] All dataset implementations support `granularity` parameter using imported `Granularity` type alias
- [ ] Single-table datasets raise `ValueError` for non-patient_level granularity (consistent validation)
- [ ] No breaking changes - existing code continues to work
- [ ] Logs appear in console with timestamps and levels

## Migration Notes

- **Backward Compatibility**: Default `granularity="patient_level"` ensures existing code works

- **Outcome Optional**: Pages already check for outcome existence before using it (e.g., Risk Factors line 101 checks `if UnifiedCohort.OUTCOME in cohort.columns`). However, we should verify all pages that require outcomes explicitly check and error clearly.

- **Logging**: Streamlit already configured with `--logger.level=info` in `run_app.sh`, so logs will appear automatically

- **Type Safety**: Importing `Granularity` from base class prevents type drift and ensures consistency across implementations

## Verification Steps

### Post-Implementation Checklist

1. **Print Statement Removal**:
   ```bash
   rg "print\(" src/clinical_analytics/
   ```
   Should return zero matches (or only in test files if intentional)

2. **Type Consistency**:
   ```bash
   rg "Literal\[\"patient_level" src/clinical_analytics/datasets/
   ```
   Should return zero matches (all should import `Granularity`)

3. **Outcome Optional**:
    - Upload dataset without outcome mapping
    - Verify Descriptive Stats and Correlations work
    - Verify Risk Factors shows clear error if outcome missing

4. **Granularity Validation**:
    - Call `dataset.get_cohort(granularity="admission_level")` on single-table dataset
    - Verify `ValueError` raised with clear message

## Alignment with Plans

- **Multi-table refactor plan**: Granularity parameter support aligns with todo #7 (update all dataset implementations)



````