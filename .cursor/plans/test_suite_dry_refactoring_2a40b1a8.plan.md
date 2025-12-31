---
name: Test Suite DRY Refactoring
overview: Refactor test suite to eliminate duplicate fixtures and hardcoded test data, consolidating common patterns into conftest.py per AGENTS.md guidelines. This will improve maintainability and reduce duplication across 21+ test files.
todos:
  - id: phase1-factory-fixture
    content: Add make_semantic_layer factory fixture to conftest.py with support for custom data, config, and workspace
    status: pending
  - id: phase1-replace-simple
    content: Replace simple mock_semantic_layer fixtures in test_queryplan_contract.py, test_queryplan_only_path.py, test_queryplan_conversion.py
    status: pending
    dependencies:
      - phase1-factory-fixture
  - id: phase1-replace-medium
    content: Replace medium-complexity fixtures in test_semantic_queryplan_execution.py, test_semantic_run_key_determinism.py, test_semantic_observability.py
    status: pending
    dependencies:
      - phase1-factory-fixture
  - id: phase1-replace-complex
    content: Replace complex fixtures in test_ask_questions_full_flow.py and remaining 14 files with semantic layer fixtures
    status: pending
    dependencies:
      - phase1-factory-fixture
  - id: phase1-verify
    content: Run make test-fast and verify all tests pass after Phase 1 replacements
    status: pending
    dependencies:
      - phase1-replace-simple
      - phase1-replace-medium
      - phase1-replace-complex
  - id: phase2-dataframe-factories
    content: Add make_patients_df, make_cohort_with_categorical, and make_multi_table_setup factory fixtures to conftest.py
    status: pending
  - id: phase2-refactor-multitable
    content: Refactor test_multi_table_handler.py to use make_multi_table_setup factory (75 hardcoded DataFrames)
    status: pending
    dependencies:
      - phase2-dataframe-factories
  - id: phase2-refactor-compute
    content: Refactor test_compute.py to extract common DataFrame patterns (32 instances)
    status: pending
    dependencies:
      - phase2-dataframe-factories
  - id: phase2-refactor-relationship
    content: Refactor test_relationship_detector.py to use DataFrame factories (21 instances)
    status: pending
    dependencies:
      - phase2-dataframe-factories
  - id: phase2-refactor-schema
    content: Refactor test_schema_conversion.py to use DataFrame factories (19 instances)
    status: pending
    dependencies:
      - phase2-dataframe-factories
  - id: phase2-verify
    content: Run make test-fast and verify all tests pass after Phase 2 refactoring
    status: pending
    dependencies:
      - phase2-refactor-multitable
      - phase2-refactor-compute
      - phase2-refactor-relationship
      - phase2-refactor-schema
  - id: phase3-import-cleanup
    content: Remove redundant import polars as pl from 32 test files (optional, verify conftest.py import works first)
    status: pending
  - id: phase4-consolidate-fixtures
    content: Review and consolidate similar fixtures (sample_cohort_with_categorical/numeric) and add documentation
    status: pending
    dependencies:
      - phase1-verify
      - phase2-verify
  - id: final-validation
    content: Run make check, verify test count/grep metrics, and confirm compliance with AGENTS.md
    status: pending
    dependencies:
      - phase1-verify
      - phase2-verify
      - phase4-consolidate-fixtures
---

# Test Suite DRY Refactoring Plan



## Overview

Refactor the test suite to eliminate violations of AGENTS.md guidelines:

- **21 files** with duplicate `mock_semantic_layer`/`semantic_layer` fixtures
- **312 hardcoded DataFrames** across 28 files that should use fixtures
- **32 files** with redundant imports (optional cleanup)

## Current State Analysis

### Duplicate Fixtures Found

- `mock_semantic_layer` defined in 8+ files with similar patterns
- `semantic_layer` defined in 13+ files (some in test classes)
- Variations include: `mock_semantic_layer_for_execution`, `semantic_layer_with_clinical_columns`, etc.

### Common Patterns Identified

1. **SemanticLayer setup**: Workspace creation, CSV data, config dict
2. **Patient DataFrames**: `patient_id`, `age`, `status` columns
3. **Cohort DataFrames**: `patient_id`, `outcome`, `age`, `treatment`
4. **Multi-table setups**: patients, medications, bridge tables

## Implementation Plan

### Phase 1: Consolidate Semantic Layer Fixtures (HIGH PRIORITY)

**Goal**: Replace 21 duplicate fixtures with factory pattern in `conftest.py`

#### Step 1.1: Add Factory Fixture to `conftest.py`

Add to [`tests/conftest.py`](tests/conftest.py) after line 523:

```python
@pytest.fixture
def make_semantic_layer(tmp_path):
    """
    Factory fixture for creating SemanticLayer instances.
    
    Usage:
        def test_example(make_semantic_layer):
            layer = make_semantic_layer(
                dataset_name="custom",
                data={"patient_id": [1, 2, 3], "age": [45, 62, 38]},
                config_overrides={"time_zero": {"value": "2024-01-01"}}
            )
    """
    from clinical_analytics.core.semantic import SemanticLayer
    
    def _make(
        dataset_name: str = "test_dataset",
        data: dict | pl.DataFrame | None = None,
        config_overrides: dict | None = None,
        workspace_name: str | None = None,
    ) -> SemanticLayer:
        workspace = tmp_path / (workspace_name or "workspace")
        workspace.mkdir()
        (workspace / "pyproject.toml").write_text("[project]\nname = 'test'")
        
        data_dir = workspace / "data" / "raw" / dataset_name
        data_dir.mkdir(parents=True)
        
        # Default data if not provided
        if data is None:
            data = {
                "patient_id": [1, 2, 3],
                "age": [45, 62, 38],
                "status": ["active", "inactive", "active"],
            }
        
        # Convert dict to DataFrame if needed
        if isinstance(data, dict):
            df = pl.DataFrame(data)
        else:
            df = data
        
        # Write CSV
        df.write_csv(data_dir / "test.csv")
        
        # Build config
        config = {
            "init_params": {"source_path": f"data/raw/{dataset_name}/test.csv"},
            "column_mapping": {"patient_id": "patient_id"},
            "time_zero": {"value": "2024-01-01"},
            "outcomes": {},
            "analysis": {"default_outcome": "outcome"},
        }
        if config_overrides:
            config.update(config_overrides)
        
        semantic = SemanticLayer(dataset_name, config=config, workspace_root=workspace)
        semantic.dataset_version = "test_v1"
        return semantic
    
    return _make
```



#### Step 1.2: Replace Duplicate Fixtures

Replace fixtures in these files (in order of complexity):**Simple replacements** (direct SemanticLayer instances):

1. `tests/core/test_queryplan_contract.py` (line 110) - Replace `mock_semantic_layer` fixture
2. `tests/core/test_queryplan_only_path.py` (line 250) - Replace `mock_semantic_layer` fixture
3. `tests/core/test_queryplan_conversion.py` (line 17) - Replace `mock_semantic_layer` fixture

**Medium complexity** (with custom data/config):

4. `tests/core/test_semantic_queryplan_execution.py` (line 77) - Replace `mock_semantic_layer_for_execution`
5. `tests/core/test_semantic_run_key_determinism.py` (lines 21, 210) - Replace class-level `semantic_layer` fixtures
6. `tests/core/test_semantic_observability.py` (lines 22, 161) - Replace class-level `semantic_layer` fixtures

**Complex replacements** (with special requirements):

7. `tests/e2e/test_ask_questions_full_flow.py` (line 27) - `mock_semantic_layer_with_statin`
8. Remaining 14 files with semantic layer fixtures

**Pattern for replacement**:

```python
# BEFORE
@pytest.fixture
def mock_semantic_layer(tmp_path):
    workspace = tmp_path / "workspace"
    # ... 30+ lines of setup ...
    return semantic

# AFTER
def test_example(make_semantic_layer):
    semantic = make_semantic_layer()
    # ... test code ...
```



#### Step 1.3: Update Test Functions

Update all test functions that use these fixtures to use the factory pattern instead.**Files to update**: 21 files total

### Phase 2: Extract Common DataFrame Patterns (MEDIUM PRIORITY)

**Goal**: Reduce 312 hardcoded DataFrames by extracting common patterns

#### Step 2.1: Add DataFrame Factory Fixtures to `conftest.py`

Add after existing DataFrame fixtures (around line 203):

```python
@pytest.fixture
def make_patients_df():
    """Factory for patient DataFrames with common columns."""
    def _make(
        patient_ids: list[str] | None = None,
        ages: list[int] | None = None,
        sexes: list[str] | None = None,
        num_patients: int = 3,
    ) -> pl.DataFrame:
        if patient_ids is None:
            patient_ids = [f"P{i:03d}" for i in range(1, num_patients + 1)]
        if ages is None:
            ages = [45, 62, 38][:num_patients]
        if sexes is None:
            sexes = ["M", "F", "M"][:num_patients]
        
        return pl.DataFrame({
            "patient_id": patient_ids,
            "age": ages,
            "sex": sexes,
        })
    return _make

@pytest.fixture
def make_cohort_with_categorical():
    """Factory for cohort DataFrames with categorical encoding."""
    def _make(
        patient_ids: list[str] | None = None,
        treatment: list[str] | None = None,
        status: list[str] | None = None,
        ages: list[int] | None = None,
    ) -> pl.DataFrame:
        if patient_ids is None:
            patient_ids = [f"P{i:03d}" for i in range(1, 6)]
        if treatment is None:
            treatment = ["1: Yes", "2: No", "1: Yes", "1: Yes", "2: No"]
        if status is None:
            status = ["1: Active", "2: Inactive", "1: Active", "1: Active", "2: Inactive"]
        if ages is None:
            ages = [45, 52, 38, 61, 49]
        
        return pl.DataFrame({
            "patient_id": patient_ids,
            "treatment": treatment,
            "status": status,
            "age": ages,
        })
    return _make

@pytest.fixture
def make_multi_table_setup():
    """Factory for multi-table test setups (patients, medications, bridge)."""
    def _make(
        num_patients: int = 3,
        num_medications: int = 3,
    ) -> dict[str, pl.DataFrame]:
        patients = pl.DataFrame({
            "patient_id": [f"P{i}" for i in range(1, num_patients + 1)],
            "name": ["Alice", "Bob", "Charlie"][:num_patients],
            "age": [30, 45, 28][:num_patients],
        })
        
        medications = pl.DataFrame({
            "medication_id": [f"M{i}" for i in range(1, num_medications + 1)],
            "drug_name": ["Aspirin", "Metformin", "Lisinopril"][:num_medications],
            "dosage": ["100mg", "500mg", "10mg"][:num_medications],
        })
        
        # Default bridge: P1->M1,M2; P2->M1; P3->M3
        patient_medications = pl.DataFrame({
            "patient_id": ["P1", "P1", "P2", "P3"][:min(4, num_patients * num_medications)],
            "medication_id": ["M1", "M2", "M1", "M3"][:min(4, num_patients * num_medications)],
            "start_date": ["2024-01-01", "2024-01-15", "2024-02-01", "2024-03-01"][:min(4, num_patients * num_medications)],
        })
        
        return {
            "patients": patients,
            "medications": medications,
            "patient_medications": patient_medications,
        }
    return _make
```



#### Step 2.2: Refactor High-Impact Files

Start with files that have the most hardcoded DataFrames:

1. **`tests/core/test_multi_table_handler.py`** (75 instances)

- Replace hardcoded `patients`, `medications`, `patient_medications` DataFrames
- Use `make_multi_table_setup()` factory

2. **`tests/analysis/test_compute.py`** (32 instances)

- Review which DataFrames are test-specific vs reusable
- Extract common patterns to fixtures

3. **`tests/core/test_relationship_detector.py`** (21 instances)

- Extract relationship test data patterns

4. **`tests/ui/test_schema_conversion.py`** (19 instances)

- Extract schema conversion test patterns

**Pattern for replacement**:

```python
# BEFORE
def test_example():
    patients = pl.DataFrame({
        "patient_id": ["P1", "P2", "P3"],
        "age": [30, 45, 28],
    })

# AFTER
def test_example(make_patients_df):
    patients = make_patients_df(
        patient_ids=["P1", "P2", "P3"],
        ages=[30, 45, 28]
    )
```



### Phase 3: Clean Up Duplicate Imports (OPTIONAL)

**Goal**: Remove redundant `import polars as pl` from test files

#### Step 3.1: Verify Import Availability

Confirm that `polars as pl` imported in `conftest.py` (line 11) is accessible to all test files.

#### Step 3.2: Remove Redundant Imports

Remove `import polars as pl` from 32 files, keeping only in `conftest.py`.**Note**: This is optional - explicit imports can be clearer. Only do this if team prefers.

### Phase 4: Update Existing Fixtures

#### Step 4.1: Consolidate Similar Fixtures

Review and potentially merge:

- `sample_cohort_with_categorical` and `sample_cohort_with_numeric` in `test_semantic_queryplan_execution.py`
- Consider if these should be factory fixtures instead

#### Step 4.2: Document Fixture Usage

Add docstring examples to factory fixtures showing common usage patterns.

## Testing Strategy

### After Each Phase

1. Run test suite: `make test-fast`
2. Verify no regressions
3. Check fixture usage: `grep -r "@pytest.fixture" tests/ | wc -l` (should decrease)
4. Verify DRY compliance: Check for duplicate fixture definitions

### Validation Commands

```bash
# Count duplicate fixtures (should decrease)
grep -r "def (mock_)?semantic_layer" tests/ --include="*.py" | grep -v conftest.py | wc -l

# Count hardcoded DataFrames (should decrease)
grep -r "pl\.DataFrame(" tests/ --include="*.py" | wc -l

# Run tests
make test-fast
make check
```



## Success Criteria

- [ ] Zero duplicate `mock_semantic_layer`/`semantic_layer` fixtures outside `conftest.py`
- [ ] Reduced hardcoded DataFrames by 50%+ (target: <150 instances)
- [ ] All tests pass: `make test-fast` and `make check`
- [ ] No regressions in test coverage
- [ ] Factory fixtures documented with usage examples

## Risk Mitigation

1. **Incremental approach**: One file at a time, verify tests pass
2. **Preserve test behavior**: Factory fixtures must produce identical results
3. **Backward compatibility**: Keep existing fixture names where possible
4. **Test isolation**: Ensure factory fixtures don't introduce shared state

## Estimated Effort

- **Phase 1**: 3-4 hours (21 files, ~30 min/file)
- **Phase 2**: 4-6 hours (4 high-impact files, ~1-1.5 hours/file)
- **Phase 3**: 1 hour (optional cleanup)
- **Phase 4**: 1 hour (documentation and consolidation)
- **Total**: 9-12 hours

## Files to Modify

### Primary Changes

- `tests/conftest.py` - Add factory fixtures
- 21 files with duplicate semantic layer fixtures