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

### 7. Create Testing Guidelines Documentation

**File:** `tests/AGENTS.md` (NEW)Create comprehensive documentation for AI agents and developers on maintaining DRY test code. The file should reference and align with all project rules:**References:**

- `.cursor/rules/101-testing-hygiene.mdc` - Testing patterns and standards
- `.cursor/rules/102-dry-principles.mdc` - DRY principles and code organization
- `.cursor/rules/104-plan-execution-hygiene.mdc` - Test-first workflow and quality gates
- `.cursor/rules/103-staff-engineer-standards.mdc` - Production-grade patterns

The file should include:

1. **Test Structure: AAA Pattern** (from 101-testing-hygiene.mdc)

- Arrange-Act-Assert with clear separation
- Code examples showing proper structure
- Why this pattern matters for maintainability

2. **Test Naming Convention** (from 101-testing-hygiene.mdc, 104-plan-execution-hygiene.mdc)

- Pattern: `test_unit_scenario_expectedBehavior`
- Examples of correct vs incorrect naming
- Descriptive names that explain what is being tested

3. **DRY Principles for Tests** (from 102-dry-principles.mdc, 104-plan-execution-hygiene.mdc)

- Single source of truth: All shared test data and fixtures in `conftest.py`
- Never duplicate fixture definitions across test files
- Extract common patterns to reusable fixtures
- Use factory fixtures for variations of similar data
- Never duplicate imports - extract to `conftest.py` if repeated

4. **Fixture Discipline** (from 101-testing-hygiene.mdc, 104-plan-execution-hygiene.mdc)

- **Before creating a new fixture**: Always check `conftest.py` first
- **Fixture scoping**:
    - Session scope: Expensive, immutable resources (e.g., database connections)
    - Module scope: Shared across tests in one file (e.g., reference data)
    - Function scope (default): Fresh per test, use for mutable state
- **Factory fixtures**: Use factory pattern for creating variations (e.g., `make_transaction()`)
- **Parametrized fixtures**: Use `@pytest.fixture(params=[...])` for multiple test cases
- **Fixture files location**: Document where fixtures belong (root `conftest.py` vs module-level)

5. **Test Isolation** (from 101-testing-hygiene.mdc, 104-plan-execution-hygiene.mdc)

- No shared mutable state between tests
- Each test must be independent
- Use fixtures for isolated test data
- Database isolation patterns (if applicable)

6. **Parameterization** (from 101-testing-hygiene.mdc)

- Use `@pytest.mark.parametrize` for variations
- Use `pytest.param` with IDs for readable test output
- Examples of proper parametrization

7. **Error Testing** (from 101-testing-hygiene.mdc)

- Use `pytest.raises` with specific exception types
- Test error messages for clarity
- Examples of proper error testing patterns

8. **Data Engineering Specific Patterns** (from 101-testing-hygiene.mdc)

- Schema contract tests
- Idempotency tests
- Null handling tests
- Examples for each pattern

9. **Polars Testing Assertions** (from 104-plan-execution-hygiene.mdc)

- Use `polars.testing.assert_frame_equal` for DataFrame comparisons
- Never use pandas assertions for Polars DataFrames
- Examples of proper Polars test assertions

10. **Common Anti-Patterns to Avoid**

    - Creating duplicate `sample_cohort` fixtures in multiple files
    - Repeating mock setup code across tests
    - Hardcoding test data in test functions instead of fixtures
    - Copy-pasting fixture definitions between test files
    - Duplicate imports across test files
    - Shared mutable state between tests

11. **Standard Fixtures Reference**

    - Document all available fixtures in `conftest.py`
    - Provide usage examples for each fixture
    - Explain when to use each fixture type
    - Document fixture dependencies

12. **Makefile Usage** (from 101-testing-hygiene.mdc, 104-plan-execution-hygiene.mdc)

    - Always use `make test` / `make test-fast` (never run pytest directly)
    - Use `make test-cov` for coverage reports
    - Use `make test-unit` / `make test-integration` for specific test types
    - Never run `pytest` or `uv run pytest` directly

13. **Test Writing Checklist**

    - [ ] Checked `conftest.py` for existing fixtures
    - [ ] Used parametrization instead of duplicate test functions
    - [ ] Extracted repeated setup to fixtures
    - [ ] Documented fixture purpose and scope
    - [ ] Used appropriate fixture scope (session/module/function)
    - [ ] Test follows AAA pattern (Arrange-Act-Assert)
    - [ ] Test name follows `test_unit_scenario_expectedBehavior` pattern
    - [ ] Test is isolated (no shared mutable state)
    - [ ] Used Polars testing assertions (not pandas)
    - [ ] Used Makefile commands for running tests

14. **Examples**

    - Show before/after examples of refactoring duplicate code
    - Demonstrate proper fixture composition
    - Show how to use factory fixtures for variations
    - Show proper AAA pattern usage
    - Show proper parametrization examples
    - Show proper error testing examples

**Important Notes:**

- AGENTS.md should cross-reference the actual rule files for detailed information
- AGENTS.md serves as a quick reference guide, but developers should consult the full rule files for comprehensive guidance
- All examples in AGENTS.md should align with patterns shown in the rule files
- AGENTS.md should emphasize the "single source of truth" principle - if it's in `conftest.py`, use it; don't duplicate

### 8. Consolidate Duplicate Fixtures Across Test Files

**Files to Audit and Refactor:**

- `tests/test_ui.py` - Has `mock_cohort` fixture (should use centralized fixture)
- `tests/unit/ui/pages/test_ask_questions_low_confidence.py` - Has `sample_cohort` fixture
- `tests/unit/ui/pages/test_ask_questions_idempotency.py` - Has `sample_cohort` and `sample_context` fixtures
- `tests/unit/ui/pages/test_ask_questions_run_key.py` - Has `sample_context` fixture
- `tests/core/test_nl_query_engine_variable_extraction.py` - Has `mock_semantic_layer` fixture
- `tests/core/test_nl_query_engine_pattern_fallback.py` - Has `mock_semantic_layer` fixture
- `tests/core/test_nl_query_engine_error_messages.py` - May have duplicate fixtures
- `tests/core/test_nl_query_engine_diagnostics.py` - May have duplicate fixtures
- `tests/core/test_clarifying_questions.py` - May have duplicate fixtures
- `tests/ui/components/test_question_engine_*.py` - Multiple files with potential duplicate fixtures
- `tests/analysis/test_compute.py` - Has `sample_numeric_df`, `sample_categorical_df`, `sample_mixed_df` fixtures

**Actions:**

1. Audit all test files for duplicate fixture definitions
2. Identify common patterns: `sample_cohort`, `mock_cohort`, `sample_context`, `mock_semantic_layer`, `sample_numeric_df`, etc.
3. Move common fixtures to `conftest.py` with appropriate scoping
4. Create factory fixtures for variations (e.g., `make_cohort()`, `make_context()`)
5. Update all test files to use centralized fixtures
6. Remove duplicate fixture definitions from individual test files

### 9. Review Additional Core Tests

**Files:**

- `tests/core/test_mapper.py` - Verify alignment with semantic layer approach
- Check for hardcoded dataset configs
- Ensure tests use registry discovery
- Verify config-driven behavior testing
- `tests/core/test_semantic_layer.py` - Verify alignment with semantic layer architecture
- Ensure tests verify interface, not implementations
- Check for any dataset-specific assumptions
- Verify config-driven SQL generation is tested

**Actions:**

1. Review both files for hardcoded dataset references
2. Ensure tests use generic patterns and registry discovery
3. Refactor if needed to align with semantic layer approach

### 10. Review Loader Test

**File:** `tests/loader/test_zip_extraction.py`

- Review for dataset-specific code
- Ensure it tests generic zip extraction behavior
- Verify it uses fixtures from `conftest.py`
- Refactor if needed to align with generic approach

### 11. Update Test Documentation

**File:** `tests/README.md`

- Remove dataset-specific references (covid_ms, sepsis, mimic3)
- Update to reflect generic, registry-based approach
- Document fixture usage from `conftest.py`
- Add reference to `AGENTS.md` for testing guidelines
- Update examples to use registry discovery

### 12. Delete Legacy Test File

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
- ✅ AGENTS.md provides comprehensive guidance aligned with all project rules:
- Testing hygiene standards (AAA pattern, fixture discipline, test isolation)
- DRY principles (single source of truth, no duplication)
- Plan execution hygiene (test-first workflow, Makefile usage)
- Staff engineer standards (where applicable to testing)
- ✅ AGENTS.md cross-references rule files for detailed information
- ✅ Common fixtures are centralized in `conftest.py`
- ✅ No duplicate fixture definitions across test files
- ✅ All test examples in AGENTS.md follow project standards
- ✅ Duplicate fixtures consolidated (sample_cohort, mock_cohort, mock_semantic_layer, etc.)
- ✅ All test files use centralized fixtures from `conftest.py`
- ✅ `tests/README.md` updated to reflect generic approach
- ✅ Additional core tests reviewed and aligned with semantic layer architecture

## Files Modified

1. `tests/core/test_dataset_interface.py` (NEW)
2. `tests/test_ui_integration.py` (MODIFY)
3. `tests/test_registry.py` (MODIFY)
4. `tests/core/test_registry.py` (MODIFY)
5. `tests/test_mapper.py` (MODIFY)
6. `tests/core/test_mapper.py` (REVIEW/MODIFY)
7. `tests/core/test_semantic_layer.py` (REVIEW/MODIFY)
8. `tests/loader/test_zip_extraction.py` (REVIEW/MODIFY)
9. `tests/conftest.py` (MODIFY - consolidate fixtures)