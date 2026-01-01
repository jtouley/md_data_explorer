# Test Performance Documentation

## Overview

This document tracks test performance, identifies slow tests (>30 seconds), and documents performance optimizations.

## Performance Optimizations

### Session-Scoped Dataset Discovery Fixture (2026-01-01)

**Problem**: Tests were calling `DatasetRegistry.discover_datasets()` and `load_dataset_config()` on every test, causing:
- Expensive module imports repeated for each test
- YAML config parsing repeated for each test
- Registry resets between tests causing redundant discovery

**Solution**: Added `discovered_datasets` session-scoped fixture in `tests/conftest.py`:
- Discovers datasets once per test session
- Pre-loads all configs to avoid repeated YAML parsing
- Caches results for all tests in the session

**Impact**:
- Mapper tests: Reduced from ~4+ minutes to ~30 seconds per test (when datasets available)
- Registry tests: Similar performance improvement
- Eliminates redundant module imports

**Usage**:
```python
def test_example(discovered_datasets):
    config = get_first_available_dataset_config(discovered_datasets)
    # Use config...
```

## Slow Tests (>30 seconds)

Tests marked with `@pytest.mark.slow` and `@pytest.mark.integration` are expected to take longer because they:
- Load real dataset data
- Make LLM calls (Ollama)
- Perform full integration workflows

### Documented Slow Tests

#### 1. LLM Query Parsing Tests
**Location**: `tests/core/test_nl_query_refinement.py`, `tests/core/test_nl_query_engine_filter_extraction.py`

**Duration**: 10-30 seconds per test (depends on LLM response time)

**Reason**: These tests make real LLM calls to Ollama to test natural language query parsing. Each test:
- Sends query to LLM
- Waits for response (5-15 seconds typical)
- Validates parsing results

**Mitigation**:
- Tests are marked `@pytest.mark.slow` to skip in fast runs
- Use `make test-fast` to skip these tests
- LLM calls are necessary to validate actual parsing behavior

**Documentation**: These tests verify that the LLM correctly:
- Recognizes refinement patterns ("remove the n/a", "exclude missing")
- Inherits previous query context
- Extracts filters from natural language

#### 2. Dataset Integration Tests
**Location**: `tests/core/test_mapper.py`, `tests/core/test_registry.py`, `tests/core/test_dataset_interface.py`

**Duration**: 5-60 seconds per test (depends on dataset size and availability)

**Reason**: These tests:
- Discover and load real dataset configs
- Instantiate dataset classes
- Load actual data files (if available)
- Validate schema compliance

**Mitigation**:
- Session-scoped fixture caches discovery (see above)
- Tests skip if data not available (`pytest.skip()`)
- Use `get_sample_datasets()` for fast tests (1-2 datasets)
- Use `get_available_datasets()` only for critical schema tests

**Documentation**: These tests verify:
- Dataset registry discovery works correctly
- Config loading and validation
- Schema compliance with UnifiedCohort
- Mapper transformations work with real configs

#### 3. End-to-End Integration Tests
**Location**: `tests/integration/test_adr002_end_to_end.py`, `tests/ui/test_integration.py`

**Duration**: 30-120 seconds per test

**Reason**: These tests perform complete workflows:
- Upload data → Store in DuckDB → Export to Parquet
- Restart application → Restore state → Query data
- Full UI workflows with dataset selection and cohort retrieval

**Mitigation**:
- Marked `@pytest.mark.slow` and `@pytest.mark.integration`
- Use temporary directories for isolation
- Skip if dependencies not available

**Documentation**: These tests verify:
- Complete data persistence (ADR002)
- State restoration after restart
- UI workflow correctness
- End-to-end system behavior

#### 4. Golden Questions Evaluation
**Location**: `tests/eval/test_golden_questions.py`

**Duration**: 60-300 seconds (depends on number of questions)

**Reason**: Evaluates NL query parsing against golden question set:
- Parses multiple queries with LLM
- Compares results to expected outputs
- Measures accuracy metrics

**Mitigation**:
- Only run in CI or manual evaluation
- Can be run with subset of questions
- Results logged for analysis

**Documentation**: This test suite:
- Validates NL query parsing accuracy
- Tracks improvements over time
- Provides metrics for regression detection

## Performance Benchmarks

### Before Optimization (2026-01-01)
- `test_mapper_initialization_with_config`: ~4+ minutes (with dataset discovery per test)
- `test_get_dataset_factory_creates_instance`: ~4+ minutes
- Full core test suite: ~12+ minutes

### After Optimization (2026-01-01)
- `test_mapper_initialization_with_config`: ~30 seconds (with cached discovery)
- `test_get_dataset_factory_creates_instance`: ~30 seconds
- Full core test suite: ~4-5 minutes (with LLM tests)

**Note**: Actual times depend on:
- Dataset availability (tests skip if no datasets)
- LLM response times (variable, 5-15 seconds per call)
- System performance

## Running Tests

### Fast Tests (Skip Slow)
```bash
make test-fast  # Skips @pytest.mark.slow tests
```

### Core Module Tests
```bash
make test-core  # All core tests (includes slow)
```

### Integration Tests Only
```bash
make test-integration  # Only integration tests
```

### Specific Test
```bash
make test-core PYTEST_ARGS="tests/core/test_mapper.py::TestColumnMapper::test_mapper_initialization_with_config -xvs"
```

## Future Optimizations

1. **Mock LLM Responses**: Consider mocking LLM calls for unit tests, keeping real calls only for integration tests
2. **Parallel Test Execution**: Use `pytest-xdist` for parallel test execution
3. **Test Data Caching**: Cache test data files to avoid repeated generation
4. **Selective Dataset Loading**: Only load datasets needed for specific tests

## Notes

- Tests that take >30 seconds should be documented here
- All slow tests must be marked with `@pytest.mark.slow`
- Integration tests must be marked with `@pytest.mark.integration`
- Use `make test-fast` for quick feedback during development

