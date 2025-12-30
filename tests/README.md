# Test Organization

Tests follow a **registry-based, generic approach** that avoids hardcoded dataset dependencies. All integration tests use dynamic dataset discovery to ensure tests work across all available datasets.

For detailed testing guidelines, see [tests/AGENTS.md](./AGENTS.md).

## Directory Structure

```
tests/
├── core/              # Core functionality tests (generic, config-driven)
│   ├── test_dataset_interface.py  # Generic dataset interface tests (parametrized)
│   ├── test_registry.py           # Dataset registry and discovery
│   ├── test_mapper.py              # Column mapping and transformations (config-driven)
│   ├── test_semantic_layer.py      # Semantic layer path resolution and granularity
│   ├── test_profiling.py           # Data profiling and quality metrics
│   └── test_schema.py              # Unified cohort schema
│
├── loader/            # Loader utility tests (shared utilities only)
│   └── test_zip_extraction.py     # ZIP file handling and multi-table extraction
│
├── analysis/          # Statistical analysis tests
│   ├── test_compute.py      # Statistical computations
│   └── test_survival.py     # Survival analysis (Kaplan-Meier, Cox regression)
│
├── ui/                # UI component tests
│   ├── components/    # UI component unit tests
│   │   ├── test_question_engine_*.py  # Question engine tests
│   ├── pages/         # Page-specific tests
│   │   └── test_ask_questions_*.py
│   ├── test_app.py    # Streamlit app integration tests
│   ├── test_integration.py  # UI integration tests (parametrized, uses sample datasets)
│   ├── test_upload_security.py  # Upload security validation
│   ├── test_user_datasets.py  # User dataset storage and security
│   ├── test_variable_detector.py  # Variable type detection
│   └── test_*.py      # Other UI component tests
│
├── conftest.py        # Pytest configuration and consolidated fixtures
├── AGENTS.md          # Comprehensive testing guidelines for AI agents
└── README.md          # This file
```

## Running Tests

> **CRITICAL**: Always use Makefile commands. Never run `pytest` directly.

Run all tests:
```bash
make test              # Run all tests
make test-fast         # Fast tests only (skip slow)
```

Run tests for a specific module:
```bash
make test-unit         # Unit tests only
make test-integration  # Integration tests only
```

Run with coverage:
```bash
make test-cov          # Tests with HTML coverage report
make test-cov-term     # Tests with terminal coverage
```

### Quality Gates (Run Before Commit)

```bash
make format        # Auto-format code
make lint-fix      # Auto-fix linting issues
make type-check    # Verify type hints
make check         # Full quality gate (format + lint + type + test)
```

> **Note**: See [.cursor/rules/000-project-setup-and-makefile.mdc](../.cursor/rules/000-project-setup-and-makefile.mdc) for complete Makefile reference.

## Test Organization Principles

### 1. **DRY (Don't Repeat Yourself)**
   - Single source of truth for fixtures in `conftest.py`
   - No duplicate fixture definitions across test files
   - Reusable helper functions for common patterns

### 2. **Registry-Based Testing**
   - Tests use `DatasetRegistry.discover_datasets()` for dynamic discovery
   - Parametrized tests use **sample datasets** (1-2) for fast unit testing
   - **Full integration tests** across all datasets are marked `@pytest.mark.slow` and `@pytest.mark.integration`
   - No hardcoded dataset names in integration tests
   
   **Test Optimization Strategy:**
   - Fast tests (`make test-fast`): Use `get_sample_datasets()` - 1-2 representative datasets
   - Integration tests (`make test-integration`): Use `get_available_datasets()` - all datasets for critical schema/compliance tests
   - Data-loading tests are marked `@pytest.mark.slow` and `@pytest.mark.integration` to skip in fast runs

### 3. **AAA Pattern**
   - All tests follow Arrange-Act-Assert structure
   - Clear separation between setup, execution, and verification
   - Inline comments marking each section for clarity

### 4. **Test Naming Convention**
   - Pattern: `test_unit_scenario_expectedBehavior`
   - Examples:
     - `test_mapper_initialization_with_config`
     - `test_get_dataset_factory_creates_instance`
     - `test_cohort_retrieval_with_default_filters`

### 5. **Comprehensive Coverage**
   - Basic functionality tests
   - Edge cases (nulls, empty data, invalid inputs)
   - Error handling and validation
   - Integration scenarios across all datasets

### 6. **Test Isolation**
   - Each test is independent and can run in any order
   - Defensive skipping with `pytest.skip()` for unavailable data
   - Tests use temporary files and test data (no dependencies on actual dataset files)

## Generic Testing Approach

### Integration Tests
Integration tests (e.g., `test_ui_integration.py`, `test_dataset_interface.py`) are parametrized across all available datasets:

```python
def get_available_datasets():
    """Helper to discover all available datasets from registry."""
    DatasetRegistry.reset()
    DatasetRegistry.discover_datasets()
    DatasetRegistry.load_config()
    return [
        name for name in DatasetRegistry.list_datasets()
        if name not in ["uploaded", "mimic3"]  # Skip special cases
    ]

@pytest.mark.parametrize("dataset_name", get_available_datasets())
def test_cohort_retrieval_with_default_filters(dataset_name):
    """Test dataset cohort retrieval with default filters."""
    dataset = DatasetRegistry.get_dataset(dataset_name)
    if not dataset.validate():
        pytest.skip(f"{dataset_name} data not available")

    cohort = dataset.get_cohort()
    assert isinstance(cohort, pd.DataFrame)
```

### Unit Tests
Unit tests (e.g., loader tests, mapper tests) use test data and temporary files:

```python
def test_load_raw_data(tmp_path):
    """Test loading raw CSV data."""
    # Arrange: Create test CSV file
    test_file = tmp_path / "test_data.csv"
    test_data = pl.DataFrame({"patient_id": ["P001", "P002"], "age": [45, 62]})
    test_data.write_csv(test_file)

    # Act: Load the data
    df = load_raw_data(test_file)

    # Assert: Verify correct loading
    assert isinstance(df, pl.DataFrame)
    assert len(df) == 2
```

## Detailed Guidelines

For comprehensive testing guidelines, including:
- Quick reference for common patterns
- Fixture discipline and scoping
- Error messages and assertions
- Polars-specific testing patterns
- Anti-patterns to avoid

See **[tests/AGENTS.md](./AGENTS.md)** for the complete guide.
