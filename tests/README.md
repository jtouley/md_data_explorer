# Test Organization

Tests are organized by function/module for better maintainability and clarity.

## Directory Structure

```
tests/
├── core/              # Core functionality tests
│   ├── test_mapper.py      # Column mapping and transformations
│   ├── test_registry.py    # Dataset registry and discovery
│   ├── test_profiling.py   # Data profiling and quality metrics
│   └── test_schema.py      # Unified cohort schema
│
├── loader/            # Dataset loader tests
│   ├── test_covid_ms_loader.py   # COVID-MS dataset loader
│   ├── test_sepsis_loader.py     # Sepsis dataset loader
│   └── test_mimic3_loader.py     # MIMIC-III dataset loader
│
├── analysis/          # Statistical analysis tests
│   ├── test_stats.py        # Logistic regression and statistical tests
│   └── test_survival.py     # Survival analysis (Kaplan-Meier, Cox regression)
│
├── ui/                # UI component tests
│   ├── test_variable_detector.py  # Variable type detection
│   └── test_user_datasets.py      # User dataset storage and security
│
├── conftest.py        # Pytest configuration and shared fixtures
└── README.md          # This file
```

## Running Tests

Run all tests:
```bash
pytest tests/
```

Run tests for a specific module:
```bash
pytest tests/core/          # Core module tests
pytest tests/loader/        # Loader tests
pytest tests/analysis/      # Analysis tests
pytest tests/ui/            # UI tests
```

Run a specific test file:
```bash
pytest tests/core/test_mapper.py
```

Run with coverage:
```bash
pytest tests/ --cov=src/clinical_analytics --cov-report=html
```

## Test Organization Principles

1. **By Function**: Tests are grouped by the functional area they test (core, loader, analysis, ui)
2. **Comprehensive Coverage**: Each module has tests for:
   - Basic functionality
   - Edge cases (nulls, empty data, invalid inputs)
   - Error handling
   - Integration scenarios
3. **Isolated Tests**: Each test is independent and can run in any order
4. **Fixtures**: Shared test fixtures are in `conftest.py`

## Legacy Tests

Some older test files remain in the root `tests/` directory:
- `test_registry.py` - Can be removed (moved to `tests/core/test_registry.py`)
- `test_ui.py` - UI integration tests
- `test_ui_integration.py` - UI integration tests
- `test_upload_security.py` - Security tests (functionality moved to `tests/ui/test_user_datasets.py`)
- `test_upload_security_manual.py` - Manual security tests
- `test_covid_ms_dataset.py` - Dataset-specific tests

These can be gradually migrated to the new structure or removed if redundant.

