# Agent Development Guidelines

## Test Data and Fixtures

**CRITICAL: Always use `conftest.py` fixtures for test data. Never create duplicate test data in individual test files.**

### Excel Test Data

Synthetic Excel files for testing are available via fixtures in `tests/conftest.py`:

- `synthetic_dexa_excel_file()` - DEXA-like file with headers in row 0 (standard format)
- `synthetic_statin_excel_file()` - Statin-like file with empty row 1, headers in row 2
- `synthetic_complex_excel_file()` - Complex file with metadata rows before headers

**Usage:**
```python
def test_excel_reading(synthetic_dexa_excel_file):
    file_bytes = synthetic_dexa_excel_file.read_bytes()
    # Use file_bytes for testing
```

### DRY Principle for Tests

**Reference: [102-dry-principles.mdc](.cursor/rules/102-dry-principles.mdc) and [101-testing-hygiene.mdc](.cursor/rules/101-testing-hygiene.mdc)**

1. **Use shared fixtures from `conftest.py`** - All reusable test data should be in `conftest.py`
2. **Never duplicate test data** - If you need test data, check `conftest.py` first
3. **Create module-level fixtures** - For test data specific to a module, add to that module's `conftest.py`
4. **Never duplicate imports** - Extract common imports to `conftest.py` if used in multiple test files

### Test Execution Workflow

**Reference: [104-plan-execution-hygiene.mdc](.cursor/rules/104-plan-execution-hygiene.mdc)**

1. **Write test first** (Red-Green-Refactor)
2. **Run test immediately** - Use module-specific commands for faster feedback:
   - `make test-core` - When working on core module
   - `make test-analysis` - When working on analysis module
   - `make test-ui` - When working on UI module
   - `make test-fast` - For quick feedback across all modules
3. **Fix code quality** - `make lint-fix` and `make format` after writing code
4. **Run full quality gate** - `make check` before committing
5. **Never commit without tests** - All new code must have tests

### Quality Gates (MANDATORY)

Before every commit:
```bash
make format        # Auto-format code
make lint-fix      # Auto-fix linting issues
make type-check    # Verify type hints
make test-fast     # Run fast tests (or use module-specific: test-core, test-analysis, test-ui, etc.)
make check         # Full quality gate (recommended)
```

### Makefile Usage (MANDATORY)

**NEVER run tools directly. Always use Makefile commands:**
- `make format` (not `ruff format`)
- `make lint-fix` (not `ruff check --fix`)
- `make test-fast` (not `pytest` directly)
- Module-specific tests: `make test-core`, `make test-analysis`, `make test-ui`, `make test-datasets`, `make test-loader`, `make test-e2e`
- `make check` (full quality gate)

**Reference: [000-project-setup-and-makefile.mdc](.cursor/rules/000-project-setup-and-makefile.mdc)**

