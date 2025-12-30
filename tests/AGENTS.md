# Testing Guidelines for MD Data Explorer

**Audience**: AI agents and developers maintaining test code

**Purpose**: Ensure DRY, maintainable test code aligned with project standards

---

## Quick Reference

**Key Principle**: Single source of truth - if it exists in `conftest.py`, use it; don't duplicate.

**Before creating ANY new fixture**: Check `conftest.py` first!

**Before running tests**: Always use `make test` or `make test-fast`, never `pytest` directly.

---

## Table of Contents

1. [Test Structure: AAA Pattern](#test-structure-aaa-pattern)
2. [Test Naming Convention](#test-naming-convention)
3. [DRY Principles for Tests](#dry-principles-for-tests)
4. [Fixture Discipline](#fixture-discipline)
5. [Test Isolation](#test-isolation)
6. [Parameterization](#parameterization)
7. [Error Testing](#error-testing)
8. [Data Engineering Specific Patterns](#data-engineering-specific-patterns)
9. [Polars Testing Assertions](#polars-testing-assertions)
10. [Common Anti-Patterns to Avoid](#common-anti-patterns-to-avoid)
11. [Standard Fixtures Reference](#standard-fixtures-reference)
12. [Makefile Usage](#makefile-usage)
13. [Test Writing Checklist](#test-writing-checklist)
14. [Examples](#examples)

---

## Test Structure: AAA Pattern

> **Reference**: [.cursor/rules/101-testing-hygiene.mdc](../.cursor/rules/101-testing-hygiene.mdc)

Every test follows **Arrange-Act-Assert** with clear separation:

```python
def test_customer_aggregation_sums_transactions():
    # Arrange: Set up test data and dependencies
    transactions = pl.DataFrame({
        "customer_id": ["A", "A", "B"],
        "amount": [100, 200, 50],
    })

    # Act: Execute the unit under test
    result = aggregate_by_customer(transactions)

    # Assert: Verify expected outcomes
    assert result.filter(pl.col("customer_id") == "A")["total"][0] == 300
    assert result.filter(pl.col("customer_id") == "B")["total"][0] == 50
```

**Why**: Separating setup, execution, and verification makes tests readable and maintainable.

---

## Test Naming Convention

> **Reference**: [.cursor/rules/101-testing-hygiene.mdc](../.cursor/rules/101-testing-hygiene.mdc), [.cursor/rules/104-plan-execution-hygiene.mdc](../.cursor/rules/104-plan-execution-hygiene.mdc)

**Pattern**: `test_unit_scenario_expectedBehavior`

### Correct Examples

```python
def test_deduplication_with_null_keys_preserves_first_occurrence():
    """Test deduplication when keys contain null values."""
    ...

def test_schema_validation_missing_required_column_raises_valueerror():
    """Test schema validation fails when required column is missing."""
    ...

def test_incremental_load_overlapping_dates_merges_correctly():
    """Test incremental load merges data when dates overlap."""
    ...
```

### Incorrect Examples

```python
# WRONG: Vague names
def test_dedup():
def test_validation():
def test_load():
```

**Why**: Descriptive names explain what is being tested without reading the code.

---

## DRY Principles for Tests

> **Reference**: [.cursor/rules/102-dry-principles.mdc](../.cursor/rules/102-dry-principles.mdc), [.cursor/rules/104-plan-execution-hygiene.mdc](../.cursor/rules/104-plan-execution-hygiene.mdc)

### Single Source of Truth

- **All shared test data and fixtures belong in `conftest.py`**
- **Never duplicate fixture definitions across test files**
- **Extract common patterns to reusable fixtures**
- **Use factory fixtures for variations of similar data**
- **Never duplicate imports** - extract to `conftest.py` if repeated

### Example: DRY Fixture Organization

```python
# conftest.py - Single source of truth
@pytest.fixture
def sample_cohort():
    """Standard cohort fixture used across all tests."""
    return pl.DataFrame({
        "patient_id": [1, 2, 3, 4, 5],
        "outcome": [0, 1, 0, 1, 0],
        "age": [45, 62, 38, 71, 55],
    })

@pytest.fixture
def make_cohort():
    """Factory fixture for creating custom cohorts."""
    def _make(num_patients=5, outcome_rate=0.5):
        return pl.DataFrame({
            "patient_id": list(range(1, num_patients + 1)),
            "outcome": [1 if i < num_patients * outcome_rate else 0 for i in range(num_patients)],
        })
    return _make
```

**Then use in tests**:

```python
# test_analysis.py - USE shared fixtures
def test_analysis_with_standard_cohort(sample_cohort):
    """Test using standard cohort fixture."""
    result = analyze(sample_cohort)
    assert result is not None

def test_analysis_with_custom_cohort(make_cohort):
    """Test using factory fixture for variation."""
    cohort = make_cohort(num_patients=100, outcome_rate=0.8)
    result = analyze(cohort)
    assert len(result) == 100
```

### NEVER Do This

```python
# test_analysis.py - WRONG: Duplicate fixture
@pytest.fixture
def sample_cohort():  # Already exists in conftest.py!
    return pl.DataFrame({...})
```

---

## Fixture Discipline

> **Reference**: [.cursor/rules/101-testing-hygiene.mdc](../.cursor/rules/101-testing-hygiene.mdc), [.cursor/rules/104-plan-execution-hygiene.mdc](../.cursor/rules/104-plan-execution-hygiene.mdc)

### Before Creating a New Fixture

**CRITICAL**: Always check `conftest.py` first!

1. Open `tests/conftest.py`
2. Search for similar fixtures (e.g., `sample_cohort`, `sample_patients_df`, `mock_semantic_layer`)
3. If fixture exists, use it
4. If fixture almost matches, use factory pattern or parametrization
5. Only create new fixture if truly unique

### Fixture Scoping

```python
# Session scope: Expensive, immutable resources (e.g., database connections)
@pytest.fixture(scope="session")
def spark_session():
    # Only created once per test run
    ...

# Module scope: Shared across tests in one file (e.g., reference data)
@pytest.fixture(scope="module")
def reference_data():
    return pl.read_parquet("tests/fixtures/reference.parquet")

# Function scope (default): Fresh per test, use for mutable state
@pytest.fixture
def empty_staging_table(test_database):
    test_database.execute("TRUNCATE staging.events")
    yield
    test_database.execute("TRUNCATE staging.events")
```

### Factory Fixtures

Use factory pattern for creating variations:

```python
@pytest.fixture
def make_transaction():
    """Factory fixture for creating test transactions."""
    def _make(
        customer_id: str = "CUST001",
        amount: float = 100.0,
        status: str = "completed",
        timestamp: datetime | None = None,
    ) -> dict:
        return {
            "customer_id": customer_id,
            "amount": amount,
            "status": status,
            "timestamp": timestamp or datetime.now(),
        }
    return _make

def test_refund_calculation(make_transaction):
    txn = make_transaction(amount=500.0, status="refunded")
    result = calculate_refund(txn)
    assert result == -500.0
```

### Parametrized Fixtures

```python
@pytest.fixture(params=["csv", "parquet", "json"])
def file_format(request):
    """Parametrized fixture for testing multiple file formats."""
    return request.param

def test_file_reader_supports_format(file_format):
    """Test runs three times: csv, parquet, json."""
    reader = FileReader(format=file_format)
    assert reader.format == file_format
```

### Fixture Files Location

```
tests/
├── conftest.py           # Shared fixtures across ALL tests
├── fixtures/             # Static test data files
│   ├── customers.parquet
│   └── transactions.parquet
├── unit/
│   ├── conftest.py       # Unit-specific fixtures
│   └── test_transforms.py
└── integration/
    ├── conftest.py       # Integration-specific fixtures
    └── test_pipeline.py
```

---

## Test Isolation

> **Reference**: [.cursor/rules/101-testing-hygiene.mdc](../.cursor/rules/101-testing-hygiene.mdc), [.cursor/rules/104-plan-execution-hygiene.mdc](../.cursor/rules/104-plan-execution-hygiene.mdc)

### No Shared Mutable State

**WRONG**:

```python
# Tests depend on execution order - BAD
class TestBadIsolation:
    results = []  # Shared state!

    def test_first(self):
        self.results.append(1)

    def test_second(self):
        assert len(self.results) == 1  # Fails if run alone
```

**CORRECT**:

```python
# Each test is independent - GOOD
def test_first(tmp_path):
    output = tmp_path / "results.json"
    process_and_save(output)
    assert output.exists()
```

### Each Test Must Be Independent

- Tests should run in any order
- Tests should not depend on other tests
- Use fixtures for isolated test data
- Clean up resources properly (use `yield` in fixtures)

---

## Parameterization

> **Reference**: [.cursor/rules/101-testing-hygiene.mdc](../.cursor/rules/101-testing-hygiene.mdc)

### Use pytest.mark.parametrize for Variations

```python
@pytest.mark.parametrize("input_status,expected_output", [
    ("active", True),
    ("inactive", False),
    ("pending", False),
    ("ACTIVE", True),  # Case insensitivity
    (None, False),     # Null handling
    ("", False),       # Empty string
])
def test_is_active_customer(input_status, expected_output):
    result = is_active_customer(input_status)
    assert result == expected_output
```

### IDs for Readable Output

```python
@pytest.mark.parametrize("date_str,expected", [
    pytest.param("2024-01-15", date(2024, 1, 15), id="iso_format"),
    pytest.param("01/15/2024", date(2024, 1, 15), id="us_format"),
    pytest.param("15-Jan-2024", date(2024, 1, 15), id="abbrev_month"),
])
def test_parse_date(date_str, expected):
    assert parse_date(date_str) == expected
```

**Output**:
```
test_parse_date[iso_format] PASSED
test_parse_date[us_format] PASSED
test_parse_date[abbrev_month] PASSED
```

---

## Error Testing

> **Reference**: [.cursor/rules/101-testing-hygiene.mdc](../.cursor/rules/101-testing-hygiene.mdc)

### Use pytest.raises with Specific Exception Types

```python
def test_validation_rejects_negative_amounts():
    invalid_df = pl.DataFrame({"amount": [-100, 50, -25]})

    with pytest.raises(ValueError, match="Negative amounts not allowed"):
        validate_transactions(invalid_df)

def test_missing_required_column_error_message():
    incomplete_df = pl.DataFrame({"id": [1, 2]})  # Missing 'amount'

    with pytest.raises(ValueError) as exc_info:
        validate_schema(incomplete_df)

    assert "amount" in str(exc_info.value)
    assert "required" in str(exc_info.value).lower()
```

---

## Data Engineering Specific Patterns

> **Reference**: [.cursor/rules/101-testing-hygiene.mdc](../.cursor/rules/101-testing-hygiene.mdc)

### Schema Contract Tests

```python
def test_output_schema_matches_contract():
    """Ensure transform output matches downstream expectations."""
    result = transform_pipeline(sample_input)

    expected_schema = {
        "customer_id": pl.Utf8,
        "total_amount": pl.Decimal,
        "last_transaction_date": pl.Date,
    }

    for col, dtype in expected_schema.items():
        assert col in result.columns, f"Missing column: {col}"
        assert result.schema[col] == dtype, f"Type mismatch for {col}"
```

### Idempotency Tests

```python
def test_pipeline_is_idempotent(test_database):
    """Running pipeline twice produces same result."""
    run_pipeline(test_database)
    first_result = test_database.read_table("output")

    run_pipeline(test_database)
    second_result = test_database.read_table("output")

    pl.testing.assert_frame_equal(first_result, second_result)
```

### Null Handling Tests

```python
@pytest.fixture
def df_with_nulls():
    return pl.DataFrame({
        "id": [1, 2, 3, 4],
        "value": [100, None, 300, None],
        "category": ["A", None, "A", "B"],
    })

def test_aggregation_handles_null_values(df_with_nulls):
    result = aggregate_by_category(df_with_nulls)
    # Verify nulls don't cause errors and are handled as expected
    assert result.filter(pl.col("category") == "A")["sum"][0] == 400
```

---

## Polars Testing Assertions

> **Reference**: [.cursor/rules/100-polars-first.mdc](../.cursor/rules/100-polars-first.mdc), [.cursor/rules/104-plan-execution-hygiene.mdc](../.cursor/rules/104-plan-execution-hygiene.mdc)

### Use Polars-Native Assertions

```python
import polars.testing as plt

# CORRECT: Use Polars testing assertions
plt.assert_frame_equal(result, expected)

# WRONG: Never use pandas assertions for Polars DataFrames
# pd.testing.assert_frame_equal(result, expected)  # NO!
```

### Polars Attribute Usage

```python
# CORRECT: Use Polars attributes
df.height  # NOT len(df)
df.width   # NOT len(df.columns)
df.schema  # NOT df.dtypes
df.to_dicts()  # NOT df.to_dict(orient="records")
```

---

## Common Anti-Patterns to Avoid

### 1. Duplicate Fixtures

**WRONG**:

```python
# tests/test_analysis.py
@pytest.fixture
def sample_cohort():  # Duplicate!
    return pl.DataFrame({...})

# tests/test_compute.py
@pytest.fixture
def sample_cohort():  # Duplicate!
    return pl.DataFrame({...})
```

**CORRECT**:

```python
# tests/conftest.py - Single source of truth
@pytest.fixture
def sample_cohort():
    return pl.DataFrame({...})

# tests/test_analysis.py - Use shared fixture
def test_analysis(sample_cohort):
    ...
```

### 2. Duplicate Imports

**WRONG**:

```python
# tests/test_analysis.py
import polars as pl
from clinical_analytics.core.schema import UnifiedCohort

# tests/test_compute.py
import polars as pl
from clinical_analytics.core.schema import UnifiedCohort

# Repeated across 10+ files...
```

**CORRECT**:

```python
# tests/conftest.py - Import once at top level
import polars as pl
from clinical_analytics.core.schema import UnifiedCohort

# Import is available to all test files automatically
```

### 3. Hardcoding Test Data in Functions

**WRONG**:

```python
def test_analysis():
    # Hardcoded data in test function
    df = pl.DataFrame({"id": [1, 2, 3], "value": [10, 20, 30]})
    result = analyze(df)
    assert result is not None
```

**CORRECT**:

```python
# conftest.py
@pytest.fixture
def sample_data():
    return pl.DataFrame({"id": [1, 2, 3], "value": [10, 20, 30]})

# test_analysis.py
def test_analysis(sample_data):
    result = analyze(sample_data)
    assert result is not None
```

### 4. Shared Mutable State

**WRONG**:

```python
GLOBAL_STATE = []

def test_first():
    GLOBAL_STATE.append(1)  # Affects other tests!

def test_second():
    assert len(GLOBAL_STATE) == 0  # Fails if test_first ran first
```

### 5. Copy-Pasting Mock Setup Code

**WRONG**:

```python
# tests/test_a.py
@patch("streamlit.session_state", {"key": "value"})
def test_a():
    ...

# tests/test_b.py
@patch("streamlit.session_state", {"key": "value"})
def test_b():
    ...
```

**CORRECT**:

```python
# conftest.py
@pytest.fixture
def mock_session_state():
    with patch("streamlit.session_state", {"key": "value"}):
        yield

# test_a.py and test_b.py
def test_a(mock_session_state):
    ...
```

---

## Standard Fixtures Reference

### Available Fixtures in `conftest.py`

See `tests/conftest.py` for complete list. Key fixtures:

#### Project Structure

- `project_root` (session): Project root directory
- `test_data_dir` (session): Test data directory

#### Dataset Fixtures

- `temp_config_file`: Temporary config file for testing

#### DataFrame Fixtures

- `sample_patients_df`: Sample patients DataFrame
- `sample_admissions_df`: Sample admissions DataFrame
- `sample_upload_df`: Sample upload DataFrame
- `sample_upload_metadata`: Sample upload metadata

#### Large Data Fixtures (for performance testing)

- `large_test_data_csv`: Large CSV data (1M records)
- `large_patients_csv`: Large patients CSV
- `large_admissions_csv`: Large admissions CSV
- `large_zip_with_csvs`: ZIP file with large CSVs

#### Excel Test Fixtures

- `synthetic_dexa_excel_file`: Synthetic DEXA Excel file
- `synthetic_statin_excel_file`: Synthetic Statin Excel file
- `synthetic_complex_excel_file`: Complex Excel file with metadata rows

### When to Use Each Fixture

- **Use `sample_patients_df`** for basic DataFrame tests
- **Use `sample_upload_df`** for testing lazy frame operations
- **Use `large_*` fixtures** for performance/streaming tests
- **Use `synthetic_*_excel_file`** for Excel parsing tests

---

## Makefile Usage

> **Reference**: [.cursor/rules/101-testing-hygiene.mdc](../.cursor/rules/101-testing-hygiene.mdc), [.cursor/rules/104-plan-execution-hygiene.mdc](../.cursor/rules/104-plan-execution-hygiene.mdc), [.cursor/rules/000-project-setup-and-makefile.mdc](../.cursor/rules/000-project-setup-and-makefile.mdc)

### CRITICAL RULE: Always Use Makefile Commands

**NEVER run pytest directly**:

```bash
# WRONG
pytest tests/
uv run pytest tests/

# CORRECT
make test
make test-fast
```

### Available Test Commands

```bash
make test              # Run all tests
make test-fast         # Fast tests only (skip slow)
make test-cov          # Tests with coverage report
make test-unit         # Unit tests only
make test-integration  # Integration tests only
```

### Quality Gates (Run Before Commit)

```bash
make format        # Auto-format code
make lint-fix      # Auto-fix linting issues
make type-check    # Verify type hints
make check         # Full quality gate (format + lint + type + test)
```

### Why Use Makefile?

1. **Consistency**: Same commands across all environments
2. **Dependency management**: Makefile handles `uv sync` and environment setup
3. **Configuration**: Makefile sets correct pytest options
4. **CI/CD alignment**: Same commands locally and in CI

---

## Test Writing Checklist

Use this checklist before committing test code:

- [ ] Checked `conftest.py` for existing fixtures
- [ ] Used parametrization instead of duplicate test functions
- [ ] Extracted repeated setup to fixtures
- [ ] Documented fixture purpose and scope
- [ ] Used appropriate fixture scope (session/module/function)
- [ ] Test follows AAA pattern (Arrange-Act-Assert)
- [ ] Test name follows `test_unit_scenario_expectedBehavior` pattern
- [ ] Test is isolated (no shared mutable state)
- [ ] Used Polars testing assertions (not pandas)
- [ ] Used Makefile commands for running tests (`make test-fast`)
- [ ] Ran `make lint-fix` and `make format` before commit
- [ ] All tests pass with `make test-fast`

---

## Examples

### Example 1: Before/After DRY Refactoring

**BEFORE** (Duplicate fixtures):

```python
# tests/test_analysis.py
@pytest.fixture
def sample_cohort():
    return pl.DataFrame({"patient_id": [1, 2, 3], "outcome": [0, 1, 0]})

def test_analysis_function_a(sample_cohort):
    result = analyze_a(sample_cohort)
    assert result is not None

# tests/test_compute.py
@pytest.fixture
def sample_cohort():  # DUPLICATE!
    return pl.DataFrame({"patient_id": [1, 2, 3], "outcome": [0, 1, 0]})

def test_compute_function_b(sample_cohort):
    result = compute_b(sample_cohort)
    assert result is not None
```

**AFTER** (DRY):

```python
# tests/conftest.py
@pytest.fixture
def sample_cohort():
    """Standard cohort fixture used across all tests."""
    return pl.DataFrame({"patient_id": [1, 2, 3], "outcome": [0, 1, 0]})

# tests/test_analysis.py
def test_analysis_function_a(sample_cohort):
    result = analyze_a(sample_cohort)
    assert result is not None

# tests/test_compute.py
def test_compute_function_b(sample_cohort):
    result = compute_b(sample_cohort)
    assert result is not None
```

### Example 2: Factory Fixture for Variations

```python
# conftest.py
@pytest.fixture
def make_cohort():
    """Factory fixture for creating custom cohorts with different sizes."""
    def _make(num_patients=5, outcome_rate=0.5):
        outcomes = [1 if i < num_patients * outcome_rate else 0
                   for i in range(num_patients)]
        return pl.DataFrame({
            "patient_id": list(range(1, num_patients + 1)),
            "outcome": outcomes,
            "age": [30 + i for i in range(num_patients)],
        })
    return _make

# test_analysis.py
def test_small_cohort_analysis(make_cohort):
    """Test with small cohort (5 patients)."""
    cohort = make_cohort(num_patients=5)
    result = analyze(cohort)
    assert len(result) == 5

def test_large_cohort_analysis(make_cohort):
    """Test with large cohort (1000 patients)."""
    cohort = make_cohort(num_patients=1000, outcome_rate=0.3)
    result = analyze(cohort)
    assert len(result) == 1000
```

### Example 3: Proper AAA Pattern

```python
def test_deduplication_preserves_first_occurrence_when_duplicates_exist():
    """Test that deduplication keeps first occurrence when duplicates exist."""
    # Arrange: Create DataFrame with duplicates
    df = pl.DataFrame({
        "id": [1, 1, 2, 2, 3],
        "timestamp": ["2024-01-01", "2024-01-02", "2024-01-01", "2024-01-03", "2024-01-01"],
        "value": [100, 200, 300, 400, 500],
    })

    # Act: Deduplicate by 'id', keeping first occurrence
    result = deduplicate_by_key(df, key="id", keep="first")

    # Assert: Verify deduplication preserves first occurrence
    assert result.height == 3
    assert result.filter(pl.col("id") == 1)["timestamp"][0] == "2024-01-01"
    assert result.filter(pl.col("id") == 1)["value"][0] == 100
    assert result.filter(pl.col("id") == 2)["timestamp"][0] == "2024-01-01"
    assert result.filter(pl.col("id") == 2)["value"][0] == 300
```

### Example 4: Parametrization with IDs

```python
@pytest.mark.parametrize("input_val,expected", [
    pytest.param(0, "inactive", id="zero_is_inactive"),
    pytest.param(1, "active", id="one_is_active"),
    pytest.param(None, "unknown", id="null_is_unknown"),
    pytest.param("1", "active", id="string_one_is_active"),
])
def test_status_mapping_handles_various_inputs(input_val, expected):
    """Test that status mapping handles various input types."""
    result = map_status(input_val)
    assert result == expected
```

### Example 5: Error Testing with Match

```python
def test_schema_validation_missing_required_column_raises_valueerror():
    """Test that schema validation raises ValueError when required column is missing."""
    # Arrange: DataFrame missing 'outcome' column
    df = pl.DataFrame({
        "patient_id": [1, 2, 3],
        "age": [30, 40, 50],
    })
    required_columns = ["patient_id", "outcome", "age"]

    # Act & Assert: Expect ValueError with specific message
    with pytest.raises(ValueError, match="outcome.*required"):
        validate_schema(df, required_columns)
```

---

## Cross-References

For detailed information, consult these project rules:

- **[.cursor/rules/101-testing-hygiene.mdc](../.cursor/rules/101-testing-hygiene.mdc)** - Testing patterns, AAA, fixtures, isolation
- **[.cursor/rules/102-dry-principles.mdc](../.cursor/rules/102-dry-principles.mdc)** - DRY principles, single source of truth
- **[.cursor/rules/103-staff-engineer-standards.mdc](../.cursor/rules/103-staff-engineer-standards.mdc)** - Production-grade patterns
- **[.cursor/rules/104-plan-execution-hygiene.mdc](../.cursor/rules/104-plan-execution-hygiene.mdc)** - Test-first workflow, quality gates
- **[.cursor/rules/000-project-setup-and-makefile.mdc](../.cursor/rules/000-project-setup-and-makefile.mdc)** - Makefile commands
- **[.cursor/rules/100-polars-first.mdc](../.cursor/rules/100-polars-first.mdc)** - Polars patterns and assertions

---

## Remember

**The Golden Rule**: If you're about to create a fixture, mock, or test data - CHECK `conftest.py` FIRST!

**Single Source of Truth**: One definition, many uses. Never duplicate.

**Use the Makefile**: `make test`, not `pytest`.

**Follow the Standards**: AAA pattern, descriptive names, isolation, Polars assertions.
