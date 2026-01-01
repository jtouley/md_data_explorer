# Testing Guidelines for MD Data Explorer

**Audience**: AI agents and developers maintaining test code

**Purpose**: Ensure DRY, maintainable test code aligned with project standards

---

## Quick Reference

**Key Principle**: Single source of truth - if it exists in `conftest.py`, use it; don't duplicate.

**Before creating ANY new fixture**: Check `conftest.py` first!

**Before running tests**: Always use `make test` or `make test-fast`, never `pytest` directly.

---

## ⚠️ MANDATORY AGENT RULES (Updated 2025-01)

**These rules are ENFORCED. Violations will result in rejected changes.**

### Rule 1: Factory Fixtures MUST Be Used
- ✅ **MUST** use `make_semantic_layer` for any SemanticLayer test setup
- ✅ **MUST** use `make_cohort_with_categorical` for patient cohorts with categorical variables
- ✅ **MUST** use `make_multi_table_setup` for 3-table relationship tests
- ❌ **FORBIDDEN** to create inline SemanticLayer instances when factory exists
- ❌ **FORBIDDEN** to create duplicate fixture definitions

### Rule 2: Fixture Discovery Is MANDATORY
- ✅ **MUST** search `conftest.py` before creating any fixture
- ✅ **MUST** use existing fixtures even if "almost" matches (use parameters)
- ❌ **FORBIDDEN** to create duplicate fixtures across test files
- ❌ **FORBIDDEN** to hardcode test data when a fixture exists

### Rule 3: Makefile Commands Are MANDATORY
- ✅ **MUST** use `make test`, `make test-fast`, or module-specific commands
- ✅ **MUST** use `make format`, `make lint-fix` before commits
- ❌ **FORBIDDEN** to run `pytest` directly
- ❌ **FORBIDDEN** to run `uv run pytest` directly
- ❌ **FORBIDDEN** to commit without running `make check-fast`

### Rule 4: Test Isolation Is MANDATORY
- ✅ **MUST** use AAA pattern (Arrange-Act-Assert) with clear separation
- ✅ **MUST** ensure tests are independent (no shared mutable state)
- ✅ **MUST** use descriptive names: `test_unit_scenario_expectedBehavior`
- ❌ **FORBIDDEN** to create tests that depend on execution order
- ❌ **FORBIDDEN** to use global state or class-level mutable variables

### Rule 5: Polars-First Is MANDATORY
- ✅ **MUST** use `pl.testing.assert_frame_equal` for DataFrame comparisons
- ✅ **MUST** use `df.height`, `df.width`, `df.to_dicts()` (Polars attributes)
- ❌ **FORBIDDEN** to use pandas for new test code (exceptions require comment)
- ❌ **FORBIDDEN** to use `len(df)` instead of `df.height`

---

## Table of Contents

1. [Test Structure: AAA Pattern](#test-structure-aaa-pattern)
2. [Test Naming Convention](#test-naming-convention)
3. [Unit Tests vs Integration Tests: Decision Criteria](#unit-tests-vs-integration-tests-decision-criteria)
4. [DRY Principles for Tests](#dry-principles-for-tests)
5. [Fixture Discipline](#fixture-discipline)
6. [Test Isolation](#test-isolation)
7. [Parameterization](#parameterization)
8. [Error Testing](#error-testing)
9. [Data Engineering Specific Patterns](#data-engineering-specific-patterns)
10. [Polars Testing Assertions](#polars-testing-assertions)
11. [Common Anti-Patterns to Avoid](#common-anti-patterns-to-avoid)
12. [Standard Fixtures Reference](#standard-fixtures-reference)
13. [Makefile Usage](#makefile-usage)
14. [Test Writing Checklist](#test-writing-checklist)
15. [Examples](#examples)

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

## Unit Tests vs Integration Tests: Decision Criteria

> **Reference**: [tests/PERFORMANCE.md](./PERFORMANCE.md) - Performance optimization details

### When to Write Unit Tests (Default - 95% of Tests)

**Write unit tests when testing:**
1. **Code logic and business rules** - How the code processes data, makes decisions, transforms inputs
2. **Error handling** - How code handles invalid inputs, edge cases, failures
3. **Data transformations** - How data is parsed, filtered, aggregated, validated
4. **Algorithm correctness** - Pattern matching, semantic matching, query parsing logic
5. **State management** - How internal state changes, how objects interact

**Characteristics:**
- ✅ Use `mock_llm_calls` fixture to mock LLM calls (prevents 30s HTTP timeouts)
- ✅ Use `nl_query_engine_with_cached_model` for fast SentenceTransformer access (2-5s speedup)
- ✅ Use test doubles (mocks, stubs, fakes) for external dependencies
- ✅ Run fast (<1s per test)
- ✅ No `@pytest.mark.integration` or `@pytest.mark.slow` markers
- ✅ Test in isolation - no real external services

**Example:**
```python
def test_parse_query_with_refinement_context_adds_filter(
    make_semantic_layer,
    mock_llm_calls,
    nl_query_engine_with_cached_model,
):
    """Test that parsing logic correctly handles refinement queries."""
    # Arrange
    semantic = make_semantic_layer(
        dataset_name="test_statins",
        data={"patient_id": ["P1", "P2"], "statin_used": [0, 1]},
    )
    engine = nl_query_engine_with_cached_model(semantic_layer=semantic)
    
    # Act
    result = engine.parse_query(
        query="remove the n/a",
        conversation_history=[{"query": "count patients", "intent": "COUNT"}],
    )
    
    # Assert: Tests logic, not real LLM
    assert result.intent_type == "COUNT"
    assert len(result.filters) >= 1
```

### When to Write Integration Tests (Exception - 5% of Tests)

**Write integration tests ONLY when testing:**
1. **Real external service behavior** - Actual LLM responses, actual API contracts
2. **End-to-end workflows** - Complete user journeys across multiple components
3. **Service availability** - Whether Ollama is running, whether models are available
4. **Performance characteristics** - Real LLM latency, real data loading times
5. **Regression detection** - Catching changes in external service behavior

**Characteristics:**
- ✅ Use real Ollama LLM calls (no mocks)
- ✅ Mark with `@pytest.mark.integration` AND `@pytest.mark.slow`
- ✅ Skip automatically if service unavailable (`skip_if_ollama_unavailable`)
- ✅ Run slow (10-30s per test) - acceptable because they're rare
- ✅ In separate files (e.g., `test_*_integration.py`)
- ✅ Test real behavior, not just logic

**Example:**
```python
@pytest.mark.integration
@pytest.mark.slow
def test_llm_parse_with_real_ollama(mock_semantic_layer, skip_if_ollama_unavailable):
    """Verify real Ollama service returns valid responses."""
    from clinical_analytics.core.nl_query_engine import NLQueryEngine
    
    engine = NLQueryEngine(mock_semantic_layer)
    intent = engine._llm_parse("complex query that needs real LLM")
    
    # Assert: Tests real LLM, not mocked
    assert intent is not None
    assert intent.confidence >= 0.5  # Real LLM should have good confidence
```

### Decision Tree

```
Start: What are you testing?
│
├─ Code logic, transformations, error handling?
│  └─> Write UNIT TEST (with mocks)
│
├─ Real external service behavior (LLM, API, database)?
│  └─> Write INTEGRATION TEST (real services)
│
├─ End-to-end user workflow?
│  └─> Write INTEGRATION TEST (real services)
│
└─ Performance/regression of external service?
   └─> Write INTEGRATION TEST (real services)
```

### Rules of Thumb

1. **Default to unit tests** - 95% of tests should be unit tests
2. **Integration tests are exceptions** - Only write when you MUST test real service behavior
3. **One integration test per concern** - Don't duplicate unit test coverage with integration tests
4. **Integration tests verify contracts** - They ensure external services still work as expected
5. **Unit tests verify logic** - They ensure your code works correctly

### Anti-Patterns

❌ **DON'T write integration tests to test code logic** - Use unit tests with mocks
❌ **DON'T write unit tests to test external service behavior** - Use integration tests
❌ **DON'T duplicate coverage** - If unit test covers the logic, integration test should only verify real service
❌ **DON'T skip mocking in unit tests** - Always use `mock_llm_calls` in unit tests
❌ **DON'T forget to mark integration tests** - Always use both `@pytest.mark.integration` and `@pytest.mark.slow`

### Performance Impact

**Unit Tests (with mocks):**
- Duration: <1s per test
- Uses: `mock_llm_calls` + `nl_query_engine_with_cached_model`
- Speedup: 30-50x faster than real LLM calls

**Integration Tests (real services):**
- Duration: 10-30s per test
- Uses: Real Ollama LLM calls
- Purpose: Verify real service behavior, catch regressions

See [tests/PERFORMANCE.md](./PERFORMANCE.md) for detailed performance optimization documentation.

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

### MANDATORY: Use Polars-Native Assertions for DataFrame Comparisons

**Rule: ALWAYS use `pl.testing.assert_frame_equal()` when comparing entire DataFrames**

```python
import polars.testing as plt

# ✅ CORRECT: Use Polars testing assertions for DataFrames
plt.assert_frame_equal(result, expected)

# ❌ WRONG: Never use list comparisons for DataFrame equality
# assert df1["col"].to_list() == df2["col"].to_list()  # NO!

# ❌ WRONG: Never use pandas assertions
# pd.testing.assert_frame_equal(result, expected)  # NO!
```

**Why**: `assert_frame_equal()` properly handles:
- Schema differences (missing/extra columns)
- Type mismatches (int vs float vs string)
- Null value semantics
- Float precision issues
- Column order differences

**When List Comparison is OK**:
- Single scalar value: `assert result == 42`
- Simple list of scalars: `assert [1, 2, 3] == [1, 2, 3]`
- String content: `assert "error message" in str(exception)`

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

#### Factory Fixtures (MUST USE THESE - Added 2025-01 Refactoring)

**Semantic Layer Factories:**
- `make_semantic_layer`: **MANDATORY** factory for SemanticLayer instances
  - Creates SemanticLayer with custom data, config, workspace
  - **Use for**: Any test requiring SemanticLayer setup
  - **Example**: `layer = make_semantic_layer(dataset_name="test", data={"patient_id": [1,2,3]})`
  - **Rule**: ❌ NEVER create SemanticLayer instances inline

- `mock_semantic_layer`: Factory for mock SemanticLayer with configurable column mappings
  - Creates MagicMock with column alias index
  - **Use for**: Tests that don't need real data loading
  - **Example**: `mock = mock_semantic_layer(columns={"age": "age", "status": "status"})`

**DataFrame Factories:**
- `make_cohort_with_categorical`: **MANDATORY** factory for patient cohorts
  - Creates cohorts with categorical variables ("1: Yes", "2: No" patterns)
  - **Use for**: Tests requiring patient cohorts with categorical encoding
  - **Example**: `cohort = make_cohort_with_categorical(patient_ids=["P001"], ages=[45])`
  - **Rule**: ❌ NEVER hardcode categorical cohort DataFrames

- `make_multi_table_setup`: **MANDATORY** factory for multi-table tests
  - Creates 3-table setup: patients, medications, patient_medications (bridge)
  - **Returns**: `{"patients": df, "medications": df, "patient_medications": df}`
  - **Example**: `tables = make_multi_table_setup(num_patients=5, num_medications=4)`
  - **Rule**: ❌ NEVER create patients/medications/bridge tables inline

**Analysis Context Fixtures:**
- `sample_context`: Direct AnalysisContext (DESCRIBE intent, confidence=0.9)
- `low_confidence_context`: AnalysisContext with confidence=0.4
- `high_confidence_context`: AnalysisContext with confidence=0.9
- `sample_context_describe`, `sample_context_compare`, `sample_context_predictor`: Intent-specific contexts

#### DataFrame Fixtures

- `sample_patients_df`: Sample patients DataFrame
- `sample_admissions_df`: Sample admissions DataFrame
- `sample_upload_df`: Sample upload DataFrame
- `sample_upload_metadata`: Sample upload metadata
- `sample_cohort`: Standard cohort (integer IDs, binary outcomes)
- `mock_cohort`: Pandas cohort for Streamlit UI tests
- `sample_numeric_df`, `sample_categorical_df`, `sample_mixed_df`: Compute test fixtures

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

**⚠️ MANDATORY: Check Factory Fixtures First!**

**Factory Fixtures (ALWAYS prefer these for variations):**
1. **`make_semantic_layer`** - ✅ REQUIRED for any SemanticLayer test
2. **`make_cohort_with_categorical`** - ✅ REQUIRED for patient cohorts with categorical vars
3. **`make_multi_table_setup`** - ✅ REQUIRED for patients/medications/bridge table tests
4. **`mock_semantic_layer`** - ✅ REQUIRED for mocked semantic layer (no data loading)

**Direct Fixtures (Use for standard cases):**
- **Use `sample_patients_df`** for basic DataFrame tests (3 patients)
- **Use `sample_cohort`** for standard cohort tests (5 patients, binary outcome)
- **Use `sample_upload_df`** for testing lazy frame operations
- **Use `large_*` fixtures** for performance/streaming tests (1M records)
- **Use `synthetic_*_excel_file`** for Excel parsing tests

**Analysis Context Fixtures:**
- **Use `sample_context`** for default DESCRIBE analysis
- **Use `low_confidence_context`** for testing clarifying questions
- **Use `high_confidence_context`** for testing auto-execution

**🚨 RULE OF THUMB:** 
- If you need variations → Use **factory fixtures**
- If standard data suffices → Use **direct fixtures**
- If unsure → **Search `conftest.py` first!**
- **NEVER create inline data** when a fixture exists

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

---

## 🚨 ENFORCEMENT: Violation Examples

### ❌ VIOLATION: Creating inline SemanticLayer when factory exists

```python
# WRONG - Will be REJECTED
def test_my_feature(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    # ... 20 lines of setup code ...
    semantic = SemanticLayer(dataset_name="test", config=config, workspace_root=workspace)
```

```python
# CORRECT - REQUIRED
def test_my_feature(make_semantic_layer):
    semantic = make_semantic_layer(dataset_name="test", data={"patient_id": [1,2,3]})
```

### ❌ VIOLATION: Creating duplicate fixture

```python
# WRONG - Will be REJECTED
# In test_analysis.py
@pytest.fixture
def sample_cohort():  # Already exists in conftest.py!
    return pl.DataFrame({...})
```

```python
# CORRECT - REQUIRED
# In test_analysis.py
def test_analysis(sample_cohort):  # Use fixture from conftest.py
    result = analyze(sample_cohort)
```

### ❌ VIOLATION: Running pytest directly

```bash
# WRONG - Will be REJECTED
pytest tests/
uv run pytest tests/core/

# CORRECT - REQUIRED
make test-fast
make test-core
```

### ❌ VIOLATION: Using pandas assertions for Polars

```python
# WRONG - Will be REJECTED
import pandas.testing as pdt
pdt.assert_frame_equal(result, expected)

# CORRECT - REQUIRED
import polars.testing as plt
plt.assert_frame_equal(result, expected)
```

### ❌ VIOLATION: Creating multi-table setup inline

```python
# WRONG - Will be REJECTED
def test_relationships():
    patients = pl.DataFrame({"patient_id": ["P1", "P2"], "age": [30, 45]})
    medications = pl.DataFrame({"medication_id": ["M1", "M2"], "drug_name": ["A", "B"]})
    # ... more setup ...
```

```python
# CORRECT - REQUIRED
def test_relationships(make_multi_table_setup):
    tables = make_multi_table_setup(num_patients=2, num_medications=2)
    patients = tables["patients"]
    medications = tables["medications"]
```

---

## 🤖 AI Agent Compliance Checklist

Before submitting ANY test code, verify:

- [ ] ✅ Searched `conftest.py` for existing fixtures
- [ ] ✅ Used factory fixtures (`make_*`) for all eligible tests
- [ ] ✅ No duplicate fixture definitions created
- [ ] ✅ All tests use `make` commands (verified in commit)
- [ ] ✅ AAA pattern followed with clear separation
- [ ] ✅ Test names follow `test_unit_scenario_expectedBehavior`
- [ ] ✅ Polars assertions used (`plt.assert_frame_equal`)
- [ ] ✅ No inline SemanticLayer, cohort, or multi-table setup
- [ ] ✅ Ran `make format && make lint-fix` before commit
- [ ] ✅ Ran `make check-fast` - all tests passing

**Failure to comply = Changes will be rejected and must be redone.**
