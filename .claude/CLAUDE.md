# Claude Code Rules

## Behavior

You are a thought partner, not a yes-machine. Accuracy over politeness.

- Validate user inputs before accepting premises
- Reject flawed logic with clear explanation
- State errors directly, then provide correct approach
- Challenge assumptions that lead to suboptimal outcomes
- Never approve code that "works but shouldn't"

## Stack

- **Polars only**. Pandas requires explicit justification comment. NEVER use map_elements.
- **Lazy by default**: `scan_*`, single `collect()` at pipeline end
- **Delta Lake** for persistence, medallion architecture (Bronze/Silver/Gold)
- **Pydantic** for external data validation at boundaries

## Code Standards

**DRY**
- Centralize column names, schemas, thresholds in config
- Extract reusable expressions to `expressions.py`
- Rule of Three: don't abstract until third instance

**Typing**
- Full type hints on all function signatures
- Use `Literal`, `TypeAlias`, `Callable` appropriately
- Runtime validation at system boundaries

**Errors**
- Fail fast, fail loud. No silent `except: pass`
- Catch specific exceptions, never bare `Exception`
- Domain exceptions: `PipelineError`, `SchemaValidationError`, `DataQualityError`

**Observability**
- Structured logging with `structlog`
- Bind context: `logger.bind(batch_id=batch_id)`

## Testing (MANDATORY)

**🚨 HARD RULES - Violations = Rejected Changes:**

### Fixture Discovery (MANDATORY)
1. ✅ **MUST** search `tests/conftest.py` before creating ANY fixture
2. ✅ **MUST** use factory fixtures for all eligible tests:
   - `make_semantic_layer` - SemanticLayer instances
   - `make_cohort_with_categorical` - Patient cohorts
   - `make_multi_table_setup` - 3-table relationship tests
   - `mock_semantic_layer` - Mocked semantic layer
3. ❌ **FORBIDDEN** to create duplicate fixtures
4. ❌ **FORBIDDEN** to create inline SemanticLayer, cohort, or multi-table setups

### Makefile Commands (MANDATORY)
1. ✅ **MUST** use `make test-fast`, `make test-core`, `make test-analysis` (never `pytest` directly)
2. ✅ **MUST** run `make format && make lint-fix` before commits
3. ✅ **MUST** run `make check-fast` to verify all changes
4. ❌ **FORBIDDEN** to run `pytest` or `uv run pytest` directly

### Test Structure (MANDATORY)
- ✅ **MUST** follow AAA pattern: Arrange, Act, Assert (with clear separation)
- ✅ **MUST** use names: `test_unit_scenario_expectedBehavior`
- ✅ **MUST** parametrize edge cases: nulls, empty, boundary values
- ✅ **MUST** use `pl.testing.assert_frame_equal` (not pandas)
- ❌ **FORBIDDEN** to use `len(df)` (use `df.height`)
- ❌ **FORBIDDEN** to use pandas in new test code

### Test Performance (MANDATORY)
- ✅ **MUST** use `get_sample_datasets()` (1-2 datasets) for fast unit tests
- ✅ **MUST** mark data-loading tests: `@pytest.mark.slow` + `@pytest.mark.integration`
- ✅ **MUST** use `get_available_datasets()` only for critical schema/compliance tests

**See `tests/AGENTS.md` for full enforcement rules and violation examples.**

## Patterns

```python
# Lazy pipeline
result = (
    pl.scan_parquet("raw/*.parquet")
    .filter(pl.col("status") == "active")
    .group_by("customer_id")
    .agg(pl.col("amount").sum().alias("total"))
    .collect()
)

# Expression reuse
sum_ab = pl.col("a") + pl.col("b")
df.with_columns((sum_ab * 2).alias("x"), (sum_ab * 3).alias("y"))

# Selector patterns
import polars.selectors as cs
df.select(cs.numeric().fill_null(0), cs.string().str.strip_chars())

# Contract validation
if value_col not in df.columns:
    raise ValueError(f"Column '{value_col}' not found")
```

## Review Checklist

1. Errors explicit and recoverable?
2. Debuggable at 3 AM?
3. Safe to run twice?
4. External data validated?
5. Edge cases tested?
