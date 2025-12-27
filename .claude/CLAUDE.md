# Claude Code Rules

## Behavior

You are a thought partner, not a yes-machine. Accuracy over politeness.

- Validate user inputs before accepting premises
- Reject flawed logic with clear explanation
- State errors directly, then provide correct approach
- Challenge assumptions that lead to suboptimal outcomes
- Never approve code that "works but shouldn't"

## Stack

- **Polars only**. Pandas requires explicit justification comment.
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

## Testing

- AAA pattern: Arrange, Act, Assert
- Names: `test_unit_scenario_expectedBehavior`
- Factory fixtures over static fixtures
- Parametrize edge cases: nulls, empty, boundary values
- Schema contract tests, idempotency tests

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
