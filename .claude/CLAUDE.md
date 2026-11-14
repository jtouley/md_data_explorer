# Claude Code Rules

## Purpose and steel thread (read first, every session)

**Purpose.** This is a local data analyst for anyone in medicine (medical students, residents, fellows, attendings, any specialty) who has a dataset and a question.
- **Any question:** they ask in their own words, about anything a capable data tool such as TextQL could answer.
- **Right answer, shown method:** they get the right answer with the method shown, from a local model (Ollama).
- **Privacy:** data never leaves the machine, and no count of 1–10 patients is ever shown.
- **Any data:** public datasets, registry exports, and hospital extracts with their dictionaries.

Use cases such as the ID-fellow research guide are examples, not the scope. Don't narrow the product to one specialty, one workflow or a fixed list of question types.

**"Works" means one thing:** right answers on real public datasets, through the real upload API, with the local model running, from a fresh app state. Passing unit tests, a green CI or a safe hold are not "works".

**Steel thread, in order.** Do the first unfinished step. Work that does not move it waits.
1. The model tier runs on uploaded datasets (#101).
2. Acceptance harness on COVID-MS (GDSI, PhysioNet): `make accept DATASET=covid_ms MODEL=...` reuses `eval/headless.py` and the presented-answer graders. It reports RIGHT, WRONG, HELD, ERROR and LEAK per question, giving the baseline.
3. The data contract becomes the model's only context: dictionary, codebooks, units, grain, keys.
4. A governed SQL path, single table: the model writes SQL, it is checked against the contract, small cells are suppressed on the output, and the SQL is shown.
5. Statistics as tools: the model picks the test, and the existing core computes it.
6. MVP. Widen to the corpus (Sepsis 2019, MIMIC demos, population sets with ranges, patient-level sets across specialties) and compare local Ollama with a hosted stand-in. The bar: 0 WRONG, 0 LEAK.

The design reference is [ADR013](../docs/implementation/ADR/ADR013.md). Take from it only what the current step needs.

**Rules that stop wasted work:**
- **Check before building.** Before writing code or briefing an agent, search the repo for what already does the job (`eval/`, `contracts/`, `analysis/`, `core/`) and the open PRs. Extend what exists. A brief must name what exists and say to reuse it.
- **One change on the thread at a time:** one PR, one independent review, one fix round. A second review only when the fix changes privacy, security or concurrency logic.
- **A change merges only if it moves the acceptance table,** or fixes a WRONG, LEAK or ERROR the table shows. Hardening for cases the corpus doesn't hit waits.
- **Report status as the latest acceptance table,** not as a list of PRs.
- **Real data:** public, openly licensed datasets only. They are fetched and checksummed at test time, never committed, and never printed row by row. Unit tests use synthetic data.

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
- **Typed Parquet** for persistence (no Delta Lake; do not add it). The layers of one uploaded dataset version:
  1. **Raw landing**: the uploaded bytes as received, `uploads/raw/{upload_id}/{dataset_version}{ext}`.
  2. **Staged/stored**: each table read as text and typed once, column by column over the whole file (`storage/staging.py`), written as `parquet/{upload_id}_{table}_{dataset_version}.parquet`.
  3. **Unified cohort**: the single table, or for a ZIP the joined cohort `parquet/{upload_id}__cohort_{dataset_version}.parquet`.
  4. **Semantic layer**: in-memory DuckDB built with `read_parquet` on that cohort (`core/semantic.py`), probed at upload.
  - The cohort Parquet is the one source of truth: queries, previews and the contract/quality checks all read it. `analytics.duckdb` holds only pre-Parquet uploads until they are deleted.
  - Every data and metadata write goes through `storage/atomic.py` (`*.part`, fsync, `os.replace`). Data files are named by `dataset_version` and never replaced; the metadata JSON is written last, as the commit point.
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
