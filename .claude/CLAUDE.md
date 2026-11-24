# Claude Code Rules

## Purpose and steel thread (read first, every session)

The purpose, what "works" means, the ordered steel thread, the merge gate and the real-data rule live in [docs/vision/PURPOSE.md](../docs/vision/PURPOSE.md). Read it first. Do the first unfinished step; work that does not move it waits.

**Rules that stop wasted work:**
- **Check before building.** Before writing code or briefing an agent, search the repo for what already does the job (`eval/`, `contracts/`, `analysis/`, `core/`, `tests/eval/`) and the open PRs. Extend what exists. A brief must name what exists and say to reuse it.
- **One change on the thread at a time:** one PR, reviewed and fixed as in [Delivery](#delivery).
- **A change merges only if it moves the acceptance table,** or fixes a WRONG, LEAK or ERROR the table shows. Hardening for cases the corpus doesn't hit waits. **Exception:** fixes for privacy, security or data loss merge on their own merits, whether or not the corpus triggers the issue.
- **Report status as the latest acceptance table,** not as a list of PRs. Give answered right and correctly held as separate numbers.
- **One issue per cause** for each reproducible WRONG, LEAK, ERROR or unacceptable HOLD, linked to the fixing PR and the oracle it must pass (PURPOSE.md, Merge gate).
- **Real data:** synthetic data is the rule. The only exception is acceptance testing on public, openly licensed, de-identified datasets: fetched at test time from their public source, checked against a pinned checksum, used under their licence, never committed, and never shown row by row. Aggregate results (counts, statistics, grades) may appear in logs, issues and pull requests. Everything else, unit tests included, uses synthetic data.

## Delivery

Tooling enforces these; a claim that they hold is not enough.

**Done before review** (the builder, before asking for one):
- `make done-check` passes; paste its summary into the PR body. It runs `check-fast` (which includes `check-eval-tokens`), `test-fast`, `mutation-changed` and `pr-size`.
- Every surviving mutant in comparison, threshold, missing-code, total-row, hold or unit logic is killed by a test, or, if equivalent, its line carries `# pragma: no mutate` with the reason.
- Property tests (`hypothesis`) state the change's invariants; example tests alone are not done.
- Where a name could change the answer (column, code, label, dataset), a test renames it and checks the result holds.
- Under 1,500 added non-test lines, or split. CI warns above 1,500 and fails above 3,000 unless the PR has the `large-pr-approved` label.

**Review is not the test suite:**
1. One review by a fresh opus subagent whose whole prompt is `/code-review PR #<n>`; its findings are posted to the PR.
2. One fix round by a fresh subagent loaded with the relevant skills.
3. A second review only when the fix changed privacy or security code.
4. Then merge on green CI, or take the decision to the user. Later findings become issues, not more rounds.

**Evaluation hygiene:**
- Builders never read sealed or held-out data, questions or answers.
- Failures from sealed runs never become rules, prompts or briefs. A holdout is used once.
- No evaluation-dataset identifier enters the repo: `make check-eval-tokens` (CI, `check-fast`) hashes added identifiers and strings against `config/eval_token_hashes.txt`. Never try to reverse that list.

**Outcome metric:** a change to the question path is judged by end-to-end tallies of right, wrong and held answers on development data (`make accept`), with the model to itself. Acceptance needs both a ceiling on wrong answers (`--fail-on WRONG`) and a floor on right ones (`--baseline`); fewer wrong answers bought by holding more is not a pass.

**One model user at a time:** every command that calls the local model (Ollama) runs through `scripts/model_lock.py`, an flock on `~/.cache/mdde/model.lock` (override: `MDDE_MODEL_LOCK`). `make accept`, `accept-fellow` and `accept-sql-arms` take it with `--wait` when a model is in use, so measurements queue. Anything else runs `uv run python scripts/model_lock.py -- <command>`, which exits 75 at once when the lock is busy instead of competing with a measurement.

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
