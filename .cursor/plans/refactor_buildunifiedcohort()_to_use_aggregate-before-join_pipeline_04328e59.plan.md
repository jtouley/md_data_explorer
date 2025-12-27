---
name: ""
overview: ""
todos: []
---

---

name: Refactor buildunifiedcohort() to Use Aggregate-Before-Join Pipeline

overview: ""

todos:

  - id: replace-build-unified-cohort

content: Replace entire build_unified_cohort() method (lines 1146-1240) with aggregate-before-join implementation. This atomic edit includes using _find_anchor_by_centrality() instead of deprecated _find_anchor_table()

status: pending

  - id: verify-no-legacy-anchor-method

content: Verify no references to deprecated _find_anchor_table() remain in codebase after refactor

status: pending

dependencies:

      - replace-build-unified-cohort
  - id: add-dimension-mart-call

content: Call _build_dimension_mart() to build dimension mart with 1:1 joins only

status: pending

dependencies:

      - replace-build-unified-cohort
  - id: add-fact-aggregation-call

content: Call _aggregate_fact_tables() with anchor grain_key to pre-aggregate fact/event tables

status: pending

dependencies:

      - replace-build-unified-cohort
  - id: add-feature-joins

content: Join each aggregated feature table to mart using Polars with validate='1:1'. Sort feature_tables.keys() for deterministic ordering, use how='left' (join_type only applies to dimension mart), and do not use suffix parameter (columns already renamed)

status: pending

dependencies:

      - replace-build-unified-cohort
  - id: fix-join-type-validation

content: Normalize and validate join_type parameter (lower().strip(), check in {'left', 'inner', 'outer'}) to fail-fast on invalid input

status: pending

dependencies:

      - replace-build-unified-cohort
  - id: fix-lazyframe-schema

content: Use collect_schema() instead of .schema on LazyFrame for reliable schema access on LazyFrame

status: pending

dependencies:

      - replace-build-unified-cohort
  - id: add-deterministic-column-ordering

content: Enforce deterministic RHS column order by selecting [grain_key] + sorted(other_cols) before joining feature tables

status: pending

dependencies:

      - replace-build-unified-cohort
  - id: add-observability-logs

content: Add logging at each step (dimension mart, aggregation, feature joins) for debugging

status: pending

dependencies:

      - replace-build-unified-cohort
  - id: add-no-legacy-path-test

content: Add test to ensure build_unified_cohort() never executes legacy DuckDB mega-join SQL path (patch duckdb.DuckDBPyConnection.execute globally to detect SELECT * FROM ... LEFT JOIN pattern)

status: pending

dependencies:

      - replace-build-unified-cohort
  - id: add-validate-join-test

content: Add micro-test proving validate='1:1' works on LazyFrame.join() by joining two LazyFrames with duplicate keys and expecting ComputeError

status: pending

dependencies:

      - replace-build-unified-cohort
  - id: add-row-count-invariant-test

content: Add test verifying result.height == mart.height (not anchor grain count, since dimension mart join_type can filter rows). Use mart.select(pl.len().alias('n')).collect()['n'][0] for mart height

status: pending

dependencies:

      - replace-build-unified-cohort
  - id: add-deterministic-columns-test

content: Add test for deterministic column ordering. Either set RNG seeds for upstream classification/aggregation, or test deterministic join ordering only (feature tables joined in sorted name order)

status: pending

dependencies:

      - replace-build-unified-cohort
  - id: add-join-type-validation-test

content: Add test verifying join_type normalization (case-insensitive) and validation (rejects invalid values, accepts 'LEFT' -> 'left')

status: pending

dependencies:

      - replace-build-unified-cohort

---

# Refactor build_unified_cohort() to Use Aggregate-Before-Join Pipeline

## Problem

The current `build_unified_cohort()` method (lines 1146-1240 in `multi_table_handler.py`) still uses the legacy DuckDB SQL join approach that builds a giant `SELECT * FROM anchor LEFT JOIN table1 ... LEFT JOIN table2 ...` query. This causes cartesian product explosions when joining one-to-many fact/event tables (like `chartevents`, `labevents`, `inputevents`) directly to the anchor, leading to OOM errors (169GB+ intermediate results).

The M3/M4 methods (`_build_dimension_mart()` and `_aggregate_fact_tables()`) are implemented and tested, but they're not being called by `build_unified_cohort()`.

## Solution

Replace the legacy SQL join logic in `build_unified_cohort()` with the aggregate-before-join pipeline:

1. **Find anchor** using `_find_anchor_by_centrality()` (not deprecated `_find_anchor_table()`)
2. **Build dimension mart** using `_build_dimension_mart()` - only joins 1:1 dimensions
3. **Aggregate fact tables** using `_aggregate_fact_tables()` - aggregates each fact/event table to grain first
4. **Join feature tables** to mart using Polars with validation (1:1 or m:1)
5. **Collect and return** the final unified DataFrame

## Implementation

### File: `src/clinical_analytics/core/multi_table_handler.py`

**Location**: Lines 1146-1240 (entire `build_unified_cohort()` method)

**Replace the entire method** with new implementation:

```python
def build_unified_cohort(
    self,
    anchor_table: Optional[str] = None,
    join_type: str = "left",
) -> pl.DataFrame:
    """
    Build unified cohort using aggregate-before-join architecture.

    Strategy:
      1) classify + detect relationships
      2) choose anchor (centrality)
      3) build dimension mart (1:1-ish joins only)
      4) aggregate fact/event tables to grain (one row per grain)
      5) join aggregated feature tables to mart (validate 1:1)
      6) collect
      
    Args:
        anchor_table: Root table for joins (auto-detected if None)
        join_type: Type of join (left, inner, outer) - only affects dimension mart
        
    Returns:
        Unified Polars DataFrame with all columns
        
    Raises:
        ValueError: If anchor not found, not unique on grain, or invalid join_type
        AggregationPolicyError: If aggregation policy violation detected
    """
    logger.info(f"Building unified cohort (anchor_table={anchor_table}, join_type={join_type})")

    # Normalize and validate join_type (fail-fast on invalid input)
    join_type = join_type.lower().strip()
    if join_type not in {"left", "inner", "outer"}:
        raise ValueError(f"Unsupported join_type: {join_type}")

    if not self.classifications:
        self.classify_tables()
    if not self.relationships:
        self.detect_relationships()

    if anchor_table is None:
        anchor_table = self._find_anchor_by_centrality()
        logger.info(f"Auto-detected anchor table: {anchor_table}")

    if anchor_table not in self.tables:
        raise ValueError(f"Anchor table '{anchor_table}' not found in tables")

    anchor_class = self.classifications.get(anchor_table)
    if anchor_class is None:
        raise ValueError(f"Anchor table '{anchor_table}' not classified")

    grain_key = anchor_class.grain_key
    if not grain_key:
        raise ValueError(f"Anchor table '{anchor_table}' has no detected grain_key")

    logger.info(f"Anchor '{anchor_table}' grain_key='{grain_key}'")

    # 1) Dimensions only (safe joins)
    logger.info("Building dimension mart...")
    mart = self._build_dimension_mart(anchor_table=anchor_table, join_type=join_type)

    # 2) Aggregate facts/events to grain
    logger.info("Aggregating fact/event tables...")
    feature_tables = self._aggregate_fact_tables(grain_key=grain_key)

    if not feature_tables:
        logger.warning("No fact/event tables to aggregate; returning dimension mart only")
        return mart.collect()

    # 3) Join features (must be 1 row per grain on both sides)
    logger.info(f"Joining {len(feature_tables)} feature tables to mart...")
    for feature_name in sorted(feature_tables.keys()):
        feature_lazy = feature_tables[feature_name]

        # Use collect_schema() instead of .schema (reliable on LazyFrame)
        schema = feature_lazy.collect_schema()
        names = schema.names()
        if grain_key not in names:
            logger.warning(
                f"Skipping '{feature_name}': grain_key '{grain_key}' not in schema "
                f"(cols={names})"
            )
            continue

        # Enforce deterministic RHS column order (grain_key first, then sorted)
        rhs_cols = [grain_key] + sorted([c for c in names if c != grain_key])
        feature_lazy = feature_lazy.select(rhs_cols)

        mart = mart.join(
            feature_lazy,
            on=grain_key,
            how="left",
            validate="1:1",  # mart unique on grain, aggregated features unique on grain
        )

    logger.info("Collecting unified cohort...")
    result = mart.collect()

    logger.info(
        f"Unified cohort built: {result.height:,} rows, {result.width} cols "
        f"(anchor={anchor_table}, grain={grain_key}, features={len(feature_tables)})"
    )
    return result
```

## Key Changes

1. **Removed DuckDB SQL construction** (lines 1215-1240) - no more `SELECT * FROM ... LEFT JOIN ...`
2. **Uses `_find_anchor_by_centrality()`** instead of deprecated `_find_anchor_table()`
3. **Calls `_build_dimension_mart()`** to build dimension mart with 1:1 joins only (validation handled inside `_build_dimension_mart()`)
4. **Calls `_aggregate_fact_tables()`** to pre-aggregate all fact/event tables
5. **Sorted feature joins** - iterate `sorted(feature_tables)` for deterministic output and failure ordering
6. **No suffix on feature joins** - aggregated columns already renamed (e.g., `heart_rate_min`), suffix would bloat names
7. **Feature joins always left** - `join_type` parameter only applies to dimension mart; features use `how="left"` to preserve cohort rows
8. **Uses Polars joins with validation** (`validate="1:1"`) to prevent row explosion and fail fast
9. **Proper logging** at each step for observability

## Implementation Refinements & Bug Fixes

### 1. Sorted Feature Joins

Iterate `sorted(feature_tables.keys())` instead of `feature_tables.items()` to ensure:

- **Deterministic output**: Same input always produces same column order
- **Deterministic failure ordering**: Errors occur in predictable sequence for debugging

### 2. No Suffix on Feature Joins

Aggregated columns are already renamed (e.g., `heart_rate_min`, `labevents_count`). Using `suffix=f"_{feature_name}"` would create bloated names like `heart_rate_min_chartevents_features` and can cause collisions. Drop the suffix parameter.

### 3. Join Type Scope & Validation

The `join_type` parameter only applies to dimension mart joins (handled inside `_build_dimension_mart()`). Feature table joins always use `how="left"` because dropping cohort rows due to missing features is usually wrong.

**Normalize and validate `join_type`** to fail-fast on invalid input:

```python
join_type = join_type.lower().strip()
if join_type not in {"left", "inner", "outer"}:
    raise ValueError(f"Unsupported join_type: {join_type}")
```

### 4. Dimension Mart Validation

Cardinality validation (`validate="m:1"`) is already handled inside `_build_dimension_mart()` (line 988). Do not add validation logic in `build_unified_cohort()` - keep it in one place.

### 5. Use `collect_schema()` Instead of `.schema` on LazyFrame

**Bug**: `feature_lazy.schema` is not reliable on LazyFrame - it can be unknown, stale, or force planning unexpectedly.

**Fix**: Use `collect_schema()` which is cheap and logical:

```python
schema = feature_lazy.collect_schema()
if grain_key not in schema.names():
    ...
```

### 6. Deterministic Column Ordering

**Issue**: Join order is deterministic, but the right-hand feature table's column order can vary depending on how aggregation expressions were built. Also, if `_aggregate_fact_tables()` uses dict iteration over tables not sorted, you may still get drift even if you sort `feature_tables.keys()` here.

**Fix**:

1. Explicitly project feature columns in sorted order before joining:
```python
rhs_cols = [grain_key] + sorted([c for c in names if c != grain_key])
feature_lazy = feature_lazy.select(rhs_cols)
```

2. Ensure `_aggregate_fact_tables()` returns feature tables in deterministic order (or normalize the output dict order there). The join sorting here helps, but upstream determinism is also required.

### 7. Validate Parameter Support

`validate="1:1"` is already used in `_build_dimension_mart()` (line 988 with `validate="m:1"`), so Polars version supports it. Keep `validate="1:1"` in feature joins for fail-fast behavior.

## Why OOM Still Happens Right Now

The OOM error occurs because the UI path (`user_datasets.save_zip_upload()`) calls `handler.build_unified_cohort()`, which still executes the legacy DuckDB SQL join. The logs show the giant SQL string being constructed and executed.

**M3/M4 being "green" in unit tests does not matter to runtime behavior** because the production call path never uses them. Until `build_unified_cohort()` is refactored, the legacy path will continue to execute.

After this refactor, the "169 GiB used" error should stop because:

- **Dimensions join stays bounded** (1:1-ish relationships only)
- **Facts/events are reduced to 1 row per grain** before any join happens
- **`validate="1:1"`** turns accidental explosions into a hard error instead of a surprise OOM

## Validation

The new implementation ensures:

- **No cartesian products**: Fact tables are aggregated before joining
- **Row count preservation**: `result.height == mart.height` (not anchor grain count, since dimension mart `join_type` can filter rows)
- **Fail-fast on errors**: Polars `validate="1:1"` will raise if row explosion detected
- **Backward compatible**: Same function signature, same return type
- **Deterministic**: Sorted feature joins ensure reproducible output (assuming upstream classification/aggregation is also deterministic)

## Testing

### High-Level Verification

After implementation, verify:

1. No more OOM errors on MIMIC-style schemas
2. Logs show "Building dimension mart..." and "Aggregating fact/event tables..." messages
3. No giant SQL queries in logs
4. Final row count equals anchor unique grain count (not exploded)

### Regression Tests

Add enforceable tests that catch regressions:

#### 1. Assertion: No DuckDB Mega-Join Path Executed

**Test**: Patch `duckdb.DuckDBPyConnection.execute` globally to detect legacy SQL pattern.

```python
def test_build_unified_cohort_does_not_use_legacy_duckdb_join():
    """Ensure build_unified_cohort() does not execute legacy DuckDB SQL join."""
    import duckdb
    
    original_execute = duckdb.DuckDBPyConnection.execute
    
    def guarded_execute(self, query, *args, **kwargs):
        q = str(query)
        if "SELECT * FROM" in q and "LEFT JOIN" in q:
            raise AssertionError(f"Legacy DuckDB mega-join detected: {q[:200]}...")
        return original_execute(self, query, *args, **kwargs)
    
    duckdb.DuckDBPyConnection.execute = guarded_execute
    
    try:
        # This should not trigger the legacy path
        result = handler.build_unified_cohort()
        assert result.height > 0
    finally:
        duckdb.DuckDBPyConnection.execute = original_execute
```

#### 2. Row-Count Invariants

**Test**: Verify feature joins do not change row count. Note: `result.height == mart.height`, not anchor grain count, since dimension mart `join_type` can filter rows.

```python
def test_feature_joins_preserve_row_count():
    """Feature joins should not change row count (1:1 validation)."""
    handler.detect_relationships()
    handler.classify_tables()
    
    anchor = handler._find_anchor_by_centrality()
    mart = handler._build_dimension_mart(anchor_table=anchor)
    mart_height = mart.select(pl.len().alias("n")).collect()["n"][0]
    
    result = handler.build_unified_cohort(anchor_table=anchor)
    
    assert result.height == mart_height, (
        f"Row count changed: {mart_height} -> {result.height}. "
        f"Feature joins may have caused row explosion."
    )
```

#### 3. Cardinality Failure Detection (Validate Proof Point)

**Test**: Self-contained test proving `validate="1:1"` works on LazyFrame.join().

```python
def test_lazy_join_validate_1_1_fails_on_duplicates():
    """Prove validate='1:1' works on LazyFrame.join() by testing duplicate keys."""
    left = pl.DataFrame({"patient_id": ["P1", "P2"]}).lazy()
    right = pl.DataFrame({"patient_id": ["P1", "P1"], "x": [1, 2]}).lazy()
    
    with pytest.raises(pl.exceptions.ComputeError):
        left.join(right, on="patient_id", how="left", validate="1:1").collect()
```

#### 4. Deterministic Columns

**Test**: Either set RNG seeds for upstream classification/aggregation, or test deterministic join ordering only.

**Option A (Preferred)**: Set seeds and ensure sorted iteration:

```python
def test_build_unified_cohort_deterministic_columns():
    """Unified cohort should have deterministic column order."""
    # Set any RNG seeds used in classification/aggregation
    # Ensure table iteration is sorted in _aggregate_fact_tables()
    
    result1 = handler.build_unified_cohort()
    result2 = handler.build_unified_cohort()
    
    assert result1.columns == result2.columns, (
        f"Column order not deterministic: "
        f"{result1.columns} != {result2.columns}"
    )
```

**Option B**: Test deterministic join ordering only (weaker signal):

```python
def test_feature_tables_joined_in_sorted_order():
    """Feature tables should be joined in sorted name order."""
    result = handler.build_unified_cohort()
    
    # Assert feature columns appear in sorted feature name order
    # (requires consistent feature naming pattern)
    feature_prefixes = [col.split("_")[0] for col in result.columns if "_features" in col]
    assert feature_prefixes == sorted(feature_prefixes)
```

#### 5. Join Type Validation

**Test**: Invalid `join_type` should raise, valid normalized values should pass.

```python
def test_build_unified_cohort_rejects_invalid_join_type():
    """Invalid join_type should raise ValueError."""
    with pytest.raises(ValueError, match="Unsupported join_type"):
        handler.build_unified_cohort(join_type="invalid")

def test_build_unified_cohort_accepts_case_insensitive_join_type():
    """join_type should normalize case-insensitively."""
    # "LEFT" should normalize to "left" and be accepted
    result = handler.build_unified_cohort(join_type="LEFT")
    assert result.height > 0
    
    result2 = handler.build_unified_cohort(join_type="left")
    assert result2.height == result.height
```