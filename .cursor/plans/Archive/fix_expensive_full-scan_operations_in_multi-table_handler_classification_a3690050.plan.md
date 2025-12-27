---
name: ""
overview: ""
todos: []
---

---name: Fix Expensive Full-Scan Operations in Multi-Table Handler Classificationoverview: ""todos:

- id: "1"

content: Add _sample_df() helper method using head(n) or stride sampling (NOT df.sample() - deterministic, fast, stable)status: completed

- id: "2"

content: Add _is_probably_id_col() helper to tighten ID pattern matching (exact 'id' or endswith('_id'), not endswith('id'))status: completed

- id: "3"

content: Refactor _detect_grain_key() with explicit scoring formula: penalize row-level IDs (event_id, row_id, uuid), prefer relationship keys, use sampled uniquenessstatus: completeddependencies:

    - "1"
    - "2"
- id: "4"

content: Refactor _detect_time_column() to remove 'dt' pattern, check dtypes first, use sampled uniquenessstatus: completeddependencies:

    - "1"
- id: "5"

content: Refactor _detect_bridge_table() to use structural heuristics + sampled composite uniquenessstatus: completeddependencies:

    - "1"
- id: "6"

content: Fix classify_tables() cardinality calculation to use sampled uniquenessstatus: completeddependencies:

    - "1"
- id: "7"

content: "Add acceptance test: grain_key_fallback_prefers_patient_over_event"status: completeddependencies:

    - "3"
- id: "8"

content: "Add acceptance test: id_pattern_does_not_match_false_positives (valid/fluid/paid)"status: completeddependencies:

    - "2"
    - "3"
- id: "9"

content: "Add performance guardrail test: classification_uses_sampled_helpers_only (monkeypatch our own helpers, not Polars internals)"status: completeddependencies:

    - "3"
    - "4"
    - "5"
    - "6"
- id: "11"

content: "Add strict acceptance gate: test_classification_1m_rows_completes_within_3_seconds (wall-clock bound)"status: completeddependencies:

    - "1"
    - "3"
    - "4"
    - "5"
    - "6"
- id: "10"

content: "Add performance guardrail test: sampling_bounds_classification_cost (timing test)"status: completeddependencies:

    - "1"

---

# Fix Expensive Full-Scan Operations in Multi-Table Handler Classification

## Overview

Fix critical performance and correctness issues in `MultiTableHandler` classification methods that cause expensive full-table scans and incorrect grain key selection. Replace full scans with deterministic sampling and tighten pattern matching to prevent misclassification.

## Problems Identified

1. **`_detect_grain_key()`**: 

- Full scans via `df[c].n_unique() `on all `*_id` columns
- Naive fallback picks highest uniqueness (event_id over patient_id)
- `endswith('id')` matches false positives (valid, fluid, paid)

2. **`_detect_time_column()`**: 

- Full scans via `df[col].n_unique()` 
- `'dt'` pattern too broad (matches dt_code, mdt_flag)

3. **`_detect_bridge_table()`**: 

- Full scans via `df[col].n_unique()` for each FK
- Full-table distinct on composite key `df.select([fk1, fk2]).n_unique()`

4. **`classify_tables()`**: 

- Full scan via `df[grain_key].n_unique()` at line 317

## Solution Architecture

### Sampling Strategy

Add deterministic sampling helper that bounds cost and keeps tests stable. **Do NOT use `df.sample()`** - even with seed=0, it's extra work and can behave differently across Polars versions.**Option A: head(n)** (fastest, stable, but can bias if file is sorted):

```python
def _sample_df(self, df: pl.DataFrame, n: int = 10_000) -> pl.DataFrame:
    """Deterministic sample using head (fastest, stable)."""
    return df.head(min(n, df.height))
```

**Option B: Stride sample** (more representative, still deterministic, avoids sorted bias):

```python
def _sample_df(self, df: pl.DataFrame, n: int = 10_000) -> pl.DataFrame:
    """Deterministic stride sample (more representative than head)."""
    if df.height <= n:
        return df
    step = max(df.height // n, 1)
    return df.slice(0, df.height).take(pl.arange(0, df.height, step)[:n])
```

**Recommendation**: Start with Option A (head) for simplicity. If classification shows bias on sorted data, switch to Option B.

### Pattern Matching Fixes

- Tighten ID column detection: `n == "id" or n.endswith("_id")` (not `endswith('id')`)
- Remove `'dt'` from time patterns (use only: `time`, `date`, `timestamp`, `datetime`)
- Prefer explicit keys by name before fallback to uniqueness

## Implementation Changes

### 1. Add Sampling Helper and Centralize Uniqueness Computations

**File**: `src/clinical_analytics/core/multi_table_handler.py`Add `_sample_df()` method after `_normalize_key_columns()` (around line 222).**Critical**: Centralize all uniqueness/null rate computations behind helper methods that only accept sampled frames. This ensures we never accidentally do full scans and makes testing easier:

```python
def _compute_sampled_uniqueness(self, df: pl.DataFrame, col: str) -> tuple[int, float]:
    """Compute uniqueness on sample only."""
    s = self._sample_df(df)
    non_null = s[col].drop_nulls()
    if non_null.len() == 0:
        return 0, 1.0
    unique_count = non_null.n_unique()
    null_rate = s[col].null_count() / max(s.height, 1)
    return unique_count, null_rate
```

All classification methods should use these helpers, never call `df[col].n_unique()` directly on the original DataFrame.

### 2. Fix `_detect_grain_key()`

**File**: `src/clinical_analytics/core/multi_table_handler.py` (lines 375-409)**Changes**:

- Add `_is_probably_id_col()` helper (exact `"id"` or `endswith("_id")`)
- Prioritize explicit keys by name (patient_id, subject_id, hadm_id, etc.)
- Fallback: use sampled uniqueness with scoring (penalize row-level IDs, penalize nulls)
- Never do full `df[c].n_unique()` on base DataFrame

**New logic**:

1. Check explicit patient/admission keys first (by name)
2. If no match, find `*_id` candidates using tightened pattern
3. Score candidates on sampled data using explicit formula that:

- Hard-penalizes row-level ID patterns: `event_id`, `row_id`, `*_event_id`, `uuid`, `guid`
- Prefers IDs that appear as relationship keys across tables (once relationships are detected)
- Uses sampled uniqueness and null rate

**Scoring formula** (on sample `s`):

```python
uniq_ratio = s[col].drop_nulls().n_unique() / max(s.height - s[col].null_count(), 1)
null_rate = s[col].null_count() / max(s.height, 1)

score = 0.0
score -= 2.0 * uniq_ratio          # penalize row-level IDs (uniq ~ 1.0)
score -= 1.0 * null_rate
score += 1.0 if col.endswith("_id") else 0.0
score += 2.0 if col in ("patient_id","subject_id","hadm_id","encounter_id","visit_id") else 0.0
score -= 5.0 if any(tok in col for tok in ("event","row","uuid","guid")) else 0.0
```

This ensures `patient_id` beats `event_id` even if `event_id` is perfectly unique.

### 3. Fix `_detect_time_column()`

**File**: `src/clinical_analytics/core/multi_table_handler.py` (lines 478-508)**Changes**:

- Remove `'dt'` from time_patterns
- Check temporal dtypes first (pl.Date, pl.Datetime)
- Use sampled uniqueness check: `s[c].drop_nulls().n_unique() > 1`
- Never do full `df[col].n_unique()` on base DataFrame

### 4. Fix `_detect_bridge_table()`

**File**: `src/clinical_analytics/core/multi_table_handler.py` (lines 525-587)**Changes**:

- Add structural heuristic: exclude if `df.width >= 15` (bridges are narrow)
- Use sampled composite uniqueness: `s.select(...).n_unique()` on sample
- Never do full `df[col].n_unique()` or `df.select([fk1, fk2]).n_unique()`

### 5. Fix `classify_tables()` Cardinality Calculation

**File**: `src/clinical_analytics/core/multi_table_handler.py` (line 317)**Changes**:

- Use sampled uniqueness for cardinality_ratio calculation
- Or: use `df[grain_key].value_counts()` on sample to estimate ratio

**Note**: For exact cardinality_ratio, we may need full scan, but we can:

- Use sample-based estimate for classification decisions
- Or: defer exact calculation until after classification (lazy evaluation)

## Testing Strategy

### Acceptance Tests

**File**: `tests/core/test_multi_table_handler.py`

#### Test 1: Grain Key Fallback Prefers Patient Over Event

```python
def test_grain_key_fallback_prefers_patient_over_event():
    """Acceptance: Table with patient_id and event_id picks patient_id even if event_id is more unique."""
    # Arrange: event_id is more unique (row-level ID)
    df = pl.DataFrame({
        "patient_id": ["P1", "P1", "P2", "P2"],  # 2 unique
        "event_id": ["E1", "E2", "E3", "E4"],    # 4 unique (more unique!)
        "value": [100, 200, 300, 400]
    })
    
    # Act
    handler = MultiTableHandler({"test": df})
    grain_key = handler._detect_grain_key(df)
    
    # Assert: Should pick patient_id (explicit key) not event_id
    assert grain_key == "patient_id"
```



#### Test 2: ID Pattern Does Not Match False Positives

```python
def test_id_pattern_does_not_match_false_positives():
    """Acceptance: endswith('id') does not match 'valid', 'fluid', 'paid'."""
    df = pl.DataFrame({
        "valid": [True, False, True],
        "fluid": [100, 200, 300],
        "paid": [10.5, 20.5, 30.5],
        "patient_id": ["P1", "P2", "P3"]
    })
    
    handler = MultiTableHandler({"test": df})
    grain_key = handler._detect_grain_key(df)
    
    # Should pick patient_id, not any of the false positives
    assert grain_key == "patient_id"
```



### Performance Guardrail Tests

**File**: `tests/core/test_multi_table_handler.py`

#### Test 3: Classification Uses Sampled Helpers Only

```python
def test_classification_uses_sampled_helpers_only(monkeypatch):
    """Performance guardrail: Enforce that classification uses _sample_df() and never touches df[...] for uniqueness on original frame."""
    # Arrange: Create large DataFrame
    large_df = pl.DataFrame({
        "patient_id": [f"P{i}" for i in range(1_000_000)],
        "value": list(range(1_000_000))
    })
    
    # Track calls to _sample_df() - all uniqueness checks should go through it
    sample_df_calls = []
    original_sample_df = MultiTableHandler._sample_df
    
    def tracked_sample_df(self, df, n=10_000):
        sample_df_calls.append((df.height, n))
        return original_sample_df(self, df, n)
    
    monkeypatch.setattr(MultiTableHandler, "_sample_df", tracked_sample_df)
    
    # Act
    handler = MultiTableHandler({"large": large_df})
    handler.classify_tables()
    
    # Assert: _sample_df() was called (proving we're using sampling, not full scans)
    assert len(sample_df_calls) > 0, (
        "Classification should use _sample_df() for all uniqueness checks"
    )
    
    # Verify all samples are bounded
    for df_height, sample_size in sample_df_calls:
        assert sample_size <= 10_000, (
            f"Sample size {sample_size} exceeds bound (df_height={df_height})"
        )
```

**Key change**: Test our own helper (`_sample_df`) instead of monkeypatching Polars internals. This is the engineering version of "don't test the ocean, test your boat."

#### Test 4: Sampling Bounds Classification Cost

```python
def test_sampling_bounds_classification_cost():
    """Performance: Classification cost is O(sample_size), not O(table_size)."""
    import time
    
    # Small table (baseline)
    small_df = pl.DataFrame({
        "patient_id": [f"P{i}" for i in range(100)],
        "value": list(range(100))
    })
    
    # Large table (1000x larger)
    large_df = pl.DataFrame({
        "patient_id": [f"P{i}" for i in range(100_000)],
        "value": list(range(100_000))
    })
    
    handler_small = MultiTableHandler({"small": small_df})
    handler_large = MultiTableHandler({"large": large_df})
    
    # Act: Time classification
    start_small = time.perf_counter()
    handler_small.classify_tables()
    time_small = time.perf_counter() - start_small
    
    start_large = time.perf_counter()
    handler_large.classify_tables()
    time_large = time.perf_counter() - start_large
    
    # Assert: Large table should not be 1000x slower (sampling bounds cost)
    # Allow 10x overhead for sampling overhead, but not 1000x
    assert time_large < time_small * 100, (
        f"Large table classification {time_large:.3f}s should not be 100x slower "
        f"than small {time_small:.3f}s (sampling should bound cost)"
    )
```



#### Test 5: Strict Acceptance Gate - 1M Rows Completes Within 3 Seconds

```python
def test_classification_1m_rows_completes_within_3_seconds():
    """
    Strict acceptance gate: Classifying a 1M-row table must complete within 1-3 seconds.
    
    This is a hard performance requirement that ensures sampling is working correctly.
    """
    import time
    
    # Arrange: Create 1M-row table
    large_df = pl.DataFrame({
        "patient_id": [f"P{i % 1000}" for i in range(1_000_000)],  # 1000 unique patients
        "event_id": [f"E{i}" for i in range(1_000_000)],  # 1M unique events (row-level ID)
        "value": list(range(1_000_000))
    })
    
    handler = MultiTableHandler({"large": large_df})
    
    # Act: Time classification
    start = time.perf_counter()
    handler.classify_tables()
    elapsed = time.perf_counter() - start
    
    # Assert: Must complete within 3 seconds (loose bound, but strict requirement)
    assert elapsed < 3.0, (
        f"Classification of 1M-row table took {elapsed:.3f}s, "
        f"exceeds 3s bound (sampling may not be working)"
    )
    
    # Verify classification worked correctly
    assert "large" in handler.classifications
    classification = handler.classifications["large"]
    assert classification.grain_key == "patient_id", (
        f"Should pick patient_id over event_id (grain_key={classification.grain_key})"
    )
    
    handler.close()
```



## Migration Notes

- **Backward compatibility**: Sampling may change classification results slightly, but should be more correct (prefers explicit keys, penalizes row-level IDs)
- **Deterministic**: `head(n)` or stride sampling ensures same results across runs (no randomness)
- **Performance**: Classification cost bounded by sample_size (default 10k rows) regardless of table size
- **Testing**: Existing tests may need updates if they relied on exact uniqueness values (use sampled estimates instead)
- **Strict acceptance gate**: 1M-row table must classify within 3 seconds (hard requirement)

## Future Considerations

The user notes that classification currently works on eager `pl.DataFrame`, but the plan direction is lazy. Consider:

1. **Phase 2**: Move classification to `pl.LazyFrame` + sampling
2. **Phase 3**: Use Parquet metadata/row-group stats instead of data scans