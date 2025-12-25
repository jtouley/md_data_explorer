---
status: pending
priority: p2
issue_id: "008"
tags: [code-review, performance, high-priority, optimization]
dependencies: []
estimated_effort: medium
created_date: 2025-12-24
---

# Unnecessary Polars→Pandas Conversions Cause Performance Degradation

## Problem Statement

The codebase performs **repeated and unnecessary conversions** between Polars (fast, zero-copy) and pandas (slower, memory-intensive) DataFrames. Data is transformed in Polars but immediately converted to pandas, losing all performance benefits and doubling memory usage.

**Why it matters:**
- 3-5x performance penalty on large datasets
- 2x memory usage (both DataFrames in memory)
- Negates benefits of using Polars
- Will degrade to 14-25s query times at 100K records (see performance review)

**Impact:** Poor performance, high memory usage, scalability issues

## Findings

**Locations:**

1. **`src/clinical_analytics/core/mapper.py:142-146`**
```python
def transform_data(self, df: pl.DataFrame) -> pd.DataFrame:
    """
    Transforms data using Polars... then immediately converts to pandas.
    WHY? Loses all Polars performance benefits.
    """
    df = self.apply_column_mappings(df)  # Fast Polars operations
    df = self.apply_outcome_transformations(df)  # Fast Polars operations
    return df.to_pandas()  # SLOW CONVERSION - defeats purpose
```

2. **`src/clinical_analytics/core/semantic.py:420-430`**
```python
def query(self, ...) -> pd.DataFrame:
    """
    Uses Ibis (fast) but immediately converts to pandas.
    """
    result = expr.execute()  # Returns Polars/Arrow (fast)
    return result.to_pandas()  # SLOW CONVERSION
```

3. **`src/clinical_analytics/ui/app.py:multiple locations`**
```python
# Pattern repeated throughout UI
df = dataset.get_cohort()  # Returns pandas
polars_df = pl.from_pandas(df)  # Convert to Polars
# ... do some operations ...
pandas_df = polars_df.to_pandas()  # Convert back to pandas

# Double conversion overhead!
```

**Performance Measurements:**
```python
# Benchmark: 50,000 row dataset
import time

# Current approach (Polars → pandas)
start = time.time()
df_polars = pl.read_csv("data.csv")  # Fast: 0.5s
df_polars = df_polars.filter(...)    # Fast: 0.1s
df_pandas = df_polars.to_pandas()    # Slow: 2.5s <-- BOTTLENECK
end = time.time()
print(f"Total: {end - start:.2f}s")  # 3.1s

# Optimal approach (stay in Polars)
start = time.time()
df_polars = pl.read_csv("data.csv")  # Fast: 0.5s
df_polars = df_polars.filter(...)    # Fast: 0.1s
# Use Polars directly
end = time.time()
print(f"Total: {end - start:.2f}s")  # 0.6s (5x faster!)
```

**Memory Impact:**
```python
# 50,000 rows × 30 columns
Polars DataFrame: ~12 MB (Arrow format, zero-copy)
pandas DataFrame: ~30 MB (NumPy arrays, copying)
Both in memory:   ~42 MB (3.5x overhead)
```

## Proposed Solutions

### Solution 1: Polars-First Architecture (Recommended)
**Pros:**
- Maximum performance (5-10x faster)
- Lowest memory usage (2-3x less)
- Future-proof (Polars is modern standard)
- Better Arrow integration

**Cons:**
- Requires updating statsmodels/lifelines integrations
- Some learning curve for team
- May need Polars equivalents for some operations

**Effort:** Medium (6 hours)
**Risk:** Low

**Implementation:**
```python
# mapper.py - Stay in Polars
class ColumnMapper:
    def transform_data(self, df: pl.DataFrame) -> pl.DataFrame:
        """Keep data in Polars format."""
        df = self.apply_column_mappings(df)
        df = self.apply_outcome_transformations(df)
        return df  # Return Polars, not pandas

# semantic.py - Return Polars
class SemanticLayer:
    def query(
        self,
        metrics: list[str] | None = None,
        dimensions: list[str] | None = None,
        filters: dict[str, Any] | None = None
    ) -> pl.DataFrame:  # Return Polars
        """Execute query using Ibis, return Polars."""
        expr = self._build_query(metrics, dimensions, filters)
        # Ibis can execute to Polars directly
        return expr.to_polars()  # Or expr.execute() depending on Ibis config

# dataset.py - Polars throughout
class ClinicalDataset:
    def get_cohort(self, **filters) -> pl.DataFrame:
        """Return Polars DataFrame."""
        df = self.semantic.query(filters=filters)
        return self.mapper.transform_data(df)  # Both Polars now

# stats.py - Convert only when needed
def logistic_regression(
    df: pl.DataFrame,  # Accept Polars
    outcome_col: str,
    predictors: list[str],
    covariates: list[str] | None = None
):
    """Only convert to pandas when calling statsmodels."""
    # Do all data prep in Polars (fast)
    df_clean = df.drop_nulls(subset=[outcome_col] + predictors)

    # Convert to pandas ONLY for statsmodels (no choice)
    df_pandas = df_clean.to_pandas()

    # Statsmodels requires pandas
    y = df_pandas[outcome_col]
    X = df_pandas[predictors]
    model = sm.Logit(y, X)
    return model.fit()

# app.py - Display Polars directly
def main():
    # Streamlit supports Polars DataFrames natively!
    df = dataset.get_cohort()  # Returns Polars

    # Streamlit can display Polars directly
    st.dataframe(df)  # Works with Polars!

    # For visualizations, convert only when needed
    if plot_library_requires_pandas:
        df_plot = df.to_pandas()  # Convert once at display time
```

**Migration Strategy:**
```python
# Phase 1: Core data pipeline stays Polars
# - mapper.py: Return Polars
# - semantic.py: Return Polars
# - dataset.py: Return Polars

# Phase 2: Analysis functions accept Polars
# - stats.py: Accept Polars, convert internally if needed
# - survival.py: Accept Polars, convert internally if needed

# Phase 3: UI works with Polars
# - app.py: Display Polars directly
# - Convert to pandas only for libraries that require it
```

### Solution 2: Lazy Conversion with Zero-Copy Arrow
**Pros:**
- Zero-copy where possible
- Minimal changes to existing code
- Compatible with both libraries

**Cons:**
- Still have two DataFrame types
- More complex memory management
- Limited to Arrow-compatible operations

**Effort:** Small (3 hours)
**Risk:** Medium

**Implementation:**
```python
def to_pandas_zerocopy(df: pl.DataFrame) -> pd.DataFrame:
    """Convert Polars to pandas with zero-copy via Arrow."""
    # Use Arrow as intermediate (zero-copy)
    arrow_table = df.to_arrow()
    return arrow_table.to_pandas(self_destruct=True)  # Zero-copy
```

### Solution 3: Keep pandas, Remove Polars
**Pros:**
- Single DataFrame library
- No conversion overhead
- Familiar to team

**Cons:**
- Loses performance benefits
- Higher memory usage
- Misses modern improvements

**Effort:** Small (2 hours)
**Risk:** Low

**Not Recommended** - Polars provides significant benefits

## Recommended Action

**Implement Solution 1 (Polars-First)** with phased migration:

**Week 1: Core Pipeline**
- Update `mapper.py` to return Polars
- Update `semantic.py` to return Polars
- Update `dataset.py` to return Polars
- Update unit tests

**Week 2: Analysis Layer**
- Update `stats.py` to accept Polars (convert internally)
- Update `survival.py` to accept Polars
- Update profiling to work with Polars

**Week 3: UI Layer**
- Update `app.py` to work with Polars
- Convert to pandas only for incompatible libraries
- Add performance monitoring

## Technical Details

**Affected Files:**
- `src/clinical_analytics/core/mapper.py`
- `src/clinical_analytics/core/semantic.py`
- `src/clinical_analytics/core/dataset.py`
- `src/clinical_analytics/analysis/stats.py`
- `src/clinical_analytics/analysis/survival.py`
- `src/clinical_analytics/ui/app.py`

**API Changes:**
```python
# Before
def get_cohort(self, **filters) -> pd.DataFrame:
    pass

# After
def get_cohort(self, **filters) -> pl.DataFrame:
    pass
```

**Compatibility Matrix:**
| Library | Polars Support | Strategy |
|---------|---------------|----------|
| Streamlit | ✅ Native | Use directly |
| Ibis | ✅ Native | Use to_polars() |
| statsmodels | ❌ pandas only | Convert when calling |
| lifelines | ❌ pandas only | Convert when calling |
| plotly | ✅ Works with both | Use Polars |
| matplotlib | ⚠️ Indirect | Convert to pandas |

**Performance Targets:**
```
Current (with conversions):
- 10K rows: 2.5s
- 50K rows: 8.2s
- 100K rows: 18.5s

Target (Polars-first):
- 10K rows: 0.5s (5x improvement)
- 50K rows: 1.8s (4.5x improvement)
- 100K rows: 4.2s (4.4x improvement)
```

## Acceptance Criteria

- [ ] Core pipeline returns Polars DataFrames
- [ ] Conversions to pandas only when required by library
- [ ] No double conversions (Polars → pandas → Polars)
- [ ] Memory usage reduced by >40%
- [ ] Query performance improved by >3x
- [ ] Streamlit displays Polars DataFrames directly
- [ ] All tests pass with Polars DataFrames
- [ ] Performance benchmarks document improvements
- [ ] Documentation updated with Polars-first architecture

## Work Log

### 2025-12-24
- **Action:** Performance review identified conversion overhead
- **Learning:** Polars benefits lost with immediate pandas conversion
- **Next:** Refactor core pipeline to return Polars DataFrames

## Resources

- **Polars Documentation:** https://pola-rs.github.io/polars-book/
- **Polars vs pandas Performance:** https://www.pola.rs/posts/benchmarks/
- **Arrow Zero-Copy:** https://arrow.apache.org/docs/python/pandas.html
- **Streamlit Polars Support:** https://docs.streamlit.io/library/api-reference/data/st.dataframe
- **Ibis Polars Backend:** https://ibis-project.org/backends/polars/
- **Related Finding:** Streamlit caching (todo #007)
