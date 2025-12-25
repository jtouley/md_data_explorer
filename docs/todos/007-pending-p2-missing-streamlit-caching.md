---
status: pending
priority: p2
issue_id: "007"
tags: [code-review, performance, high-priority]
dependencies: []
estimated_effort: small
created_date: 2025-12-24
---

# Missing Streamlit Caching for Expensive Operations

## Problem Statement

The Streamlit UI (`src/clinical_analytics/ui/app.py`) performs expensive operations on **every widget interaction** without caching. Loading datasets, running analyses, and generating visualizations execute repeatedly, causing poor user experience and wasted compute resources.

**Why it matters:**
- Dataset loading takes 2-5 seconds on each interaction
- Statistical analyses recompute unnecessarily
- Users experience lag on every slider/button click
- Wastes compute resources on duplicate work

**Impact:** Poor UX, slow application, inefficient resource usage

## Findings

**Location:** `src/clinical_analytics/ui/app.py` (entire file)

**Uncached Expensive Operations:**

1. **Dataset Loading (Lines 119-140):**
```python
def load_dataset(dataset_name: str):
    """NO CACHING - reloads on every widget interaction."""
    dataset = get_dataset(dataset_name)  # Expensive
    dataset.load()  # Reads entire CSV
    return dataset
```

2. **Statistical Analysis (Lines 309-333):**
```python
# NO CACHING - recomputes regression on every interaction
if analysis_type == "Logistic Regression":
    result = logistic_regression(
        df=cohort_df,
        outcome_col=outcome,
        predictors=predictors
    )  # Expensive computation repeated
```

3. **Data Profiling (Lines 463-492):**
```python
# NO CACHING - regenerates profile on every interaction
profile = generate_profile_report(df)  # Very expensive
st.write(profile)
```

**Performance Impact:**
```
Current behavior (without caching):
- Select dataset → Load CSV (2s)
- Change filter → Reload CSV (2s)
- Adjust slider → Reload CSV (2s)
- Click button → Reload CSV (2s)

Expected behavior (with caching):
- Select dataset → Load CSV (2s, cached)
- Change filter → Use cached data (0.1s)
- Adjust slider → Use cached data (0.1s)
- Click button → Use cached data (0.1s)
```

## Proposed Solutions

### Solution 1: Streamlit @st.cache_data Decorator (Recommended)
**Pros:**
- Built-in Streamlit solution
- Simple decorator syntax
- Automatic cache invalidation
- TTL support for stale data

**Cons:**
- Need to ensure functions are pure
- Must handle cache misses gracefully

**Effort:** Small (2 hours)
**Risk:** Low

**Implementation:**
```python
import streamlit as st
from functools import lru_cache

# Cache dataset loading
@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_dataset(dataset_name: str) -> pd.DataFrame:
    """Load dataset with caching."""
    dataset = get_dataset(dataset_name)
    dataset.load()
    return dataset.df

# Cache cohort generation
@st.cache_data
def get_cohort_cached(
    dataset_name: str,
    filters: dict
) -> pd.DataFrame:
    """Get cohort with caching based on filters."""
    dataset = load_dataset(dataset_name)
    # Convert dict to hashable tuple for cache key
    filter_key = tuple(sorted(filters.items()))
    return dataset.get_cohort(**filters)

# Cache statistical analysis
@st.cache_data
def run_logistic_regression_cached(
    df_hash: str,  # Use hash of dataframe as cache key
    outcome_col: str,
    predictors: tuple,  # Must be hashable
    covariates: tuple = None
):
    """Run logistic regression with caching."""
    # Note: df passed separately, hash used for caching
    return logistic_regression(
        df=st.session_state['current_df'],
        outcome_col=outcome_col,
        predictors=list(predictors),
        covariates=list(covariates) if covariates else None
    )

# Cache data profiling
@st.cache_data(ttl=7200)  # Cache for 2 hours
def generate_profile_cached(df_hash: str):
    """Generate profile report with caching."""
    df = st.session_state['current_df']
    return generate_profile_report(df)

# Usage in main app
def main():
    st.title("Clinical Analytics Platform")

    # Cached dataset load
    dataset_choice = st.sidebar.selectbox("Dataset", datasets)
    df = load_dataset(dataset_choice)  # Only loads once

    # Cached cohort generation
    filters = build_filters(df)
    cohort = get_cohort_cached(dataset_choice, filters)  # Cached by filters

    # Cached analysis
    if st.button("Run Regression"):
        df_hash = hash(tuple(cohort.values.flatten()))
        result = run_logistic_regression_cached(
            df_hash=str(df_hash),
            outcome_col=outcome,
            predictors=tuple(predictors)
        )
```

### Solution 2: Manual Session State Caching
**Pros:**
- More control over cache behavior
- Can implement custom eviction policies
- Works for non-serializable objects

**Cons:**
- More code to maintain
- Manual cache invalidation logic
- Error-prone

**Effort:** Medium (4 hours)
**Risk:** Medium

### Solution 3: External Cache (Redis)
**Pros:**
- Shared across Streamlit instances
- Persistent cache
- Advanced cache strategies

**Cons:**
- Additional infrastructure dependency
- Overkill for single-user app
- More complex deployment

**Effort:** Large (8 hours)
**Risk:** Medium

## Recommended Action

**Implement Solution 1** with these caching strategies:

1. **Dataset loading:** Cache with 1-hour TTL
2. **Cohort generation:** Cache by dataset + filters
3. **Statistical analysis:** Cache by data hash + parameters
4. **Visualizations:** Cache by data hash + plot params
5. **Data profiling:** Cache with 2-hour TTL

**Cache Key Design:**
```python
# Good cache keys (deterministic, hashable)
@st.cache_data
def load_dataset(dataset_name: str):  # String is hashable
    pass

@st.cache_data
def get_cohort(dataset_name: str, age_min: int, age_max: int):  # All hashable
    pass

# Bad cache keys (non-hashable, mutable)
@st.cache_data
def process_data(df: pd.DataFrame):  # DataFrame not hashable - use hash
    pass

# Solution: Use hash of DataFrame
@st.cache_data
def process_data(df_hash: str):
    df = st.session_state['current_df']
    pass
```

## Technical Details

**Affected Files:**
- `src/clinical_analytics/ui/app.py` (add caching decorators)

**Cache Configuration:**
```python
# At top of app.py
import streamlit as st

# Configure cache settings
st.set_page_config(
    page_title="Clinical Analytics",
    layout="wide"
)

# Clear cache button in sidebar
if st.sidebar.button("Clear Cache"):
    st.cache_data.clear()
    st.rerun()
```

**Functions to Cache:**
1. `load_dataset(dataset_name)` - TTL 1 hour
2. `get_cohort_cached(dataset, filters)` - No TTL (invalidate on filter change)
3. `run_logistic_regression_cached(df_hash, params)` - No TTL
4. `run_survival_analysis_cached(df_hash, params)` - No TTL
5. `generate_profile_cached(df_hash)` - TTL 2 hours
6. `create_plot_cached(df_hash, plot_type, params)` - No TTL

## Acceptance Criteria

- [ ] Dataset loading cached with @st.cache_data
- [ ] Cohort generation cached by filters
- [ ] Statistical analyses cached by data hash + parameters
- [ ] Data profiling cached with TTL
- [ ] Visualizations cached by parameters
- [ ] Cache clear button in UI
- [ ] Page loads <1s after initial dataset load
- [ ] Widget interactions <200ms response time
- [ ] Cache hit rate >80% during typical session
- [ ] Memory usage monitored (cache doesn't grow unbounded)
- [ ] Documentation updated with caching strategy

## Work Log

### 2025-12-24
- **Action:** Performance review identified missing caching
- **Learning:** Streamlit reruns entire script on every interaction
- **Next:** Add @st.cache_data decorators to expensive operations

## Resources

- **Streamlit Caching:** https://docs.streamlit.io/library/advanced-features/caching
- **Cache Decorators:** https://docs.streamlit.io/library/api-reference/performance/st.cache_data
- **Streamlit Performance:** https://docs.streamlit.io/library/advanced-features/performance
- **Related Finding:** Performance optimization for Polars conversions (todo #008)
