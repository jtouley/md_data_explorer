# Quick Wins: High-Impact, Low-Effort Fixes

**Target:** Fixes that can be completed in <2 hours each
**Total Estimated Time:** 6-8 hours
**Impact:** Immediate improvement to code quality and security

---

## 1. Remove Premature collect() (30 minutes)

**File:** `src/clinical_analytics/core/multi_table_handler.py:1237`
**Impact:** Better performance, keeps pipeline lazy
**Difficulty:** ⭐ Very Easy

**Current:**
```python
if not feature_tables:
    return mart.collect()  # ❌ Premature materialization
```

**Fix:**
```python
if not feature_tables:
    return mart  # ✅ Keep lazy, caller decides when to collect
```

**Testing:** Run existing multi-table tests, verify same results

---

## 2. Fix Class Attribute Mutation (1 hour)

**File:** `src/clinical_analytics/core/dataset.py:31`
**Impact:** Prevents bugs with multiple dataset instances
**Difficulty:** ⭐⭐ Easy

**Current:**
```python
semantic: Optional["SemanticLayer"] = None  # Shared!
```

**Fix:**
```python
def __init__(self, ...):
    self._semantic: Optional[SemanticLayer] = None

@property
def semantic(self) -> SemanticLayer:
    if self._semantic is None:
        raise ValueError("Call load() first")
    return self._semantic
```

---

## 3. Add SQL Injection Protection (1 hour)

**File:** `src/clinical_analytics/core/semantic.py:202`
**Impact:** CRITICAL security fix
**Difficulty:** ⭐ Very Easy

**Current:**
```python
f"CREATE TABLE {table_name} AS ..."  # Vulnerable
```

**Fix:**
```python
f'CREATE TABLE "{table_name}" AS ...'  # Quoted identifier
```

**Apply to:** All 5 locations where table_name is interpolated

---

## 4. Replace print() with logger (2 hours)

**Files:** 12 files use `print()` instead of `logger`
**Impact:** Proper logging, debuggable in production
**Difficulty:** ⭐ Very Easy

**Find & Replace:**
```bash
# Find all print() calls:
grep -r "print(" src/ --include="*.py"

# Replace with logger.info/warning/error
```

**Example:**
```python
# Before:
print(f"Warning: {message}")

# After:
logger.warning(message)
```

---

## 5. Use Vectorized Operations in stats.py (1 hour)

**File:** `src/clinical_analytics/analysis/stats.py:46-47`
**Impact:** 10-50x faster execution
**Difficulty:** ⭐ Very Easy

**Current:**
```python
odds_ratios = params.apply(lambda x: np.exp(x))  # Row-wise
conf_or = conf.apply(lambda x: np.exp(x))
```

**Fix:**
```python
odds_ratios = np.exp(params)  # Vectorized
conf_or = np.exp(conf)
```

---

## 6. Remove Unused imports (1 hour)

**Files:** Multiple files have unused imports
**Impact:** Cleaner code, faster startup
**Difficulty:** ⭐ Very Easy

**Tool:** Use `ruff` or `autoflake`:
```bash
# Install ruff
uv pip install ruff

# Remove unused imports
ruff check --select F401 --fix src/
```

---

## 7. Add Type Hints to stats.py (1 hour)

**File:** `src/clinical_analytics/analysis/stats.py:7-11`
**Impact:** Better IDE support, type safety
**Difficulty:** ⭐⭐ Easy

**Current:**
```python
def run_logistic_regression(
    df: pd.DataFrame,
    outcome_col: str,
    predictors: List[str]
) -> Tuple[Any, pd.DataFrame]:  # ❌ Any
```

**Fix:**
```python
from statsmodels.discrete.discrete_model import BinaryResultsWrapper

def run_logistic_regression(
    df: pd.DataFrame,
    outcome_col: str,
    predictors: List[str]
) -> Tuple[BinaryResultsWrapper, pd.DataFrame]:  # ✅ Specific type
```

---

## 8. Consolidate Error Messages (1-2 hours)

**Files:** Multiple UI pages repeat error messages
**Impact:** DRY principle, consistent messaging
**Difficulty:** ⭐⭐ Easy

**Create:** `src/clinical_analytics/ui/messages.py`
```python
class UIMessages:
    """Centralized error/warning messages for UI."""

    NO_DATASETS = "No datasets available. Please upload data first."
    VALIDATION_FAILED = "Dataset validation failed. Please check data quality."
    SELECT_VARIABLES = "Please select both outcome and predictor variables."
    UPLOAD_SUCCESS = "Dataset uploaded successfully"
    UPLOAD_FAILED = "Upload failed: {error}"
```

**Usage:**
```python
from clinical_analytics.ui.messages import UIMessages

st.error(UIMessages.NO_DATASETS)
st.success(UIMessages.UPLOAD_SUCCESS)
```

---

## Quick Win Checklist

### Total Time: ~8 hours
- [ ] Remove premature collect() (30 min)
- [ ] Fix class attribute mutation (1 hour)
- [ ] Add SQL injection protection (1 hour)
- [ ] Replace print() with logger (2 hours)
- [ ] Use vectorized operations (1 hour)
- [ ] Remove unused imports (1 hour)
- [ ] Add type hints to stats.py (1 hour)
- [ ] Consolidate error messages (1-2 hours)

### Expected Impact
- ✅ Improved security (SQL injection fix)
- ✅ Better performance (vectorized ops, lazy evaluation)
- ✅ Cleaner code (no unused imports)
- ✅ Better debugging (proper logging)
- ✅ Type safety (complete type hints)
- ✅ DRY principle (centralized messages)

### Can Be Done In One Day
All these fixes can be completed by one engineer in a single focused day, providing immediate value with minimal risk.
