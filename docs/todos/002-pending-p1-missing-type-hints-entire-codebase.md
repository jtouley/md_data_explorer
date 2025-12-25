---
status: pending
priority: p1
issue_id: "002"
tags: [code-review, python-quality, type-hints, maintainability]
dependencies: []
estimated_effort: large
created_date: 2025-12-24
---

# Missing Type Hints Across Entire Codebase

## Problem Statement

**Complete failure in type hint coverage** across the entire codebase (~2,846 lines of Python). No functions have proper type annotations, using old `typing.List/Dict` instead of modern `list/dict`, and extensive use of `Any` types. This is unacceptable for Python 3.11+ code and blocks IDE support, type checking, and maintainability.

**Why it matters:**
- No IDE autocomplete or type checking
- Runtime type errors not caught during development
- Difficult for new contributors to understand APIs
- MyPy configured with `disallow_untyped_defs = false` (permissive)

**Impact:** Code quality grade F, maintenance nightmare, runtime errors

## Findings

**Critical Files Without Type Hints:**

1. **`src/clinical_analytics/core/semantic.py` (490 lines)**
   - Line 26-44: `__init__` missing return type
   - Line 88-161: `get_base_view()` missing return type
   - Line 163-213: `apply_filters()` missing parameter type for `view`
   - Line 417-488: `query()` using old `List[str]`, `Dict[str, Any]`

2. **`src/clinical_analytics/ui/app.py` (497 lines)**
   - ZERO type hints on any function
   - Lines 21-116, 119-188, 309-333, 463-492

3. **`src/clinical_analytics/core/mapper.py` (451 lines)**
   - Using old typing module everywhere
   - `Dict` instead of `dict`, `List` instead of `list`

4. **`src/clinical_analytics/analysis/stats.py`**
   - Using `Any` for model return type instead of `LogitResults`
   - Using `Tuple` instead of `tuple`

**Evidence:**
```python
# BAD - Current code
def query(
    self,
    metrics: Optional[List[str]] = None,  # Old typing
    dimensions: Optional[List[str]] = None,
    filters: Optional[Dict[str, Any]] = None
) -> pd.DataFrame:

# GOOD - Should be
def query(
    self,
    metrics: list[str] | None = None,  # Modern Python 3.10+
    dimensions: list[str] | None = None,
    filters: dict[str, Any] | None = None
) -> pd.DataFrame:
```

## Proposed Solutions

### Solution 1: Systematic Module-by-Module Addition (Recommended)
**Pros:**
- Thorough and complete
- Can be done incrementally
- Tests verify correctness

**Cons:**
- Time-consuming (20-30 hours total)
- Requires understanding all APIs

**Effort:** Large (25 hours)
**Risk:** Low

**Priority Order:**
1. Core modules (dataset.py, schema.py, registry.py) - 4 hours
2. Mapper and semantic layer - 6 hours
3. Analysis modules (stats.py, survival.py) - 4 hours
4. Dataset implementations - 4 hours
5. UI modules - 6 hours
6. Loaders - 2 hours

### Solution 2: Automated Tool + Manual Review
**Pros:**
- Faster initial pass
- Tools like MonkeyType can infer types from runtime
- Still requires human review

**Cons:**
- Generated types may be incorrect
- Requires good test coverage for inference
- Still significant manual work

**Effort:** Medium (15 hours)
**Risk:** Medium (incorrect inferences)

**Tools:**
- MonkeyType: Runtime type collection
- Pytype: Google's type inferencer
- mypy --suggest: Type suggestion mode

### Solution 3: Enforce via MyPy Strict Mode + Fix
**Pros:**
- Forces addressing all type issues
- Catches problems immediately
- No partial solutions

**Cons:**
- Blocks all development until fixed
- All-or-nothing approach

**Effort:** Large (20 hours)
**Risk:** High (blocks work)

## Recommended Action

**Solution 1** with phased implementation:

**Phase 1 (Week 1):** Core abstractions
- `schema.py`, `dataset.py`, `registry.py`
- Enable strict mode for these modules only

**Phase 2 (Week 2):** Data layer
- `mapper.py`, `semantic.py`
- All loader files

**Phase 3 (Week 3):** Analysis and UI
- `stats.py`, `survival.py`, `profiling.py`
- `app.py` (UI module)

**Phase 4 (Week 4):** Enable strict mode globally
```toml
[tool.mypy]
disallow_untyped_defs = true  # ENFORCE
check_untyped_defs = true
strict_optional = true
```

## Technical Details

**Affected Files:** ALL Python files (~23 modules)

**Key Type Definitions Needed:**
```python
from typing import Any, Protocol
from pathlib import Path
import pandas as pd
import polars as pl
import ibis

# Common type aliases
ConfigDict = dict[str, Any]
FilterDict = dict[str, Any]
ColumnMapping = dict[str, str]

# Protocol for dataset-like objects
class DatasetLike(Protocol):
    def validate(self) -> bool: ...
    def load(self) -> None: ...
    def get_cohort(self, **filters: Any) -> pd.DataFrame: ...
```

**MyPy Configuration Update:**
```toml
[tool.mypy]
python_version = "3.11"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true  # ENABLE THIS
check_untyped_defs = true
strict_optional = true
warn_redundant_casts = true
warn_unused_ignores = true
```

## Acceptance Criteria

- [ ] All functions have parameter type hints
- [ ] All functions have return type hints
- [ ] Use `list/dict/tuple` not `List/Dict/Tuple`
- [ ] Use `| None` not `Optional[]` (Python 3.10+ style)
- [ ] Use specific types not `Any` where possible
- [ ] MyPy strict mode enabled (`disallow_untyped_defs = true`)
- [ ] All MyPy errors resolved
- [ ] CI/CD runs MyPy on every commit
- [ ] No `# type: ignore` comments without justification

## Work Log

### 2025-12-24
- **Action:** Code review identified complete lack of type hints
- **Learning:** Python 3.11 projects must use type hints for maintainability
- **Next:** Begin with core modules (schema, dataset, registry)

## Resources

- **Python Typing Documentation:** https://docs.python.org/3/library/typing.html
- **MyPy Documentation:** https://mypy.readthedocs.io/
- **PEP 484 (Type Hints):** https://peps.python.org/pep-0484/
- **PEP 585 (Standard Collections):** https://peps.python.org/pep-0585/
- **Code Review Report:** Python quality section
