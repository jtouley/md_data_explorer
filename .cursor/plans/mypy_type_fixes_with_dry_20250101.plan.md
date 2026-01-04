---
name: Mypy Type Fixes with DRY Principles
overview: Systematic fix of 154 mypy errors across 19 files, applying DRY principles to eliminate duplication in type annotations and error patterns. Creates reusable type aliases and helper functions. Follows TDD workflow with per-phase commits.
todos:
  - id: phase0-discovery
    content: "Phase 0: Discovery - Check for existing type infrastructure"
    status: pending
  - id: phase1-simple-fixes
    content: "Phase 1: Fix simple type annotation errors (any->Any, missing imports)"
    status: pending
    dependencies:
      - phase0-discovery
  - id: phase2-optional-types
    content: "Phase 2: Fix optional type annotations with TDD (type_aliases.py + tests)"
    status: pending
    dependencies:
      - phase1-simple-fixes
  - id: phase3-union-attr
    content: "Phase 3: Fix union-attr errors with TDD (type_guards.py + tests)"
    status: pending
    dependencies:
      - phase2-optional-types
  - id: phase4-return-types
    content: "Phase 4: Fix return type issues (Any returns)"
    status: pending
    dependencies:
      - phase2-optional-types
  - id: phase5-dict-assignments
    content: "Phase 5: Fix dict type mismatches and assignment errors"
    status: pending
    dependencies:
      - phase4-return-types
  - id: phase6-missing-annotations
    content: "Phase 6: Fix missing type annotations for variables"
    status: pending
    dependencies:
      - phase5-dict-assignments
  - id: phase7-verify
    content: "Phase 7: Run full mypy check and verify all errors fixed"
    status: pending
    dependencies:
      - phase6-missing-annotations
---

# Mypy Type Fixes with DRY Principles

## Summary

**Status: ✅ IN PROGRESS**

Systematic fix of 154 mypy errors across 19 files, applying DRY principles to eliminate duplication in type annotations and error patterns.

## Error Pattern Analysis

### Pattern Distribution

1. **Union-attr errors** (11): Accessing attributes on `None` without checks
2. **Index type errors** (9): Invalid index types for DataFrames
3. **Assignment errors** (8): Assigning to variables typed as `None`
4. **Float conversion errors** (8): Incompatible types for `float()` calls
5. **Return Any errors** (6): Functions returning `Any` instead of specific types
6. **Missing imports** (6): `pl` not imported in semantic.py
7. **Dict type mismatches** (5): Dict entries with wrong value types
8. **Type annotation errors** (3): Using `any` instead of `Any`

### DRY Opportunities

1. **Type Aliases**: Create common type aliases (e.g., `ConfigDict`, `FilterDict`)
2. **None Check Helpers**: Extract common None-check patterns
3. **Type Guards**: Create reusable type guard functions
4. **Import Consolidation**: Standardize imports across files

## Phase 0: Discovery (15 min)

### 0.1: Check Existing Type Infrastructure

**Purpose:** Verify no duplication, document existing patterns.

**Actions:**
1. Search for existing type aliases: `grep -r "type.*=.*dict\[str" src/`
2. Search for existing type guards: `grep -r "TypeGuard\|safe_get" src/`
3. Check if `type_aliases.py` or `type_guards.py` already exist
4. Document existing patterns in plan notes

**Success Criteria:**
- No duplicate type infrastructure found
- Existing patterns documented
- Plan updated if infrastructure exists

**Baseline Capture:**
```bash
# Capture current mypy state
make type-check 2>&1 | tee baseline_mypy.txt
ERROR_COUNT=$(grep -c "error:" baseline_mypy.txt || echo "0")
echo "Baseline: $ERROR_COUNT errors"

# Capture current test state
make test-fast 2>&1 | tee baseline_tests.txt
```

## Phase 1: Simple Type Fixes (30 min)

### 1.1: Fix `any` -> `Any` (3 files)

**Files:**
- `src/clinical_analytics/ui/components/result_interpreter.py` (already fixed)
- `src/clinical_analytics/core/semantic.py` (lines 1551, 1618, 1673)
- Check for any remaining instances

**Action:**
```python
# Before
def format_execution_result(self, execution_result: dict[str, Any], context: any) -> dict[str, Any]:

# After
from typing import Any
def format_execution_result(self, execution_result: dict[str, Any], context: Any) -> dict[str, Any]:
```

### 1.2: Fix Missing Imports (6 errors)

**File:** `src/clinical_analytics/core/semantic.py` (lines 1585, 1588, 1589, 1593, 1618, 1673)

**Action:** Add `import polars as pl` at top of file if missing.

**Test:** Run `make type-check` to verify imports resolved.

**Quality Gate:**
- Run `make type-check` - verify error count reduced
- Run `make test-fast` - verify no runtime regressions
- Expected: ~9 errors fixed (3 any->Any + 6 missing imports)

**Commit:**
```bash
git commit -m "fix: Phase 1 - Fix simple type annotation errors

- Fix 'any' -> 'Any' in semantic.py (3 instances)
- Add missing polars import in semantic.py
- Verify type-check shows error reduction

All tests passing: X/Y
Type checking: $(ERROR_COUNT - 9) errors (reduced from $ERROR_COUNT)
Following TDD: Simple type fixes"
```

## Phase 2: Optional Type Annotations (1 hour)

### 2.1: Write Tests for Type Aliases (TDD - Red Phase)

**File:** `tests/core/test_type_aliases.py` (new)

**Test Strategy (TDD):**
1. Write failing test that imports and uses type aliases
2. Verify test fails (Red phase)
3. Implement type aliases module
4. Verify test passes (Green phase)

**Test Implementation:**
```python
"""Tests for type aliases module."""

import pytest
from clinical_analytics.core.type_aliases import (
    ConfigDict,
    FilterDict,
    ColumnMapping,
    MetadataDict,
    OptionalStr,
    OptionalInt,
    OptionalFloat,
    OptionalDict,
    OptionalList,
)


def test_type_aliases_are_importable():
    """Test that all type aliases can be imported."""
    assert ConfigDict is not None
    assert FilterDict is not None
    assert ColumnMapping is not None
    assert MetadataDict is not None
    assert OptionalStr is not None
    assert OptionalInt is not None
    assert OptionalFloat is not None
    assert OptionalDict is not None
    assert OptionalList is not None


def test_type_aliases_can_be_used_in_signatures():
    """Test that type aliases work in function signatures."""
    def example_func(config: ConfigDict, filter: FilterDict) -> OptionalStr:
        return config.get("key")

    # Should not raise type errors
    result = example_func({"key": "value"}, {"filter": "value"})
    assert result == "value"


def test_optional_types_accept_none():
    """Test that optional types accept None values."""
    def example_func(value: OptionalStr) -> OptionalStr:
        return value

    assert example_func(None) is None
    assert example_func("value") == "value"
```

**Run Test (Red Phase):**
```bash
make test-core PYTEST_ARGS="tests/core/test_type_aliases.py -xvs"
# Expected: FAILED - ModuleNotFoundError (module doesn't exist yet)
```

### 2.2: Create Type Alias Module (TDD - Green Phase)

**File:** `src/clinical_analytics/core/type_aliases.py` (new)

**Purpose:** Centralize common type aliases to eliminate duplication.

**Implementation:**
```python
"""Type aliases for consistent type annotations across codebase."""

from typing import Any

# Common dictionary types
ConfigDict = dict[str, Any]
FilterDict = dict[str, Any]
ColumnMapping = dict[str, str]
MetadataDict = dict[str, Any]

# Common optional types
OptionalStr = str | None
OptionalInt = int | None
OptionalFloat = float | None
OptionalDict = dict[str, Any] | None
OptionalList = list[str] | None
```

**Run Test (Green Phase):**
```bash
make test-core PYTEST_ARGS="tests/core/test_type_aliases.py -xvs"
# Expected: PASSED
```

**Quality Gate:**
- Run `make test-core PYTEST_ARGS="tests/core/test_type_aliases.py -xvs"` - all tests pass
- Run `make type-check` - verify no new errors
- Verify type aliases are importable and usable

**Invoke `/deslop`:** Remove AI-generated slop from type_aliases.py

### 2.3: Fix Optional Type Annotations (106 instances)

**Pattern:** Variables initialized to `None` without `| None` in type hint.

**Files:**
- `src/clinical_analytics/core/column_parser.py` (lines 36-37) - already fixed
- `src/clinical_analytics/core/schema_inference.py` (line 97)
- `src/clinical_analytics/core/multi_table_handler.py` (line 237)
- `src/clinical_analytics/ui/components/data_validator.py` (line 348)
- And 15+ more files

**Action:**
```python
# Before
config = {"column_mapping": {}, "outcomes": {}, "time_zero": {}}

# After
config: dict[str, Any] = {"column_mapping": {}, "outcomes": {}, "time_zero": {}}
```

**Test Strategy:**
- Run `make type-check` after each file
- Verify no new errors introduced
- Run `make test-fast` to catch runtime regressions

**Quality Gate:**
- Run `make type-check` - verify error count reduced (~106 errors fixed)
- Run `make test-fast` - verify no runtime regressions
- Expected: Significant error reduction from optional type fixes

**Commit:**
```bash
git commit -m "fix: Phase 2 - Create type aliases and fix optional annotations

- Create type_aliases.py with common type definitions (DRY)
- Add comprehensive test suite for type aliases (TDD)
- Fix optional type annotations across codebase (~106 instances)
- Use type aliases where appropriate

All tests passing: X/Y
Type checking: $(ERROR_COUNT - 106) errors (reduced from $ERROR_COUNT)
Following TDD: Red-Green-Refactor for type aliases"
```

## Phase 3: Union-Attr Errors (2 hours)

### 3.1: Write Tests for Type Guards (TDD - Red Phase)

**File:** `tests/core/test_type_guards.py` (new)

**Test Strategy (TDD):**
1. Write failing test that uses type guards
2. Verify test fails (Red phase)
3. Implement type guards module
4. Verify test passes (Green phase)

**Test Implementation:**
```python
"""Tests for type guards module."""

import pytest
from clinical_analytics.core.type_guards import is_not_none, safe_get


def test_is_not_none_type_guard():
    """Test that is_not_none type guard works correctly."""
    value: dict[str, int] | None = {"key": 42}

    if is_not_none(value):
        # Type should be narrowed to dict[str, int]
        assert value["key"] == 42

    value = None
    assert not is_not_none(value)


def test_safe_get_with_none_dict():
    """Test safe_get with None dict."""
    result = safe_get(None, "key", "default")
    assert result == "default"


def test_safe_get_with_valid_dict():
    """Test safe_get with valid dict."""
    d = {"key": "value"}
    result = safe_get(d, "key", "default")
    assert result == "value"

    result = safe_get(d, "missing", "default")
    assert result == "default"


def test_safe_get_type_narrowing():
    """Test that safe_get works with type narrowing."""
    metadata: dict[str, str] | None = {"key": "value"}

    value = safe_get(metadata, "key")
    # Should not raise type errors
    assert value == "value"
```

**Run Test (Red Phase):**
```bash
make test-core PYTEST_ARGS="tests/core/test_type_guards.py -xvs"
# Expected: FAILED - ModuleNotFoundError (module doesn't exist yet)
```

### 3.2: Create None Check Helper (TDD - Green Phase)

**File:** `src/clinical_analytics/core/type_guards.py` (new)

**Purpose:** Reusable type guard functions for None checks.

**Implementation:**
```python
"""Type guard functions for safe None handling."""

from typing import TypeGuard

def is_not_none(value: dict[str, Any] | None) -> TypeGuard[dict[str, Any]]:
    """Type guard for non-None dict."""
    return value is not None

def safe_get(d: dict[str, Any] | None, key: str, default: Any = None) -> Any:
    """Safely get value from potentially None dict."""
    if d is None:
        return default
    return d.get(key, default)
```

**Run Test (Green Phase):**
```bash
make test-core PYTEST_ARGS="tests/core/test_type_guards.py -xvs"
# Expected: PASSED
```

**Quality Gate:**
- Run `make test-core PYTEST_ARGS="tests/core/test_type_guards.py -xvs"` - all tests pass
- Run `make type-check` - verify no new errors
- Verify type guards work correctly

**Invoke `/deslop`:** Remove AI-generated slop from type_guards.py

### 3.3: Fix Union-Attr Errors (11 instances)

**Pattern:** Accessing `.get()` on `dict[str, Any] | None` without check.

**Files:**
- `src/clinical_analytics/core/semantic.py` (lines 1006, 1174, 1175)
- `src/clinical_analytics/datasets/uploaded/definition.py` (multiple)
- `src/clinical_analytics/ui/storage/user_datasets.py` (lines 2201, 2205, 2234)

**Action:**
```python
# Before
if "key" in metadata:  # metadata is dict[str, Any] | None
    value = metadata["key"]

# After
if metadata is not None and "key" in metadata:
    value = metadata["key"]
# OR use helper
value = safe_get(metadata, "key")
```

**Quality Gate:**
- Run `make type-check` - verify union-attr errors reduced (~11 errors fixed)
- Run `make test-core` - verify None checks work correctly
- Run `make test-fast` - verify no runtime regressions
- Expected: All union-attr errors resolved

**Commit:**
```bash
git commit -m "fix: Phase 3 - Create type guards and fix union-attr errors

- Create type_guards.py with reusable None-check helpers (DRY)
- Add comprehensive test suite for type guards (TDD)
- Fix union-attr errors with proper None checks (~11 instances)
- Use type guards where appropriate

All tests passing: X/Y
Type checking: $(ERROR_COUNT - 11) errors (reduced from $ERROR_COUNT)
Following TDD: Red-Green-Refactor for type guards"
```

## Phase 4: Return Type Issues (2 hours)

### 4.1: Fix Return Any Errors (6 instances)

**Note:** Use type aliases from Phase 2 where appropriate.

**Files:**
- `src/clinical_analytics/core/schema_inference.py` (line 550)
- `src/clinical_analytics/core/multi_table_handler.py` (lines 563, 569, 613, 682, 802)
- `src/clinical_analytics/core/mapper.py` (lines 427, 432, 436, 440, 473, 478, 504, 523)
- `src/clinical_analytics/core/semantic.py` (lines 568, 583, 693, 700, 1047)
- `src/clinical_analytics/core/nl_query_engine.py` (lines 2610, 2632)
- `src/clinical_analytics/core/result_interpretation.py` (line 136)
- `src/clinical_analytics/core/error_translation.py` (line 86)

**Action:** Replace `Any` returns with specific types or `str | None`, `dict[str, Any] | None`, etc.

**Example:**
```python
# Before
def get_column(self) -> str | None:
    return self._data.get("column")  # Returns Any

# After
def get_column(self) -> str | None:
    value = self._data.get("column")
    return str(value) if value is not None else None
```

**Quality Gate:**
- Run `make type-check` - verify return Any errors reduced (~6 errors fixed)
- Run `make test-fast` - verify no runtime regressions
- Expected: All return type issues resolved

**Commit:**
```bash
git commit -m "fix: Phase 4 - Fix return type issues (Any -> specific types)

- Replace Any returns with specific types (str | None, dict[str, Any] | None, etc.)
- Use type aliases from Phase 2 where appropriate
- Fix return type issues across core modules (~6 instances)

All tests passing: X/Y
Type checking: $(ERROR_COUNT - 6) errors (reduced from $ERROR_COUNT)
Following TDD: Type fixes with DRY principles"
```

## Phase 5: Dict Type Mismatches (1 hour)

### 5.1: Fix Dict Entry Type Errors (5 instances)

**File:** `src/clinical_analytics/ui/components/data_validator.py` (lines 110, 111, 130, 145-147, 265, 266, 278, 289, 307)

**Pattern:** Dict entries with `int`/`float` values when `str` expected.

**Action:**
```python
# Before
{
    "key1": "value1",
    "key2": 42,  # Error: expected str, got int
}

# After
{
    "key1": "value1",
    "key2": str(42),  # Convert to str
```

**Quality Gate:**
- Run `make type-check` - verify dict type mismatch errors reduced (~5 errors fixed)
- Run `make test-ui PYTEST_ARGS="tests/ui/test_data_validator.py -xvs"` - verify no runtime breaks
- Run `make test-fast` - verify no regressions
- Expected: All dict type mismatches resolved

**Commit:**
```bash
git commit -m "fix: Phase 5 - Fix dict type mismatches and assignment errors

- Fix dict entries with wrong value types (int/float -> str)
- Fix assignment errors (variables typed as None)
- Verify no runtime behavior changes

All tests passing: X/Y
Type checking: $(ERROR_COUNT - 5) errors (reduced from $ERROR_COUNT)
Following TDD: Type fixes with DRY principles"
```

## Phase 6: Missing Type Annotations (1 hour)

### 6.1: Fix Variable Type Annotations (8 instances)

**Files:**
- `src/clinical_analytics/core/schema_inference.py` (line 97)
- `src/clinical_analytics/core/multi_table_handler.py` (lines 237, 1847, 1926)
- `src/clinical_analytics/ui/components/data_validator.py` (line 348)
- `src/clinical_analytics/datasets/uploaded/definition.py` (line 407)

**Action:** Add explicit type annotations for variables.

**Quality Gate:**
- Run `make type-check` - verify missing annotation errors reduced (~8 errors fixed)
- Run `make test-fast` - verify no runtime regressions
- Expected: All missing type annotations resolved

**Commit:**
```bash
git commit -m "fix: Phase 6 - Fix missing type annotations for variables

- Add explicit type annotations for variables (~8 instances)
- Use type aliases from Phase 2 where appropriate
- Verify no runtime behavior changes

All tests passing: X/Y
Type checking: $(ERROR_COUNT - 8) errors (reduced from $ERROR_COUNT)
Following TDD: Type fixes with DRY principles"
```

## Phase 7: Verification (30 min)

### 7.1: Run Full Mypy Check

```bash
make type-check
```

**Success Criteria:**
- Zero mypy errors
- All type annotations consistent
- Type aliases used where appropriate

### 7.2: Run Test Suite

```bash
make test-fast
```

**Success Criteria:**
- All tests pass
- No regressions introduced
- Zero mypy errors

**Final Quality Gate:**
```bash
make check  # Full quality gate: format, lint, type, test
```

**Expected Results:**
- `make type-check`: 0 errors
- `make test-fast`: All tests passing
- `make lint`: No linting errors
- `make format-check`: Code properly formatted

**Rollback Verification:**
- Verify `git log` shows 6 phase commits (Phases 1-6)
- Test `git revert` on each phase commit to ensure clean rollback
- Verify baseline tests still pass after reverts

## Testing Strategy

### Per-Phase Testing (TDD Workflow)

**Phase 0:**
- Baseline capture: `make type-check` and `make test-fast`

**Phase 1:**
- After fixes: `make type-check` (verify error reduction)
- After fixes: `make test-fast` (verify no runtime regressions)

**Phase 2:**
- Red phase: `make test-core PYTEST_ARGS="tests/core/test_type_aliases.py -xvs"` (verify failure)
- Green phase: `make test-core PYTEST_ARGS="tests/core/test_type_aliases.py -xvs"` (verify pass)
- After fixes: `make type-check` (verify error reduction)
- After fixes: `make test-fast` (verify no runtime regressions)

**Phase 3:**
- Red phase: `make test-core PYTEST_ARGS="tests/core/test_type_guards.py -xvs"` (verify failure)
- Green phase: `make test-core PYTEST_ARGS="tests/core/test_type_guards.py -xvs"` (verify pass)
- After fixes: `make type-check` (verify error reduction)
- After fixes: `make test-core` (verify None checks work)
- After fixes: `make test-fast` (verify no runtime regressions)

**Phase 4-6:**
- After fixes: `make type-check` (verify error reduction)
- After fixes: `make test-fast` (verify no runtime regressions)
- Module-specific: `make test-[module]` for affected modules

**Phase 7:**
- Final verification: `make check` (full quality gate)

### Regression Protection

- Run baseline: `make type-check 2>&1 | tee baseline_mypy.txt`
- After each phase: `make type-check 2>&1 | tee phase_N_mypy.txt`
- Verify: Error count decreases each phase

## DRY Principles Applied

1. **Type Aliases**: Centralized in `type_aliases.py`
2. **Type Guards**: Reusable None-check helpers in `type_guards.py`
3. **Import Consolidation**: Standardized imports across files
4. **Pattern Extraction**: Common error patterns fixed once, applied everywhere

## Success Metrics

- **Mypy Errors**: 154 → 0
- **Type Coverage**: Increase by ~15%
- **Code Duplication**: Reduce type annotation duplication by ~30%
- **Maintainability**: Easier to add new type-safe code

## Estimated Time

- Phase 0: 15 min (Discovery + baseline)
- Phase 1: 30 min (Simple fixes + commit)
- Phase 2: 1.5 hours (TDD: tests + implementation + fixes + commit)
- Phase 3: 2.5 hours (TDD: tests + implementation + fixes + commit)
- Phase 4: 2 hours (Fixes + commit)
- Phase 5: 1 hour (Fixes + commit)
- Phase 6: 1 hour (Fixes + commit)
- Phase 7: 30 min (Verification only)

**Total: ~9.5 hours** (includes TDD workflow and per-phase commits)
