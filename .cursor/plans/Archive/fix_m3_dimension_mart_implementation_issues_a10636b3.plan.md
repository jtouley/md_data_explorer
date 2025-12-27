---
name: Fix M3 dimension mart implementation issues
overview: "Fix three critical issues in `_build_dimension_mart()`: ensure relationships exist before BFS, add runtime cardinality validation to joins, and use deque with deterministic traversal order."
todos:
  - id: add-relationships-guard
    content: Add guard to ensure self.relationships exists before BFS traversal in _build_dimension_mart()
    status: pending
  - id: add-validate-parameter
    content: Add validate='m:1' parameter to Polars join to enforce cardinality at runtime
    status: pending
  - id: use-deque-and-sort
    content: Replace list queue with deque, use popleft(), and sort relationships for deterministic traversal
    status: pending
---

# Fix M3 Dimension Mart Implementation Issues

## Overview

Fix three critical issues in `_build_dimension_mart()` that could cause silent failures or runtime errors:

1. **Missing relationships guard**: Ensure `self.relationships` exists before BFS traversal
2. **Runtime cardinality validation**: Add `validate="m:1"` to Polars joins to fail fast on row explosion
3. **BFS performance and determinism**: Use `deque` for O(1) queue operations and sort relationships for stable join order

## Changes

### 1. Add Relationships Guard

**File**: `src/clinical_analytics/core/multi_table_handler.py`**Location**: After line 841 (after classifications check)Add guard to ensure relationships are detected before BFS:

```python
# Ensure relationships exist
if not self.relationships:
    self.detect_relationships()
```

This prevents silent no-op when BFS loop finds no relationships to traverse.

### 2. Add Runtime Cardinality Validation

**File**: `src/clinical_analytics/core/multi_table_handler.py`**Location**: Line 938-944 (join operation)Add `validate="m:1"` parameter to the join:

```python
mart = mart.join(
    next_lazy,
    left_on=join_key_left,
    right_on=join_key_right,
    how=join_type,
    suffix=f"_{next_table}",
    validate="m:1",  # fail fast if RHS key is not unique
)
```

This enforces many-to-one cardinality at execution time, turning row explosion into a hard error.

### 3. Use Deque and Deterministic Traversal

**File**: `src/clinical_analytics/core/multi_table_handler.py`**Location**:

- Line 19 (imports): Add `from collections import deque`
- Line 873: Change `queue = [anchor_table]` to `queue = deque([anchor_table])`
- Line 877: Change `current = queue.pop(0)` to `current = queue.popleft()`
- Line 880: Sort relationships for deterministic order

Update the relationship iteration to be deterministic:

```python
for rel in sorted(
    self.relationships,
    key=lambda r: (r.parent_table, r.child_table, r.parent_key, r.child_key)
):
```

This ensures:

- O(1) queue operations instead of O(n)
- Stable join order regardless of relationship detection order

## Testing

These changes maintain existing behavior while adding safety guards. The `validate="m:1"` parameter will cause joins to fail at execution time if the RHS key is not unique, which is the desired behavior for preventing OOMs.

## Impact

- **Safety**: Prevents silent failures when relationships aren't detected
- **Correctness**: Runtime validation catches cardinality violations that sampling might miss
- **Performance**: O(1) queue operations instead of O(n)