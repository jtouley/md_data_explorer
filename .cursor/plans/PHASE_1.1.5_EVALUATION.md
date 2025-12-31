# Phase 1.1.5 Evaluation: Verify run_key Determinism in All Execution Paths

## Status: ❌ NOT COMPLETE

**Date**: 2025-12-30  
**Branch**: Current working branch  
**Issue**: Same query produces different run_keys across execution paths

---

## Problem Statement

From terminal output (lines 66 vs 75):
- **Line 66**: `run_key=user_upload_20251229_225650_45c58677_70856f45196718d6`
- **Line 75**: `run_key=eea89e674c902e37` (different)

**Root Cause**: Multiple code paths generate run_keys using different algorithms, causing non-deterministic behavior.

---

## Current State Analysis

### Three Different run_key Generation Methods Found

#### 1. `semantic._generate_run_key()` ✅ (Canonical - should be source of truth)
**Location**: `src/clinical_analytics/core/semantic.py:1331`

**Algorithm**:
```python
hash_input = f"{dataset_version}|{canonical_plan_json}|{query_signature}"
run_key = hashlib.sha256(hash_input.encode("utf-8")).hexdigest()[:16]
```

**Inputs**:
- `dataset_version`
- `canonical_plan_json` (intent, metric, group_by, filters, entity_key, scope)
- `query_signature` (normalized query text)

**Output**: 16-char hash

**Used in**: `execute_query_plan()` at line 1216

---

#### 2. `Ask_Questions.generate_run_key()` ❌ (Legacy - should be removed)
**Location**: `src/clinical_analytics/ui/pages/3_💬_Ask_Questions.py:193`

**Algorithm**:
```python
payload = {
    "dataset_version": dataset_version,
    "query": query_text,
    "intent": context.inferred_intent.value,
    "vars": material_vars,  # primary, grouping, predictors, time, event
    "scope": canonicalize_scope(scope),
}
run_key = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
```

**Inputs**:
- `dataset_version`
- `query_text` (normalized)
- `context.inferred_intent.value`
- `material_vars` (primary_variable, grouping_variable, predictors, time_variable, event_variable)
- `scope` (canonicalized)

**Output**: Full 64-char SHA256 hash

**Used in**: 
- Line 1681: `run_key = query_plan.run_key or generate_run_key(...)`
- Line 1687: `run_key = generate_run_key(...)`
- Line 1994: `run_key = query_plan.run_key or generate_run_key(...)`
- Line 1997: `run_key = generate_run_key(...)`

---

#### 3. `nl_query_engine._intent_to_plan()` ⚠️ (Partial - missing query text)
**Location**: `src/clinical_analytics/core/nl_query_engine.py:1794`

**Algorithm**:
```python
normalized_plan = {
    "intent": plan.intent,
    "metric": plan.metric,
    "group_by": plan.group_by,
    "filters": [...],
}
plan_hash = hashlib.sha256(json.dumps(normalized_plan, sort_keys=True).encode()).hexdigest()[:16]
plan.run_key = f"{dataset_version}_{plan_hash}"
```

**Inputs**:
- `dataset_version`
- `normalized_plan` (intent, metric, group_by, filters) - **NO query text**

**Output**: `{dataset_version}_{16-char-hash}`

**Used in**: Sets `plan.run_key` directly, which is then used at:
- Line 1681: `run_key = query_plan.run_key or generate_run_key(...)`
- Line 1994: `run_key = query_plan.run_key or generate_run_key(...)`

---

## Execution Flow Analysis

### Path 1: QueryPlan with run_key (from nl_query_engine)
```
1. nl_query_engine.parse_query() → QueryIntent
2. nl_query_engine._intent_to_plan() → QueryPlan with run_key = "{dataset_version}_{plan_hash}"
3. Ask_Questions.py line 1681: Uses query_plan.run_key
4. semantic.execute_query_plan() line 1216: Generates NEW run_key using _generate_run_key()
5. Ask_Questions.py line 1717: Overwrites with execution_result["run_key"]
```

**Result**: Two different run_keys generated for same query:
- First: `user_upload_20251229_225650_45c58677_70856f45196718d6` (from nl_query_engine)
- Second: `eea89e674c902e37` (from semantic._generate_run_key())

---

### Path 2: QueryPlan without run_key (fallback)
```
1. QueryPlan exists but run_key is None
2. Ask_Questions.py line 1681: Calls generate_run_key() (legacy method)
3. semantic.execute_query_plan() line 1216: Generates NEW run_key using _generate_run_key()
4. Ask_Questions.py line 1717: Overwrites with execution_result["run_key"]
```

**Result**: Two different run_keys generated:
- First: Full 64-char hash from `generate_run_key()` (uses context variables)
- Second: 16-char hash from `semantic._generate_run_key()` (uses plan + query)

---

### Path 3: No QueryPlan (legacy path)
```
1. No QueryPlan available
2. Ask_Questions.py line 1687: Calls generate_run_key() (legacy method)
3. Uses legacy execution path (not execute_query_plan())
```

**Result**: Uses legacy run_key generation (should not exist after Phase 3)

---

## What Phase 1.1.5 Requires

### Objective
Ensure all code paths (legacy and QueryPlan) use the same deterministic run_key generation.

### Tasks
1. ✅ **Write integration tests** - Verify same query produces same run_key regardless of execution path
2. ❌ **Audit all run_key generation points** - Ensure they all call `_generate_run_key()` with canonical inputs
3. ❌ **Fix any divergent paths** - Consolidate to single source of truth

---

## What's Currently in Branch

### ✅ Completed
1. **Tests exist** (`tests/core/test_semantic_run_key_determinism.py`):
   - Tests for `semantic._generate_run_key()` determinism
   - Tests for same plan + query → same key
   - Tests for different queries → different keys
   - Tests for whitespace normalization
   - Tests for scope inclusion

2. **Canonical method exists** (`semantic._generate_run_key()`):
   - Includes all plan fields
   - Includes normalized query text
   - Includes dataset_version
   - Includes scope (from plan.scope)
   - Deterministic JSON serialization

### ❌ Missing
1. **Integration tests** - No tests verifying all execution paths produce same run_key
2. **Consolidation** - Multiple run_key generation methods still exist
3. **nl_query_engine fix** - Still generates run_key without query text
4. **UI path fix** - Still uses legacy `generate_run_key()` as fallback
5. **Execution flow fix** - Still overwrites run_key from execution result

---

## Required Fixes

### Fix 1: Remove run_key generation from nl_query_engine
**File**: `src/clinical_analytics/core/nl_query_engine.py:1794`

**Current**:
```python
plan.run_key = f"{dataset_version}_{plan_hash}"
```

**Should be**:
```python
# Don't set run_key here - let semantic layer generate it deterministically
plan.run_key = None
```

**Rationale**: `nl_query_engine` doesn't have access to normalized query text, so it can't generate a deterministic run_key that matches `semantic._generate_run_key()`.

---

### Fix 2: Remove legacy `generate_run_key()` function
**File**: `src/clinical_analytics/ui/pages/3_💬_Ask_Questions.py:193`

**Action**: Delete `generate_run_key()` function entirely.

**Replace all usages** with:
```python
# Always use semantic layer's run_key from execution result
# Don't generate run_key in UI - let semantic layer handle it
```

**Rationale**: UI should not generate run_keys. Only `semantic._generate_run_key()` should generate run_keys.

---

### Fix 3: Always use execution result's run_key
**File**: `src/clinical_analytics/ui/pages/3_💬_Ask_Questions.py:1681, 1717`

**Current**:
```python
# Line 1681: Generate run_key before execution
run_key = query_plan.run_key or generate_run_key(dataset_version, normalized_query, context)

# Line 1717: Overwrite with execution result
if execution_result.get("run_key"):
    run_key = execution_result["run_key"]
```

**Should be**:
```python
# Don't generate run_key before execution - always use execution result's run_key
# Line 1717: Always use execution result's run_key (never generate in UI)
run_key = execution_result.get("run_key")
if not run_key:
    raise ValueError("Execution result must include run_key")
```

**Rationale**: Only `semantic._generate_run_key()` should generate run_keys. UI should always use the run_key from execution result.

---

### Fix 4: Add integration tests
**File**: `tests/core/test_semantic_run_key_determinism.py` (add new tests)

**Add**:
```python
def test_all_execution_paths_produce_same_run_key():
    """All execution paths should produce same run_key for same query."""
    # Arrange: Same query, same plan
    query = "average age by status"
    plan = QueryPlan(intent="COUNT", metric="age", group_by="status", confidence=0.9)
    
    # Act: Generate run_key from different paths
    # Path 1: semantic._generate_run_key() directly
    key1 = semantic_layer._generate_run_key(plan, query)
    
    # Path 2: execute_query_plan() (which calls _generate_run_key())
    result = semantic_layer.execute_query_plan(plan, query_text=query)
    key2 = result["run_key"]
    
    # Path 3: nl_query_engine._intent_to_plan() should NOT set run_key
    # (should be None, then semantic layer generates it)
    
    # Assert: All paths produce same key
    assert key1 == key2
    assert plan.run_key is None  # nl_query_engine should not set run_key

def test_query_plan_run_key_always_from_semantic_layer():
    """QueryPlan.run_key should always come from semantic layer, never from nl_query_engine."""
    # Arrange: QueryIntent from nl_query_engine
    intent = QueryIntent(...)
    
    # Act: Convert to plan
    plan = nl_query_engine._intent_to_plan(intent, dataset_version)
    
    # Assert: run_key should be None (semantic layer will generate it)
    assert plan.run_key is None
```

---

## Implementation Plan

### Step 1: Write failing integration tests
- Add tests to `tests/core/test_semantic_run_key_determinism.py`
- Test that all execution paths produce same run_key
- Test that nl_query_engine doesn't set run_key
- Run tests (should fail)

### Step 2: Fix nl_query_engine
- Remove `plan.run_key = f"{dataset_version}_{plan_hash}"` from `_intent_to_plan()`
- Set `plan.run_key = None` instead
- Run tests (should pass for nl_query_engine)

### Step 3: Remove legacy generate_run_key()
- Delete `generate_run_key()` function from `Ask_Questions.py`
- Replace all usages with `execution_result["run_key"]`
- Run tests (should pass)

### Step 4: Fix execution flow
- Remove run_key generation before execution
- Always use `execution_result["run_key"]` after execution
- Add assertion that run_key exists in execution result
- Run tests (should pass)

### Step 5: Run full test suite
```bash
make test-core
make test-ui
make check
```

---

## Success Criteria

- [ ] Same query produces same run_key regardless of execution path
- [ ] Only `semantic._generate_run_key()` generates run_keys
- [ ] `nl_query_engine` does not set `plan.run_key`
- [ ] UI does not generate run_keys (only uses execution result's run_key)
- [ ] All integration tests passing
- [ ] Terminal output shows same run_key for same query (no more collisions)

---

## Terminal Output Analysis

**Current Problem**:
```
Line 66: run_key=user_upload_20251229_225650_45c58677_70856f45196718d6
Line 75: run_key=eea89e674c902e37 (different)
```

**Expected After Fix**:
```
Line 66: run_key=eea89e674c902e37
Line 75: run_key=eea89e674c902e37 (same)
```

**Root Cause**: 
- Line 66 uses `query_plan.run_key` from nl_query_engine (format: `{dataset_version}_{hash}`)
- Line 75 uses `execution_result["run_key"]` from semantic layer (format: 16-char hash)

**Fix**: Remove run_key generation from nl_query_engine, always use semantic layer's run_key.

---

## Related Files

- `src/clinical_analytics/core/semantic.py` - `_generate_run_key()` (canonical method)
- `src/clinical_analytics/core/nl_query_engine.py` - `_intent_to_plan()` (remove run_key generation)
- `src/clinical_analytics/ui/pages/3_💬_Ask_Questions.py` - Remove `generate_run_key()`, always use execution result
- `tests/core/test_semantic_run_key_determinism.py` - Add integration tests

---

## Next Steps

1. **Create integration tests** (test-first approach)
2. **Fix nl_query_engine** (remove run_key generation)
3. **Remove legacy generate_run_key()** (consolidate to semantic layer)
4. **Fix execution flow** (always use execution result's run_key)
5. **Run quality gates** (`make check`)
6. **Commit** (Phase 1.1.5 complete)

