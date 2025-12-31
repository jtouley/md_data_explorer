---
name: ""
overview: ""
todos: []
---

---
name: QueryPlan as Product + Critical Fixes (MASTER PLAN)
overview: **MASTER EXECUTION PLAN - Priority 0**. Make QueryPlan the end-to-end product by completing Phase 3 integration, removing confirmation gating, fixing critical security/stability issues, and establishing LLM-as-planner architecture. This is the single execution stream until results are deterministic, QueryPlan is the only path, warnings are visible/logged, and "it works sometimes" is dead. All other plans are either merged into this (as sub-phases) or explicitly deferred behind it.
todos:
  - id: phase0-security-data-loss
    content: "Phase 0: Fix silent data loss in value mapping (mapper.py) - default to NULL, validate unmapped values"
    status: pending
  - id: phase0-security-sql-injection
    content: "Phase 0: Review and strengthen SQL injection mitigation in semantic.py"
    status: pending
  - id: phase0-stability-refactor
    content: "Phase 0: Extract RelationshipDetector from multi_table_handler.py (stability/refactor - improves testability and maintainability)"
    status: pending
  - id: phase0-tests
    content: "Phase 0: Write tests for security fixes and refactoring"
    status: pending
    dependencies:
      - phase0-security-data-loss
      - phase0-security-sql-injection
      - phase0-stability-refactor
  - id: phase0-quality-gates
    content: "Phase 0: Run quality gates (format, lint, type-check, test) and commit"
    status: pending
    dependencies:
      - phase0-tests
  - id: phase1-run-key-determinism
    content: "Phase 1: Fix run_key determinism - canonical plan + normalized query + dataset_version + canonical scope (filters, cohort, grouping, entity grain, optionally engine version)"
    status: pending
    dependencies:
      - phase0-quality-gates
  - id: phase1-pr20-cache-primitive
    content: "Phase 1: PR20 P0 - Fix cache primitive (st.cache_data → st.cache_resource for semantic layer)"
    status: pending
    dependencies:
      - phase0-quality-gates
  - id: phase1-pr20-empty-query
    content: "Phase 1: PR20 P0 - Reject empty queries at ingestion (normalize returns \"\" for None, should reject)"
    status: pending
    dependencies:
      - phase0-quality-gates
  - id: phase1-pr20-scope-canonical
    content: "Phase 1: PR20 P0 - Recursive deterministic scope canonicalization (handle nested dicts/lists, enums, dataclasses)"
    status: pending
    dependencies:
      - phase0-quality-gates
  - id: phase1-tests
    content: "Phase 1: Write tests for run_key determinism and PR20 P0 fixes"
    status: pending
    dependencies:
      - phase1-run-key-determinism
      - phase1-pr20-cache-primitive
      - phase1-pr20-empty-query
      - phase1-pr20-scope-canonical
  - id: phase1-quality-gates
    content: "Phase 1: Run quality gates and commit"
    status: pending
    dependencies:
      - phase1-tests
  - id: phase2-observability
    content: "Phase 2: Add observability before removing gating - warnings list, 'why confidence is low', plan diff visibility, persistent 'executed with warnings' record"
    status: pending
    dependencies:
      - phase1-quality-gates
  - id: phase2-remove-gating
    content: "Phase 2: Remove confirmation gating from execute_query_plan() - convert to warnings (includes Phase 1.6: move validation to warnings)"
    status: pending
    dependencies:
      - phase2-observability
  - id: phase2-update-return
    content: "Phase 2: Update return signature - remove requires_confirmation, add warnings list"
    status: pending
    dependencies:
      - phase2-remove-gating
  - id: phase2-remove-confirmation-ui
    content: "Phase 2: Delete _render_confirmation_ui() and remove all confirmation checks"
    status: pending
    dependencies:
      - phase2-update-return
  - id: phase2-pending-state
    content: "Phase 2.4: Implement pending state pattern (Phase 1.5) - prevent accidental execution when warnings present"
    status: pending
    dependencies:
      - phase2-remove-confirmation-ui
  - id: phase2-tests
    content: "Phase 2: Write tests for warning-based execution (no gating) and pending state pattern"
    status: pending
    dependencies:
      - phase2-pending-state
  - id: phase2-quality-gates
    content: "Phase 2: Run quality gates and commit"
    status: pending
    dependencies:
      - phase2-tests
  - id: phase2-5-thinking-indicators
    content: "Phase 2.5: Add progressive thinking indicators (stepwise UI narration, retry UX) - after gating removal is safe"
    status: pending
    dependencies:
      - phase2-quality-gates
  - id: phase2-5-retry-logic
    content: "Phase 2.5: Add retry logic for backend errors with exponential backoff"
    status: pending
    dependencies:
      - phase2-5-thinking-indicators
  - id: phase2-5-tests
    content: "Phase 2.5: Write tests for thinking indicators and retry logic"
    status: pending
    dependencies:
      - phase2-5-retry-logic
  - id: phase2-5-quality-gates
    content: "Phase 2.5: Run quality gates and commit"
    status: pending
    dependencies:
      - phase2-5-tests
  - id: phase3-remove-legacy-paths
    content: "Phase 3: Remove legacy execution paths - ensure all queries use QueryPlan"
    status: pending
    dependencies:
      - phase2-quality-gates
  - id: phase3-enforce-contract
    content: "Phase 3: Enforce QueryPlan contract - add assertions/logging to catch bypasses"
    status: pending
    dependencies:
      - phase3-remove-legacy-paths
  - id: phase3-tests
    content: "Phase 3: Write tests to verify QueryPlan is only execution path"
    status: pending
    dependencies:
      - phase3-enforce-contract
  - id: phase3-quality-gates
    content: "Phase 3: Run quality gates and commit"
    status: pending
    dependencies:
      - phase3-tests
  - id: phase4-multipart-parsing
    content: "Phase 4: Enhance multi-part query parsing - extract primary/secondary variables, detect grouping"
    status: pending
    dependencies:
      - phase3-quality-gates
  - id: phase4-type-aware
    content: "Phase 4: Add type-aware logic - distinguish categorical vs numeric, validate grouping"
    status: pending
    dependencies:
      - phase4-multipart-parsing
  - id: phase4-tests
    content: "Phase 4: Write tests for multi-part query parsing and type-aware logic"
    status: pending
    dependencies:
      - phase4-type-aware
  - id: phase4-quality-gates
    content: "Phase 4: Run quality gates and commit"
    status: pending
    dependencies:
      - phase4-tests
  - id: phase5-llm-structured-json
    content: "Phase 5: Require LLM to return QueryPlan JSON schema (not freeform)"
    status: pending
    dependencies:
      - phase4-quality-gates
  - id: phase5-validate-schema
    content: "Phase 5: Validate QueryPlan JSON against dataclass, reject malformed plans"
    status: pending
    dependencies:
      - phase5-llm-structured-json
  - id: phase5-deterministic-compiler
    content: "Phase 5: Ensure engine compiles QueryPlan → Ibis/SQL deterministically"
    status: pending
    dependencies:
      - phase5-validate-schema
  - id: phase5-tests
    content: "Phase 5: Write tests for constrained QueryPlan JSON output and deterministic compilation"
    status: pending
    dependencies:
      - phase5-deterministic-compiler
  - id: phase5-quality-gates
    content: "Phase 5: Run quality gates and commit"
    status: pending
    dependencies:
      - phase5-tests
  - id: phase6-reorder-pages
    content: "Phase 6: Reorder pages (Upload → Summary → Ask Questions) and gate legacy pages"
    status: pending
    dependencies:
      - phase5-quality-gates
  - id: phase6-upload-progress
    content: "Phase 6: Add real upload progress and move validation to warnings"
    status: pending
    dependencies:
      - phase6-reorder-pages
  - id: phase6-tests
    content: "Phase 6: Write tests for page ordering and upload progress"
    status: pending
    dependencies:
      - phase6-upload-progress
  - id: phase6-quality-gates
    content: "Phase 6: Run quality gates and commit"
    status: pending
    dependencies:
      - phase6-tests
  - id: phase7-enhanced-logging
    content: "Phase 7: Enhance query logging - comprehensive context, execution details, failures"
    status: pending
    dependencies:
      - phase6-quality-gates
  - id: phase7-golden-questions
    content: "Phase 7: Create golden questions YAML and eval harness runner"
    status: pending
    dependencies:
      - phase7-enhanced-logging
  - id: phase7-tests
    content: "Phase 7: Write tests for eval harness (golden questions → QueryPlan comparison)"
    status: pending
    dependencies:
      - phase7-golden-questions
  - id: phase7-quality-gates
    content: "Phase 7: Run quality gates and commit"
    status: pending
    dependencies:
      - phase7-tests
  - id: phase8-documentation-dry
    content: "Phase 8: Deferred from staff_engineer_feedback plan - State machine documentation, UI DRY refactoring (non-blocking)"
    status: pending
    dependencies:
      - phase7-quality-gates
  - id: phase8-tests
    content: "Phase 8: Write tests for UI DRY refactoring"
    status: pending
    dependencies:
      - phase8-documentation-dry
  - id: phase8-quality-gates
    content: "Phase 8: Run quality gates and commit"
    status: pending
    dependencies:
      - phase8-tests
---

# QueryPlan as Product: End-to-End Implementation + Critical Fixes

## Strategic Context

You have a real product with solid architecture (Polars, Ibis, DuckDB, clean separation of concerns), but it's suffering from:

1. **Too many UI surfaces** creating friction
2. **NL engine that can't reliably build correct plans** (regex-heavy, falling through tiers)
3. **Phase 3 QueryPlan execution implemented but not committed/complete** (still has confirmation gating, rerun weirdness)
4. **Critical security/stability issues** blocking production readiness

This plan makes QueryPlan the **only execution path**, fixes the planner architecture, and addresses all critical issues.

## Master Plan Status: Priority 0

**This is the single execution stream until:**
1. Results are deterministic
2. QueryPlan is the only path
3. Warnings are visible and logged
4. "It works sometimes" is dead

**All other plans are either:**
- **Merged into this plan** (as sub-phases) - see "Plan Integration" section below
- **Explicitly deferred** (post-MVP) - see "Deferred Plans" section below

## Plan Integration: How Other Plans Map to This

### Merged Plans

| Original Plan | Status | Location in Master Plan |
|---------------|--------|------------------------|
| `remove_confirmation_logic_and_add_progressive_thinking_indicators` | **Merged** | Phase 2 (confirmation removal) + Phase 2.5 (progressive thinking indicators) |
| `address_staff_engineer_feedback...` (P0 items) | **Merged** | Phase 1 (cache primitive, empty query rejection, scope canonicalization) |
| `fix_streamlit_chat_transcript_and_run_key_collisions` | **Covered** | Phase 1 (run_key determinism, pending state) |

### Deferred Plans (Post-MVP)

| Original Plan | Status | Reason |
|--------------|--------|--------|
| `address_staff_engineer_feedback...` (documentation, UI DRY) | **Deferred to Phase 8** | Non-blocking; can be done after core is stable |
| `reusable_visualization_framework_for_correlation_analysis` | **Deferred** | Visualization framework is premature until QueryPlan outputs are deterministic |
| `doctor-friendly_macos_dmg_installer` | **Deferred** | Distribution is Phase 10 problem; we're in Phase 0 |

**Note**: Minimal QueryPlan-driven visualization (chart_spec in result artifact) is included in Phase 3 to provide visual trust without building a full framework.

**Plan Execution Rules:**

- Follow test-first workflow (Red-Green-Refactor)
- Use Makefile commands only (`make format`, `make lint-fix`, `make type-check`, `make test-fast`, `make check`)
- Commit after each phase (implementation + tests)
- All quality gates must pass before proceeding
- Follow Polars-first patterns (no pandas in new code)
- Follow DRY principles (extract common patterns)

---

## Phase 0: Critical Security & Stability Fixes (Do First)

**Objective**: Fix critical security and stability issues blocking production readiness.

**Definition of Done (DoD)**:
- [ ] No silent mapping drops; unmapped values are surfaced (warning or error) and covered by tests
- [ ] SQL injection risk eliminated or minimized (strict validation, length limits, parameterized queries)
- [ ] RelationshipDetector extracted and testable independently
- [ ] All tests passing (`make test-core`)
- [ ] All quality gates passing (`make check`)

### Phase 0.1: Fix Silent Data Loss in Value Mapping

**Files**:

- `src/clinical_analytics/core/mapper.py` - `apply_outcome_transformations()`
- `src/clinical_analytics/core/semantic.py` - `get_base_view()` (Ibis mapping)

**Test-First Approach**:

1. **Write failing tests** (`tests/core/test_mapper_value_mapping.py`):
   ```python
   def test_unmapped_values_default_to_null_not_zero():
       """Unmapped values should become NULL, not 0."""
       # Arrange: DataFrame with unmapped value
       df = pl.DataFrame({
           "status": ["yes", "no", "unknown", None]
       })
       mapper = ColumnMapper({"outcome": {"source_column": "status", "type": "binary", "mapping": {"yes": 1, "no": 0}}})
       
       # Act
       result = mapper.apply_outcome_transformations(df)
       
       # Assert: "unknown" and None should be NULL, not 0
       assert result["outcome"].null_count() == 2
       assert (result["outcome"] == 0).sum() == 1  # Only "no" maps to 0
   
   def test_unmapped_values_raise_data_quality_error():
       """Unmapped values should raise DataQualityError with examples."""
       # Arrange: DataFrame with unmapped values
       df = pl.DataFrame({
           "status": ["yes", "no", "unknown", "pending"]
       })
       mapper = ColumnMapper({"outcome": {"source_column": "status", "type": "binary", "mapping": {"yes": 1, "no": 0}}})
       
       # Act & Assert
       with pytest.raises(DataQualityError) as exc_info:
           mapper.apply_outcome_transformations(df)
       assert "unknown" in str(exc_info.value)
       assert "pending" in str(exc_info.value)
   ```

2. **Run tests** (should fail):
   ```bash
   make test-core
   ```

3. **Implement fix**:

   - Change `expr = pl.lit(0)` → `expr = pl.lit(None)` in `mapper.py`
   - Add validation to count unmapped values before casting
   - Raise `DataQualityError` if unmapped values found
   - Include examples in error message

4. **Run tests again** (should pass):
   ```bash
   make test-core
   ```

5. **Run quality gates**:
   ```bash
   make format
   make lint-fix
   make type-check
   make check
   ```


### Phase 0.2: Review and Strengthen SQL Injection Mitigation

**Files**:

- `src/clinical_analytics/core/semantic.py` - `_register_source()` and `_validate_table_identifier()`

**Test-First Approach**:

1. **Write tests** (`tests/core/test_semantic_sql_injection.py`):
   ```python
   def test_table_identifier_rejects_sql_injection_attempts():
       """Table identifiers with SQL injection attempts should be rejected."""
       # Arrange: Malicious table names
       malicious_names = [
           "table; DROP TABLE users;",
           "table' OR '1'='1",
           "table\" UNION SELECT * FROM users",
           "table`; DELETE FROM users",
       ]
       
       # Act & Assert
       for name in malicious_names:
           with pytest.raises(ValueError, match="invalid.*identifier"):
               _validate_table_identifier(name)
   
   def test_table_identifier_enforces_length_limit():
       """Table identifiers should have reasonable length limits."""
       # Arrange: Very long name
       long_name = "a" * 1000
       
       # Act & Assert
       with pytest.raises(ValueError, match="too long"):
           _validate_table_identifier(long_name)
   ```

2. **Run tests** (may pass, but verify):
   ```bash
   make test-core
   ```

3. **Review and strengthen**:

   - Verify `_SQL_IDENTIFIER_PATTERN` is strict (alphanumeric + underscore only)
   - Add length limits (e.g., max 64 characters)
   - Consider using Ibis `table()` methods exclusively

4. **Run quality gates**:
   ```bash
   make check
   ```


### Phase 0.3: Extract RelationshipDetector from Multi-Table Handler (Stability/Refactor)

**Files**:

- `src/clinical_analytics/core/multi_table_handler.py` - Extract to `src/clinical_analytics/core/relationship_detector.py`

**Note**: This is a stability/refactor task (not security). It improves testability and maintainability by isolating graph logic from execution logic.

**Test-First Approach**:

1. **Write tests** (`tests/core/test_relationship_detector.py`):
   ```python
   def test_relationship_detector_finds_foreign_keys():
       """RelationshipDetector should find foreign key relationships."""
       # Arrange: Tables with relationships
       tables = {
           "patients": pl.DataFrame({"patient_id": [1, 2, 3]}),
           "admissions": pl.DataFrame({"patient_id": [1, 2], "admission_id": [1, 2]}),
       }
       detector = RelationshipDetector(tables)
       
       # Act
       relationships = detector.detect_relationships()
       
       # Assert: Should find patient_id relationship
       assert len(relationships) > 0
       assert any(r.parent_key == "patient_id" for r in relationships)
   ```

2. **Run tests** (should fail - class doesn't exist):
   ```bash
   make test-core
   ```

3. **Extract RelationshipDetector**:

   - Create `src/clinical_analytics/core/relationship_detector.py`
   - Move relationship detection logic from `MultiTableHandler`
   - Update `MultiTableHandler` to use `RelationshipDetector`

4. **Run tests again** (should pass):
   ```bash
   make test-core
   ```

5. **Run quality gates**:
   ```bash
   make check
   ```


### Phase 0 Commit

**Before committing**:

- [ ] All tests written and passing (`make test-core`)
- [ ] All quality gates passing (`make check`)
- [ ] No duplicate imports
- [ ] Code formatted (`make format`)

**Commit message**:

```
feat: Phase 0 - Critical security fixes and refactoring

- Fix silent data loss in value mapping (unmapped → NULL + error)
- Strengthen SQL injection mitigation (length limits, strict validation)
- Extract RelationshipDetector from MultiTableHandler

Tests: 12 tests passing (test_mapper_value_mapping, test_semantic_sql_injection, test_relationship_detector)
All quality gates passing
```

---

## Phase 1: Immediate Fixes (Blocking UX) + PR20 P0 Fixes

**Objective**: Fix immediate UX issues (run_key determinism, pending state, schema warnings) and PR20 P0 correctness fixes.

**Definition of Done (DoD)**:
- [ ] Same query + same scope yields same run_key every time
- [ ] Different scope yields different run_key
- [ ] Cache primitive fixed (st.cache_resource for semantic layer)
- [ ] Empty queries rejected at ingestion
- [ ] Scope canonicalization handles nested structures recursively
- [ ] All tests passing (`make test-core` and `make test-ui`)
- [ ] All quality gates passing (`make check`)

**Note**: Phase 1.5 (pending state pattern) and Phase 1.6 (move validation to warnings) have been moved to Phase 2:
- Phase 1.6 → Merged into Phase 2.2 (Remove Gating Logic - same work)
- Phase 1.5 → Phase 2.4 (Pending State Pattern - after confirmation UI removed)

### Phase 1.1: Fix run_key Determinism (Including Canonical Scope)

**Files**:

- `src/clinical_analytics/core/semantic.py` - `_generate_run_key()`
- `src/clinical_analytics/core/nl_query_engine.py` - `_intent_to_plan()`
- `src/clinical_analytics/ui/pages/3_💬_Ask_Questions.py` - `canonicalize_scope()`

**Test-First Approach**:

1. **Write tests** (`tests/core/test_semantic_run_key_determinism.py`):
   ```python
   def test_run_key_deterministic_for_same_plan_and_scope():
       """Same QueryPlan + same scope should produce same run_key."""
       # Arrange: Same plan, same scope
       plan1 = QueryPlan(intent="COUNT", metric="age", group_by="status", confidence=0.9)
       plan2 = QueryPlan(intent="COUNT", metric="age", group_by="status", confidence=0.9)
       scope = {"filters": {"status": "active"}, "cohort": "all"}
       
       # Act
       key1 = semantic_layer._generate_run_key(plan1, "average age by status", scope)
       key2 = semantic_layer._generate_run_key(plan2, "average age by status", scope)
       
       # Assert
       assert key1 == key2
   
   def test_run_key_different_for_different_scopes():
       """Different scopes should produce different run_keys."""
       # Arrange: Same plan, different scopes
       plan = QueryPlan(intent="COUNT", metric="age", group_by="status", confidence=0.9)
       scope1 = {"filters": {"status": "active"}}
       scope2 = {"filters": {"status": "inactive"}}
       
       # Act
       key1 = semantic_layer._generate_run_key(plan, "query", scope1)
       key2 = semantic_layer._generate_run_key(plan, "query", scope2)
       
       # Assert
       assert key1 != key2  # Different scopes should produce different keys
   
   def test_run_key_includes_all_plan_fields_and_scope():
       """Run key should include all QueryPlan fields + canonical scope for determinism."""
       # Arrange: Plans with different fields
       plan1 = QueryPlan(intent="COUNT", metric="age", group_by="status", entity_key="patient_id")
       plan2 = QueryPlan(intent="COUNT", metric="age", group_by="status", entity_key=None)
       scope = {"filters": {}, "cohort": "all"}
       
       # Act
       key1 = semantic_layer._generate_run_key(plan1, "query", scope)
       key2 = semantic_layer._generate_run_key(plan2, "query", scope)
       
       # Assert
       assert key1 != key2  # Different entity_key should produce different key
   ```

2. **Run tests** (may fail):
   ```bash
   make test-core
   ```

3. **Implement fix**:

   - Ensure `_generate_run_key()` includes:
     - Canonical plan (intent, metric, group_by, filters, entity_key, scope) - sorted keys
     - Normalized query text (lowercase, whitespace normalized)
     - Dataset version
     - **Canonical scope** (filters, cohort, grouping, entity grain) - recursively canonicalized
     - Optionally: engine version or QueryPlan schema version (for cache invalidation)
   - Normalize query text consistently (lowercase, whitespace)
   - Sort all fields for determinism
   - Ensure same logic in `_intent_to_plan()` and `_generate_run_key()`


### Phase 1.2: PR20 P0 - Fix Cache Primitive

**Files**:
- `src/clinical_analytics/ui/pages/3_💬_Ask_Questions.py` - `get_cached_semantic_layer()`

**Test-First Approach**:

1. **Write tests** (`tests/ui/pages/test_cache_primitive.py`):
   ```python
   def test_semantic_layer_uses_cache_resource():
       """Semantic layer should use st.cache_resource, not st.cache_data."""
       # Arrange: Get function
       import inspect
       from clinical_analytics.ui.pages.ask_questions import get_cached_semantic_layer
       
       # Act: Check decorator
       decorators = [d for d in inspect.getmembers(get_cached_semantic_layer) if d[0] == '__wrapped__']
       
       # Assert: Should use cache_resource
       # (This is a code inspection test - verify decorator is @st.cache_resource)
   ```

2. **Run tests**:
   ```bash
   make test-ui
   ```

3. **Implement fix**:
   - Change `@st.cache_data` → `@st.cache_resource` for `get_cached_semantic_layer()`
   - Add docstring explaining why (non-picklable objects: DB/ibis connections, lazy backends)

4. **Run tests again**:
   ```bash
   make test-ui
   ```

### Phase 1.3: PR20 P0 - Reject Empty Queries at Ingestion

**Files**:
- `src/clinical_analytics/ui/pages/3_💬_Ask_Questions.py` - Main query flow

**Test-First Approach**:

1. **Write tests** (`tests/ui/pages/test_empty_query_rejection.py`):
   ```python
   def test_empty_query_rejected_at_ingestion():
       """Empty queries should be rejected before run_key generation."""
       # Arrange: Empty/None queries
       empty_queries = [None, "", "   ", "\t\n"]
       
       # Act & Assert: Should not generate run_key, not append message, not compute
       for query in empty_queries:
           # Simulate UI flow
           # Assert: No run_key generated, no message appended, no computation
   ```

2. **Run tests**:
   ```bash
   make test-ui
   ```

3. **Implement fix**:
   - Add early return/no-op for empty normalized queries
   - Reject after `normalize_query()` call, before run_key generation
   - Don't generate run_key, don't append message, don't compute

4. **Run tests again**:
   ```bash
   make test-ui
   ```

### Phase 1.4: PR20 P0 - Recursive Deterministic Scope Canonicalization

**Files**:
- `src/clinical_analytics/ui/pages/3_💬_Ask_Questions.py` - `canonicalize_scope()`

**Test-First Approach**:

1. **Write tests** (`tests/core/test_scope_canonicalization.py`):
   ```python
   def test_scope_canonicalization_handles_nested_dicts():
       """Scope canonicalization should handle nested dicts recursively."""
       # Arrange: Nested scope
       scope = {
           "filters": {"status": "active", "age": {"min": 18, "max": 65}},
           "cohort": "all"
       }
       
       # Act
       canonical1 = canonicalize_scope(scope)
       canonical2 = canonicalize_scope(scope)  # Same input
       
       # Assert: Should be identical
       assert canonical1 == canonical2
   
   def test_scope_canonicalization_handles_nested_lists():
       """Scope canonicalization should handle nested lists."""
       # Arrange: Scope with lists
       scope = {"filters": {"ids": [3, 1, 2]}}
       
       # Act
       canonical = canonicalize_scope(scope)
       
       # Assert: Lists should be sorted if order doesn't matter
       # OR preserved if order matters semantically
   ```

2. **Run tests**:
   ```bash
   make test-core
   ```

3. **Implement fix**:
   - Implement recursive canonicalization:
     - Handle nested dicts (sort keys, recurse on values)
     - Handle nested lists (sort if order doesn't matter, preserve if it does)
     - Handle enums (convert to value)
     - Handle dataclasses (convert to dict, recurse)
     - Handle non-JSON-native values (normalize appropriately)

4. **Run tests again**:
   ```bash
   make test-core
   ```



### Phase 1 Commit

**Before committing**:

- [ ] All tests written and passing (`make test-core` and `make test-ui`)
- [ ] All quality gates passing (`make check`)

**Commit message**:

```
feat: Phase 1 - Immediate UX fixes + PR20 P0 fixes

- Fix run_key determinism (canonical plan + normalized query + canonical scope)
- PR20 P0: Fix cache primitive (st.cache_data → st.cache_resource)
- PR20 P0: Reject empty queries at ingestion
- PR20 P0: Recursive deterministic scope canonicalization

Note: Phase 1.5 (pending state) and Phase 1.6 (validation warnings) moved to Phase 2

Tests: 15+ tests passing (test_run_key_determinism, test_cache_primitive, test_empty_query_rejection, test_scope_canonicalization)
All quality gates passing
```

---

## Phase 2: Remove Confirmation Gating (After Observability)

**Objective**: Remove all confirmation gating, convert to warnings, always execute. **CRITICAL**: Add observability first to prevent silent bad plan execution.

**Definition of Done (DoD)**:
- [ ] No `requires_confirmation` exists anywhere in codebase
- [ ] Warnings are visible in UI and logged
- [ ] Execution always proceeds deterministically (no blocking)
- [ ] "Why confidence is low" explanation available
- [ ] Plan diff visibility in UI/logs
- [ ] Persistent "executed with warnings" record
- [ ] Pending state pattern implemented (prevents accidental execution when warnings present)
- [ ] Validation moved to warnings (non-blocking, from Phase 1.6)
- [ ] All tests passing (`make test-core` and `make test-ui`)
- [ ] All quality gates passing (`make check`)

### Phase 2.1: Add Observability Before Removing Gating

**Files**:
- `src/clinical_analytics/core/semantic.py` - `execute_query_plan()`
- `src/clinical_analytics/ui/pages/3_💬_Ask_Questions.py` - UI display

**Test-First Approach**:

1. **Write tests** (`tests/core/test_semantic_observability.py`):
   ```python
   def test_warnings_list_included_in_result():
       """Execution result should include warnings list."""
       # Arrange: QueryPlan with low confidence
       plan = QueryPlan(intent="COUNT", confidence=0.3)
       
       # Act
       result = semantic_layer.execute_query_plan(plan)
       
       # Assert: Warnings present
       assert "warnings" in result
       assert isinstance(result["warnings"], list)
   
   def test_confidence_explanation_included():
       """Warnings should explain why confidence is low."""
       # Arrange: QueryPlan with low confidence
       plan = QueryPlan(intent="COUNT", confidence=0.3)
       
       # Act
       result = semantic_layer.execute_query_plan(plan)
       
       # Assert: Warning explains low confidence
       assert any("confidence" in w.lower() for w in result["warnings"])
   ```

2. **Run tests**:
   ```bash
   make test-core
   ```

3. **Implement observability**:
   - Add warnings list to execution result
   - Include "why confidence is low" explanation in warnings
   - Add plan diff visibility (show what changed from expected)
   - Add persistent "executed with warnings" record (log to query logger)
   - Display warnings in UI with clear explanations

4. **Run tests again**:
   ```bash
   make test-core
   ```

### Phase 2.2: Remove Gating Logic from execute_query_plan() (Includes Phase 1.6)

**Files**:

- `src/clinical_analytics/core/semantic.py` - `execute_query_plan()`

**Note**: This phase includes **Phase 1.6: Move validation to warnings** - same work, merged here.

**Test-First Approach**:

1. **Write tests** (`tests/core/test_semantic_no_gating.py`):
   ```python
   def test_execute_query_plan_always_attempts_execution():
       """execute_query_plan() should always attempt execution, never block."""
       # Arrange: Low confidence plan
       plan = QueryPlan(intent="COUNT", confidence=0.3)  # Below threshold
       
       # Act
       result = semantic_layer.execute_query_plan(plan)
       
       # Assert: No requires_confirmation, warnings present
       assert "requires_confirmation" not in result
       assert "warnings" in result
       assert len(result["warnings"]) > 0
       # Execution attempted (may succeed or fail)
   
   def test_execute_query_plan_returns_warnings_not_blocking():
       """Confidence/completeness/validation issues should produce warnings, not block."""
       # Arrange: Incomplete plan
       plan = QueryPlan(intent="COUNT")  # Missing entity_key
       
       # Act
       result = semantic_layer.execute_query_plan(plan)
       
       # Assert: Warnings present, execution attempted
       assert "warnings" in result
       assert "requires_confirmation" not in result
   
   def test_validation_warnings_non_blocking():
       """Validation issues should produce warnings, not block execution (Phase 1.6)."""
       # Arrange: QueryPlan with missing column
       plan = QueryPlan(intent="COUNT", metric="nonexistent_column")
       
       # Act
       result = semantic_layer.execute_query_plan(plan)
       
       # Assert: Warnings present, but execution attempted
       assert "warnings" in result
       assert len(result["warnings"]) > 0
       # Execution may fail, but not blocked by validation
   ```

2. **Run tests** (should fail - gating still exists):
   ```bash
   make test-core
   ```

3. **Implement fix**:

   - Remove confidence gating (convert to warning)
   - Remove completeness gating (convert to warning)
   - **Remove validation gating (convert to warning) - Phase 1.6**
   - **Move validation checks to warnings (non-blocking) - Phase 1.6**
   - **Store warnings in execution result - Phase 1.6**
   - Always attempt execution
   - Update return signature: remove `requires_confirmation`, add `warnings: list[str]`

4. **Run tests again**:
   ```bash
   make test-core
   ```


### Phase 2.3: Remove Confirmation UI

**Files**:

- `src/clinical_analytics/ui/pages/3_💬_Ask_Questions.py`

**Test-First Approach**:

1. **Write tests** (`tests/ui/pages/test_ask_questions_no_confirmation.py`):
   ```python
   def test_confirmation_ui_removed():
       """_render_confirmation_ui() should not exist."""
       # Assert: Function should not exist
       assert not hasattr(module, "_render_confirmation_ui")
   
   def test_execution_flow_no_confirmation_checks():
       """Execution flow should not check requires_confirmation."""
       # Arrange: QueryPlan with warnings
       plan = QueryPlan(intent="COUNT", confidence=0.5)
       execution_result = {"success": True, "warnings": ["Low confidence"]}
       
       # Act: Simulate UI flow
       
       # Assert: No confirmation UI rendered, warnings shown inline
   ```

2. **Run tests**:
   ```bash
   make test-ui
   ```

3. **Implement fix**:

- Delete `_render_confirmation_ui()` function
- Remove all `requires_confirmation` checks
- Always execute immediately, show warnings inline

4. **Run tests again**:
   ```bash
   make test-ui
   ```

### Phase 2.4: Implement Pending State Pattern (Phase 1.5)

**Files**:

- `src/clinical_analytics/ui/pages/3_💬_Ask_Questions.py` - Execution flow

**Note**: This is **Phase 1.5** moved to Phase 2.4. It runs after confirmation UI is removed, providing a way to prevent accidental execution when warnings are present.

**Test-First Approach**:

1. **Write tests** (`tests/ui/pages/test_ask_questions_pending_state.py`):
   ```python
   def test_pending_plan_stored_in_session_state():
       """Pending plan with warnings should be stored in session state."""
       # Arrange: QueryPlan with warnings
       plan = QueryPlan(intent="COUNT", confidence=0.5)  # Low confidence
       execution_result = {"success": False, "warnings": ["Low confidence"]}
       
       # Act: Simulate UI flow
       # (Use mock for Streamlit session state)
       
       # Assert: Pending plan stored
       assert f"pending_plan_{dataset_version}" in st.session_state
   
   def test_pending_plan_requires_explicit_confirmation():
       """Pending plan should not execute until explicit confirmation."""
       # Arrange: Pending plan in session state
       st.session_state[f"pending_plan_{dataset_version}"] = plan
       
       # Act: Attempt execution
       
       # Assert: Execution blocked until confirmation
   ```

2. **Run tests**:
   ```bash
   make test-ui
   ```

3. **Implement fix**:

   - Store pending plan in `st.session_state[f"pending_plan_{dataset_version}"]` when warnings present
   - Check if plan is pending before re-executing
   - Only execute on explicit "Confirm and Run" button click (or similar user action)
   - Clear pending state after confirmation
   - Display warnings inline with pending plan

4. **Run tests again**:
   ```bash
   make test-ui
   ```


### Phase 2 Commit

**Before committing**:

- [ ] All tests written and passing
- [ ] All quality gates passing (`make check`)

**Commit message**:

```
feat: Phase 2 - Remove confirmation gating (with observability)

- Add observability: warnings list, "why confidence is low", plan diff visibility, persistent "executed with warnings" record
- Convert confidence/completeness/validation gating to warnings (includes Phase 1.6: move validation to warnings)
- Remove requires_confirmation from return signature
- Delete _render_confirmation_ui() and all confirmation checks
- Implement pending state pattern (Phase 1.5) - prevent accidental execution when warnings present
- Always execute immediately, show warnings inline

Tests: 15+ tests passing (test_observability, test_no_gating, test_validation_warnings, test_no_confirmation, test_pending_state)
All quality gates passing
```

---

## Phase 2.5: Progressive Thinking Indicators (After Gating Removal is Safe)

**Objective**: Add progressive thinking indicators and retry UX to improve user experience after core execution is deterministic.

**Definition of Done (DoD)**:
- [ ] Progressive thinking steps displayed (parsing, planning, validation, execution)
- [ ] Retry logic for backend errors with exponential backoff
- [ ] Stepwise UI narration shows query plan interpretation as it processes
- [ ] All tests passing (`make test-ui` and `make test-core`)
- [ ] All quality gates passing (`make check`)

**Note**: This phase is merged from `remove_confirmation_logic_and_add_progressive_thinking_indicators` plan. It runs after Phase 2 (gating removal) is complete and safe.

### Phase 2.5.1: Add Progressive Thinking Indicator

**Files**:
- `src/clinical_analytics/ui/pages/3_💬_Ask_Questions.py` - Add `_render_thinking_indicator()`

**Test-First Approach**:

1. **Write tests** (`tests/ui/pages/test_thinking_indicator.py`):
   ```python
   def test_thinking_indicator_shows_progressive_steps():
       """Thinking indicator should show progressive steps as query processes."""
       # Arrange: Steps list
       steps = [
           {"status": "completed", "text": "Parsed query", "details": "..."},
           {"status": "processing", "text": "Building query plan", "details": "..."},
           {"status": "pending", "text": "Executing query", "details": "..."},
       ]
       
       # Act: Render thinking indicator
       # (Use mock for Streamlit rendering)
       
       # Assert: Steps displayed correctly
   ```

2. **Run tests**:
   ```bash
   make test-ui
   ```

3. **Implement fix**:
   - Create `_render_thinking_indicator()` function
   - Show progressive steps: parsing → planning → validation → execution → processing
   - Display query plan interpretation as it processes
   - Use `st.status()` or custom container for step-by-step display

4. **Run tests again**:
   ```bash
   make test-ui
   ```

### Phase 2.5.2: Add Retry Logic for Backend Errors

**Files**:
- `src/clinical_analytics/core/semantic.py` - Add `_execute_plan_with_retry()`

**Test-First Approach**:

1. **Write tests** (`tests/core/test_semantic_retry_logic.py`):
   ```python
   def test_retry_logic_handles_backend_errors():
       """Retry logic should handle backend errors with exponential backoff."""
       # Arrange: Mock backend error
       
       # Act: Execute with retry
       
       # Assert: Retries with backoff, eventually succeeds or fails after max retries
   ```

2. **Run tests**:
   ```bash
   make test-core
   ```

3. **Implement fix**:
   - Add `_execute_plan_with_retry()` method with exponential backoff
   - Handle `AttributeError: '_record_batch_readers_consumed'` (backend init issue)
   - Retry on connection errors and transient execution errors
   - Log retry attempts

4. **Run tests again**:
   ```bash
   make test-core
   ```

### Phase 2.5 Commit

**Before committing**:
- [ ] All tests written and passing
- [ ] All quality gates passing (`make check`)

**Commit message**:
```
feat: Phase 2.5 - Progressive thinking indicators and retry logic

- Add progressive thinking indicator (stepwise UI narration)
- Add retry logic for backend errors with exponential backoff
- Display query plan interpretation as it processes

Tests: 6 tests passing (test_thinking_indicator, test_retry_logic)
All quality gates passing
```

---

## Phase 3: Make QueryPlan the Only Execution Path

**Objective**: Ensure all queries use QueryPlan, remove legacy execution paths.

**Definition of Done (DoD)**:
- [ ] Grep-able assertions or tests prevent any non-QueryPlan execution path from running
- [ ] All queries go through `nl_query_engine.parse_query()` → `QueryPlan`
- [ ] All execution goes through `semantic_layer.execute_query_plan()`
- [ ] No legacy execution paths exist
- [ ] All tests passing (`make test-core` and `make test-ui`)
- [ ] All quality gates passing (`make check`)

### Phase 3.1: Remove Legacy Execution Paths

**Files**:

- `src/clinical_analytics/ui/pages/3_💬_Ask_Questions.py`
- `src/clinical_analytics/core/nl_query_engine.py`

**Test-First Approach**:

1. **Write tests** (`tests/core/test_queryplan_only_path.py`):
   ```python
   def test_question_engine_always_produces_queryplan():
       """QuestionEngine should always produce QueryPlan, never None."""
       # Arrange: Any query
       query = "how many patients?"
       
       # Act
       plan = question_engine.parse_query(query)
       
       # Assert: Always QueryPlan
       assert plan is not None
       assert isinstance(plan, QueryPlan)
   
   def test_no_legacy_execution_paths():
       """No code should execute queries without QueryPlan."""
       # This is a code review test - verify no direct execution
       # Check that all execution goes through semantic_layer.execute_query_plan()
   ```

2. **Run tests**:
   ```bash
   make test-core
   ```

3. **Implement fix**:

   - Remove any code that executes queries without QueryPlan
- Ensure `QuestionEngine` always produces QueryPlan (never returns None)
- Remove fallback execution logic

4. **Run tests again**:
   ```bash
   make test-core
   ```


### Phase 3.2: Enforce QueryPlan Contract

**Files**:

- `src/clinical_analytics/core/semantic.py`
- `src/clinical_analytics/core/nl_query_engine.py`

**Test-First Approach**:

1. **Write tests** (`tests/core/test_queryplan_contract.py`):
   ```python
   def test_all_queries_go_through_queryplan():
       """All queries must go through nl_query_engine.parse_query() → QueryPlan."""
       # Integration test: Verify full flow
       query = "how many patients?"
       plan = nl_query_engine.parse_query(query)
       result = semantic_layer.execute_query_plan(plan)
       
       # Assert: QueryPlan used throughout
       assert isinstance(plan, QueryPlan)
       assert "run_key" in result
   ```

2. **Run tests**:
   ```bash
   make test-core
   ```

3. **Implement fix**:

- Add assertions/logging to catch any bypasses
   - Ensure all execution goes through `semantic_layer.execute_query_plan()`

4. **Run tests again**:
   ```bash
   make test-core
   ```

### Phase 3.3: Add Minimal QueryPlan-Driven Visualization

**Files**:
- `src/clinical_analytics/core/query_plan.py` - Add `ChartSpec` to result artifact
- `src/clinical_analytics/ui/pages/3_💬_Ask_Questions.py` - Render chart_spec

**Objective**: Add minimal visualization without building a full framework. Provides visual trust while keeping scope manageable. This is NOT the full visualization framework (deferred).

**Test-First Approach**:

1. **Write tests** (`tests/core/test_chart_spec.py`):
   ```python
   def test_result_artifact_includes_chart_spec():
       """Execution result should include chart_spec for visualization."""
       # Arrange: QueryPlan execution
       plan = QueryPlan(intent="COUNT", group_by="status")
       
       # Act
       result = semantic_layer.execute_query_plan(plan)
       
       # Assert: Chart spec present
       assert "chart_spec" in result
       assert result["chart_spec"]["type"] in ["bar", "line", "hist"]
       assert "title" in result["chart_spec"]
   
   def test_chart_spec_deterministic_from_plan():
       """Chart spec should be deterministic from QueryPlan."""
       # Arrange: Same plan
       plan = QueryPlan(intent="COUNT", group_by="status")
       
       # Act: Generate chart spec multiple times
       spec1 = generate_chart_spec(plan)
       spec2 = generate_chart_spec(plan)
       
       # Assert: Same spec
       assert spec1 == spec2
   ```

2. **Run tests** (should fail):
   ```bash
   make test-core
   ```

3. **Implement fix**:
   - Add `ChartSpec` dataclass:
     ```python
     @dataclass
     class ChartSpec:
         type: Literal["bar", "line", "hist"]
         x: str | None
         y: str | None
         group_by: str | None
         title: str
     ```
   - Generate chart_spec from QueryPlan (deterministic, tied to plan)
   - Include chart_spec in execution result artifact
   - Render basic charts from chart_spec in UI (simple bar/line/hist)

4. **Run tests again**:
   ```bash
   make test-core
   ```

**Note**: This is minimal visualization tied directly to QueryPlan. Full visualization framework (`reusable_visualization_framework_for_correlation_analysis`) is deferred until Phase 3 (QueryPlan-only) and Phase 7 (eval harness) are complete.


### Phase 3 Commit

**Before committing**:

- [ ] All tests written and passing
- [ ] All quality gates passing (`make check`)

**Commit message**:

```
feat: Phase 3 - Make QueryPlan the only execution path + minimal visualization

- Remove legacy execution paths
- Ensure QuestionEngine always produces QueryPlan
- Enforce QueryPlan contract with assertions/logging
- Add minimal QueryPlan-driven visualization (chart_spec in result artifact)

Tests: 8 tests passing (test_queryplan_only_path, test_queryplan_contract, test_chart_spec)
All quality gates passing
```

---

## Phase 4: Improve Multi-Part Query Parsing

**Objective**: Enhance planner to correctly identify primary/secondary variables and grouping logic.

### Phase 4.1: Enhance Multi-Part Query Parsing

**Files**:

- `src/clinical_analytics/core/nl_query_engine.py` - `_llm_parse()` and `_intent_to_plan()`

**Test-First Approach**:

1. **Write tests** (`tests/core/test_nl_query_multipart.py`):
   ```python
   def test_parse_compare_x_by_y_extracts_primary_and_grouping():
       """'compare X by Y' should extract metric=X, group_by=Y."""
       # Arrange
       query = "compare mortality by treatment"
       
       # Act
       plan = nl_query_engine.parse_query(query)
       
       # Assert
       assert plan.intent == "COMPARE_GROUPS"
       assert plan.metric == "mortality"
       assert plan.group_by == "treatment"
   
   def test_parse_most_common_sets_requires_grouping():
       """'most common X' should set requires_grouping=True."""
       # Arrange
       query = "what was the most common HIV regiment?"
       
       # Act
       plan = nl_query_engine.parse_query(query)
       
       # Assert
       assert plan.requires_grouping is True
       assert plan.group_by is not None
   
   def test_parse_what_predicts_extracts_metric():
       """'what predicts Y' should extract metric=Y."""
       # Arrange
       query = "what predicts mortality?"
       
       # Act
       plan = nl_query_engine.parse_query(query)
       
       # Assert
       assert plan.intent == "FIND_PREDICTORS"
       assert plan.metric == "mortality"
   ```

2. **Run tests** (should fail):
   ```bash
   make test-core
   ```

3. **Implement fix**:

- Extract primary vs secondary variables explicitly
- Detect grouping logic from query patterns ("most common", "by", "per", "breakdown")
   - Set `requires_grouping=True` when grouping is implied

4. **Run tests again**:
   ```bash
   make test-core
   ```


### Phase 4.2: Add Type-Aware Logic

**Files**:

- `src/clinical_analytics/core/nl_query_engine.py`

**Test-First Approach**:

1. **Write tests** (`tests/core/test_nl_query_type_aware.py`):
   ```python
   def test_type_aware_sets_appropriate_intent():
       """Type-aware logic should set appropriate intent based on variable types."""
       # Arrange: Categorical variable
       query = "how many patients by status?"
       
       # Act
       plan = nl_query_engine.parse_query(query)
       
       # Assert: COUNT for categorical grouping
       assert plan.intent == "COUNT"
   
   def test_type_aware_validates_grouping_variable_categorical():
       """Grouping variable should be validated as categorical."""
       # Arrange: Numeric variable used for grouping
       query = "compare mortality by age"  # age is numeric, should group by age categories
       
       # Act
       plan = nl_query_engine.parse_query(query)
       
       # Assert: Should handle numeric grouping appropriately
   ```

2. **Run tests**:
   ```bash
   make test-core
   ```

3. **Implement fix**:

- Distinguish categorical vs numeric variables in plan
   - Set appropriate intent based on variable types
- Validate grouping variable is categorical before grouping

4. **Run tests again**:
   ```bash
   make test-core
   ```


### Phase 4 Commit

**Before committing**:

- [ ] All tests written and passing
- [ ] All quality gates passing (`make check`)

**Commit message**:

```
feat: Phase 4 - Improve multi-part query parsing

- Extract primary vs secondary variables explicitly
- Detect grouping logic from query patterns
- Add type-aware logic (categorical vs numeric)
- Set requires_grouping flag appropriately

Tests: 10 tests passing (test_multipart, test_type_aware)
All quality gates passing
```

---

## Phase 5: LLM as Constrained Planner

**Objective**: Require LLM to output constrained QueryPlan JSON, not freeform code.

### Phase 5.1: Require QueryPlan JSON Schema

**Files**:

- `src/clinical_analytics/core/nl_query_engine.py` - `_build_llm_prompt()` and `_extract_query_intent_from_llm_response()`

**Test-First Approach**:

1. **Write tests** (`tests/core/test_nl_query_llm_constrained.py`):
   ```python
   def test_llm_returns_queryplan_json_schema():
       """LLM should return JSON matching QueryPlan schema."""
       # Arrange: Query requiring LLM fallback
       query = "complex multi-part query"
       
       # Act
       plan = nl_query_engine.parse_query(query)
       
       # Assert: Valid QueryPlan
       assert isinstance(plan, QueryPlan)
       assert plan.intent in VALID_INTENT_TYPES
   
   def test_llm_rejects_malformed_plans():
       """LLM should reject malformed plans with clear error."""
       # Arrange: Mock LLM returning invalid JSON
       
       # Act & Assert
       with pytest.raises(ValueError, match="invalid.*plan"):
           nl_query_engine.parse_query(query)
   ```

2. **Run tests**:
   ```bash
   make test-core
   ```

3. **Implement fix**:

   - Update prompt to require strict JSON matching QueryPlan schema
- Validate JSON structure before accepting
- Reject malformed plans with clear error

4. **Run tests again**:
   ```bash
   make test-core
   ```


### Phase 5.2: Validate QueryPlan Against Dataclass

**Files**:

- `src/clinical_analytics/core/query_plan.py` - Add validation
- `src/clinical_analytics/core/nl_query_engine.py` - Use validation

**Test-First Approach**:

1. **Write tests** (`tests/core/test_queryplan_validation.py`):
   ```python
   def test_queryplan_validation_rejects_invalid_fields():
       """QueryPlan validation should reject invalid fields."""
       # Arrange: Invalid intent
       invalid_plan_dict = {"intent": "INVALID_INTENT", "metric": "age"}
       
       # Act & Assert
       with pytest.raises(ValueError, match="invalid.*intent"):
           QueryPlan.from_dict(invalid_plan_dict)
   
   def test_queryplan_validation_rejects_missing_required_fields():
       """QueryPlan validation should reject missing required fields."""
       # Arrange: Missing intent
       incomplete_plan_dict = {"metric": "age"}
       
       # Act & Assert
       with pytest.raises(ValueError, match="missing.*intent"):
           QueryPlan.from_dict(incomplete_plan_dict)
   ```

2. **Run tests**:
   ```bash
   make test-core
   ```

3. **Implement fix**:

   - Add `from_dict()` class method to QueryPlan
- Validate all fields against QueryPlan dataclass
   - Reject plans with invalid or missing required fields

4. **Run tests again**:
   ```bash
   make test-core
   ```


### Phase 5.3: Ensure Deterministic Compilation

**Files**:

- `src/clinical_analytics/core/semantic.py` - `execute_query_plan()`

**Test-First Approach**:

1. **Write tests** (`tests/core/test_semantic_deterministic_compilation.py`):
   ```python
   def test_queryplan_compiles_to_ibis_deterministically():
       """QueryPlan should compile to Ibis expressions deterministically."""
       # Arrange: Same QueryPlan
       plan = QueryPlan(intent="COUNT", metric="age", group_by="status")
       
       # Act: Compile multiple times
       expr1 = semantic_layer._compile_queryplan_to_ibis(plan)
       expr2 = semantic_layer._compile_queryplan_to_ibis(plan)
       
       # Assert: Same expression
       assert str(expr1) == str(expr2)
   ```

2. **Run tests**:
   ```bash
   make test-core
   ```

3. **Implement fix**:

   - Ensure `execute_query_plan()` compiles QueryPlan to Ibis expressions deterministically
   - No freeform code generation
   - All execution paths are testable

4. **Run tests again**:
   ```bash
   make test-core
   ```


### Phase 5 Commit

**Before committing**:

- [ ] All tests written and passing
- [ ] All quality gates passing (`make check`)

**Commit message**:

```
feat: Phase 5 - LLM as constrained planner

- Require LLM to return QueryPlan JSON schema (not freeform)
- Validate QueryPlan against dataclass
- Ensure deterministic compilation to Ibis/SQL

Tests: 8 tests passing (test_llm_constrained, test_queryplan_validation, test_deterministic_compilation)
All quality gates passing
```

---

## Phase 6: UX Reduction and Page Gating

**Objective**: Reduce UI surface area, reorder pages, add upload progress. **Important**: Page gating is not just "nice UX" - it reduces state complexity and rerun bugs. It's a reliability move disguised as product focus.

**Definition of Done (DoD)**:
- [ ] Pages 2-6 hidden/gated (V1 MVP mode)
- [ ] Pages reordered: Upload → Summary → Ask Questions
- [ ] Real upload progress shown
- [ ] State complexity reduced (fewer pages = fewer rerun bugs)
- [ ] All tests passing (`make test-ui`)
- [ ] All quality gates passing (`make check`)

### Phase 6.1: Page Gating / V1 MVP Mode

**Files**:
- `src/clinical_analytics/ui/pages/` - Page visibility/gating logic

**Test-First Approach**:

1. **Write tests** (`tests/ui/pages/test_page_gating.py`):
   ```python
   def test_pages_2_6_gated_in_v1_mvp_mode():
       """Pages 2-6 should be hidden/gated in V1 MVP mode."""
       # Arrange: V1 MVP mode enabled
       
       # Act: Check page visibility
       
       # Assert: Only Upload, Summary, Ask Questions visible
   ```

2. **Run tests**:
   ```bash
   make test-ui
   ```

3. **Implement fix**:
   - Add V1 MVP mode flag
   - Hide/gate Pages 2-6 (Descriptive Statistics, Correlation Analysis, etc.)
   - Reduce state complexity (fewer pages = fewer rerun bugs)

4. **Run tests again**:
   ```bash
   make test-ui
   ```

### Phase 6.2: Reorder Pages

**Files**:

- `src/clinical_analytics/ui/pages/` - Page ordering

**Test-First Approach**:

1. **Write tests** (`tests/ui/pages/test_page_ordering.py`):
   ```python
   def test_pages_ordered_upload_summary_ask():
       """Pages should be ordered: Upload → Summary → Ask Questions."""
       # Arrange: Get page order
       
       # Act
       page_order = get_page_order()
       
       # Assert
       assert page_order[0] == "Upload"
       assert page_order[1] == "Summary"
       assert page_order[2] == "Ask Questions"
   ```

2. **Run tests**:
   ```bash
   make test-ui
   ```

3. **Implement fix**:

   - Reorder pages: Upload → Summary → Ask Questions
   - Gate legacy pages (hide/disable pages not in V1 MVP)

4. **Run tests again**:
   ```bash
   make test-ui
   ```


### Phase 6.3: Add Upload Progress

**Files**:

- `src/clinical_analytics/ui/pages/1_📤_Upload_Data.py`

**Test-First Approach**:

1. **Write tests** (`tests/ui/pages/test_upload_progress.py`):
   ```python
   def test_upload_shows_progress():
       """Upload should show actual file processing progress."""
       # Integration test: Verify progress indicators
   ```

2. **Run tests**:
   ```bash
   make test-ui
   ```

3. **Implement fix**:

   - Add real upload progress (show actual file processing)
   - Move "doctor-can't-fix" validation to warnings

4. **Run tests again**:
   ```bash
   make test-ui
   ```


### Phase 6 Commit

**Before committing**:

- [ ] All tests written and passing
- [ ] All quality gates passing (`make check`)

**Commit message**:

```
feat: Phase 6 - UX reduction and page gating

- Page gating / V1 MVP mode (hide Pages 2-6, reduce state complexity)
- Reorder pages (Upload → Summary → Ask Questions)
- Add real upload progress
- Move validation to warnings

Tests: 6 tests passing (test_page_gating, test_page_ordering, test_upload_progress)
All quality gates passing
```

---

## Phase 7: Eval Harness

**Objective**: Add comprehensive logging and eval harness for data-driven improvement.

### Phase 7.1: Enhance Query Logging

**Files**:

- `src/clinical_analytics/storage/query_logger.py`
- `src/clinical_analytics/core/semantic.py`
- `src/clinical_analytics/core/nl_query_engine.py`

**Test-First Approach**:

1. **Write tests** (`tests/storage/test_query_logger_enhanced.py`):
   ```python
   def test_log_comprehensive_query_context():
       """Query logger should log comprehensive query context."""
       # Arrange: Query with full context
       query = "how many patients?"
       plan = QueryPlan(intent="COUNT", confidence=0.9)
       
       # Act
       query_logger.log_query_with_context(query, plan)
       
       # Assert: All context logged
       # Verify log entry contains: query text, matched vars, confidence, execution path, etc.
   ```

2. **Run tests**:
   ```bash
   make test-storage
   ```

3. **Implement fix**:

   - Log comprehensive query context (query text, matched vars, confidence, execution path, aliasing)
   - Log execution details (run_key, QueryPlan JSON, execution time, result row count, warnings)
   - Log failures (query text, parsing tier, failure reason, suggested fixes)

4. **Run tests again**:
   ```bash
   make test-storage
   ```


### Phase 7.2: Create Eval Harness

**Files**:

- `tests/eval/golden_questions.yaml` - New file
- `tests/eval/query_eval.py` - New file

**Test-First Approach**:

1. **Write tests** (`tests/eval/test_eval_harness.py`):
   ```python
   def test_eval_harness_runs_golden_questions():
       """Eval harness should run golden questions and compare plans."""
       # Arrange: Golden questions
       questions = load_golden_questions()
       
       # Act
       results = run_eval_harness(questions)
       
       # Assert: Results contain accuracy metrics
       assert "accuracy" in results
       assert "plan_matches" in results
   ```

2. **Run tests**:
   ```bash
   make test-eval  # May need to add this to Makefile
   ```

3. **Implement fix**:

   - Create `tests/eval/golden_questions.yaml` with production query failures
   - Create `tests/eval/query_eval.py` to run golden questions
   - Compare produced QueryPlan to expected
   - Report accuracy metrics

4. **Run tests again**:
   ```bash
   make test-eval
   ```


### Phase 7 Commit

**Before committing**:

- [ ] All tests written and passing
- [ ] All quality gates passing (`make check`)

**Commit message**:

```
feat: Phase 7 - Eval harness and enhanced logging

- Enhance query logging (comprehensive context, execution details, failures)
- Create golden questions YAML
- Create eval harness runner
- Report accuracy metrics

Tests: 6 tests passing (test_query_logger_enhanced, test_eval_harness)
All quality gates passing
```

---

## Phase 8: Documentation and UI DRY Refactoring (Deferred)

**Objective**: Address non-blocking items from `address_staff_engineer_feedback` plan: state machine documentation, UI DRY refactoring.

**Definition of Done (DoD)**:
- [ ] State machine documented (session state transitions)
- [ ] UI DRY refactoring complete (extract duplicated dataset loading)
- [ ] All tests passing (`make test-ui`)
- [ ] All quality gates passing (`make check`)

**Note**: This phase is deferred from the staff engineer feedback plan. It's non-blocking and can be done after core functionality is stable.

### Phase 8.1: Document Session State Machine

**Files**:
- `src/clinical_analytics/ui/pages/3_💬_Ask_Questions.py` - Add state machine documentation

**Changes**:
- Add comprehensive state machine documentation at top of `main()` function
- Document state keys (`analysis_context`, `intent_signal`, `use_nl_query`)
- Document allowed transitions (None → "nl_parsed" → "executed")
- Document fragility notes (order matters, reruns can invalidate assumptions)
- Reference future refactor (ADR008)

### Phase 8.2: Extract Dataset Loading Component (UI DRY)

**Files**:
- `src/clinical_analytics/ui/components/dataset_loader.py` - New file
- `src/clinical_analytics/ui/pages/` - Use new component

**Changes**:
- Extract duplicated dataset loading logic into reusable component
- Eliminate 350-560 lines of duplication across UI pages
- High ROI refactoring (reduces maintenance burden)

**Test-First Approach**:

1. **Write tests** (`tests/ui/components/test_dataset_loader.py`):
   ```python
   def test_dataset_loader_encapsulates_loading_logic():
       """Dataset loader should encapsulate all dataset loading logic."""
       # Arrange: Mock dataset registry
       
       # Act: Use dataset loader component
       
       # Assert: Loading logic centralized
   ```

2. **Run tests**:
   ```bash
   make test-ui
   ```

3. **Implement fix**:
   - Create `dataset_loader.py` component
   - Move duplicated loading logic from all pages
   - Update pages to use new component

4. **Run tests again**:
   ```bash
   make test-ui
   ```

### Phase 8 Commit

**Before committing**:
- [ ] All tests written and passing
- [ ] All quality gates passing (`make check`)

**Commit message**:
```
feat: Phase 8 - Documentation and UI DRY refactoring

- Document session state machine (transitions, fragility notes)
- Extract dataset loading component (eliminate 350-560 lines of duplication)
- High ROI refactoring (reduces maintenance burden)

Tests: 4 tests passing (test_state_machine_docs, test_dataset_loader)
All quality gates passing
```

---

## Deferred Plans (Post-MVP)

These plans are explicitly deferred until core functionality is stable. They remain as separate plans and should not be executed until this master plan is complete through Phase 7 (eval harness).

### 1. Reusable Visualization Framework
**Plan**: `reusable_visualization_framework_for_correlation_analysis_68d33cb3.plan.md`
**Status**: Deferred
**Reason**: Visualization framework is premature until QueryPlan outputs are deterministic. Minimal QueryPlan-driven visualization (chart_spec) is included in Phase 3 to provide visual trust without building a full framework.

**When to revisit**: After Phase 3 (QueryPlan-only path) and Phase 7 (eval harness) are complete. Only then will we know which visualizations are actually useful.

### 2. Doctor-Friendly macOS DMG Installer
**Plan**: `doctor-friendly_macos_dmg_installer_6f0341eb.plan.md`
**Status**: Deferred
**Reason**: Distribution is a Phase 10 problem; we're in Phase 0. Shipping a DMG for a tool that sometimes answers BMI when asked about LDL is premature.

**When to revisit**: After Phase 3 (QueryPlan-only path) and Phase 7 (eval harness) are complete. The tool must be reliable before distribution.

---

## Success Criteria

- [ ] QueryPlan is the only execution path (no legacy bypasses)
- [ ] No confirmation gating (warnings only, always execute)
- [ ] No accidental low-confidence execution (pending state pattern)
- [ ] Deterministic run_key (same query + plan = same key)
- [ ] No silent data loss (unmapped values → NULL + error)
- [ ] SQL injection risk eliminated
- [ ] Multi-table handler refactored (RelationshipDetector extracted)
- [ ] LLM outputs constrained QueryPlan JSON (not freeform code)
- [ ] Multi-part queries parsed correctly (primary vs secondary variables)
- [ ] Comprehensive logging for eval harness
- [ ] All tests passing (`make test`)
- [ ] Quality gates passing (`make check`)

---

## Quality Gate Checklist (Before Each Commit)

**MANDATORY - Never commit without:**

1. **Tests written and passing**:
   ```bash
   make test-fast        # Quick feedback
   # OR module-specific:
   make test-core        # Core module tests
   make test-ui          # UI module tests
   make test-storage     # Storage module tests
   ```

2. **Code quality**:
   ```bash
   make format           # Auto-format
   make lint-fix         # Auto-fix linting
   make type-check       # Type checking
   ```

3. **Full quality gate**:
   ```bash
   make check            # All checks (format, lint, type, test)
   ```

4. **Verify**:

   - [ ] No duplicate imports
   - [ ] All tests passing
   - [ ] Code formatted
   - [ ] No linting errors
   - [ ] Type checking passes
   - [ ] Commit includes both implementation AND tests

---

## Makefile Commands Reference

**Always use Makefile commands (never run tools directly):**

- `make format` / `make format-check` - Format code
- `make lint` / `make lint-fix` - Lint code
- `make type-check` - Type checking
- `make test-fast` - Fast tests (skip slow)
- `make test-core` - Core module tests
- `make test-ui` - UI module tests
- `make test-storage` - Storage module tests
- `make test` - All tests
- `make check` - Full quality gate (format, lint, type, test)
- `make ci` - CI-specific checks

---

## Testing Patterns

**Follow AAA pattern (Arrange-Act-Assert):**

```python
def test_example():
    # Arrange: Set up test data and dependencies
    df = pl.DataFrame({"col": [1, 2, 3]})
    
    # Act: Execute the unit under test
    result = transform(df)
    
    # Assert: Verify expected outcomes
    assert result.height == 3
```

**Use Polars testing assertions:**

```python
import polars.testing as plt

plt.assert_frame_equal(result, expected)
```

**Use shared fixtures from `conftest.py` (DRY principle):**

```python
def test_example(sample_cohort, mock_semantic_layer):
    # Use fixtures instead of duplicating setup
    result = process(sample_cohort, mock_semantic_layer)
    assert result is not None
```