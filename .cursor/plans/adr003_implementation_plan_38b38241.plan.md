---
name: ADR003 Implementation Plan
overview: "Implement ADR003: Clinical Trust Protocol + Adaptive Alias Persistence. Three phases: (1) Trust UI with verification expanders and patient-level export, (2) Adaptive alias persistence scoped per dataset, (3) Semantic layer QueryPlan execution with type-aware validation and confidence gating."
todos:
  - id: phase1-tests
    content: Write Phase 1 test specifications (11 tests) in tests/ui/test_trust_ui.py
    status: pending
  - id: phase1-trust-ui
    content: Implement _render_trust_verification() function in 3_💬_Ask_Questions.py
    status: pending
    dependencies:
      - phase1-tests
  - id: phase1-integration
    content: Integrate trust UI into result rendering functions (_render_focused_descriptive, render_comparison_analysis, count analysis)
    status: pending
    dependencies:
      - phase1-trust-ui
  - id: phase1-commit
    content: Run quality gates (make check) and commit Phase 1 with all tests passing
    status: pending
    dependencies:
      - phase1-integration
  - id: phase2-tests
    content: Write Phase 2 test specifications (8 tests) in tests/core/test_semantic_alias_persistence.py
    status: pending
  - id: phase2-alias-persistence
    content: Extend SemanticLayer to load and persist user aliases (add_user_alias, load on init)
    status: pending
    dependencies:
      - phase2-tests
  - id: phase2-ui
    content: Add 'Add alias?' UI in error handling (3_💬_Ask_Questions.py)
    status: pending
    dependencies:
      - phase2-alias-persistence
  - id: phase2-commit
    content: Run quality gates (make check) and commit Phase 2 with all tests passing
    status: pending
    dependencies:
      - phase2-ui
  - id: phase3-tests
    content: Write Phase 3 test specifications (15 tests) in tests/core/test_semantic_queryplan_execution.py
    status: pending
  - id: phase3-queryplan-extend
    content: Extend QueryPlan with ADR003 contract fields (requires_filters, requires_grouping, entity_key, scope)
    status: pending
    dependencies:
      - phase3-tests
  - id: phase3-execution
    content: Implement execute_query_plan() with confidence gating, validation, type-aware execution
    status: pending
    dependencies:
      - phase3-queryplan-extend
  - id: phase3-integration
    content: Integrate QueryPlan execution into analysis flow (execute_analysis_with_idempotency)
    status: pending
    dependencies:
      - phase3-execution
  - id: phase3-commit
    content: Run quality gates (make check) and commit Phase 3 with all tests passing
    status: pending
    dependencies:
      - phase3-integration
---

# ADR003 Implementation Plan: Clinical Trust Protocol + Adaptive Alias Persistence

## Overview

This plan implements ADR003 in three phases following test-first development, phase commits, and quality gates per `.cursor/rules/104-plan-execution-hygiene.mdc`.**Architecture**: ADR003 extends existing infrastructure:

- **QueryPlan** (`src/clinical_analytics/core/query_plan.py`) - Extend with ADR003 contract fields
- **SemanticLayer** (`src/clinical_analytics/core/semantic.py`) - Add alias persistence and QueryPlan execution
- **UI Pages** (`src/clinical_analytics/ui/pages/3_💬_Ask_Questions.py`) - Add trust verification UI
- **Metadata Storage** (`src/clinical_analytics/ui/storage/user_datasets.py`) - Persist aliases per `(upload_id, dataset_version)`

**Critical Constraints**:

- ADR003 does NOT own NLU parsing (ADR001's responsibility)
- Semantic layer executes QueryPlans, does not parse natural language
- All validation is against explicit QueryPlan contract fields, not inferred from NL

## Phase 1: Trust Layer (P0 - Mandatory Verification UI)

### 1.1 Test Specifications

**File**: `tests/ui/test_trust_ui.py`**Test Cases**:

1. `test_trust_ui_shows_query_plan_raw_fields` - Verify raw QueryPlan (intent, metric, filters) displayed
2. `test_trust_ui_shows_alias_resolved_plan` - Verify canonical column names shown after alias resolution
3. `test_trust_ui_shows_effective_execution` - Verify effective execution display (dataset, entity_key, resolved columns, effective filters, cohort size)
4. `test_trust_ui_shows_run_key_and_audit_trail` - Verify run_key + query text displayed
5. `test_trust_ui_patient_level_export_capped` - Verify patient-level export limited to 100 rows by default
6. `test_trust_ui_patient_level_export_full_requires_confirmation` - Verify full export requires explicit confirmation
7. `test_trust_ui_cohort_size_calculation` - Verify `count_total` and `count_filtered` computed correctly for percentage reporting
8. `test_trust_ui_tautology_detection` - Verify non-restrictive filters labeled or dropped from effective filters display
9. `test_trust_ui_integration_with_descriptive_analysis` - Verify trust UI appears in `_render_focused_descriptive()`
10. `test_trust_ui_integration_with_comparison_analysis` - Verify trust UI appears in `render_comparison_analysis()`
11. `test_trust_ui_integration_with_count_analysis` - Verify trust UI appears in count analysis rendering

**Fixtures** (add to `tests/conftest.py`):

- `sample_query_plan()` - Factory for QueryPlan with various configurations
- `sample_analysis_result()` - Factory for analysis result dicts
- `mock_semantic_layer_with_aliases()` - SemanticLayer with alias index populated

### 1.2 Implementation Tasks

**File**: `src/clinical_analytics/ui/pages/3_💬_Ask_Questions.py`**Task 1.2.1**: Create `_render_trust_verification()` function

- Input: `query_plan: QueryPlan`, `result: dict`, `cohort: pl.DataFrame`, `semantic_layer: SemanticLayer`, `dataset_version: str`
- Display expander "🔎 Verify: Show source patients"
- Show sections:
- **Raw QueryPlan**: Intent, metric, filters (as parsed)
- **Alias-Resolved Plan**: Canonical column names (post-alias resolution)
- **Effective Execution**:
    - Dataset used (from `dataset_version`)
    - Entity key used for COUNT (if applicable)
    - Resolved canonical column names
    - Effective filters (post-normalization, with actual values)
    - Cohort size: `count_total` (denominator), `count_filtered` (numerator), percentage
    - Run-key and audit trail: `run_key` + "generated from: [query]"
- **Patient-Level Export**: DataFrame with audit columns (patient_id, primary_variable, filter columns), capped at 100 rows, download button for full export

**Task 1.2.2**: Integrate trust UI into result rendering functions

- Modify `_render_focused_descriptive()` to call `_render_trust_verification()` after main results
- Modify `render_comparison_analysis()` to call `_render_trust_verification()` after main results
- Modify count analysis rendering (in `render_analysis_by_type()`) to call `_render_trust_verification()`

**Task 1.2.3**: Implement effective filter normalization

- Function: `_normalize_effective_filters(filters: list[FilterSpec], semantic_layer: SemanticLayer) -> list[dict]`
- Resolve aliases to canonical column names
- Detect tautologies (non-restrictive filters, e.g., `IN [1..9]` when all valid codes)
- Return list of effective filters with "non-restrictive" label for tautologies

**Task 1.2.4**: Implement cohort size calculation

- Function: `_calculate_cohort_size(cohort: pl.DataFrame, filters: list[FilterSpec], entity_key: str | None) -> dict[str, int]`
- Compute `count_total`: Same entity_key, no filters (denominator)
- Compute `count_filtered`: With filters applied (numerator)
- Return `{"count_total": int, "count_filtered": int, "percentage": float}`

**File**: `src/clinical_analytics/ui/components/trust_ui.py` (NEW)**Task 1.2.5**: Extract trust UI components to reusable module

- Move `_render_trust_verification()` to `TrustUI.render_verification()` (class-based for testability)
- Move filter normalization and cohort size calculation to pure functions (no Streamlit dependencies)
- Follow DRY: Single source of truth for trust UI rendering

### 1.3 Quality Gates

**Before Phase 1 Commit**:

```bash
make format
make lint-fix
make type-check
make test-ui  # Run UI tests including new trust UI tests
make check    # Full quality gate
```

**Commit Message Template**:

```javascript
feat: Phase 1 - Trust Layer UI (ADR003)

- Add trust verification expander with QueryPlan display
- Show alias-resolved plan and effective execution
- Implement patient-level export (capped at 100 rows)
- Calculate cohort size (count_total, count_filtered, percentage)
- Integrate trust UI into descriptive, comparison, and count analysis rendering

All tests passing: 11/11
```



## Phase 2: Adaptive Alias Persistence (P0 - Error-Driven Learning)

### 2.1 Test Specifications

**File**: `tests/core/test_semantic_alias_persistence.py`**Test Cases**:

1. `test_add_user_alias_persists_to_metadata` - Verify `add_user_alias()` saves to metadata JSON
2. `test_load_user_aliases_on_initialization` - Verify SemanticLayer loads persisted aliases on init
3. `test_user_aliases_override_system_aliases` - Verify user aliases take precedence over system aliases
4. `test_alias_scope_per_dataset` - Verify aliases scoped to `(upload_id, dataset_version)` don't leak to other datasets
5. `test_orphaned_alias_handling` - Verify aliases marked orphaned when target column missing after schema change
6. `test_alias_collision_detection` - Verify collisions surfaced in UI, never silently remapped
7. `test_alias_persistence_format` - Verify metadata JSON format matches ADR002 schema
8. `test_alias_normalization_consistency` - Verify user aliases normalized consistently with system aliases

**Fixtures** (add to `tests/conftest.py`):

- `sample_metadata_with_aliases()` - Metadata dict with alias_mappings structure
- `mock_metadata_storage()` - Mock metadata file I/O for testing persistence

### 2.2 Implementation Tasks

**File**: `src/clinical_analytics/core/semantic.py`**Task 2.2.1**: Extend `SemanticLayer.__init__()` to load user aliases

- Load metadata JSON from `data/uploads/metadata/{upload_id}.json`
- Extract `alias_mappings.user_aliases` dict
- Merge with system aliases (user aliases override system for same normalized key)
- Handle orphaned aliases (target column missing) - mark and ignore

**Task 2.2.2**: Add `add_user_alias()` method

- Signature: `add_user_alias(term: str, column: str, upload_id: str, dataset_version: str) -> None`
- Validate column exists in schema
- Normalize term using `_normalize_alias()`
- Check for collisions (multiple columns match same alias)
- Persist to metadata JSON: `metadata["alias_mappings"]["user_aliases"][term] = column`
- Update in-memory alias index (for current session)

**Task 2.2.3**: Extend `_build_alias_index()` to merge user aliases

- Load user aliases from metadata (if available)
- Merge with system aliases (user aliases override)
- Handle collisions: Surface in UI, never silently remap

**File**: `src/clinical_analytics/ui/pages/3_💬_Ask_Questions.py`**Task 2.2.4**: Add "Add alias?" UI in error handling

- When unknown term detected, show error: `st.error(f"Unknown term: '{term}'")`
- Add expander "💡 Add alias?" with:
- Selectbox: "Map to column:" (available columns)
- Button: "Save"
- On save: Call `semantic_layer.add_user_alias(term, col, upload_id, dataset_version)`
- Show success: "Retry your query"

**File**: `src/clinical_analytics/ui/storage/user_datasets.py`**Task 2.2.5**: Extend metadata JSON schema for alias persistence

- Ensure `alias_mappings` structure in metadata:
  ```json
    {
      "alias_mappings": {
        "user_aliases": {"VL": "viral_load", "LDL": "LDL mg/dL"},
        "system_aliases": {...}
      }
    }
  ```

- Load aliases on metadata read
- Save aliases on metadata write (atomic update)

### 2.3 Quality Gates

**Before Phase 2 Commit**:

```bash
make format
make lint-fix
make type-check
make test-core  # Run core tests including alias persistence tests
make check      # Full quality gate
```

**Commit Message Template**:

```javascript
feat: Phase 2 - Adaptive Alias Persistence (ADR003)

- Extend SemanticLayer to load and persist user aliases
- Add add_user_alias() method with collision detection
- Implement alias scope per (upload_id, dataset_version)
- Add "Add alias?" UI in error handling
- Extend metadata JSON schema for alias_mappings

All tests passing: 8/8
```



## Phase 3: Semantic Layer QueryPlan Execution (P1 - Type-Aware Execution)

### 3.1 Test Specifications

**File**: `tests/core/test_semantic_queryplan_execution.py`**Test Cases**:

1. `test_execute_query_plan_validates_columns_exist` - Verify executor checks columns exist after alias resolution
2. `test_execute_query_plan_validates_operators` - Verify executor validates operators are supported
3. `test_execute_query_plan_validates_type_compatibility` - Verify executor checks type compatibility (e.g., can't use ">" on categorical)
4. `test_execute_query_plan_count_scope_validation` - Verify executor refuses `scope="all"` with filters, refuses `scope="filtered"` with empty filters
5. `test_execute_query_plan_count_entity_key_validation` - Verify executor requires entity_key for COUNT (or defaults to dataset primary key)
6. `test_execute_query_plan_confidence_gating` - Verify executor refuses execution when `confidence < threshold`
7. `test_execute_query_plan_completeness_gating` - Verify executor refuses execution when required fields missing (COUNT requires entity_key OR grouping_variable)
8. `test_execute_query_plan_type_aware_categorical` - Verify categorical columns return frequency tables, not mean/median
9. `test_execute_query_plan_type_aware_numeric` - Verify numeric columns return descriptive statistics
10. `test_execute_query_plan_count_intent_execution` - Verify COUNT intent uses SQL aggregation with entity_key
11. `test_execute_query_plan_breakdown_validation` - Verify executor refuses `grouping_variable=entity_key` when query implies categorical breakdown
12. `test_execute_query_plan_high_cardinality_detection` - Verify executor refuses high-cardinality grouping (near-unique values)
13. `test_execute_query_plan_filter_deduplication` - Verify executor detects and warns/deduplicates redundant filters (filtering and grouping on same field)
14. `test_execute_query_plan_run_key_determinism` - Verify run_key generated deterministically from canonical plan + query text
15. `test_execute_query_plan_refuses_invalid_plans` - Verify executor raises clear errors for contract violations

**Fixtures** (add to `tests/conftest.py`):

- `sample_query_plan_count()` - QueryPlan with COUNT intent
- `sample_query_plan_describe()` - QueryPlan with DESCRIBE intent
- `sample_cohort_with_categorical()` - Polars DataFrame with categorical encoding ("1: Yes 2: No")
- `sample_cohort_with_numeric()` - Polars DataFrame with numeric columns

### 3.2 Implementation Tasks

**File**: `src/clinical_analytics/core/query_plan.py`**Task 3.2.1**: Extend QueryPlan with ADR003 contract fields

- Add `requires_filters: bool = False` - True if query explicitly requires filters
- Add `requires_grouping: bool = False` - True if query pattern implies breakdown
- Add `entity_key: str | None = None` - For COUNT: entity to count
- Add `scope: Literal["all", "filtered"] = "all"` - For COUNT: count all rows vs filtered cohort

**File**: `src/clinical_analytics/core/semantic.py`**Task 3.2.2**: Add `execute_query_plan()` method

- Signature: `execute_query_plan(plan: QueryPlan, confidence_threshold: float = 0.75) -> dict[str, Any]`
- **Confidence and Completeness Gating** (hard gate):
- Gate: `confidence >= threshold AND is_complete AND validation_passes`
- `is_complete` = required fields present for intent:
    - COUNT: requires `entity_key` OR `grouping_variable`
    - DESCRIBE: requires `primary_variable` (metric)
    - COMPARE_GROUPS: requires `primary_variable` AND `grouping_variable`
- If gate fails: Return error dict with `requires_confirmation=True`, `failure_reason` explaining what's missing
- No side effects: Do NOT cache, store, or execute when gate fails
- **Validate QueryPlan Contract**:
- Check columns exist (after alias resolution)
- Validate operators are supported
- Check type compatibility
- **COUNT-specific validation**:
    - Refuse `scope="all"` with filters
    - Refuse `scope="filtered"` with empty filters (if `requires_filters=True`)
    - Require `entity_key` (default to dataset primary key if known, else refuse)
- **Breakdown validation**:
    - Refuse `grouping_variable=entity_key` with error "Cannot group by entity key"
    - Refuse high-cardinality grouping (near-unique values) with error "Grouping by [field] yields ~[N] groups"
    - Refuse `requires_grouping=True` with `grouping_variable=None` with error "This query requires grouping"
- **Filter Deduplication**:
- Detect redundant filters (filtering and grouping on same field)
- Warn and deduplicate (keep grouping, remove redundant filter)
- **Type-Aware Execution**:
- Categorical columns: Detect encoding patterns ("1: Yes 2: No") → return frequency tables
- Numeric columns: Return descriptive statistics (mean, median, std dev)
- COUNT intent: Use `query(metrics=["count"], filters={...}, entity_key=...)`
- DESCRIBE intent: Use `query(metrics=["avg_X"])` for numeric, frequency table for categorical
- **Run-Key Generation**:
- Generate deterministic `run_key = hash(dataset_version + canonical_plan_json + query_text_signature)`
- `canonical_plan_json` includes: intent, metric (canonical), group_by (canonical), filters (normalized, sorted), entity_key, scope
- `query_text_signature` is normalized query text (lowercase, whitespace normalized)

**Task 3.2.3**: Implement categorical detection

- Function: `_detect_categorical_encoding(column: pl.Series) -> bool`
- Detect patterns: "1: Yes 2: No", numeric codes with limited distinct values (< 20), string categories
- Return True if categorical, False if numeric

**Task 3.2.4**: Implement type-aware aggregation

- Function: `_execute_type_aware_aggregation(df: pl.DataFrame, metric: str, intent: str) -> dict[str, Any]`
- If categorical: Return frequency table (value counts, percentages)
- If numeric: Return descriptive statistics (mean, median, std dev, min, max)
- Use Polars-native operations (no pandas)

**File**: `src/clinical_analytics/ui/pages/3_💬_Ask_Questions.py`**Task 3.2.5**: Integrate QueryPlan execution into analysis flow

- Modify `execute_analysis_with_idempotency()` to call `semantic_layer.execute_query_plan()` if QueryPlan present
- Handle `requires_confirmation=True` response: Show plan with confidence score, completeness status, missing fields, require explicit "Confirm and Run" button
- Update trust UI to show effective execution from QueryPlan execution result

### 3.3 Quality Gates

**Before Phase 3 Commit**:

```bash
make format
make lint-fix
make type-check
make test-core  # Run core tests including QueryPlan execution tests
make test-ui    # Run UI tests to verify integration
make check      # Full quality gate
```

**Commit Message Template**:

```javascript
feat: Phase 3 - Semantic Layer QueryPlan Execution (ADR003)

- Extend QueryPlan with ADR003 contract fields (requires_filters, requires_grouping, entity_key, scope)
- Add execute_query_plan() with confidence and completeness gating
- Implement type-aware execution (categorical → frequency, numeric → stats)
- Add COUNT intent validation and execution
- Implement breakdown validation and filter deduplication
- Generate deterministic run_key from canonical plan + query text

All tests passing: 15/15
```



## Cross-Phase Considerations

### DRY Principles

**Configuration Extraction**:

- Extract confidence thresholds to `src/clinical_analytics/core/nl_query_config.py` (already exists)
- Extract alias persistence schema to shared constants module

**Reusable Components**:

- Trust UI components in `src/clinical_analytics/ui/components/trust_ui.py` (Phase 1)
- Alias persistence logic in `SemanticLayer` (Phase 2)
- QueryPlan execution in `SemanticLayer` (Phase 3)

### Testing Hygiene

**Test Structure**:

- Use AAA pattern (Arrange-Act-Assert) in all tests
- Descriptive test names: `test_unit_scenario_expectedBehavior`
- Use fixtures from `tests/conftest.py` (no duplicate imports)

**Test Isolation**:

- Each test is independent (no shared mutable state)
- Mock external dependencies (file I/O, Streamlit session state)

### Polars-First

**Data Processing**:

- Use `pl.DataFrame` for all data processing (no pandas except at render boundary)
- Use Polars-native operations: `df.height` (not `len(df)`), `df.to_dicts()` (not `to_dict(orient="records")`)
- Lazy execution by default: Use `pl.scan_parquet()` when possible

**Pandas Exception** (only at render boundary):

```python
# PANDAS EXCEPTION: Required for Streamlit st.dataframe display
# TODO: Remove when Streamlit supports Polars natively
import pandas as pd
```



### Staff Engineer Standards

**Error Handling**:

- Fail fast, fail loud: Raise explicit exceptions with clear messages
- Never catch generic `Exception` - catch specific exceptions
- Validate at boundaries: Check QueryPlan contract before execution

**Observability**:

- Structured logging with `structlog` (already in codebase)
- Log confidence scores, completeness status, validation failures

**Idempotency**:

- Run-key generation is deterministic (same plan + query = same run_key)
- Results cached by run_key (no side effects from repeated execution)

## Implementation Order

1. **Phase 1** (Trust UI) - Enables user verification immediately
2. **Phase 2** (Alias Persistence) - Enables error-driven learning
3. **Phase 3** (QueryPlan Execution) - Enables type-aware execution and validation

**Dependencies**:

- Phase 1: Independent (can start immediately)
- Phase 2: Independent (can start immediately, but benefits from Phase 1 error UI)
- Phase 3: Requires Phase 2 (needs alias resolution) and Phase 1 (needs trust UI for validation display)

## Success Criteria

**Phase 1 Complete**:

- [ ] All 11 trust UI tests passing
- [ ] Trust verification expander appears in all result rendering functions
- [ ] Patient-level export works (capped at 100 rows, full export requires confirmation)
- [ ] Cohort size calculation correct (count_total, count_filtered, percentage)

**Phase 2 Complete**:

- [ ] All 8 alias persistence tests passing
- [ ] User aliases persist to metadata JSON per `(upload_id, dataset_version)`
- [ ] Aliases load on SemanticLayer initialization
- [ ] "Add alias?" UI appears in error handling

**Phase 3 Complete**:

- [ ] All 15 QueryPlan execution tests passing
- [ ] Confidence and completeness gating enforced (hard gate, no side effects)
- [ ] Type-aware execution works (categorical → frequency, numeric → stats)
- [ ] COUNT intent validation and execution correct