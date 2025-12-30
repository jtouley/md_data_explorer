---
name: ADR003 Implementation Plan
overview: "Implement ADR003: Clinical Trust Protocol + Adaptive Alias Persistence. Four phases: (0) Tier 3 LLM fallback with local Ollama, (1) Trust UI with verification expanders and patient-level export, (2) Adaptive alias persistence scoped per dataset, (3) Semantic layer QueryPlan execution with type-aware validation and confidence gating."
progress: "Phase 0 ✅ (Commit f979c96) - 12/12 tests passing. Tier 3 LLM Fallback with local Ollama implemented. Phase 1 ✅ (Commit af46ca9) - 11/11 tests passing. Trust UI with patient-level export implemented."
todos:
  - id: phase0-tests
    content: Write Phase 0 test specifications (12 tests) in tests/core/test_llm_fallback.py
    status: completed
  - id: phase0-ollama-client
    content: Implement OllamaClient class with connection handling and model management
    status: completed
    dependencies:
      - phase0-tests
  - id: phase0-rag-context
    content: Implement RAG context builder from semantic layer metadata
    status: completed
    dependencies:
      - phase0-ollama-client
  - id: phase0-structured-parsing
    content: Implement structured JSON extraction with schema validation and retries
    status: completed
    dependencies:
      - phase0-rag-context
  - id: phase0-integration
    content: Integrate _llm_parse() into NLQueryEngine with error handling and timeouts
    status: completed
    dependencies:
      - phase0-structured-parsing
  - id: phase0-commit
    content: Run quality gates (make check) and commit Phase 0 with all tests passing
    status: completed
    dependencies:
      - phase0-integration
  - id: phase1-tests
    content: Write Phase 1 test specifications (11 tests) in tests/ui/test_trust_ui.py
    status: completed
  - id: phase1-trust-ui
    content: Implement _render_trust_verification() function in 3_💬_Ask_Questions.py
    status: completed
    dependencies:
      - phase1-tests
  - id: phase1-integration
    content: Integrate trust UI into result rendering functions (_render_focused_descriptive, render_comparison_analysis, count analysis)
    status: completed
    dependencies:
      - phase1-trust-ui
  - id: phase1-commit
    content: Run quality gates (make check) and commit Phase 1 with all tests passing
    status: completed
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
    dependencies:
      - phase0-commit
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

This plan implements ADR003 in four phases following test-first development, phase commits, and quality gates per `.cursor/rules/104-plan-execution-hygiene.mdc`.**Current Status**: Phase 0 ✅ and Phase 1 ✅ complete. Phase 2 (Alias Persistence) can start immediately. Phase 3 (QueryPlan Execution) is now unblocked since Phase 0 provides valid QueryPlans with confidence >= 0.75.**Architecture**: ADR003 extends existing infrastructure:

- **QueryPlan** (`src/clinical_analytics/core/query_plan.py`) - Extend with ADR003 contract fields
- **SemanticLayer** (`src/clinical_analytics/core/semantic.py`) - Add alias persistence and QueryPlan execution
- **UI Pages** (`src/clinical_analytics/ui/pages/3_💬_Ask_Questions.py`) - Add trust verification UI
- **Metadata Storage** (`src/clinical_analytics/ui/storage/user_datasets.py`) - Persist aliases per `(upload_id, dataset_version)`

**Critical Constraints**:

- ADR003 does NOT own NLU parsing (ADR001's responsibility)
- Semantic layer executes QueryPlans, does not parse natural language
- All validation is against explicit QueryPlan contract fields, not inferred from NL
- **Phase 0 Dependency**: Tier 3 LLM fallback must be functional before Phase 3 execution layer can work (confidence gating requires valid QueryPlans)

## Phase 0: Tier 3 LLM Fallback with Local Ollama (P0 - Bootstrap Dependency)

### 0.1 Test Specifications

**File**: `tests/core/test_llm_fallback.py`**Test Cases**:

1. `test_ollama_client_connection_success` - Verify OllamaClient connects to local Ollama service
2. `test_ollama_client_connection_failure_handles_gracefully` - Verify connection failure returns None without crashing
3. `test_ollama_client_model_available` - Verify model availability check (llama3.1:8b or llama3.2:3b)
4. `test_ollama_client_model_not_available_handles_gracefully` - Verify missing model handled gracefully
5. `test_rag_context_builder_includes_columns` - Verify RAG context includes available columns from semantic layer
6. `test_rag_context_builder_includes_aliases` - Verify RAG context includes alias mappings
7. `test_rag_context_builder_includes_examples` - Verify RAG context includes example queries
8. `test_structured_json_extraction_valid_schema` - Verify JSON extraction with valid schema returns QueryIntent
9. `test_structured_json_extraction_invalid_json_retries` - Verify invalid JSON triggers retry (up to 3 attempts)
10. `test_structured_json_extraction_timeout_handling` - Verify timeout handling (5s default)
11. `test_llm_parse_returns_query_intent_with_confidence` - Verify _llm_parse() returns QueryIntent with confidence >= 0.5
12. `test_llm_parse_fallback_to_stub_on_error` - Verify _llm_parse() falls back to stub (confidence=0.3) on unrecoverable errors

**Fixtures** (add to `tests/conftest.py`):

- `mock_ollama_client()` - Mock OllamaClient for testing without actual Ollama service
- `sample_rag_context()` - Factory for RAG context dict with columns, aliases, examples
- `sample_semantic_layer_metadata()` - Mock semantic layer with metadata for RAG context

### 0.2 Implementation Tasks

**File**: `src/clinical_analytics/core/llm_client.py` (NEW)**Task 0.2.1**: Create `OllamaClient` class

- Signature: `class OllamaClient:`
- Methods:
- `__init__(model: str = "llama3.1:8b", base_url: str = "http://localhost:11434", timeout: float = 5.0)`
- `is_available() -> bool` - Check if Ollama service is running
- `is_model_available(model: str) -> bool` - Check if model is downloaded
- `generate(prompt: str, system_prompt: str | None = None, json_mode: bool = True) -> str | None` - Generate response with JSON mode
- `_check_connection() -> bool` - Internal method to check Ollama service
- Error handling: Return None on connection failures, don't raise exceptions
- Timeout: Default 5 seconds, configurable
- Privacy: All data stays local (no external API calls)

**Task 0.2.2**: Add Ollama dependency (optional)

- Add `ollama>=0.1.0` to `pyproject.toml` under `[project.optional-dependencies]` as `llm`
- Or use `requests` library (already in dependencies via other packages) to call Ollama REST API directly
- **Decision**: Use `requests` to avoid new dependency (Ollama exposes REST API at `http://localhost:11434`)

**File**: `src/clinical_analytics/core/nl_query_engine.py`**Task 0.2.3**: Implement RAG context builder

- Function: `_build_rag_context(self, query: str) -> dict[str, Any]`
- Extract from semantic layer:
- Available columns (from `semantic_layer.get_base_view().columns`)
- Alias mappings (from `semantic_layer.get_column_alias_index()`)
- Example queries (from query templates)
- Format as structured context for LLM prompt
- Return dict with keys: `columns`, `aliases`, `examples`, `query`

**Task 0.2.4**: Implement structured prompt builder

- Function: `_build_llm_prompt(self, query: str, context: dict[str, Any]) -> tuple[str, str]`
- System prompt: Instructions for JSON schema extraction
- User prompt: Query + RAG context (columns, aliases, examples)
- Return tuple: `(system_prompt, user_prompt)`
- JSON schema: Define expected QueryIntent structure (intent_type, primary_variable, grouping_variable, filters, confidence)

**Task 0.2.5**: Implement structured JSON extraction with retries

- Function: `_extract_query_intent_from_llm_response(self, response: str, max_retries: int = 3) -> QueryIntent | None`
- Parse JSON response with schema validation
- Retry on invalid JSON (up to 3 attempts with exponential backoff)
- Validate required fields: `intent_type`, `confidence`
- Return `QueryIntent` if valid, `None` if all retries fail
- Use `json.loads()` with defensive error handling

**Task 0.2.6**: Replace `_llm_parse()` stub with full implementation

- Signature: `_llm_parse(self, query: str) -> QueryIntent`
- Steps:

1. Check if Ollama is available (lazy check, cache result)
2. Build RAG context from semantic layer
3. Build structured prompt (system + user)
4. Call OllamaClient.generate() with JSON mode
5. Extract QueryIntent from response (with retries)
6. Fallback to stub (confidence=0.3) if all steps fail

- Error handling: Catch all exceptions, log with structlog, return stub on failure
- Timeout: Use OllamaClient timeout (5s default)
- Confidence: Set confidence based on extraction quality (0.5-0.9 range)

**File**: `src/clinical_analytics/core/nl_query_config.py`**Task 0.2.7**: Add LLM configuration constants

- `OLLAMA_BASE_URL: str = "http://localhost:11434"`
- `OLLAMA_DEFAULT_MODEL: str = "llama3.1:8b"`
- `OLLAMA_FALLBACK_MODEL: str = "llama3.2:3b"`
- `OLLAMA_TIMEOUT_SECONDS: float = 5.0`
- `OLLAMA_MAX_RETRIES: int = 3`
- `OLLAMA_JSON_MODE: bool = True`

### 0.3 Quality Gates

**Before Phase 0 Commit**:

```bash
make format
make lint-fix
make type-check
make test-core  # Run core tests including LLM fallback tests
make check      # Full quality gate
```

**Note**: Tests should mock OllamaClient to avoid requiring actual Ollama service running. Integration tests (optional) can test against real Ollama if available.**Commit Message Template**:

```javascript
feat: Phase 0 - Tier 3 LLM Fallback with Local Ollama (ADR003)

- Implement OllamaClient for local LLM integration
- Add RAG context builder from semantic layer metadata
- Implement structured JSON extraction with retries and validation
- Replace _llm_parse() stub with full implementation
- Add LLM configuration constants to nl_query_config.py
- Privacy-preserving: All data stays on-device

All tests passing: 12/12
```



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

- LLM client in `src/clinical_analytics/core/llm_client.py` (Phase 0)
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

**Current Status**:

- ✅ **Phase 0** (Tier 3 LLM Fallback) - **COMPLETE** (Commit f979c96)
- ✅ **Phase 1** (Trust UI) - **COMPLETE** (Commit af46ca9)
- ⏳ **Phase 2** (Alias Persistence) - **PENDING** - Can start immediately
- ⏳ **Phase 3** (QueryPlan Execution) - **PENDING** - Unblocked (Phase 0 complete), still requires Phase 2

**Dependencies**:

- **Phase 0**: ✅ **COMPLETE** - Tier 3 LLM fallback with local Ollama, RAG context, structured JSON extraction
- **Phase 1**: ✅ **COMPLETE** - Trust UI with verification expanders and patient-level export
- **Phase 2**: Independent (can start immediately, benefits from Phase 1 error UI which is already complete)
- **Phase 3**: **UNBLOCKED** - Phase 0 ✅ complete. Still requires:
- Phase 2 (needs alias resolution)
- Phase 1 ✅ (needs trust UI for validation display - already complete)

## Success Criteria

**Phase 0 Complete** ✅:

- [x] All 12 LLM fallback tests passing
- [x] OllamaClient connects to local Ollama service (or handles gracefully if unavailable)
- [x] RAG context builder extracts columns, aliases, and examples from semantic layer
- [x] Structured JSON extraction works with retries and validation
- [x] _llm_parse() returns QueryIntent with confidence >= 0.5 (or falls back to stub gracefully)
- [x] Timeout handling works (5s default)
- [x] Privacy-preserving: No external API calls, all data stays on-device
- [x] Created `src/clinical_analytics/core/llm_client.py` (OllamaClient class)
- [x] Implemented `_build_rag_context()`, `_build_llm_prompt()`, `_extract_query_intent_from_llm_response()` in `nl_query_engine.py`
- [x] Replaced `_llm_parse()` stub with full implementation (blast shield exception handling)
- [x] Added LLM configuration constants to `nl_query_config.py`
- [x] Type errors fixed, linting/formatting applied
- [x] Committed: f979c96

**Phase 1 Complete** ✅:

- [x] All 11 trust UI tests passing
- [x] Trust verification expander appears in all result rendering functions
- [x] Patient-level export works (capped at 100 rows, full export requires confirmation)
- [x] Cohort size calculation correct (count_total, count_filtered, percentage)
- [x] Created `src/clinical_analytics/ui/components/trust_ui.py` (362 lines)
- [x] Integrated into `execute_analysis_with_idempotency()`
- [x] Type errors fixed, linting/formatting applied
- [x] Committed: af46ca9

**Phase 2 Complete**:

- [ ] All 8 alias persistence tests passing
- [ ] User aliases persist to metadata JSON per `(upload_id, dataset_version)`
- [ ] Aliases load on SemanticLayer initialization
- [ ] "Add alias?" UI appears in error handling

**Phase 3 Complete**:

- [ ] All 15 QueryPlan execution tests passing
- [ ] Confidence and completeness gating enforced (hard gate, no side effects)
- [ ] Type-aware execution works (categorical → frequency, numeric → stats)