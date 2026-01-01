---
name: ADR009 LLM-Enhanced UX Implementation
overview: "Implement LLM-enhanced user experience features per ADR009: context-aware follow-ups, query explanations, result interpretation, error translation, enhanced filter extraction, and automated golden question generation. All phases follow test-first workflow with quality gates."
todos:
  - id: prephase-llm-json
    content: Create centralized LLM JSON parsing/validation module (llm_json.py)
    status: pending
  - id: prephase-llm-feature
    content: Create LLMFeature enum and unified call_llm() wrapper (llm_feature.py)
    status: pending
    dependencies:
      - prephase-llm-json
  - id: prephase-observability
    content: Define observability event schema with required fields and query pattern sanitization
    status: pending
    dependencies:
      - prephase-llm-feature
  - id: prephase-timeout-config
    content: Define timeout configuration with hard cap (LLM_TIMEOUT_*_S config keys)
    status: pending
  - id: prephase-infrastructure-tests
    content: Add tests for LLM infrastructure (JSON parsing, unified wrapper, sanitization)
    status: pending
    dependencies:
      - prephase-observability
      - prephase-timeout-config
  - id: phase1-schema
    content: Extend QueryPlan schema with follow_ups and follow_up_explanation fields
    status: pending
    dependencies:
      - prephase-infrastructure-tests
  - id: phase1-prompt
    content: Enhance _build_llm_prompt() to request follow-ups in JSON response
    status: pending
    dependencies:
      - phase1-schema
  - id: phase1-timeout
    content: Configure timeout for follow-up generation (10-15s recommended, make configurable)
    status: pending
    dependencies:
      - phase1-prompt
  - id: phase1-extraction
    content: Update _extract_query_intent_from_llm_response() to parse follow-ups from LLM response
    status: pending
    dependencies:
      - phase1-prompt
  - id: phase1-ui
    content: Implement _render_llm_follow_ups() UI function to replace disabled _suggest_follow_ups()
    status: pending
    dependencies:
      - phase1-extraction
  - id: phase1-observability
    content: Add structured logging for LLM follow-up generation (success/failure rates, Tier 3 usage tracking)
    status: pending
    dependencies:
      - phase1-ui
  - id: phase1-tests
    content: Add comprehensive tests for follow-up generation and UI rendering
    status: pending
    dependencies:
      - phase1-observability
  - id: phase2-schema
    content: Add interpretation and confidence_explanation fields to QueryPlan schema
    status: pending
  - id: phase2-prompt
    content: Enhance _build_llm_prompt() to request interpretation and confidence_explanation
    status: pending
    dependencies:
      - phase2-schema
  - id: phase2-ui
    content: Implement _render_query_interpretation() UI function with expander
    status: pending
    dependencies:
      - phase2-prompt
  - id: phase2-observability
    content: Add structured logging for interpretation generation (quality metrics, confidence tracking)
    status: pending
    dependencies:
      - phase2-ui
  - id: phase2-tests
    content: Add tests for interpretation generation and UI rendering
    status: pending
    dependencies:
      - phase2-observability
  - id: phase3-function
    content: Create interpret_result_with_llm() function for clinical insights
    status: pending
  - id: phase3-timeout
    content: Configure timeout for result interpretation (15-20s recommended, make configurable)
    status: pending
    dependencies:
      - phase3-function
  - id: phase3-integration
    content: Integrate result interpretation into execute_analysis_with_idempotency() flow
    status: pending
    dependencies:
      - phase3-function
  - id: phase3-ui
    content: Implement _render_result_interpretation() UI function
    status: pending
    dependencies:
      - phase3-integration
  - id: phase3-feature-flag
    content: Add ENABLE_RESULT_INTERPRETATION feature flag for optional interpretation
    status: pending
    dependencies:
      - phase3-ui
  - id: phase3-observability
    content: Add structured logging for result interpretation (latency, clinical context quality)
    status: pending
    dependencies:
      - phase3-feature-flag
  - id: phase3-tests
    content: Add tests for result interpretation generation and UI rendering
    status: pending
    dependencies:
      - phase3-observability
  - id: phase4-function
    content: Create translate_error_with_llm() function for user-friendly error messages
    status: pending
  - id: phase4-integration
    content: Integrate error translation into execute_query_plan() error handling
    status: pending
    dependencies:
      - phase4-function
  - id: phase4-observability
    content: Add structured logging for error translation (success rates, common error types)
    status: pending
    dependencies:
      - phase4-integration
  - id: phase4-tests
    content: Add tests for error translation quality and graceful degradation
    status: pending
    dependencies:
      - phase4-observability
  - id: phase5-function
    content: Create _extract_filters_with_llm() function for complex filter extraction
    status: pending
  - id: phase5-timeout
    content: Configure timeout for filter extraction (5-10s recommended, make configurable)
    status: pending
    dependencies:
      - phase5-function
  - id: phase5-integration
    content: Integrate LLM filter extraction into _llm_parse() flow
    status: pending
    dependencies:
      - phase5-function
  - id: phase5-prompt
    content: Enhance _build_llm_prompt() to emphasize filter extraction with examples
    status: pending
    dependencies:
      - phase5-integration
  - id: phase5-observability
    content: Add structured logging for filter extraction (LLM vs regex usage, complex pattern detection)
    status: pending
    dependencies:
      - phase5-prompt
  - id: phase5-tests
    content: Add tests for complex filter extraction patterns and fallback to regex
    status: pending
    dependencies:
      - phase5-observability
  - id: phase6-generation
    content: Create generate_golden_questions_from_logs() function
    status: pending
  - id: phase6-coverage
    content: Create analyze_golden_question_coverage() function for gap analysis
    status: pending
    dependencies:
      - phase6-generation
  - id: phase6-maintenance
    content: Create maintain_golden_questions_automatically() function
    status: pending
    dependencies:
      - phase6-coverage
  - id: phase6-cli
    content: Create CLI tool tests/eval/maintain_golden_questions.py
    status: pending
    dependencies:
      - phase6-maintenance
  - id: phase6-validation
    content: Add validation for generated golden questions (schema, uniqueness)
    status: pending
    dependencies:
      - phase6-cli
  - id: phase6-tests
    content: Add comprehensive tests for golden question generation and maintenance
    status: pending
    dependencies:
      - phase6-validation
---

# ADR009: LLM-Enhanced User Experience Imple

mentation Plan

## Overview

This plan implements ADR009's six phases to enhance the user experience using the local Ollama LLM. The implementation follows test-first development, phase-by-phase commits, and mandatory quality gates per project rules.

**Timeline Note:** Effort estimates are order-of-magnitude approximations. Real-world LLM edge cases, prompt refinement, and user feedback will require learning buffers. Treat Phase 2-3 estimates as optimistic baselines that may expand once real queries hit the system.

**Staff-Engineer Review Feedback (Incorporated):**

This plan has been updated based on two rounds of staff-engineer-level review feedback.

**First Review (Initial):** Approved with caveats, noting it's "well-structured, phased, test-first implementation that shows rare discipline around scope control." Key concerns addressed: timeline realism, LLM observability, UI coupling, scope gravity.

**Second Review (Final):** Approved for execution. Verdict: "This is now a high-quality, execution-ready plan. Not 'good for a side project.' Good, period." The review noted: "You addressed every material risk from the prior review without bloating the architecture or turning this into an LLM science experiment. The plan now has explicit guardrails, observability, and escape hatches."

**Key Feedback Points Addressed:**

1. **Timeline Realism**: All effort estimates now explicitly marked as "order-of-magnitude" with learning buffer notes. Phases 2-3 may expand with prompt iteration. The review noted timeline is "optimistic unless single-threaded by a very senior engineer" - acknowledged.

2. **LLM Observability (Hard Guarantees)**: Added structured logging requirements for all LLM operations:
   - Tier 3 invocation rate tracking (health metric) - **treat as health metric**
   - What queries trigger Tier 3 (query patterns logged)
   - Whether Tier 3 materially improves confidence (track confidence deltas)
   - Follow-up generation success/failure rates
   - Filter extraction LLM vs regex usage
   - Error translation success rates
   - Result interpretation latency metrics
   - **Critical**: Without explicit observability, "LLM usage will quietly creep"

3. **UI Coupling Management**: Added architectural boundary comments requirement. Future refactoring would extract to `QueryProcessor`/`ResultRenderer`/`ResultInterpreter` interfaces. The review noted: "This is fine for MVP, but it will become brittle once conversational follow-ups arrive" - acknowledged with explicit boundary markers.

4. **LLM Surface Creep Prevention**: Explicit monitoring requirement - if Tier 3 usage exceeds 50% of queries, investigate Tier 1/2 failures. The review emphasized: "Emit structured logs or counters for tier usage" - implemented.

5. **Architectural Discipline**: Clear separation markers for where UI orchestration ends and parsing logic begins, leaving breadcrumbs for future refactoring. The review noted: "Make it obvious where a future refactor would split responsibilities" - implemented.

6. **Scope Gravity Warning**: The review's final verdict: "The biggest risk is not architectural. It's scope gravity once real users start asking real questions. If you hold the phase boundaries and resist 'just one more UX tweak,' this will land well." - acknowledged in risk mitigation.

**Second Review (Final Approval):** The plan received final approval: "This is now a high-quality, execution-ready plan. Not 'good for a side project.' Good, period." Phase-by-phase notes incorporated:

**Third Review (Final Operational Gaps):** The plan received final approval with six critical operational improvements:

1. **Operationally Usable Observability**: Added explicit event schema with required fields (event, timestamp, run_key, query_hash, dataset_version, tier, model, feature, timeout_s, latency_ms, success, error_type, error_message). Enables downstream metric computation (e.g., tier3_rate_rolling_100).

2. **Query Pattern Sanitization (Privacy Protection)**: Never log raw query text. Log only query_hash (SHA256) and pattern_tags (controlled vocabulary: contains_negation, mentions_missingness, etc.). Enforced by code, not good intentions.

3. **Centralized LLM JSON Parsing/Validation**: Added `llm_json.py` module with `parse_json_response()` and `validate_shape()`. Single choke point for LLM formatting failures. Prevents duplicated brittle parsing across features.

4. **Timeout Policy Design**: Explicit config keys (LLM_TIMEOUT_*_S) with hard cap (25s). Always log timeout_used_s per call. Prevents "fixing" issues by making everything 2 minutes.

5. **Phase 3 Safety Rail**: Numeric contradiction detection - heuristic check for direction mismatches (increase/decrease), drop insights if mismatch detected. Prevents worst-case trust failure: confident wrong interpretation.

6. **Phase 5 Filter Validation Refinement**: Validate each filter independently; apply only valid filters; log invalid ones; reduce confidence if any invalid. Keeps system useful while being conservative.

7. **Unified LLM Call Wrapper**: Added `LLMFeature` enum and `call_llm()` wrapper. Consistent logging, fallback behavior, timeouts, and metrics across all LLM calls. Prevents copy-paste architecture.

**Pre-Phase Infrastructure Setup**: Added required infrastructure phase before all feature phases to establish centralized LLM infrastructure.

- **Phase 1 (Follow-Ups)**: Follow-ups must be suggestions, not endorsements. Avoid clinical advice territory.
- **Phase 2 (Interpretation)**: Cap interpretation verbosity (2-3 sentences, not essays).
- **Phase 3 (Result Interpretation)**: Critical constraint - interpretation must not mutate or contradict numeric results (interpretation only, not recomputation).
- **Phase 4 (Error Translation)**: Technical error must always be logged, even if UI never shows it.
- **Phase 5 (Filter Extraction)**: Strict validation - invalid filters must be dropped, not best-effort applied.

## Architecture Context

**Current State:**

- LLM (Ollama) currently only used for Tier 3 query parsing fallback
- `QueryPlan` schema defined in [`src/clinical_analytics/core/query_plan.py`](src/clinical_analytics/core/query_plan.py)
- LLM infrastructure ready: `OllamaClient`, `_build_rag_context()`, `_build_llm_prompt()`
- Follow-ups disabled: `_suggest_follow_ups()` in [`src/clinical_analytics/ui/pages/03_💬_Ask_Questions.py`](src/clinical_analytics/ui/pages/03_💬_Ask_Questions.py) returns early
- Execution via `SemanticLayer.execute_query_plan()` returns standardized result dict
- **Known Issue**: Current 5-second timeout may be too aggressive for ADR009 features (follow-ups, interpretation require longer generation)
- **Known Issue**: Complex filter patterns like "get rid of the n/a" currently fail filter extraction (Phase 5 addresses this)

**Key Files:**

- [`src/clinical_analytics/core/query_plan.py`](src/clinical_analytics/core/query_plan.py) - QueryPlan schema (extend with new fields)
- [`src/clinical_analytics/core/nl_query_engine.py`](src/clinical_analytics/core/nl_query_engine.py) - LLM parsing logic
- [`src/clinical_analytics/core/llm_client.py`](src/clinical_analytics/core/llm_client.py) - OllamaClient (no changes needed)
- [`src/clinical_analytics/core/llm_json.py`](src/clinical_analytics/core/llm_json.py) - **NEW**: Centralized LLM JSON parsing/validation
- [`src/clinical_analytics/core/llm_feature.py`](src/clinical_analytics/core/llm_feature.py) - **NEW**: LLMFeature enum and unified call wrapper
- [`src/clinical_analytics/ui/pages/03_💬_Ask_Questions.py`](src/clinical_analytics/ui/pages/03_💬_Ask_Questions.py) - UI rendering
- [`src/clinical_analytics/core/semantic.py`](src/clinical_analytics/core/semantic.py) - Execution layer (error handling)

## Pre-Phase: Infrastructure Setup (Required Before All Phases)

**Goal:** Establish centralized LLM infrastructure to prevent duplicated parsing, inconsistent logging, and privacy leaks.

**Tasks:**

1. **Create LLM JSON Parsing/Validation Module**
   - File: `src/clinical_analytics/core/llm_json.py` (new)
   - Functions:
     - `parse_json_response(raw: str) -> dict | None` - Centralized JSON parsing with error handling
     - `validate_shape(payload: dict, schema_name: str) -> ValidationResult` - Schema validation
     - Standardized fallback + logging hooks
   - **Purpose**: Single choke point for LLM formatting failures (the only thing LLMs are reliably bad at)

2. **Create LLM Feature Enum and Unified Call Wrapper**
   - File: `src/clinical_analytics/core/llm_feature.py` (new)
   - Enum: `LLMFeature` with values: `PARSE`, `FOLLOWUPS`, `INTERPRETATION`, `RESULT_INTERPRETATION`, `ERROR_TRANSLATION`, `FILTER_EXTRACTION`
   - Function: `call_llm(feature: LLMFeature, system: str, user: str, timeout_s: int) -> LLMCallResult`
   - Return type: `LLMCallResult` with fields:
     - `raw_text: str | None`
     - `payload: dict | None` (parsed JSON)
     - `latency_ms: float`
     - `timed_out: bool`
     - `error: str | None`
   - **Purpose**: Consistent logging, fallback behavior, timeouts, and metrics across all LLM calls

3. **Define Observability Event Schema**
   - File: `src/clinical_analytics/core/llm_observability.py` (new)
   - **Required fields for ALL LLM events:**
     - `event: str` (event name)
     - `timestamp: datetime`
     - `run_key: str | None`
     - `query_hash: str` (SHA256 hash, never raw query text)
     - `dataset_version: str`
     - `tier: int` (1|2|3)
     - `model: str` (e.g., "llama3.1:8b")
     - `feature: str` (followups|interpretation|result_interpretation|error_translation|filter_extraction)
     - `timeout_s: float`
     - `latency_ms: float`
     - `success: bool`
     - `error_type: str | None`
     - `error_message: str | None` (sanitized, no PII)
   - **Query Pattern Sanitization (Privacy Protection):**
     - **Rule**: Never log raw query text
     - **Log only**:
       - `query_hash: str` (SHA256)
       - `pattern_tags: list[str]` (controlled vocabulary):
         - `contains_negation`
         - `mentions_missingness`
         - `multi_table_join_request`
         - `contains_numeric_range`
         - `contains_value_exclusion`
         - `contains_comparison`
         - `contains_grouping`
       - `token_count: int` (optional)
     - **Enforcement**: Code-level validation, not good intentions

4. **Define Timeout Configuration**
   - File: [`src/clinical_analytics/core/nl_query_config.py`](src/clinical_analytics/core/nl_query_config.py)
   - Add config keys:
     - `LLM_TIMEOUT_PARSE_S: float = 5.0`
     - `LLM_TIMEOUT_FOLLOWUPS_S: float = 15.0`
     - `LLM_TIMEOUT_INTERPRETATION_S: float = 10.0`
     - `LLM_TIMEOUT_RESULT_INTERPRETATION_S: float = 20.0`
     - `LLM_TIMEOUT_ERROR_TRANSLATION_S: float = 5.0`
     - `LLM_TIMEOUT_FILTER_EXTRACTION_S: float = 10.0`
   - **Hard cap**: `LLM_TIMEOUT_MAX_S: float = 25.0` (prevents "fixing" issues by making everything 2 minutes)
   - **Logging**: Always log `timeout_used_s` per call

5. **Add Tests**
   - File: `tests/core/test_llm_infrastructure.py` (new)
   - Test JSON parsing with malformed input
   - Test schema validation
   - Test unified call wrapper (timeout, error handling, logging)
   - Test query pattern sanitization (no raw text logged)
   - Test timeout enforcement (hard cap)

**Definition of Done:**
- [ ] Centralized LLM JSON parsing/validation module exists
- [ ] Unified LLM call wrapper with LLMFeature enum
- [ ] Observability event schema defined with required fields
- [ ] Query pattern sanitization implemented (no raw text, only hash + tags)
- [ ] Timeout configuration defined with hard cap
- [ ] All LLM calls use unified wrapper
- [ ] Tests verify infrastructure works correctly
- [ ] All quality gates pass

**Estimated Effort:** 1-2 days

---

## Implementation Phases

### Phase 1: LLM-Generated Follow-Ups (Priority 0)

**Goal:** Replace disabled hardcoded follow-ups with LLM-generated, context-aware suggestions.**Tasks:**

1. **Extend QueryPlan Schema**

- File: [`src/clinical_analytics/core/query_plan.py`](src/clinical_analytics/core/query_plan.py)
- Add fields: `follow_ups: list[str] = field(default_factory=list)`, `follow_up_explanation: str = ""`
- Update `from_dict()` to handle new fields (with defaults)

2. **Enhance LLM Prompt for Follow-Ups**

- File: [`src/clinical_analytics/core/nl_query_engine.py`](src/clinical_analytics/core/nl_query_engine.py)
- Use unified `call_llm()` wrapper with `LLMFeature.FOLLOWUPS`
- Modify `_build_llm_prompt()` to request follow-ups in JSON response
- Add context: previous result (if available), available columns, clinical patterns
- Update system prompt to include follow-up generation instructions
- **Critical UX constraint**: Follow-ups must be treated as suggestions, not endorsements. Prompt must emphasize:
  - These are exploratory questions, not clinical advice
  - Tone should be helpful, not authoritative
  - Avoid language that implies clinical recommendations
- **Timeout**: Use `LLM_TIMEOUT_FOLLOWUPS_S` from config (default 15s, hard cap 25s)
- **JSON parsing**: Use centralized `parse_json_response()` and `validate_shape()` from `llm_json.py`

3. **Parse Follow-Ups from LLM Response**

- File: [`src/clinical_analytics/core/nl_query_engine.py`](src/clinical_analytics/core/nl_query_engine.py)
- Update `_extract_query_intent_from_llm_response()` to extract `follow_ups` and `follow_up_explanation`
- Preserve follow-ups when converting QueryPlan → QueryIntent → QueryPlan

4. **Implement UI Rendering**

- File: [`src/clinical_analytics/ui/pages/03_💬_Ask_Questions.py`](src/clinical_analytics/ui/pages/03_💬_Ask_Questions.py)
- Create `_render_llm_follow_ups(plan: QueryPlan, run_key: str)` function
- Replace `_suggest_follow_ups()` call with `_render_llm_follow_ups()` (only if `plan.follow_ups` non-empty)
- Use same button rendering pattern (columns, prefilled_query, rerun)
- **Architectural boundary:** Add clear comment marking where UI orchestration ends and parsing logic begins. Future refactor would extract to `QueryProcessor`/`ResultRenderer` interfaces (deferred per plan).

5. **Observability (Automatic via Unified Wrapper)**

- File: [`src/clinical_analytics/core/nl_query_engine.py`](src/clinical_analytics/core/nl_query_engine.py)
- **Automatic**: Unified `call_llm()` wrapper emits structured logs with required event schema:
  - All required fields (event, timestamp, run_key, query_hash, dataset_version, tier, model, feature, timeout_s, latency_ms, success, error_type, error_message)
  - Query pattern tags (not raw query text) via sanitization layer
- **Metrics**: Tier 3 invocation rate computed downstream from logs (tier3_rate_rolling_100)
- **No manual logging needed**: Wrapper handles all observability requirements

6. **Add Tests**

- File: `tests/core/test_queryplan_followups.py` (new)
- Test QueryPlan schema with follow_ups field
- Test LLM prompt includes follow-up request
- Test follow-up extraction from LLM response
- Test UI rendering (mock Streamlit)
- Test graceful degradation (no follow-ups if LLM unavailable)
- Test observability metrics (log events emitted correctly)

**Definition of Done:**

- [ ] QueryPlan schema includes `follow_ups` and `follow_up_explanation`
- [ ] LLM generates context-aware follow-ups in JSON response
- [ ] **UX constraint**: Follow-ups are suggestions, not endorsements (tone is helpful, not authoritative)
- [ ] UI renders LLM follow-ups (replaces disabled hardcoded function)
- [ ] Observability: Structured logs for follow-up generation (success/failure rates)
- [ ] Tests verify follow-up quality and relevance
- [ ] Graceful degradation: no follow-ups shown if LLM unavailable
- [ ] Architectural boundaries marked with comments for future refactoring
- [ ] All quality gates pass (`make check`)

**Estimated Effort:** 2-3 days (order-of-magnitude; add learning buffer for prompt refinement)---

### Phase 2: Query Explanation and Interpretation (Priority 1)

**Goal:** Help users understand what the system interpreted from their query.**Tasks:**

1. **Enhance QueryPlan Schema**

- File: [`src/clinical_analytics/core/query_plan.py`](src/clinical_analytics/core/query_plan.py)
- Enhance existing `explanation` field usage (already exists)
- Add `interpretation: str = ""` - Natural language interpretation
- Add `confidence_explanation: str = ""` - Why confidence is high/low

2. **Implement Interpretation Prompt Builder**

- File: [`src/clinical_analytics/core/nl_query_engine.py`](src/clinical_analytics/core/nl_query_engine.py)
- Create `_build_interpretation_prompt(plan: QueryPlan, query: str) -> tuple[str, str]`
- System prompt: Instructions for explaining query interpretation
- User prompt: Query + QueryPlan fields (intent, metric, group_by, filters, confidence)

3. **Generate Interpretation (Optional Enhancement)**

- File: [`src/clinical_analytics/core/nl_query_engine.py`](src/clinical_analytics/core/nl_query_engine.py)
- Option A: Include interpretation in main LLM parse call (single call)
- Option B: Separate call after parsing (adds latency, better quality)
- **Decision:** Start with Option A (include in main prompt), can optimize later

4. **Update LLM Prompt to Include Interpretation**

- File: [`src/clinical_analytics/core/nl_query_engine.py`](src/clinical_analytics/core/nl_query_engine.py)
- Use unified `call_llm()` wrapper with `LLMFeature.INTERPRETATION`
- Modify `_build_llm_prompt()` to request `interpretation` and `confidence_explanation` fields
- Update JSON schema documentation in system prompt
- **Critical constraint**: Cap interpretation verbosity explicitly in the prompt. Request summaries (2-3 sentences), not essays. Example: "Provide a brief 2-3 sentence interpretation, not a detailed explanation."
- **Timeout**: Use `LLM_TIMEOUT_INTERPRETATION_S` from config (default 10s, hard cap 25s)
- **JSON parsing**: Use centralized `parse_json_response()` and `validate_shape()` from `llm_json.py`

5. **Implement UI Rendering**

- File: [`src/clinical_analytics/ui/pages/03_💬_Ask_Questions.py`](src/clinical_analytics/ui/pages/03_💬_Ask_Questions.py)
- Create `_render_query_interpretation(plan: QueryPlan)` function
- Use expander: "How I interpreted your query"
- Show interpretation + confidence_explanation (if available)
- **Architectural boundary:** Mark UI rendering boundary with comment for future extraction

6. **Observability (Automatic via Unified Wrapper)**

- File: [`src/clinical_analytics/core/nl_query_engine.py`](src/clinical_analytics/core/nl_query_engine.py)
- **Automatic**: Unified `call_llm()` wrapper emits structured logs with required event schema
- All required fields logged automatically (event, timestamp, run_key, query_hash, dataset_version, tier, model, feature, timeout_s, latency_ms, success, error_type, error_message)
- Query pattern tags logged (not raw query text) via sanitization layer
- **No manual logging needed**: Wrapper handles all observability requirements

7. **Add Tests**

- File: `tests/core/test_queryplan_interpretation.py` (new)
- Test interpretation extraction from LLM response
- Test UI rendering (mock Streamlit)
- Test interpretation quality (descriptive, not generic)
- Test observability metrics

**Definition of Done:**

- [ ] QueryPlan includes `interpretation` and `confidence_explanation` fields
- [ ] LLM generates interpretation in JSON response
- [ ] **Verbosity constraint**: Interpretation is capped (2-3 sentences, not essays)
- [ ] UI renders interpretation in expander
- [ ] Observability: Structured logs for interpretation generation
- [ ] Tests verify interpretation quality
- [ ] All quality gates pass

**Estimated Effort:** 1-2 days (order-of-magnitude; may expand with prompt refinement)---

### Phase 3: Result Interpretation and Clinical Insights (Priority 2)

**Goal:** Provide clinical context for query results.**Tasks:**

1. **Create Result Interpretation Function**

- File: [`src/clinical_analytics/core/nl_query_engine.py`](src/clinical_analytics/core/nl_query_engine.py)
- Create `interpret_result_with_llm(plan: QueryPlan, result: dict, semantic_layer: SemanticLayer) -> dict[str, str]`
- Returns: `{"summary": str, "clinical_context": str, "insights": str}`
- Use unified `call_llm()` wrapper with `LLMFeature.RESULT_INTERPRETATION`
- Build prompt with result context (formatted result dict)
- **Critical constraint**: Do not let interpretation mutate or "re-explain" numeric results in ways that disagree with the tables. The prompt must emphasize:
  - Interpretation only, not recomputation
  - Must align with the actual numeric results shown
  - Cannot contradict or "correct" the displayed data
  - Should provide context and insights, not alternative calculations
- **Safety rail - Numeric Contradiction Detection**:
  - If result is a table with named metrics, include a "facts" section in the prompt:
    - Top-level numbers
    - Directionality (increased/decreased/higher/lower)
  - After generation, run heuristic check:
    - If output contains words like "increase/decrease/higher/lower" and result deltas exist, compare direction
    - If mismatch detected: drop insights, keep only summary (or show warning "interpretation skipped due to mismatch risk")
  - This prevents worst-case trust failure: confident wrong interpretation
- **Timeout**: Use `LLM_TIMEOUT_RESULT_INTERPRETATION_S` from config (default 20s, hard cap 25s)
- **JSON parsing**: Use centralized `parse_json_response()` and `validate_shape()` from `llm_json.py`

2. **Integrate into Execution Flow**

- File: [`src/clinical_analytics/ui/pages/03_💬_Ask_Questions.py`](src/clinical_analytics/ui/pages/03_💬_Ask_Questions.py)
- In `execute_analysis_with_idempotency()`, after result computation
- Call `interpret_result_with_llm()` (optional, feature flag)
- Store interpretation in result dict or session state
- **Architectural boundary:** Mark where execution orchestration ends and interpretation logic begins (future: extract to `ResultInterpreter` interface)

3. **Implement UI Rendering**

- File: [`src/clinical_analytics/ui/pages/03_💬_Ask_Questions.py`](src/clinical_analytics/ui/pages/03_💬_Ask_Questions.py)
- Create `_render_result_interpretation(interpretation: dict)` function
- Render summary, clinical_context (info box), insights (success box)
- Show after main results, before follow-ups

4. **Add Feature Flag**

- File: [`src/clinical_analytics/core/nl_query_config.py`](src/clinical_analytics/core/nl_query_config.py)
- Add `ENABLE_RESULT_INTERPRETATION: bool = True` config
- Allow disabling for fast queries (reduces latency)

5. **Observability (Automatic via Unified Wrapper)**

- File: [`src/clinical_analytics/core/nl_query_engine.py`](src/clinical_analytics/core/nl_query_engine.py)
- **Automatic**: Unified `call_llm()` wrapper emits structured logs with required event schema
- All required fields logged automatically (event, timestamp, run_key, query_hash, dataset_version, tier, model, feature, timeout_s, latency_ms, success, error_type, error_message)
- Query pattern tags logged (not raw query text) via sanitization layer
- **No manual logging needed**: Wrapper handles all observability requirements

6. **Add Tests**

- File: `tests/core/test_result_interpretation.py` (new)
- Test interpretation generation with sample results
- Test UI rendering (mock Streamlit)
- Test graceful degradation (no interpretation if LLM unavailable)
- Test observability metrics

**Definition of Done:**

- [ ] `interpret_result_with_llm()` generates clinical insights
- [ ] **Critical constraint**: Interpretation does not mutate or contradict numeric results (interpretation only, not recomputation)
- [ ] **Safety rail**: Numeric contradiction detection implemented (heuristic check for direction mismatches, drop insights if mismatch detected)
- [ ] Interpretation integrated into execution flow (optional, feature flag)
- [ ] UI renders interpretation after results
- [ ] Observability: Automatic via unified wrapper (latency, success rates logged)
- [ ] Tests verify interpretation quality, alignment with displayed results, and contradiction detection
- [ ] Feature flag allows disabling for performance
- [ ] All quality gates pass

**Estimated Effort:** 2-3 days (order-of-magnitude; clinical context quality may require prompt iteration)

### Phase 4: Error Message Translation (Priority 2)

**Goal:** Translate technical errors into user-friendly explanations.**Tasks:**

1. **Create Error Translation Function**

- File: [`src/clinical_analytics/core/nl_query_engine.py`](src/clinical_analytics/core/nl_query_engine.py)
- Create `translate_error_with_llm(error: Exception, query: str, plan: QueryPlan | None, semantic_layer: SemanticLayer) -> str`
- Use unified `call_llm()` wrapper with `LLMFeature.ERROR_TRANSLATION`
- Build prompt with error context (error message, query, plan)
- **Timeout**: Use `LLM_TIMEOUT_ERROR_TRANSLATION_S` from config (default 5s, hard cap 25s)
- **JSON parsing**: Use centralized `parse_json_response()` from `llm_json.py` (or plain text if JSON not required)

2. **Integrate into Error Handling**

- File: [`src/clinical_analytics/core/semantic.py`](src/clinical_analytics/core/semantic.py)
- In `execute_query_plan()` error handling, wrap exceptions
- Call `translate_error_with_llm()` before returning error dict
- Return both `error` (user-friendly) and `technical_error` (for debugging)
- **Critical logging requirement**: Always log the technical error, even if the UI never shows it. This is essential for debugging and must not be optional.

3. **Update Error Response Schema**

- File: [`src/clinical_analytics/core/semantic.py`](src/clinical_analytics/core/semantic.py)
- Ensure error dict includes: `error` (user-friendly), `technical_error` (original)
- UI should show `error` to user, log `technical_error`

4. **Observability (Automatic via Unified Wrapper)**

- File: [`src/clinical_analytics/core/nl_query_engine.py`](src/clinical_analytics/core/nl_query_engine.py)
- **Automatic**: Unified `call_llm()` wrapper emits structured logs with required event schema
- All required fields logged automatically (event, timestamp, run_key, query_hash, dataset_version, tier, model, feature, timeout_s, latency_ms, success, error_type, error_message)
- **Critical**: Technical error always logged (even if UI never shows it) - enforced by unified wrapper
- Query pattern tags logged (not raw query text) via sanitization layer
- **No manual logging needed**: Wrapper handles all observability requirements

5. **Add Tests**

- File: `tests/core/test_error_translation.py` (new)
- Test translation of common errors (column not found, invalid operator, etc.)
- Test graceful degradation (returns original error if LLM unavailable)
- Test error message quality (actionable, user-friendly)
- Test observability metrics

**Definition of Done:**

- [ ] `translate_error_with_llm()` translates technical errors
- [ ] Error handling in `execute_query_plan()` uses translation
- [ ] Error response includes both user-friendly and technical error
- [ ] **Critical logging**: Technical error always logged, even if UI never shows it
- [ ] Observability: Structured logs for error translation (success rates, common error types)
- [ ] Tests verify error translation quality
- [ ] All quality gates pass

**Estimated Effort:** 1-2 days (order-of-magnitude)---

### Phase 5: Enhanced Filter Extraction (Priority 1 - Phase 5 Alignment)

**Goal:** Use LLM for complex filter extraction instead of regex.**Tasks:**

1. **Create LLM Filter Extraction Function**

- File: [`src/clinical_analytics/core/nl_query_engine.py`](src/clinical_analytics/core/nl_query_engine.py)
- Create `_extract_filters_with_llm(query: str, context: dict, semantic_layer: SemanticLayer) -> list[FilterSpec]`
- Use unified `call_llm()` wrapper with `LLMFeature.FILTER_EXTRACTION`
- Build prompt with filter extraction instructions
- Request JSON array of filter objects
- **Critical validation requirement**: Validate each filter independently (not all-or-nothing):
  - Column exists in semantic layer
  - Operator is valid for column type
  - Value type matches column type
  - FilterSpec construction succeeds
  - **Apply only valid filters** (not "drop all if any fail")
  - **Log invalid filters** with reasons
  - **If any invalid filter exists**: Set `plan.confidence = min(plan.confidence, 0.6)` and populate `confidence_explanation` with validation failures
  - This keeps system useful while being conservative and transparent
- **JSON parsing**: Use centralized `parse_json_response()` and `validate_shape()` from `llm_json.py`
- **Timeout**: Use `LLM_TIMEOUT_FILTER_EXTRACTION_S` from config (default 10s, hard cap 25s)
- **Real-world test case**: "get rid of the n/a" currently fails filter extraction - ensure LLM extraction handles this pattern

2. **Integrate into LLM Parse Flow**

- File: [`src/clinical_analytics/core/nl_query_engine.py`](src/clinical_analytics/core/nl_query_engine.py)
- In `_llm_parse()`, call `_extract_filters_with_llm()` before building QueryPlan
- Use LLM-extracted filters in QueryPlan construction
- Keep regex as fallback for simple patterns (Tier 1 still works)

3. **Update LLM Prompt**

- File: [`src/clinical_analytics/core/nl_query_engine.py`](src/clinical_analytics/core/nl_query_engine.py)
- Enhance `_build_llm_prompt()` to emphasize filter extraction
- Include examples of complex filters: "remove n/a (0)", "exclude missing"

4. **Observability (Automatic via Unified Wrapper)**

- File: [`src/clinical_analytics/core/nl_query_engine.py`](src/clinical_analytics/core/nl_query_engine.py)
- **Automatic**: Unified `call_llm()` wrapper emits structured logs with required event schema
- All required fields logged automatically (event, timestamp, run_key, query_hash, dataset_version, tier, model, feature, timeout_s, latency_ms, success, error_type, error_message)
- Query pattern tags logged (not raw query text) via sanitization layer
- Filter validation failures logged separately (invalid filters with reasons)
- **No manual logging needed**: Wrapper handles all observability requirements

5. **Add Tests**

- File: `tests/core/test_filter_extraction.py` (new)
- Test complex filter patterns: "remove n/a (0)", "exclude missing", "patients on statins, exclude n/a"
- **Real-world test case**: "get rid of the n/a" (currently fails filter extraction - see terminal logs)
- Test filter extraction quality (correct column, operator, value)
- Test fallback to regex for simple patterns
- Test timeout handling (LLM may timeout on complex patterns - ensure graceful degradation)
- Test observability metrics

**Definition of Done:**

- [ ] `_extract_filters_with_llm()` handles complex filter patterns
- [ ] **Critical validation**: Each filter validated independently; apply only valid filters; log invalid ones; reduce confidence if any invalid
- [ ] LLM parse flow uses LLM-extracted filters
- [ ] Regex still works for simple patterns (Tier 1)
- [ ] Observability: Automatic via unified wrapper (LLM vs regex usage, validation failures logged)
- [ ] Tests verify filter extraction quality and validation (partial filter application, confidence reduction)
- [ ] All quality gates pass

**Estimated Effort:** 2-3 days (order-of-magnitude; complex filter patterns may require prompt iteration)---

### Phase 6: Automated Golden Question Generation (Priority 2 - Self-Improvement)

**Goal:** Use LLM to automatically generate and maintain golden questions from query logs.**Tasks:**

1. **Create Golden Question Generation Function**

- File: `src/clinical_analytics/core/golden_question_generator.py` (new)
- Create `generate_golden_questions_from_logs(query_logger: QueryLogger, semantic_layer: SemanticLayer, llm_client: OllamaClient, min_confidence: float = 0.75, min_frequency: int = 3) -> list[dict[str, Any]]`
- Load recent query logs (30 days)
- Filter successful queries (confidence >= min_confidence, success=True)
- Build LLM prompt to analyze logs and generate golden questions
- Return list of golden question dicts

2. **Create Coverage Gap Analysis Function**

- File: `src/clinical_analytics/core/golden_question_generator.py` (new)
- Create `analyze_golden_question_coverage(golden_questions: list[dict], query_logs: list[dict], llm_client: OllamaClient) -> dict[str, Any]`
- Use LLM to compare golden questions to query logs
- Identify missing patterns, high-frequency missing queries, edge cases

3. **Create Maintenance Function**

- File: `src/clinical_analytics/core/golden_question_generator.py` (new)
- Create `maintain_golden_questions_automatically(query_logger: QueryLogger, golden_questions_path: Path, semantic_layer: SemanticLayer, llm_client: OllamaClient, dry_run: bool = False) -> dict[str, Any]`
- Load existing golden questions
- Generate new questions from logs
- Analyze coverage gaps
- Use LLM to merge (identify duplicates, suggest updates, flag obsolete)
- Apply recommendations (unless dry_run)

4. **Create CLI Tool**

- File: `tests/eval/maintain_golden_questions.py` (new)
- CLI tool to run maintenance manually
- Arguments: `--dry-run`, `--log-dir`, `--output`
- Print summary: new questions, updated questions, removed questions, coverage gaps

5. **Add Validation**

- File: `src/clinical_analytics/core/golden_question_generator.py` (new)
- Validate generated questions: schema, uniqueness, required fields
- Ensure questions pass eval harness validation

6. **Add Tests**

- File: `tests/core/test_golden_question_generation.py` (new)
- Test generation from sample query logs
- Test coverage gap analysis
- Test maintenance function (dry-run mode)
- Test validation (schema, uniqueness)

**Definition of Done:**

- [ ] LLM generates golden questions from query logs
- [ ] LLM identifies coverage gaps
- [ ] CLI tool can maintain golden questions automatically
- [ ] Generated questions pass validation (schema, eval harness)
- [ ] Tests verify generation quality
- [ ] Documentation explains maintenance workflow
- [ ] All quality gates pass

**Estimated Effort:** 3-4 days---

## Implementation Guidelines

### Test-First Workflow (MANDATORY)

Per [104-plan-execution-hygiene.mdc](.cursor/rules/104-plan-execution-hygiene.mdc):

1. **Write failing test** (Red)
2. **Run test immediately** - Use `make test-core` or `make test-fast`
3. **Implement minimum code to pass** (Green)
4. **Run test again** - Verify it passes
5. **Fix code quality** - `make lint-fix`, `make format`
6. **Refactor** (Refactor)
7. **Run full test suite** - `make test-fast` before commit

### Quality Gates (MANDATORY)

**Before every commit:**

```bash
make format        # Auto-format code
make lint-fix      # Auto-fix linting issues
make type-check    # Verify type hints
make test-fast     # Run fast tests (or module-specific: test-core, test-ui)
make check         # Full quality gate (recommended)
```

**Never commit code that fails these checks.**

### Phase Commit Discipline

**Before starting next phase:**

1. Write tests for current phase
2. Run tests immediately (`make test-core` or `make test-fast`)
3. Fix any test failures
4. Run `make check` (all quality gates)
5. Commit all changes (implementation + tests)

**Commit message template:**

```javascript
feat: Phase N - [Brief description]

- [Key change 1]
- [Key change 2]
- [Key change 3]
- Add comprehensive test suite (X tests passing)

All tests passing: X/Y
```



### Code Quality Standards

Per [103-staff-engineer-standards.mdc](.cursor/rules/103-staff-engineer-standards.mdc):

- **Error handling**: Explicit failures, typed exceptions
- **Observability**: Structured logging with `structlog`
- **Defensive programming**: Validate at boundaries, fail fast
- **Type hints**: All function signatures typed

### LLM Integration Patterns

- **Graceful degradation**: All LLM features must work without LLM (fallbacks)
- **Privacy**: Local Ollama only, no external API calls
- **Latency**: Make interpretation optional (feature flag) for fast queries
- **Error handling**: Return None/empty on LLM failures, don't crash
- **Timeout configuration**: Current 5-second timeout may be insufficient for ADR009 features:
  - Follow-up generation: May need 10-15 seconds (longer context)
  - Result interpretation: May need 15-20 seconds (clinical analysis)
  - Filter extraction: 5-10 seconds (complex pattern analysis)
  - **Action**: Make timeout configurable per feature type, or increase default for ADR009 features
- **Model considerations**: Model size affects latency/quality:
  - `llama3.2:3b` (current fallback): Faster but may timeout on complex tasks
  - `llama3.1:8b` (recommended): Better quality, handles complex tasks better
  - **Action**: Document model requirements in implementation, consider model size in timeout configuration
- **Observability**: Emit structured logs for all LLM operations (success/failure, latency, usage patterns). Track Tier 3 invocation rate as health metric.
- **Usage tracking**: Monitor LLM feature usage to prevent silent creep. If Tier 3 usage exceeds 50% of queries, investigate Tier 1/2 failures.

### Testing Patterns

Per [101-testing-hygiene.mdc](.cursor/rules/101-testing-hygiene.mdc):

- **AAA pattern**: Arrange-Act-Assert with clear separation
- **Naming**: `test_unit_scenario_expectedBehavior`
- **Fixtures**: Use shared fixtures from `conftest.py` (DRY)
- **Mocking**: Mock Streamlit, OllamaClient for UI/LLM tests
- **Isolation**: No shared mutable state between tests

### Architectural Boundaries

**UI Coupling Management:**

- Add clear comment boundaries in UI code marking where orchestration ends and parsing logic begins
- Future refactoring would extract to:
  - `QueryProcessor` interface (parsing, interpretation)
  - `ResultRenderer` interface (result display, follow-ups)
  - `ResultInterpreter` interface (clinical insights)
- Current coupling is acceptable for MVP but will become brittle with conversational follow-ups
- Leave breadcrumbs for Future You: explicit comments, not just implicit structure

## Risk Mitigation

1. **LLM Unavailability**: All features degrade gracefully (no follow-ups, original errors, etc.)
2. **Latency**: Result interpretation optional (feature flag), can be disabled
3. **Quality Variability**: Validation, confidence thresholds, user feedback
4. **Resource Usage**: Local only, user controls Ollama resource usage
5. **LLM Surface Creep (Critical)**: 
   - **Problem**: Review noted "Without explicit observability, LLM usage will quietly creep"
   - **Mitigation**: Hard observability guarantees - structured logs for all LLM operations
   - **Health Metric**: Tier 3 invocation rate tracked and monitored
   - **Threshold**: If Tier 3 usage exceeds 50% of queries, investigate Tier 1/2 failures
   - **Tracking**: Log what queries trigger Tier 3, whether it materially improves confidence
6. **Timeline Optimism**: 
   - **Problem**: Review noted timeline is "optimistic unless single-threaded by a very senior engineer"
   - **Mitigation**: Effort estimates are order-of-magnitude. Real-world LLM edge cases and prompt refinement will require learning buffers, especially in Phases 2-3.
   - **Reality Check**: "Phase 2 and Phase 3 will expand once real queries hit the system"
7. **UI Coupling**: 
   - **Problem**: Review noted "This is fine for MVP, but it will become brittle once conversational follow-ups arrive"
   - **Mitigation**: Architectural boundaries marked with comments for future refactoring. Current UI orchestration is acceptable for MVP but will need extraction to `QueryProcessor`/`ResultRenderer` interfaces when conversational follow-ups arrive.
   - **Breadcrumbs**: "Make it obvious where a future refactor would split responsibilities"
10. **Scope Gravity (Critical)**:
    - **Problem**: Review's final verdict: "The biggest risk is not architectural. It's scope gravity once real users start asking real questions."
    - **Mitigation**: Hold phase boundaries strictly. Resist "just one more UX tweak" - this is explicitly called out as the biggest risk.
    - **Discipline**: "If you hold the phase boundaries and resist 'just one more UX tweak,' this will land well. If you don't, you'll still learn a lot, just louder and later."
8. **Timeout Issues**: Current 5-second timeout may be insufficient for ADR009 features:
   - **Mitigation**: Make timeout configurable per feature type (follow-ups: 10-15s, interpretation: 15-20s, filter extraction: 5-10s)
   - **Fallback**: All features must handle timeout gracefully (return empty/partial results, don't crash)
   - **Monitoring**: Track timeout rates as observability metric - high timeout rate indicates model size or prompt complexity issues
9. **Model Size Considerations**: Smaller models (3B) may timeout on complex tasks. Document model requirements and consider timeout scaling with model size.

## Success Criteria

1. **Follow-Ups**: Users find LLM follow-ups more relevant than hardcoded (user testing)
2. **Query Explanation**: Users understand what system interpreted (reduced confusion)
3. **Result Interpretation**: Users find clinical insights valuable
4. **Error Messages**: Users can fix errors based on LLM explanations (reduced support)
5. **Filter Extraction**: Complex filters extracted correctly (test coverage)
6. **Golden Questions**: Test suite grows automatically from real usage (coverage metrics improve)
7. **Observability**: Tier 3 invocation rate tracked and monitored. If >50% of queries use Tier 3, investigate Tier 1/2 failures.
8. **LLM Usage Health**: All LLM features emit structured logs. Success/failure rates queryable for monitoring.

## References

- **ADR009**: [docs/implementation/ADR/ADR009.md](docs/implementation/ADR/ADR009.md)
- **ADR001**: Query Plan Producer - defines QueryPlan schema