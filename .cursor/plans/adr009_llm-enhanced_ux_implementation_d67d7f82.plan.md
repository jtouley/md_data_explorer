---
name: ADR009 LLM-Enhanced UX Implementation
overview: "Implement LLM-enhanced user experience features per ADR009: context-aware follow-ups, query explanations, result interpretation, error translation, enhanced filter extraction, and automated golden question generation. All phases follow test-first workflow with quality gates."
todos:
  - id: phase1-schema
    content: Extend QueryPlan schema with follow_ups and follow_up_explanation fields
    status: pending
  - id: phase1-prompt
    content: Enhance _build_llm_prompt() to request follow-ups in JSON response
    status: pending
    dependencies:
      - phase1-schema
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

This plan has been updated based on staff-engineer-level review feedback:

1. **Timeline Realism**: All effort estimates now explicitly marked as "order-of-magnitude" with learning buffer notes. Phases 2-3 may expand with prompt iteration.

2. **LLM Observability**: Added structured logging requirements for all LLM operations:
   - Tier 3 invocation rate tracking (health metric)
   - Follow-up generation success/failure rates
   - Filter extraction LLM vs regex usage
   - Error translation success rates
   - Result interpretation latency metrics

3. **UI Coupling Management**: Added architectural boundary comments requirement. Future refactoring would extract to `QueryProcessor`/`ResultRenderer`/`ResultInterpreter` interfaces.

4. **LLM Surface Creep Prevention**: Explicit monitoring requirement - if Tier 3 usage exceeds 50% of queries, investigate Tier 1/2 failures.

5. **Architectural Discipline**: Clear separation markers for where UI orchestration ends and parsing logic begins, leaving breadcrumbs for future refactoring.

## Architecture Context

**Current State:**

- LLM (Ollama) currently only used for Tier 3 query parsing fallback
- `QueryPlan` schema defined in [`src/clinical_analytics/core/query_plan.py`](src/clinical_analytics/core/query_plan.py)
- LLM infrastructure ready: `OllamaClient`, `_build_rag_context()`, `_build_llm_prompt()`
- Follow-ups disabled: `_suggest_follow_ups()` in [`src/clinical_analytics/ui/pages/03_💬_Ask_Questions.py`](src/clinical_analytics/ui/pages/03_💬_Ask_Questions.py) returns early
- Execution via `SemanticLayer.execute_query_plan()` returns standardized result dict

**Key Files:**

- [`src/clinical_analytics/core/query_plan.py`](src/clinical_analytics/core/query_plan.py) - QueryPlan schema (extend with new fields)
- [`src/clinical_analytics/core/nl_query_engine.py`](src/clinical_analytics/core/nl_query_engine.py) - LLM parsing logic
- [`src/clinical_analytics/core/llm_client.py`](src/clinical_analytics/core/llm_client.py) - OllamaClient (no changes needed)
- [`src/clinical_analytics/ui/pages/03_💬_Ask_Questions.py`](src/clinical_analytics/ui/pages/03_💬_Ask_Questions.py) - UI rendering
- [`src/clinical_analytics/core/semantic.py`](src/clinical_analytics/core/semantic.py) - Execution layer (error handling)

## Implementation Phases

### Phase 1: LLM-Generated Follow-Ups (Priority 0)

**Goal:** Replace disabled hardcoded follow-ups with LLM-generated, context-aware suggestions.**Tasks:**

1. **Extend QueryPlan Schema**

- File: [`src/clinical_analytics/core/query_plan.py`](src/clinical_analytics/core/query_plan.py)
- Add fields: `follow_ups: list[str] = field(default_factory=list)`, `follow_up_explanation: str = ""`
- Update `from_dict()` to handle new fields (with defaults)

2. **Enhance LLM Prompt for Follow-Ups**

- File: [`src/clinical_analytics/core/nl_query_engine.py`](src/clinical_analytics/core/nl_query_engine.py)
- Modify `_build_llm_prompt()` to request follow-ups in JSON response
- Add context: previous result (if available), available columns, clinical patterns
- Update system prompt to include follow-up generation instructions

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

5. **Add Observability**

- File: [`src/clinical_analytics/core/nl_query_engine.py`](src/clinical_analytics/core/nl_query_engine.py)
- Emit structured log events for LLM follow-up generation:
  - `llm_followups_generated` (count, query_hash, success)
  - `llm_followups_failed` (reason, fallback_used)
- Track Tier 3 invocation rate as health metric (already logged, but ensure it's queryable)

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
- Modify `_build_llm_prompt()` to request `interpretation` and `confidence_explanation` fields
- Update JSON schema documentation in system prompt

5. **Implement UI Rendering**

- File: [`src/clinical_analytics/ui/pages/03_💬_Ask_Questions.py`](src/clinical_analytics/ui/pages/03_💬_Ask_Questions.py)
- Create `_render_query_interpretation(plan: QueryPlan)` function
- Use expander: "How I interpreted your query"
- Show interpretation + confidence_explanation (if available)
- **Architectural boundary:** Mark UI rendering boundary with comment for future extraction

6. **Add Observability**

- File: [`src/clinical_analytics/core/nl_query_engine.py`](src/clinical_analytics/core/nl_query_engine.py)
- Emit structured log events for interpretation generation:
  - `llm_interpretation_generated` (query_hash, confidence, has_explanation)
  - Track interpretation quality metrics (length, specificity)

7. **Add Tests**

- File: `tests/core/test_queryplan_interpretation.py` (new)
- Test interpretation extraction from LLM response
- Test UI rendering (mock Streamlit)
- Test interpretation quality (descriptive, not generic)
- Test observability metrics

**Definition of Done:**

- [ ] QueryPlan includes `interpretation` and `confidence_explanation` fields
- [ ] LLM generates interpretation in JSON response
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
- Build prompt with result context (formatted result dict)
- Call OllamaClient (separate call, adds latency)

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

5. **Add Observability**

- File: [`src/clinical_analytics/core/nl_query_engine.py`](src/clinical_analytics/core/nl_query_engine.py)
- Emit structured log events for result interpretation:
  - `llm_result_interpretation_generated` (query_hash, latency_ms, has_clinical_context)
  - `llm_result_interpretation_failed` (reason, fallback_used)
- Track interpretation latency as performance metric

6. **Add Tests**

- File: `tests/core/test_result_interpretation.py` (new)
- Test interpretation generation with sample results
- Test UI rendering (mock Streamlit)
- Test graceful degradation (no interpretation if LLM unavailable)
- Test observability metrics

**Definition of Done:**

- [ ] `interpret_result_with_llm()` generates clinical insights
- [ ] Interpretation integrated into execution flow (optional, feature flag)
- [ ] UI renders interpretation after results
- [ ] Tests verify interpretation quality
- [ ] Feature flag allows disabling for performance
- [ ] All quality gates pass

**Estimated Effort:** 2-3 days---

### Phase 4: Error Message Translation (Priority 2)

**Goal:** Translate technical errors into user-friendly explanations.**Tasks:**

1. **Create Error Translation Function**

- File: [`src/clinical_analytics/core/nl_query_engine.py`](src/clinical_analytics/core/nl_query_engine.py)
- Create `translate_error_with_llm(error: Exception, query: str, plan: QueryPlan | None, semantic_layer: SemanticLayer) -> str`
- Build prompt with error context (error message, query, plan)
- Call OllamaClient (separate call, but fast - simple translation)

2. **Integrate into Error Handling**

- File: [`src/clinical_analytics/core/semantic.py`](src/clinical_analytics/core/semantic.py)
- In `execute_query_plan()` error handling, wrap exceptions
- Call `translate_error_with_llm()` before returning error dict
- Return both `error` (user-friendly) and `technical_error` (for debugging)

3. **Update Error Response Schema**

- File: [`src/clinical_analytics/core/semantic.py`](src/clinical_analytics/core/semantic.py)
- Ensure error dict includes: `error` (user-friendly), `technical_error` (original)
- UI should show `error` to user, log `technical_error`

4. **Add Observability**

- File: [`src/clinical_analytics/core/nl_query_engine.py`](src/clinical_analytics/core/nl_query_engine.py)
- Emit structured log events for error translation:
  - `llm_error_translated` (error_type, query_hash, latency_ms)
  - `llm_error_translation_failed` (reason, original_error_preserved)
- Track translation success rate as health metric

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
- [ ] Observability: Structured logs for error translation (success rates, common error types)
- [ ] Tests verify error translation quality
- [ ] All quality gates pass

**Estimated Effort:** 1-2 days (order-of-magnitude)---

### Phase 5: Enhanced Filter Extraction (Priority 1 - Phase 5 Alignment)

**Goal:** Use LLM for complex filter extraction instead of regex.**Tasks:**

1. **Create LLM Filter Extraction Function**

- File: [`src/clinical_analytics/core/nl_query_engine.py`](src/clinical_analytics/core/nl_query_engine.py)
- Create `_extract_filters_with_llm(query: str, context: dict, semantic_layer: SemanticLayer) -> list[FilterSpec]`
- Build prompt with filter extraction instructions
- Request JSON array of filter objects
- Parse response into `FilterSpec` objects

2. **Integrate into LLM Parse Flow**

- File: [`src/clinical_analytics/core/nl_query_engine.py`](src/clinical_analytics/core/nl_query_engine.py)
- In `_llm_parse()`, call `_extract_filters_with_llm()` before building QueryPlan
- Use LLM-extracted filters in QueryPlan construction
- Keep regex as fallback for simple patterns (Tier 1 still works)

3. **Update LLM Prompt**

- File: [`src/clinical_analytics/core/nl_query_engine.py`](src/clinical_analytics/core/nl_query_engine.py)
- Enhance `_build_llm_prompt()` to emphasize filter extraction
- Include examples of complex filters: "remove n/a (0)", "exclude missing"

4. **Add Observability**

- File: [`src/clinical_analytics/core/nl_query_engine.py`](src/clinical_analytics/core/nl_query_engine.py)
- Emit structured log events for LLM filter extraction:
  - `llm_filters_extracted` (query_hash, filter_count, complex_patterns_detected)
  - `llm_filter_extraction_failed` (reason, fallback_to_regex)
- Track LLM vs regex filter extraction ratio (Tier 1 vs Tier 3 for filters)

5. **Add Tests**

- File: `tests/core/test_filter_extraction.py` (new)
- Test complex filter patterns: "remove n/a (0)", "exclude missing", "patients on statins, exclude n/a"
- Test filter extraction quality (correct column, operator, value)
- Test fallback to regex for simple patterns
- Test observability metrics

**Definition of Done:**

- [ ] `_extract_filters_with_llm()` handles complex filter patterns
- [ ] LLM parse flow uses LLM-extracted filters
- [ ] Regex still works for simple patterns (Tier 1)
- [ ] Observability: Structured logs for filter extraction (LLM vs regex usage)
- [ ] Tests verify filter extraction quality
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
5. **LLM Surface Creep**: Observability metrics track Tier 3 usage rates - treat as health metric. If Tier 3 invocation rate exceeds 50% of queries, investigate why Tier 1/2 are failing.
6. **Timeline Optimism**: Effort estimates are order-of-magnitude. Real-world LLM edge cases and prompt refinement will require learning buffers, especially in Phases 2-3.
7. **UI Coupling**: Architectural boundaries marked with comments for future refactoring. Current UI orchestration is acceptable for MVP but will need extraction to `QueryProcessor`/`ResultRenderer` interfaces when conversational follow-ups arrive.

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