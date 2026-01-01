---
name: ADR009 LLM-Enhanced UX Implementation
overview: "Implement LLM-enhanced user experience features per ADR009: context-aware follow-ups, query explanations, result interpretation, error translation, enhanced filter extraction, and automated golden question generation. All phases follow test-first workflow with quality gates."
todos:
  - id: prephase-llm-json
    content: Create centralized LLM JSON parsing/validation module (llm_json.py)
    status: done
  - id: prephase-llm-feature
    content: Create LLMFeature enum and unified call_llm() wrapper (llm_feature.py)
    status: done
    dependencies:
      - prephase-llm-json
  - id: prephase-observability
    content: Define observability event schema with required fields and query pattern sanitization
    status: done
    dependencies:
      - prephase-llm-feature
  - id: prephase-timeout-config
    content: Define timeout configuration with hard cap (LLM_TIMEOUT_*_S config keys)
    status: done
  - id: prephase-infrastructure-tests
    content: Add tests for LLM infrastructure (JSON parsing, unified wrapper, sanitization)
    status: done
    dependencies:
      - prephase-observability
      - prephase-timeout-config
  - id: phase1-schema
    content: Extend QueryPlan schema with follow_ups and follow_up_explanation fields
    status: done
    dependencies:
      - prephase-infrastructure-tests
  - id: phase1-prompt
    content: Enhance _build_llm_prompt() to request follow-ups in JSON response
    status: done
    dependencies:
      - phase1-schema
  - id: phase1-timeout
    content: Configure timeout for follow-up generation (10-15s recommended, make configurable)
    status: done
    dependencies:
      - phase1-prompt
  - id: phase1-extraction
    content: Update _extract_query_intent_from_llm_response() to parse follow-ups from LLM response
    status: done
    dependencies:
      - phase1-prompt
  - id: phase1-ui
    content: Implement _render_llm_follow_ups() UI function to replace disabled _suggest_follow_ups()
    status: done
    dependencies:
      - phase1-extraction
  - id: phase1-observability
    content: Add structured logging for LLM follow-up generation (success/failure rates, Tier 3 usage tracking)
    status: done
    dependencies:
      - phase1-ui
  - id: phase1-tests
    content: Add comprehensive tests for follow-up generation and UI rendering
    status: done
    dependencies:
      - phase1-observability
  - id: phase2-schema
    content: Add interpretation and confidence_explanation fields to QueryPlan schema
    status: done
  - id: phase2-prompt
    content: Enhance _build_llm_prompt() to request interpretation and confidence_explanation
    status: done
    dependencies:
      - phase2-schema
  - id: phase2-ui
    content: Implement _render_query_interpretation() UI function with expander
    status: done
    dependencies:
      - phase2-prompt
  - id: phase2-observability
    content: Add structured logging for interpretation generation (quality metrics, confidence tracking)
    status: done
    dependencies:
      - phase2-ui
  - id: phase2-tests
    content: Add tests for interpretation generation and UI rendering
    status: done
    dependencies:
      - phase2-observability
  - id: phase3-function
    content: Create interpret_result_with_llm() function for clinical insights
    status: done
  - id: phase3-timeout
    content: Configure timeout for result interpretation (15-20s recommended, make configurable)
    status: done
    dependencies:
      - phase3-function
  - id: phase3-integration
    content: Integrate result interpretation into execute_analysis_with_idempotency() flow
    status: done
    dependencies:
      - phase3-function
  - id: phase3-ui
    content: Implement _render_result_interpretation() UI function
    status: done
    dependencies:
      - phase3-integration
  - id: phase3-feature-flag
    content: Add ENABLE_RESULT_INTERPRETATION feature flag for optional interpretation
    status: done
    dependencies:
      - phase3-ui
  - id: phase3-observability
    content: Add structured logging for result interpretation (latency, clinical context quality)
    status: done
    dependencies:
      - phase3-feature-flag
  - id: phase3-tests
    content: Add tests for result interpretation generation and UI rendering
    status: done
    dependencies:
      - phase3-observability
  - id: phase4-function
    content: Create translate_error_with_llm() function for user-friendly error messages
    status: done
  - id: phase4-integration
    content: Integrate error translation into execute_query_plan() error handling
    status: done
    dependencies:
      - phase4-function
  - id: phase4-observability
    content: Add structured logging for error translation (success rates, common error types)
    status: done
    dependencies:
      - phase4-integration
  - id: phase4-tests
    content: Add tests for error translation quality and graceful degradation
    status: done
    dependencies:
      - phase4-observability
  - id: phase5-function
    content: Create _extract_filters_with_llm() function for complex filter extraction
    status: done
  - id: phase5-timeout
    content: Configure timeout for filter extraction (5-10s recommended, make configurable)
    status: done
    dependencies:
      - phase5-function
  - id: phase5-integration
    content: Integrate LLM filter extraction into _llm_parse() flow
    status: done
    dependencies:
      - phase5-function
  - id: phase5-prompt
    content: Enhance _build_llm_prompt() to emphasize filter extraction with examples
    status: done
    dependencies:
      - phase5-integration
  - id: phase5-observability
    content: Add structured logging for filter extraction (LLM vs regex usage, complex pattern detection)
    status: done
    dependencies:
      - phase5-prompt
  - id: phase5-tests
    content: Add tests for complex filter extraction patterns and fallback to regex
    status: done
    dependencies:
      - phase5-observability
  - id: phase6-generation
    content: Create generate_golden_questions_from_logs() function
    status: done
  - id: phase6-coverage
    content: Create analyze_golden_question_coverage() function for gap analysis
    status: done
    dependencies:
      - phase6-generation
  - id: phase6-maintenance
    content: Create maintain_golden_questions_automatically() function
    status: done
    dependencies:
      - phase6-coverage
  - id: phase6-cli
    content: Create CLI tool tests/eval/maintain_golden_questions.py
    status: done
    dependencies:
      - phase6-maintenance
  - id: phase6-validation
    content: Add validation for generated golden questions (schema, uniqueness)
    status: done
    dependencies:
      - phase6-cli
  - id: phase6-tests
    content: Add comprehensive tests for golden question generation and maintenance
    status: done
    dependencies:
      - phase6-validation
