"""NL Query Engine Configuration Constants.

Single source of truth for confidence thresholds and parsing parameters.
These are domain config, not code - adjust without code changes.

Configuration is loaded from config/nl_query.yaml via config_loader.
Ollama configuration values are also in config/nl_query.yaml (loaded from config/ollama.yaml during migration).
Environment variables take precedence over YAML config.
"""

import logging

from clinical_analytics.core.config_loader import load_nl_query_config

logger = logging.getLogger(__name__)

# Load configuration from YAML via config_loader
# This loads from config/nl_query.yaml with environment variable overrides
_config = load_nl_query_config()

# Confidence thresholds for tier matching
TIER_1_PATTERN_MATCH_THRESHOLD: float = _config["tier_1_pattern_match_threshold"]
TIER_2_SEMANTIC_MATCH_THRESHOLD: float = _config["tier_2_semantic_match_threshold"]
CLARIFYING_QUESTIONS_THRESHOLD: float = _config["clarifying_questions_threshold"]

# Auto-execute threshold (intentionally same as TIER_2_SEMANTIC_MATCH_THRESHOLD)
# because semantic match is the minimum confidence for auto-execution
AUTO_EXECUTE_CONFIDENCE_THRESHOLD: float = _config["auto_execute_confidence_threshold"]

# Performance/timeout settings
TIER_TIMEOUT_SECONDS: float = _config["tier_timeout_seconds"]
ENABLE_PARALLEL_TIER_MATCHING: bool = _config["enable_parallel_tier_matching"]

# Semantic matching parameters
SEMANTIC_SIMILARITY_THRESHOLD: float = _config["semantic_similarity_threshold"]
FUZZY_MATCH_CUTOFF: float = _config["fuzzy_match_cutoff"]

# Feature flags
ENABLE_CLARIFYING_QUESTIONS: bool = _config["enable_clarifying_questions"]
ENABLE_PROGRESSIVE_FEEDBACK: bool = _config["enable_progressive_feedback"]

# Tier 3 LLM Fallback Configuration (ADR003 Phase 0)
# Privacy-preserving: Local Ollama only, no external API calls
# Environment variables take precedence over YAML config
OLLAMA_BASE_URL: str = _config["ollama_base_url"]
OLLAMA_DEFAULT_MODEL: str = _config["ollama_default_model"]
OLLAMA_FALLBACK_MODEL: str = _config["ollama_fallback_model"]
OLLAMA_TIMEOUT_SECONDS: float = _config["ollama_timeout_seconds"]
OLLAMA_MAX_RETRIES: int = _config["ollama_max_retries"]
OLLAMA_JSON_MODE: bool = _config["ollama_json_mode"]

# Tier 3 confidence thresholds (Phase 0 success vs Phase 3 execution gate)
TIER_3_MIN_CONFIDENCE: float = _config["tier_3_min_confidence"]
TIER_3_EXECUTION_THRESHOLD: float = _config["tier_3_execution_threshold"]

# ADR009: LLM Feature-Specific Timeout Configuration (Pre-Phase)
# Each feature has its own timeout based on complexity
# Error translation should be fast (5s) - users are already frustrated by error
# Result interpretation can be longer (20s) - users expect thoughtful analysis
# Query parsing is most complex (30s) - requires understanding schema and intent
LLM_TIMEOUT_PARSE_S: float = _config["llm_timeout_parse_s"]
LLM_TIMEOUT_FOLLOWUPS_S: float = _config["llm_timeout_followups_s"]
LLM_TIMEOUT_INTERPRETATION_S: float = _config["llm_timeout_interpretation_s"]
LLM_TIMEOUT_RESULT_INTERPRETATION_S: float = _config["llm_timeout_result_interpretation_s"]
LLM_TIMEOUT_ERROR_TRANSLATION_S: float = _config["llm_timeout_error_translation_s"]
LLM_TIMEOUT_FILTER_EXTRACTION_S: float = _config["llm_timeout_filter_extraction_s"]

# Hard cap: prevents increasing timeouts to "fix" issues
# If any feature needs more than 30s, investigate model size or prompt complexity
LLM_TIMEOUT_MAX_S: float = _config["llm_timeout_max_s"]

# ADR009: Feature Flags
# Enable/disable LLM-enhanced features independently
ENABLE_RESULT_INTERPRETATION: bool = _config["enable_result_interpretation"]

# ADR004: Feature Flags for Surgical Rollback
# Enable/disable each phase independently for operational safety
# Defaults to True for backward compatibility - phases are enabled unless explicitly disabled
ADR004_ENABLE_DOC_EXTRACTION: bool = _config.get("adr004_enable_doc_extraction", True)
ADR004_ENABLE_SCHEMA_CONTEXT: bool = _config.get("adr004_enable_schema_context", True)
ADR004_ENABLE_AUTOCONTEXT: bool = _config.get("adr004_enable_autocontext", True)
ADR004_ENABLE_QUESTION_GENERATION: bool = _config.get("adr004_enable_question_generation", False)

# Legacy alias for Phase 4 (maintained for backward compatibility)
ENABLE_PROACTIVE_QUESTIONS: bool = _config["enable_proactive_questions"]

# ADR004 Phase 4: Proactive Question Generation Timeout
LLM_TIMEOUT_QUESTION_GENERATION_S: float = _config["llm_timeout_question_generation_s"]
