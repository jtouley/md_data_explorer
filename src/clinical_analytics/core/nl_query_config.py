"""NL Query Engine Configuration Constants.

Single source of truth for confidence thresholds and parsing parameters.
These are domain config, not code - adjust without code changes.
"""

import os

# Confidence thresholds for tier matching
TIER_1_PATTERN_MATCH_THRESHOLD = 0.9  # Pattern matching requires high confidence
TIER_2_SEMANTIC_MATCH_THRESHOLD = 0.75  # Semantic matching threshold
CLARIFYING_QUESTIONS_THRESHOLD = 0.5  # Below this, ask clarifying questions

# Auto-execute threshold (intentionally same as TIER_2_SEMANTIC_MATCH_THRESHOLD)
# because semantic match is the minimum confidence for auto-execution
AUTO_EXECUTE_CONFIDENCE_THRESHOLD = TIER_2_SEMANTIC_MATCH_THRESHOLD

# Performance/timeout settings
TIER_TIMEOUT_SECONDS = 5.0  # Fail fast if any tier takes too long
ENABLE_PARALLEL_TIER_MATCHING = False  # Future optimization (not implemented yet)

# Semantic matching parameters
SEMANTIC_SIMILARITY_THRESHOLD = 0.7  # Minimum cosine similarity for semantic match
FUZZY_MATCH_CUTOFF = 0.7  # difflib cutoff for fuzzy variable matching

# Feature flags (env vars with defaults)
ENABLE_CLARIFYING_QUESTIONS = os.getenv("ENABLE_CLARIFYING_QUESTIONS", "true").lower() == "true"
ENABLE_PROGRESSIVE_FEEDBACK = os.getenv("ENABLE_PROGRESSIVE_FEEDBACK", "true").lower() == "true"

# Tier 3 LLM Fallback Configuration (ADR003 Phase 0)
# Privacy-preserving: Local Ollama only, no external API calls
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_DEFAULT_MODEL = os.getenv("OLLAMA_DEFAULT_MODEL", "llama3.1:8b")
OLLAMA_FALLBACK_MODEL = os.getenv("OLLAMA_FALLBACK_MODEL", "llama3.2:3b")
OLLAMA_TIMEOUT_SECONDS = float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "5.0"))
OLLAMA_MAX_RETRIES = int(os.getenv("OLLAMA_MAX_RETRIES", "3"))
OLLAMA_JSON_MODE = os.getenv("OLLAMA_JSON_MODE", "true").lower() == "true"

# Tier 3 confidence thresholds (Phase 0 success vs Phase 3 execution gate)
TIER_3_MIN_CONFIDENCE = 0.5  # Minimum for Phase 0 success (valid parse)
TIER_3_EXECUTION_THRESHOLD = 0.75  # Minimum for Phase 3 execution (trusted execution)
