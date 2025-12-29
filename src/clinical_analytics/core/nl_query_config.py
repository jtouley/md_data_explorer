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
