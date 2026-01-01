"""NL Query Engine Configuration Constants.

Single source of truth for confidence thresholds and parsing parameters.
These are domain config, not code - adjust without code changes.

Ollama configuration is loaded from ollama_config.yaml in the project root.
Environment variables take precedence over YAML config.
"""

import logging
import os
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

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


def _load_ollama_config() -> dict[str, Any]:
    """
    Load Ollama configuration from YAML file with fallback to defaults.

    Returns:
        Dictionary with Ollama configuration values
    """
    # Default values (used if YAML missing or incomplete)
    defaults = {
        "base_url": "http://localhost:11434",
        "default_model": "llama3.1:8b",
        "fallback_model": "llama3.2:3b",
        "timeout_seconds": 30.0,
        "max_retries": 3,
        "json_mode": True,
        "min_confidence": 0.5,
        "execution_threshold": 0.75,
    }

    # Try to find project root (where ollama_config.yaml should be)
    # Start from this file and go up to find project root
    current_file = Path(__file__)
    project_root = current_file.parent.parent.parent.parent

    config_path = project_root / "ollama_config.yaml"

    if not config_path.exists():
        logger.debug(f"Ollama config file not found at {config_path}, using defaults")
        return defaults

    try:
        with open(config_path) as f:
            config = yaml.safe_load(f) or {}

        # Extract ollama section
        ollama_config = config.get("ollama", {})
        confidence_config = config.get("confidence", {})

        # Merge with defaults
        result = {
            "base_url": ollama_config.get("base_url", defaults["base_url"]),
            "default_model": ollama_config.get("default_model", defaults["default_model"]),
            "fallback_model": ollama_config.get("fallback_model", defaults["fallback_model"]),
            "timeout_seconds": float(ollama_config.get("timeout_seconds", defaults["timeout_seconds"])),
            "max_retries": int(ollama_config.get("max_retries", defaults["max_retries"])),
            "json_mode": bool(ollama_config.get("json_mode", defaults["json_mode"])),
            "min_confidence": float(confidence_config.get("min_confidence", defaults["min_confidence"])),
            "execution_threshold": float(confidence_config.get("execution_threshold", defaults["execution_threshold"])),
        }

        logger.debug(f"Loaded Ollama config from {config_path}")
        return result

    except Exception as e:
        logger.warning(f"Failed to load Ollama config from {config_path}: {e}, using defaults")
        return defaults


# Load Ollama config from YAML
_ollama_config = _load_ollama_config()

# Tier 3 LLM Fallback Configuration (ADR003 Phase 0)
# Privacy-preserving: Local Ollama only, no external API calls
# Environment variables take precedence over YAML config
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", _ollama_config["base_url"])
OLLAMA_DEFAULT_MODEL = os.getenv("OLLAMA_DEFAULT_MODEL", _ollama_config["default_model"])
OLLAMA_FALLBACK_MODEL = os.getenv("OLLAMA_FALLBACK_MODEL", _ollama_config["fallback_model"])
OLLAMA_TIMEOUT_SECONDS = float(os.getenv("OLLAMA_TIMEOUT_SECONDS", str(_ollama_config["timeout_seconds"])))
OLLAMA_MAX_RETRIES = int(os.getenv("OLLAMA_MAX_RETRIES", str(_ollama_config["max_retries"])))
OLLAMA_JSON_MODE = (
    os.getenv("OLLAMA_JSON_MODE", str(_ollama_config["json_mode"])).lower() == "true"
    if isinstance(os.getenv("OLLAMA_JSON_MODE"), str)
    else _ollama_config["json_mode"]
)

# Tier 3 confidence thresholds (Phase 0 success vs Phase 3 execution gate)
TIER_3_MIN_CONFIDENCE = _ollama_config["min_confidence"]
TIER_3_EXECUTION_THRESHOLD = _ollama_config["execution_threshold"]

# ADR009: LLM Feature-Specific Timeout Configuration (Pre-Phase)
# Each feature has its own timeout based on complexity
# Error translation should be fast (5s) - users are already frustrated by error
# Result interpretation can be longer (20s) - users expect thoughtful analysis
# Query parsing is most complex (30s) - requires understanding schema and intent
LLM_TIMEOUT_PARSE_S: float = float(os.getenv("LLM_TIMEOUT_PARSE_S", "30.0"))
LLM_TIMEOUT_FOLLOWUPS_S: float = float(os.getenv("LLM_TIMEOUT_FOLLOWUPS_S", "30.0"))
LLM_TIMEOUT_INTERPRETATION_S: float = float(os.getenv("LLM_TIMEOUT_INTERPRETATION_S", "30.0"))
LLM_TIMEOUT_RESULT_INTERPRETATION_S: float = float(os.getenv("LLM_TIMEOUT_RESULT_INTERPRETATION_S", "20.0"))
LLM_TIMEOUT_ERROR_TRANSLATION_S: float = float(os.getenv("LLM_TIMEOUT_ERROR_TRANSLATION_S", "5.0"))
LLM_TIMEOUT_FILTER_EXTRACTION_S: float = float(os.getenv("LLM_TIMEOUT_FILTER_EXTRACTION_S", "30.0"))

# Hard cap: prevents increasing timeouts to "fix" issues
# If any feature needs more than 30s, investigate model size or prompt complexity
LLM_TIMEOUT_MAX_S: float = float(os.getenv("LLM_TIMEOUT_MAX_S", "30.0"))

# ADR009: Feature Flags
# Enable/disable LLM-enhanced features independently
ENABLE_RESULT_INTERPRETATION: bool = os.getenv("ENABLE_RESULT_INTERPRETATION", "true").lower() == "true"
