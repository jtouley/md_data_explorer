# SPDX-License-Identifier: MIT
"""NL Query Engine Configuration Constants.

Single source of truth for confidence thresholds and parsing parameters.
These are domain config, not code - adjust without code changes.

Configuration is loaded from resources/config/nl_query.yaml via config_loader.
Ollama configuration values are also read from resources/config/nl_query.yaml
(resources/config/ollama.yaml is a reference copy and is not read).
Environment variables take precedence over YAML config.
"""

import logging
import os
import re
from dataclasses import dataclass
from typing import Literal, TypeAlias

from clinical_analytics.core.config_loader import load_nl_query_config

logger = logging.getLogger(__name__)

# Load configuration from YAML via config_loader
# This loads from resources/config/nl_query.yaml with environment variable overrides
_config = load_nl_query_config()

# Confidence thresholds for tier matching
TIER_1_PATTERN_MATCH_THRESHOLD: float = _config["tier_1_pattern_match_threshold"]
TIER_2_SEMANTIC_MATCH_THRESHOLD: float = _config["tier_2_semantic_match_threshold"]
CLARIFYING_QUESTIONS_THRESHOLD: float = _config["clarifying_questions_threshold"]

# Auto-execute threshold (intentionally same as TIER_2_SEMANTIC_MATCH_THRESHOLD)
# because semantic match is the minimum confidence for auto-execution
AUTO_EXECUTE_CONFIDENCE_THRESHOLD: float = _config["auto_execute_confidence_threshold"]

# Small-cell suppression for displayed patient counts
SMALL_CELL_THRESHOLD: int = int(_config["small_cell_threshold"])

# A numeric column with more distinct values than this is a measurement, never a grouping:
# one group per exact value (289 LDL levels) identifies patients and answers nothing.
MAX_NUMERIC_GROUP_LEVELS: int = 10

# Stored values a question can never name are not matched as words (core/value_index.py): a
# numeric column with more than MAX_NUMERIC_GROUP_LEVELS values, and an identifier with at
# least this many values (the patient id, or a column with a different value in every row).
# Below it a small table's every column is unique per row, and its values must still match.
VALUE_MATCH_MIN_IDENTIFIER_VALUES: int = 50

# Shared stem: two words match when they share at least this many leading letters (and 75% of
# the shorter). word_accounting.stem_match and the value index's prefix buckets
# (core/value_index.py) both read it, so the index stays an exact superset of stem_match.
STEM_MATCH_MIN_PREFIX: int = 5

MemoryUnit: TypeAlias = Literal["%", "KB", "MB", "GB", "TB", "KiB", "MiB", "GiB", "TiB"]
_UNITS: dict[str, MemoryUnit] = {
    "KB": "KB",
    "MB": "MB",
    "GB": "GB",
    "TB": "TB",
    "KIB": "KiB",
    "MIB": "MiB",
    "GIB": "GiB",
    "TIB": "TiB",
}
_MEMORY_LIMIT = re.compile(r"(\d+(?:\.\d+)?)\s*(%|[KMGT]i?B)", re.IGNORECASE)


@dataclass(frozen=True)
class DuckdbMemoryLimit:
    """A DuckDB memory_limit: a positive size, or a share of RAM above 0 and at most 100%."""

    amount: float
    unit: MemoryUnit

    @property
    def sql(self) -> str:
        """The value for ``SET memory_limit``, built from the parsed number and unit only."""
        return f"{self.amount:.6f}".rstrip("0").rstrip(".") + self.unit


def parse_duckdb_memory_limit(value: object, name: str = "duckdb_memory_budget") -> DuckdbMemoryLimit:
    """Parse ``"20%"``, ``"2GB"`` or ``"512 MiB"`` into a number and a unit. ``name`` is the
    setting named in the error.

    Raises:
        ValueError: not a size or a percentage, zero or less (DuckDB accepts "0GB" and then
            fails every query), or a percentage above 100.
    """
    text = value.strip() if isinstance(value, str) else ""
    match = _MEMORY_LIMIT.fullmatch(text)
    if match is None:
        raise ValueError(f"{name} must be a size such as '2GB' or a share such as '20%', got {value!r}")
    amount = float(match.group(1))
    raw_unit = match.group(2)
    unit: MemoryUnit = "%" if raw_unit == "%" else _UNITS[raw_unit.upper()]
    if amount <= 0:
        raise ValueError(f"{name} must be above 0, got {value!r}")
    if unit == "%" and amount > 100:
        raise ValueError(f"{name} must be at most 100%, got {value!r}")
    return DuckdbMemoryLimit(amount, unit)


def resolve_duckdb_threads(value: object) -> int:
    """The DuckDB thread count: a positive count as given, 0 for half the CPUs (at least 1).

    Raises:
        ValueError: not a whole number (a YAML boolean included), or below 0.
    """
    if isinstance(value, bool) or not isinstance(value, int | str):
        raise ValueError(f"duckdb_threads must be a whole number, got {value!r}")
    try:
        threads = int(value)
    except ValueError:
        raise ValueError(f"duckdb_threads must be a whole number, got {value!r}") from None
    if threads < 0:
        raise ValueError(f"duckdb_threads must be 0 (half the CPUs) or more, got {threads}")
    return threads or max(1, (os.cpu_count() or 2) // 2)


_BYTES_PER_UNIT: dict[str, int] = {
    "KB": 1000,
    "MB": 1000**2,
    "GB": 1000**3,
    "TB": 1000**4,
    "KiB": 1024,
    "MiB": 1024**2,
    "GiB": 1024**3,
    "TiB": 1024**4,
}
# Below these a layer's DuckDB cannot hold a dataset's working set: every question would fail.
MIN_LAYER_MEMORY_BYTES: int = 64 * 1024**2
MIN_LAYER_MEMORY_PERCENT: float = 1.0


def resolve_semantic_layer_cache_size(value: object) -> int:
    """How many semantic layers (one in-memory DuckDB per dataset) the API keeps open: 1 or more.

    Raises:
        ValueError: not a whole number (a YAML boolean included), or below 1. There is no
            "unbounded" value: the DuckDB memory budget is split across this many layers.
    """
    if isinstance(value, bool) or not isinstance(value, int | str):
        raise ValueError(f"semantic_layer_cache_size must be a whole number of 1 or more, got {value!r}")
    try:
        size = int(value)
    except ValueError:
        raise ValueError(f"semantic_layer_cache_size must be a whole number of 1 or more, got {value!r}") from None
    if size < 1:
        raise ValueError(f"semantic_layer_cache_size must be 1 or more, got {size}")
    return size


def per_layer_memory_limit(budget: DuckdbMemoryLimit, cache_size: int) -> DuckdbMemoryLimit:
    """Each open layer's DuckDB memory_limit: the total budget shared evenly by ``cache_size``
    layers, so the layers the cache keeps open use at most the budget together.

    Raises:
        ValueError: the share is too small to answer questions (under 64 MiB, or under 1% of RAM).
    """
    limit = DuckdbMemoryLimit(budget.amount / cache_size, budget.unit)
    too_small = (
        limit.amount < MIN_LAYER_MEMORY_PERCENT
        if limit.unit == "%"
        else limit.amount * _BYTES_PER_UNIT[limit.unit] < MIN_LAYER_MEMORY_BYTES
    )
    if too_small:
        raise ValueError(
            f"duckdb_memory_budget {budget.sql} shared by {cache_size} open datasets leaves {limit.sql} each, "
            "too little to answer questions: raise the budget or lower semantic_layer_cache_size"
        )
    return limit


def _setting(key: str) -> object:
    """A setting's raw value: its environment variable if set, else the YAML (or default) value.

    Both are raw (config_loader.RAW_VALIDATED_KEYS skips the YAML coercion), so a bad value is
    rejected here, never replaced by the default or truncated ("four", 2.5).
    """
    env_value = os.environ.get(key.upper())
    return env_value if env_value is not None else _config[key]


if os.environ.get("DUCKDB_MEMORY_LIMIT") is not None or "duckdb_memory_limit" in _config:
    # Was a per-layer limit; it is now a total. Ignoring it would silently change the memory used.
    raise ValueError(
        "duckdb_memory_limit (DUCKDB_MEMORY_LIMIT) was replaced by duckdb_memory_budget "
        "(DUCKDB_MEMORY_BUDGET), a total shared by all open datasets: set that instead"
    )

# The API keeps at most this many semantic layers (datasets) open; see api/layer_cache.py.
SEMANTIC_LAYER_CACHE_SIZE: int = resolve_semantic_layer_cache_size(_setting("semantic_layer_cache_size"))
# DuckDB memory for ALL the layers the API's cache keeps open together. Each layer is its own
# in-memory DuckDB, so each cached layer gets CACHED_LAYER_MEMORY_LIMIT = budget / cache size
# (set by the API when its cache builds the layer). Threads are per layer. Validated here, at
# import, so a bad setting stops the process at startup instead of failing the first question.
DUCKDB_MEMORY_BUDGET: DuckdbMemoryLimit = parse_duckdb_memory_limit(_setting("duckdb_memory_budget"))
CACHED_LAYER_MEMORY_LIMIT: DuckdbMemoryLimit = per_layer_memory_limit(DUCKDB_MEMORY_BUDGET, SEMANTIC_LAYER_CACHE_SIZE)
# Every other layer (an upload's probe, the Streamlit app, the CLI, evals) keeps the standalone
# per-layer limit it had before the budget existed; applied in SemanticLayer.__init__.
DUCKDB_MEMORY_LIMIT: DuckdbMemoryLimit = DuckdbMemoryLimit(20.0, "%")
DUCKDB_THREADS: int = resolve_duckdb_threads(_setting("duckdb_threads"))

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

# Shown when the language-model step raised: the question is refused with this, never answered
# from a low-confidence DESCRIBE stub that looks like a normal parse.
MODEL_STEP_FAILED_REASON: str = (
    "The local language model couldn’t be used for this question, so it was not answered. "
    "Ask again naming the columns you mean, or restart the app if it keeps happening."
)
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
# DEPRECATED: Use ADR004_ENABLE_QUESTION_GENERATION instead
ENABLE_PROACTIVE_QUESTIONS: bool = _config.get("enable_proactive_questions", False)

# Log deprecation warning if legacy key is explicitly set in config
if "enable_proactive_questions" in _config and "adr004_enable_question_generation" not in _config:
    logger.warning(
        "Config key 'enable_proactive_questions' is deprecated. "
        "Use 'adr004_enable_question_generation' instead. Will be removed in v2.0."
    )

# ADR004 Phase 4: Proactive Question Generation Timeout
LLM_TIMEOUT_QUESTION_GENERATION_S: float = _config["llm_timeout_question_generation_s"]
