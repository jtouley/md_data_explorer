"""Centralized configuration loader for YAML-based configuration.

This module provides functions to load configuration from YAML files with:
- Environment variable overrides (env var → YAML → defaults)
- Type coercion (string to float, bool, int)
- Schema validation using dataclasses
- Graceful degradation (missing files use defaults)
"""

import logging
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml  # type: ignore

logger = logging.getLogger(__name__)


def get_project_root() -> Path:
    """
    Get project root directory.

    Uses pattern: config_loader.py → core/ → clinical_analytics/ → src/ → project_root
    This matches the existing codebase pattern in nl_query_config.py.

    Validates that the config/ directory exists to ensure correct project root detection.

    Returns:
        Path to project root directory

    Raises:
        ValueError: If config/ directory is not found at the detected project root
    """
    current_file = Path(__file__)
    # Go up 4 levels: config_loader.py → core/ → clinical_analytics/ → src/ → project_root
    project_root = current_file.parent.parent.parent.parent

    # Validate that config/ directory exists
    config_dir = project_root / "config"
    if not config_dir.is_dir():
        raise ValueError(
            f"Project root detection failed: config/ directory not found at {config_dir}. "
            f"Detected project root: {project_root}. "
            f"If project structure has changed, update get_project_root() in config_loader.py"
        )

    return project_root


def _coerce_type(value: Any, target_type: type) -> Any:
    """
    Coerce value to target type.

    Handles:
    - String "30.0" → float 30.0
    - String "true"/"false" → bool True/False (case-insensitive)
    - String "123" → int 123

    Args:
        value: Value to coerce
        target_type: Target type (float, bool, int, str)

    Returns:
        Coerced value

    Raises:
        ValueError: If coercion fails
    """
    if value is None:
        return None

    if isinstance(value, target_type):
        return value

    if target_type is bool:
        if isinstance(value, str):
            return value.lower() in ("true", "1", "yes", "on")
        return bool(value)

    if target_type is float:
        if isinstance(value, str):
            return float(value)
        return float(value)

    if target_type is int:
        if isinstance(value, str):
            return int(float(value))  # Handle "30.0" → 30
        return int(value)

    if target_type is str:
        return str(value)

    return value


def _get_env_var(key: str, default: Any = None) -> str | None:
    """
    Get environment variable value.

    Args:
        key: Environment variable name
        default: Default value if not set

    Returns:
        Environment variable value or default
    """
    return os.getenv(key, default)


def _apply_env_overrides(config: dict[str, Any], env_mapping: dict[str, str] | None = None) -> dict[str, Any]:
    """
    Apply environment variable overrides to config.

    Supports two modes:
    1. Explicit mapping: env_mapping provides env var name → config key mapping
    2. Automatic mapping: Any env var matching config key (uppercase with underscores) overrides

    Args:
        config: Configuration dictionary
        env_mapping: Optional mapping of env var names to config keys

    Returns:
        Config with env var overrides applied
    """
    result = config.copy()

    # Apply explicit mappings first
    if env_mapping:
        for env_key, config_key in env_mapping.items():
            env_value = _get_env_var(env_key)
            if env_value is not None:
                # Coerce type based on existing config value type
                if config_key in result:
                    target_type = type(result[config_key])
                    try:
                        result[config_key] = _coerce_type(env_value, target_type)
                    except (ValueError, TypeError) as e:
                        logger.warning(f"Failed to coerce env var {env_key}={env_value} to {target_type.__name__}: {e}")
                else:
                    # New key, try to infer type from common patterns
                    if env_key.endswith("_ENABLED") or env_key.startswith("ENABLE_"):
                        result[config_key] = _coerce_type(env_value, bool)
                    elif env_key.endswith("_S") or env_key.endswith("_SECONDS"):
                        result[config_key] = _coerce_type(env_value, float)
                    elif env_key.endswith("_MB") or env_key.endswith("_RETRIES"):
                        result[config_key] = _coerce_type(env_value, int)
                    else:
                        result[config_key] = env_value

    # Apply automatic overrides: any env var matching config key (uppercase)
    for config_key in result.keys():
        # Convert config key to env var format: snake_case → SNAKE_CASE
        env_key = config_key.upper()
        env_value = _get_env_var(env_key)
        if env_value is not None:
            # Skip if already overridden by explicit mapping
            if env_mapping and env_key in env_mapping:
                continue
            # Coerce type based on existing config value type
            target_type = type(result[config_key])
            try:
                result[config_key] = _coerce_type(env_value, target_type)
            except (ValueError, TypeError) as e:
                logger.warning(f"Failed to coerce env var {env_key}={env_value} to {target_type.__name__}: {e}")

    return result


@dataclass
class NLQueryConfigDefaults:
    """Default values for NL query configuration."""

    tier_1_pattern_match_threshold: float = 0.9
    tier_2_semantic_match_threshold: float = 0.75
    clarifying_questions_threshold: float = 0.5
    auto_execute_confidence_threshold: float = 0.75  # Same as tier_2_semantic_match_threshold
    tier_timeout_seconds: float = 5.0
    enable_parallel_tier_matching: bool = False
    semantic_similarity_threshold: float = 0.7
    fuzzy_match_cutoff: float = 0.7
    enable_clarifying_questions: bool = True
    enable_progressive_feedback: bool = True
    ollama_base_url: str = "http://localhost:11434"
    ollama_default_model: str = "llama3.1:8b"
    ollama_fallback_model: str = "llama3.2:3b"
    ollama_timeout_seconds: float = 30.0
    ollama_max_retries: int = 3
    ollama_json_mode: bool = True
    tier_3_min_confidence: float = 0.5
    tier_3_execution_threshold: float = 0.75
    llm_timeout_parse_s: float = 30.0
    llm_timeout_followups_s: float = 30.0
    llm_timeout_interpretation_s: float = 30.0
    llm_timeout_result_interpretation_s: float = 20.0
    llm_timeout_error_translation_s: float = 5.0
    llm_timeout_filter_extraction_s: float = 30.0
    llm_timeout_max_s: float = 30.0
    enable_result_interpretation: bool = True
    enable_proactive_questions: bool = False
    llm_timeout_question_generation_s: float = 5.0
    enable_multi_layer_validation: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert dataclass to dictionary."""
        return {
            "tier_1_pattern_match_threshold": self.tier_1_pattern_match_threshold,
            "tier_2_semantic_match_threshold": self.tier_2_semantic_match_threshold,
            "clarifying_questions_threshold": self.clarifying_questions_threshold,
            "auto_execute_confidence_threshold": self.auto_execute_confidence_threshold,
            "tier_timeout_seconds": self.tier_timeout_seconds,
            "enable_parallel_tier_matching": self.enable_parallel_tier_matching,
            "semantic_similarity_threshold": self.semantic_similarity_threshold,
            "fuzzy_match_cutoff": self.fuzzy_match_cutoff,
            "enable_clarifying_questions": self.enable_clarifying_questions,
            "enable_progressive_feedback": self.enable_progressive_feedback,
            "ollama_base_url": self.ollama_base_url,
            "ollama_default_model": self.ollama_default_model,
            "ollama_fallback_model": self.ollama_fallback_model,
            "ollama_timeout_seconds": self.ollama_timeout_seconds,
            "ollama_max_retries": self.ollama_max_retries,
            "ollama_json_mode": self.ollama_json_mode,
            "tier_3_min_confidence": self.tier_3_min_confidence,
            "tier_3_execution_threshold": self.tier_3_execution_threshold,
            "llm_timeout_parse_s": self.llm_timeout_parse_s,
            "llm_timeout_followups_s": self.llm_timeout_followups_s,
            "llm_timeout_interpretation_s": self.llm_timeout_interpretation_s,
            "llm_timeout_result_interpretation_s": self.llm_timeout_result_interpretation_s,
            "llm_timeout_error_translation_s": self.llm_timeout_error_translation_s,
            "llm_timeout_filter_extraction_s": self.llm_timeout_filter_extraction_s,
            "llm_timeout_max_s": self.llm_timeout_max_s,
            "enable_result_interpretation": self.enable_result_interpretation,
            "enable_proactive_questions": self.enable_proactive_questions,
            "llm_timeout_question_generation_s": self.llm_timeout_question_generation_s,
            "enable_multi_layer_validation": self.enable_multi_layer_validation,
        }


def _is_critical_config(key: str) -> bool:
    """
    Check if a config key is critical (should raise ValueError on type coercion failure).

    Critical configs are those that could cause serious issues if wrong:
    - Confidence thresholds (tier_*_threshold, *_confidence_threshold, tier_3_*)
    - Timeouts (tier_timeout_seconds, llm_timeout_*, ollama_timeout_seconds)

    Args:
        key: Config key name

    Returns:
        True if config is critical, False otherwise
    """
    critical_patterns = [
        "threshold",
        "confidence",
        "timeout",
        "tier_3_",
    ]
    return any(pattern in key.lower() for pattern in critical_patterns)


def load_nl_query_config(config_path: Path | None = None) -> dict[str, Any]:
    """
    Load NL query config from YAML with env var overrides.

    Precedence: Environment variable → YAML value → Default value

    Args:
        config_path: Optional path to config file. If None, uses default location.

    Returns:
        dict with keys matching current constant names (snake_case)

    Raises:
        ValueError: If YAML is invalid or schema validation fails
    """
    defaults = NLQueryConfigDefaults().to_dict()

    # Determine config file path
    if config_path is None:
        project_root = get_project_root()
        config_path = project_root / "config" / "nl_query.yaml"

    # Load YAML if file exists
    config = defaults.copy()
    if config_path.exists():
        try:
            with open(config_path) as f:
                yaml_data = yaml.safe_load(f) or {}
                # Merge YAML data into defaults
                for key, value in yaml_data.items():
                    if key in defaults:
                        # Coerce type based on default value type
                        target_type = type(defaults[key])
                        try:
                            config[key] = _coerce_type(value, target_type)
                        except (ValueError, TypeError) as e:
                            # For critical configs, raise ValueError instead of warning
                            if _is_critical_config(key):
                                raise ValueError(
                                    f"Type coercion failed for critical config {key}={value}: "
                                    f"expected {target_type.__name__}, got {type(value).__name__}. "
                                    f"Error: {e}"
                                ) from e
                            # For non-critical configs, log warning and use default
                            logger.warning(
                                f"Failed to coerce YAML value {key}={value} to "
                                f"{target_type.__name__}: {e}, using default"
                            )
                            # Keep default value
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in {config_path}: {e}") from e
        except ValueError:
            # Re-raise ValueError (e.g., from critical config type coercion failure)
            raise
        except Exception as e:
            logger.warning(f"Failed to load config from {config_path}: {e}, using defaults")
    else:
        logger.debug(f"Config file not found at {config_path}, using defaults")

    # Environment variable mapping
    env_mapping = {
        "ENABLE_CLARIFYING_QUESTIONS": "enable_clarifying_questions",
        "ENABLE_PROGRESSIVE_FEEDBACK": "enable_progressive_feedback",
        "OLLAMA_BASE_URL": "ollama_base_url",
        "OLLAMA_DEFAULT_MODEL": "ollama_default_model",
        "OLLAMA_FALLBACK_MODEL": "ollama_fallback_model",
        "OLLAMA_TIMEOUT_SECONDS": "ollama_timeout_seconds",
        "OLLAMA_MAX_RETRIES": "ollama_max_retries",
        "OLLAMA_JSON_MODE": "ollama_json_mode",
        "LLM_TIMEOUT_PARSE_S": "llm_timeout_parse_s",
        "LLM_TIMEOUT_FOLLOWUPS_S": "llm_timeout_followups_s",
        "LLM_TIMEOUT_INTERPRETATION_S": "llm_timeout_interpretation_s",
        "LLM_TIMEOUT_RESULT_INTERPRETATION_S": "llm_timeout_result_interpretation_s",
        "LLM_TIMEOUT_ERROR_TRANSLATION_S": "llm_timeout_error_translation_s",
        "LLM_TIMEOUT_FILTER_EXTRACTION_S": "llm_timeout_filter_extraction_s",
        "LLM_TIMEOUT_MAX_S": "llm_timeout_max_s",
        "ENABLE_RESULT_INTERPRETATION": "enable_result_interpretation",
        "ENABLE_PROACTIVE_QUESTIONS": "enable_proactive_questions",
        "LLM_TIMEOUT_QUESTION_GENERATION_S": "llm_timeout_question_generation_s",
    }

    # Apply environment variable overrides
    config = _apply_env_overrides(config, env_mapping)

    return config


@dataclass
class UIConfigDefaults:
    """Default values for UI configuration."""

    multi_table_enabled: bool = False
    v1_mvp_mode: bool = True
    log_level: str = "INFO"
    max_upload_size_mb: int = 100
    ask_questions_page: str = "pages/03_💬_Ask_Questions.py"

    def to_dict(self) -> dict[str, Any]:
        """Convert dataclass to dictionary."""
        return {
            "multi_table_enabled": self.multi_table_enabled,
            "v1_mvp_mode": self.v1_mvp_mode,
            "log_level": self.log_level,
            "max_upload_size_mb": self.max_upload_size_mb,
            "ask_questions_page": self.ask_questions_page,
        }


def load_ui_config(config_path: Path | None = None) -> dict[str, Any]:
    """
    Load UI config from YAML with env var overrides.

    Precedence: Environment variable → YAML value → Default value

    Args:
        config_path: Optional path to config file. If None, uses default location.

    Returns:
        dict with keys:
        - multi_table_enabled: bool
        - v1_mvp_mode: bool
        - log_level: str
        - max_upload_size_mb: int
        - ask_questions_page: str

    Raises:
        ValueError: If YAML is invalid
    """
    defaults = UIConfigDefaults().to_dict()

    # Determine config file path
    if config_path is None:
        project_root = get_project_root()
        config_path = project_root / "config" / "ui.yaml"

    # Load YAML if file exists
    config = defaults.copy()
    if config_path.exists():
        try:
            with open(config_path) as f:
                yaml_data = yaml.safe_load(f) or {}
                # Merge YAML data into defaults
                for key, value in yaml_data.items():
                    if key in defaults:
                        # Coerce type based on default value type
                        target_type = type(defaults[key])
                        try:
                            config[key] = _coerce_type(value, target_type)
                        except (ValueError, TypeError) as e:
                            logger.warning(
                                f"Failed to coerce YAML value {key}={value} to "
                                f"{target_type.__name__}: {e}, using default"
                            )
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in {config_path}: {e}") from e
        except Exception as e:
            logger.warning(f"Failed to load config from {config_path}: {e}, using defaults")
    else:
        logger.debug(f"Config file not found at {config_path}, using defaults")

    # Environment variable mapping
    env_mapping = {
        "MULTI_TABLE_ENABLED": "multi_table_enabled",
        "V1_MVP_MODE": "v1_mvp_mode",
        "LOG_LEVEL": "log_level",
        "MAX_UPLOAD_SIZE_MB": "max_upload_size_mb",
    }

    # Apply environment variable overrides
    config = _apply_env_overrides(config, env_mapping)

    return config


@dataclass
class LoggingConfigDefaults:
    """Default values for logging configuration."""

    root_level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    module_levels: dict[str, str] | None = None
    reduce_noise: dict[str, str] | None = None

    def __post_init__(self) -> None:
        """Initialize default dict values."""
        if self.module_levels is None:
            self.module_levels = {
                "clinical_analytics.core.semantic": "INFO",
                "clinical_analytics.core.registry": "INFO",
                "clinical_analytics.core.multi_table_handler": "INFO",
                "clinical_analytics.ui.storage.user_datasets": "INFO",
                "clinical_analytics.datasets": "INFO",
            }
        if self.reduce_noise is None:
            self.reduce_noise = {
                "streamlit": "WARNING",
                "urllib3": "WARNING",
            }

    def to_dict(self) -> dict[str, Any]:
        """Convert dataclass to dictionary."""
        return {
            "root_level": self.root_level,
            "format": self.format,
            "module_levels": self.module_levels.copy() if self.module_levels else {},
            "reduce_noise": self.reduce_noise.copy() if self.reduce_noise else {},
        }


def load_logging_config(config_path: Path | None = None) -> dict[str, Any]:
    """
    Load logging config from YAML.

    Args:
        config_path: Optional path to config file. If None, uses default location.

    Returns:
        dict with keys:
        - root_level: str
        - format: str
        - module_levels: dict[str, str]
        - reduce_noise: dict[str, str]

    Raises:
        ValueError: If YAML is invalid
    """
    defaults = LoggingConfigDefaults().to_dict()

    # Determine config file path
    if config_path is None:
        project_root = get_project_root()
        config_path = project_root / "config" / "logging.yaml"

    # Load YAML if file exists
    config = defaults.copy()
    if config_path.exists():
        try:
            with open(config_path) as f:
                yaml_data = yaml.safe_load(f) or {}
                # Merge YAML data into defaults
                for key, value in yaml_data.items():
                    if key in defaults:
                        if key in ("module_levels", "reduce_noise"):
                            # Merge dicts
                            if isinstance(value, dict):
                                config[key].update(value)
                        else:
                            config[key] = value
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in {config_path}: {e}") from e
        except Exception as e:
            logger.warning(f"Failed to load config from {config_path}: {e}, using defaults")
    else:
        logger.debug(f"Config file not found at {config_path}, using defaults")

    return config


@dataclass
class ValidationConfigDefaults:
    """Default values for validation configuration."""

    validation_layers: dict[str, Any] | None = None
    validation_rules: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        """Initialize default dict values."""
        if self.validation_layers is None:
            self.validation_layers = {
                "dba": {
                    "system_prompt": "You are a DBA reviewing query plans for type safety.",
                    "response_schema": {"is_valid": "bool", "errors": "list[str]", "warnings": "list[str]"},
                },
                "analyst": {
                    "system_prompt": "You are an analyst reviewing query plans for business logic.",
                    "response_schema": {"is_valid": "bool", "errors": "list[str]", "warnings": "list[str]"},
                },
                "manager": {
                    "system_prompt": "You are a manager reviewing query plans for final approval.",
                    "response_schema": {"approved": "bool", "reason": "str", "confidence_adjustment": "float"},
                },
                "retry": {
                    "system_prompt": "Fix the errors and return corrected query plan.",
                    "response_schema": {"intent_type": "str", "filters": "list[dict]"},
                },
            }
        if self.validation_rules is None:
            self.validation_rules = {
                "max_retries": 1,
                "confidence_threshold": 0.6,
                "timeout_seconds": 20.0,
            }

    def to_dict(self) -> dict[str, Any]:
        """Convert dataclass to dictionary."""
        return {
            "validation_layers": self.validation_layers.copy() if self.validation_layers else {},
            "validation_rules": self.validation_rules.copy() if self.validation_rules else {},
        }


@dataclass
class PathsConfigDefaults:
    """Default values for path configuration."""

    prompt_overlay_dir: str = "/tmp/nl_query_learning"
    query_logs_dir: str = "data/query_logs"
    analytics_db: str = "data/analytics.duckdb"
    uploads_dir: str = "data/uploads"
    config_dir: str = "config"
    golden_questions: str = "tests/eval/golden_questions.yaml"

    def to_dict(self) -> dict[str, str]:
        """Convert dataclass to dictionary."""
        return {
            "prompt_overlay_dir": self.prompt_overlay_dir,
            "query_logs_dir": self.query_logs_dir,
            "analytics_db": self.analytics_db,
            "uploads_dir": self.uploads_dir,
            "config_dir": self.config_dir,
            "golden_questions": self.golden_questions,
        }


def load_paths_config(config_path: Path | None = None) -> dict[str, Path]:
    """
    Load paths from config/paths.yaml with env var resolution.

    Environment variables in paths are resolved using os.path.expandvars().
    If an env var is not set (path still contains $VAR), the default is used.

    Precedence: Resolved env var → YAML value → Default value

    Args:
        config_path: Optional path to config file. If None, uses default location.

    Returns:
        dict with path keys mapped to Path objects

    Raises:
        ValueError: If YAML is invalid
    """
    defaults = PathsConfigDefaults().to_dict()

    # Determine config file path
    if config_path is None:
        project_root = get_project_root()
        config_path = project_root / "config" / "paths.yaml"

    # Load YAML if file exists
    paths_config: dict[str, str] = {}
    yaml_defaults: dict[str, str] = {}

    if config_path.exists():
        try:
            with open(config_path) as f:
                yaml_data = yaml.safe_load(f) or {}
                paths_config = yaml_data.get("paths", {})
                yaml_defaults = yaml_data.get("defaults", {})
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in {config_path}: {e}") from e
        except Exception as e:
            logger.warning(f"Failed to load config from {config_path}: {e}, using defaults")

    # Build result with env var resolution
    result: dict[str, Path] = {}

    for key in defaults:
        # Start with hardcoded default
        value = defaults[key]

        # Override with YAML default if present
        if key in yaml_defaults:
            value = yaml_defaults[key]

        # Override with YAML path if present
        if key in paths_config:
            raw_value = paths_config[key]
            # Resolve environment variables
            resolved = os.path.expandvars(raw_value)
            # If env var was not set, expandvars returns the original string with $VAR
            # In that case, use the default
            if resolved.startswith("$") or "${" in resolved:
                # Env var not resolved, use default
                logger.debug(f"Path config {key}: env var not set, using default {value}")
            else:
                value = resolved

        result[key] = Path(value)

    return result


@lru_cache(maxsize=1)
def load_patterns_config(config_path: Path | None = None) -> dict[str, list[dict]]:
    """
    Load and compile regex patterns from config.

    Uses LRU cache to avoid recompiling patterns on every call.

    Args:
        config_path: Optional path to config file. If None, uses default location.

    Returns:
        dict mapping intent type to list of compiled pattern dicts.
        Each pattern dict has: regex (compiled), groups (dict), confidence (float)

    Raises:
        ValueError: If YAML is invalid
    """
    # Determine config file path
    if config_path is None:
        project_root = get_project_root()
        config_path = project_root / "config" / "nl_query_patterns.yaml"

    # Load YAML if file exists
    if not config_path.exists():
        logger.debug(f"Patterns config file not found at {config_path}, using empty patterns")
        return {}

    try:
        with open(config_path) as f:
            yaml_data = yaml.safe_load(f) or {}
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML in {config_path}: {e}") from e
    except Exception as e:
        logger.warning(f"Failed to load patterns from {config_path}: {e}")
        return {}

    # Compile patterns
    compiled: dict[str, list[dict]] = {}
    patterns_data = yaml_data.get("patterns", {})

    for intent, patterns in patterns_data.items():
        compiled[intent] = []
        for p in patterns:
            try:
                compiled_pattern = {
                    "regex": re.compile(p["pattern"], re.IGNORECASE),
                    "groups": p.get("groups", {}),
                    "confidence": p.get("confidence", 0.9),
                }
                compiled[intent].append(compiled_pattern)
            except re.error as e:
                logger.warning(f"Failed to compile pattern '{p.get('pattern')}': {e}")
                continue

    logger.debug(
        "patterns_loaded: intent_count=%d, total_patterns=%d",
        len(compiled),
        sum(len(p) for p in compiled.values()),
    )

    return compiled


def load_validation_config(config_path: Path | None = None) -> dict[str, Any]:
    """
    Load validation config from YAML with env var overrides.

    Precedence: Environment variable → YAML value → Default value

    Args:
        config_path: Optional path to config file. If None, uses default location.

    Returns:
        dict with keys:
        - validation_layers: dict (dba, analyst, manager, retry)
        - validation_rules: dict (max_retries, confidence_threshold, timeout_seconds)

    Raises:
        FileNotFoundError: If config file missing
        ValueError: If YAML is invalid
    """
    # Determine config file path
    if config_path is None:
        project_root = get_project_root()
        config_path = project_root / "config" / "validation.yaml"

    # Validation config is required - raise if missing
    if not config_path.exists():
        raise FileNotFoundError(f"Validation config file not found at {config_path}")

    # Load YAML
    try:
        with open(config_path) as f:
            config = yaml.safe_load(f) or {}
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML in {config_path}: {e}") from e

    # Merge with defaults for any missing keys
    defaults = ValidationConfigDefaults().to_dict()
    if "validation_layers" not in config:
        config["validation_layers"] = defaults["validation_layers"]
    if "validation_rules" not in config:
        config["validation_rules"] = defaults["validation_rules"]

    # Apply environment variable overrides for validation rules
    env_mapping = {
        "VALIDATION_MAX_RETRIES": "max_retries",
        "VALIDATION_CONFIDENCE_THRESHOLD": "confidence_threshold",
        "VALIDATION_TIMEOUT_SECONDS": "timeout_seconds",
    }

    rules = config.get("validation_rules", {})
    for env_key, config_key in env_mapping.items():
        env_value = _get_env_var(env_key)
        if env_value is not None:
            if config_key in rules:
                target_type = type(rules[config_key])
                try:
                    rules[config_key] = _coerce_type(env_value, target_type)
                except (ValueError, TypeError) as e:
                    logger.warning(f"Failed to coerce env var {env_key}={env_value} to {target_type.__name__}: {e}")

    return config
