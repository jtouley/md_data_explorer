# Configuration Directory

This directory contains all YAML configuration files for the Clinical Analytics Platform.

## Design Principles

1. **Config is Data**: Values that change by environment → YAML
2. **Code is Behavior**: Orchestration, validation, side effects → Python
3. **Environment Variable Precedence**: Env vars override YAML (maintain existing behavior)
4. **Backward Compatibility**: Public API unchanged (same constant names)
5. **Schema Validation**: Validate YAML structure and types
6. **Idempotent Loading**: Safe to call multiple times
7. **Clear Error Messages**: Fail fast with helpful errors

## Configuration Files

### `nl_query.yaml`
Natural Language Query Engine configuration. Contains:
- Confidence thresholds for tier matching
- Semantic matching parameters
- Feature flags (clarifying questions, progressive feedback)
- LLM timeout settings
- Ollama configuration

### `ui.yaml`
UI configuration. Contains:
- Feature flags (multi-table support, V1 MVP mode)
- Logging level
- Upload size limits
- Page names

### `logging.yaml`
Logging configuration. Contains:
- Root log level
- Format string
- Module-specific log levels
- Noise reduction settings (loggers to set to WARNING)

### `ollama.yaml`
Ollama LLM service configuration. Contains:
- Base URL
- Default and fallback models
- Timeout and retry settings
- JSON mode flag
- Confidence thresholds

### `prompt_learning.yaml`
Prompt learning and optimization configuration. Contains:
- Intent keyword patterns
- Refinement phrases
- Valid intents
- Failure pattern detection rules
- Prompt template

### `datasets.yaml`
Dataset configurations. Contains:
- Dataset metadata (name, description, source)
- Column mappings
- Outcome definitions
- Analysis configurations
- Metrics and dimensions

## Environment Variable Precedence

Configuration values follow this precedence order (highest to lowest):

1. **Environment Variables** (highest priority)
2. **YAML File Values**
3. **Default Values** (lowest priority)

### Automatic Environment Variable Override

**IMPORTANT**: The config loader automatically overrides YAML values with environment variables if the environment variable name matches the config key (uppercase). This means:

- Any environment variable matching a config key will override the YAML value
- Example: Setting `TIER_2_SEMANTIC_MATCH_THRESHOLD=0.80` will override the YAML value
- This behavior applies to ALL config keys, not just those explicitly listed below

**Use with caution**: Accidental environment variables (e.g., `TIER_1_PATTERN_MATCH_THRESHOLD=0.5`) will override YAML values, which could cause unexpected behavior. Always verify environment variables before deployment.

### Environment Variable Mapping

The following environment variables override YAML values:

#### NL Query Configuration
- `ENABLE_CLARIFYING_QUESTIONS` → `enable_clarifying_questions` (bool)
- `ENABLE_PROGRESSIVE_FEEDBACK` → `enable_progressive_feedback` (bool)
- `OLLAMA_BASE_URL` → `ollama_base_url` (str)
- `OLLAMA_DEFAULT_MODEL` → `ollama_default_model` (str)
- `OLLAMA_FALLBACK_MODEL` → `ollama_fallback_model` (str)
- `OLLAMA_TIMEOUT_SECONDS` → `ollama_timeout_seconds` (float)
- `OLLAMA_MAX_RETRIES` → `ollama_max_retries` (int)
- `OLLAMA_JSON_MODE` → `ollama_json_mode` (bool)
- `LLM_TIMEOUT_PARSE_S` → `llm_timeout_parse_s` (float)
- `LLM_TIMEOUT_FOLLOWUPS_S` → `llm_timeout_followups_s` (float)
- `LLM_TIMEOUT_INTERPRETATION_S` → `llm_timeout_interpretation_s` (float)
- `LLM_TIMEOUT_RESULT_INTERPRETATION_S` → `llm_timeout_result_interpretation_s` (float)
- `LLM_TIMEOUT_ERROR_TRANSLATION_S` → `llm_timeout_error_translation_s` (float)
- `LLM_TIMEOUT_FILTER_EXTRACTION_S` → `llm_timeout_filter_extraction_s` (float)
- `LLM_TIMEOUT_MAX_S` → `llm_timeout_max_s` (float)
- `ENABLE_RESULT_INTERPRETATION` → `enable_result_interpretation` (bool)

#### UI Configuration
- `MULTI_TABLE_ENABLED` → `multi_table_enabled` (bool)
- `V1_MVP_MODE` → `v1_mvp_mode` (bool)
- `LOG_LEVEL` → `log_level` (str)
- `MAX_UPLOAD_SIZE_MB` → `max_upload_size_mb` (int)

### Type Coercion Rules

Environment variables are automatically coerced to the correct type:

- String `"30.0"` → `float` 30.0
- String `"true"`/`"false"` → `bool` True/False (case-insensitive)
- String `"123"` → `int` 123
- Missing env var → use YAML value → use default

#### Boolean Coercion Accepted Values

Boolean values accept the following string values (case-insensitive):
- **Truthy**: `"true"`, `"1"`, `"yes"`, `"on"`
- **Falsy**: `"false"`, `"0"`, `"no"`, `"off"`

For example:
- `ENABLE_CLARIFYING_QUESTIONS=true` → `True`
- `ENABLE_CLARIFYING_QUESTIONS=1` → `True`
- `ENABLE_CLARIFYING_QUESTIONS=yes` → `True`
- `ENABLE_CLARIFYING_QUESTIONS=false` → `False`
- `ENABLE_CLARIFYING_QUESTIONS=0` → `False`
- `ENABLE_CLARIFYING_QUESTIONS=no` → `False`

## Project Root Detection

The config loader automatically detects the project root using this pattern:

```
config_loader.py → core/ → clinical_analytics/ → src/ → project_root
```

Config files are located at: `{project_root}/config/*.yaml`

This matches the existing codebase pattern used in `nl_query_config.py`:
```python
current_file = Path(__file__)
project_root = current_file.parent.parent.parent.parent
```

## Schema Validation

All YAML files are validated using dataclasses with type coercion (matching existing codebase patterns). The config loader:

- Validates types match expected types (with automatic coercion)
- Coerces string values to appropriate types (float, bool, int)
- Raises `ValueError` for critical config type coercion failures (thresholds, timeouts)
- Logs warnings for non-critical config type coercion failures (uses defaults)

**Note**: Value range validation (e.g., confidence thresholds 0.0-1.0) is not currently implemented. The config loader focuses on type safety and graceful degradation.

## Error Handling

### Missing YAML File
- Logs warning
- Uses default values
- Does not raise exception (graceful degradation)

### Invalid YAML Syntax
- Raises `ValueError` with clear message
- Includes file path and line number if available

### Schema Validation Failure
- Raises `ValueError` with field-level errors
- Lists all validation failures (not just first)

### Type Coercion Failure

**Critical Configs** (thresholds, timeouts):
- Raises `ValueError` with field name and expected type
- Includes the invalid value for debugging
- Prevents silent failures that could cause serious issues

**Non-Critical Configs** (feature flags, strings):
- Logs warning and uses default value
- Allows graceful degradation for less critical settings

## Usage

Configuration is loaded via the centralized `config_loader` module:

```python
from clinical_analytics.core.config_loader import (
    load_nl_query_config,
    load_ui_config,
    load_logging_config,
)

# Load NL query config
nl_config = load_nl_query_config()
threshold = nl_config["tier_2_semantic_match_threshold"]

# Load UI config
ui_config = load_ui_config()
multi_table_enabled = ui_config["multi_table_enabled"]

# Load logging config
logging_config = load_logging_config()
root_level = logging_config["root_level"]
```

## Backward Compatibility

All existing Python modules maintain their public API:

- `nl_query_config.py` exports the same constant names
- `ui/config.py` exports the same constant names
- `logging_config.py` keeps the `configure_logging()` function

Import statements continue to work without changes:

```python
from clinical_analytics.core.nl_query_config import TIER_2_SEMANTIC_MATCH_THRESHOLD
from clinical_analytics.ui.config import MULTI_TABLE_ENABLED
```

## Migration Notes

This configuration system was migrated from Python constants to YAML files to:

1. Separate configuration data from code
2. Enable environment-specific configuration without code changes
3. Improve maintainability and clarity
4. Support schema validation

The migration maintains 100% backward compatibility - all existing imports continue to work.
