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

All YAML files are validated using dataclasses with manual validation (matching existing codebase patterns). The config loader:

- Validates required fields are present
- Validates types match expected types
- Validates value ranges (e.g., confidence thresholds 0.0-1.0)
- Raises `ValueError` with field-level error messages on validation failure

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
- Raises `ValueError` with field name and expected type
- Includes the invalid value for debugging

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

