---
name: Config Migration to YAML
overview: Migrate all configuration from Python files to YAML files in a dedicated config/ folder at root level, with a centralized config loader, schema validation, and comprehensive documentation.
todos:
  - id: "1"
    content: Create config/ folder structure and README.md with documentation
    status: pending
  - id: "2"
    content: Write failing tests for config_loader (TDD Red phase)
    status: pending
    dependencies:
      - "1"
  - id: "3"
    content: Create config_loader.py module with YAML loading, schema validation, and env var support (TDD Green phase)
    status: pending
    dependencies:
      - "2"
  - id: "4"
    content: Run tests for config_loader and fix quality issues (TDD Refactor phase)
    status: pending
    dependencies:
      - "3"
  - id: "5"
    content: Move ollama_config.yaml to config/ollama.yaml
    status: pending
    dependencies:
      - "1"
  - id: "6"
    content: Move prompt_learning_config.yaml to config/prompt_learning.yaml
    status: pending
    dependencies:
      - "1"
  - id: "7"
    content: Move data/configs/datasets.yaml to config/datasets.yaml
    status: pending
    dependencies:
      - "1"
  - id: "8"
    content: Create config/nl_query.yaml with all NL query engine constants
    status: pending
    dependencies:
      - "1"
      - "3"
  - id: "9"
    content: Create config/ui.yaml with UI configuration values
    status: pending
    dependencies:
      - "1"
      - "3"
  - id: "10"
    content: Create config/logging.yaml with logging configuration values
    status: pending
    dependencies:
      - "1"
      - "3"
  - id: "11"
    content: Write failing tests for nl_query_config refactor (TDD Red phase)
    status: pending
    dependencies:
      - "8"
  - id: "12"
    content: Refactor nl_query_config.py to load from YAML via config_loader (TDD Green phase)
    status: pending
    dependencies:
      - "3"
      - "8"
      - "5"
      - "11"
  - id: "13"
    content: Run tests for nl_query_config and fix quality issues (TDD Refactor phase)
    status: pending
    dependencies:
      - "12"
  - id: "14"
    content: Write failing tests for ui/config refactor (TDD Red phase)
    status: pending
    dependencies:
      - "9"
  - id: "15"
    content: Refactor ui/config.py to load from YAML via config_loader (TDD Green phase)
    status: pending
    dependencies:
      - "3"
      - "9"
      - "14"
  - id: "16"
    content: Run tests for ui/config and fix quality issues (TDD Refactor phase)
    status: pending
    dependencies:
      - "15"
  - id: "17"
    content: Write failing tests for logging_config refactor (TDD Red phase)
    status: pending
    dependencies:
      - "10"
  - id: "18"
    content: Refactor logging_config.py to load values from YAML, keep function (TDD Green phase)
    status: pending
    dependencies:
      - "3"
      - "10"
      - "17"
  - id: "19"
    content: Run tests for logging_config and fix quality issues (TDD Refactor phase)
    status: pending
    dependencies:
      - "18"
  - id: "20"
    content: Update prompt_optimizer.py to use new config path
    status: pending
    dependencies:
      - "6"
  - id: "21"
    content: Update mapper.py to use new config path
    status: pending
    dependencies:
      - "7"
  - id: "22"
    content: Update nl_query_config import statements (8 files), run tests
    status: pending
    dependencies:
      - "13"
  - id: "23"
    content: Update ui.config import statements (5 files), run tests
    status: pending
    dependencies:
      - "16"
  - id: "24"
    content: Update logging_config import statements (2 files), run tests
    status: pending
    dependencies:
      - "19"
  - id: "25"
    content: Add backward compatibility integration test
    status: pending
    dependencies:
      - "22"
      - "23"
      - "24"
  - id: "26"
    content: Run full test suite and verify all tests pass
    status: pending
    dependencies:
      - "25"
---

# Configuration Migration to YAML

## Overview

Migrate all configuration constants from Python files to YAML files in a dedicated `config/` folder at the project root. This follows the principle: **Config is data, code is behavior**. Configuration values that change by environment belong in YAML; orchestration logic stays in Python.

## Target Structure

```javascript
config/
├── README.md                    # Configuration documentation
├── nl_query.yaml               # NL query engine config (from nl_query_config.py)
├── ui.yaml                     # UI config (from ui/config.py)
├── logging.yaml                # Logging config values (extracted from logging_config.py)
├── ollama.yaml                 # Ollama LLM config (moved from ollama_config.yaml)
├── prompt_learning.yaml        # Prompt learning config (moved from prompt_learning_config.yaml)
└── datasets.yaml               # Dataset configs (moved from data/configs/datasets.yaml)
```



## Implementation Plan

### Config Loader API Contract

**Function Signatures** (to be implemented in `config_loader.py`):

```python
def load_nl_query_config(config_path: Path | None = None) -> dict[str, Any]:
    """Load NL query config from YAML with env var overrides.
    
    Precedence: Environment variable → YAML value → Default value
    
    Returns:
        dict with keys matching current constant names (snake_case):
        - tier_1_pattern_match_threshold: float
        - tier_2_semantic_match_threshold: float
        - clarifying_questions_threshold: float
        - auto_execute_confidence_threshold: float
        - tier_timeout_seconds: float
        - enable_parallel_tier_matching: bool
        - semantic_similarity_threshold: float
        - fuzzy_match_cutoff: float
        - enable_clarifying_questions: bool
        - enable_progressive_feedback: bool
        - ollama_base_url: str
        - ollama_default_model: str
        - ollama_fallback_model: str
        - ollama_timeout_seconds: float
        - ollama_max_retries: int
        - ollama_json_mode: bool
        - tier_3_min_confidence: float
        - tier_3_execution_threshold: float
        - llm_timeout_parse_s: float
        - llm_timeout_followups_s: float
        - llm_timeout_interpretation_s: float
        - llm_timeout_result_interpretation_s: float
        - llm_timeout_error_translation_s: float
        - llm_timeout_filter_extraction_s: float
        - llm_timeout_max_s: float
        - enable_result_interpretation: bool
    
    Raises:
        FileNotFoundError: If config file missing and no defaults available
        ValueError: If YAML is invalid or schema validation fails
    """

def load_ui_config(config_path: Path | None = None) -> dict[str, Any]:
    """Load UI config from YAML with env var overrides.
    
    Returns:
        dict with keys:
        - multi_table_enabled: bool
        - v1_mvp_mode: bool
        - log_level: str
        - max_upload_size_mb: int
        - ask_questions_page: str
    """

def load_logging_config(config_path: Path | None = None) -> dict[str, Any]:
    """Load logging config from YAML.
    
    Returns:
        dict with keys:
        - root_level: str
        - format: str
        - module_levels: dict[str, str]
        - reduce_noise: dict[str, str]  # loggers to set to WARNING
    """
```

**Environment Variable Mapping**:

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
- `MULTI_TABLE_ENABLED` → `multi_table_enabled` (bool)
- `V1_MVP_MODE` → `v1_mvp_mode` (bool)
- `LOG_LEVEL` → `log_level` (str)
- `MAX_UPLOAD_SIZE_MB` → `max_upload_size_mb` (int)

**Type Coercion Rules**:
- String "30.0" → float 30.0
- String "true"/"false" → bool True/False (case-insensitive)
- String "123" → int 123
- Missing env var → use YAML value → use default

**Error Handling**:
- Missing YAML file → log warning, use defaults
- Invalid YAML → raise ValueError with clear message
- Schema validation failure → raise ValueError with field-level errors
- Type coercion failure → raise ValueError with field name and expected type

**Project Root Detection**:
- Use `Path(__file__).parent.parent.parent.parent` pattern (match existing codebase pattern)
- Config loader located at `src/clinical_analytics/core/config_loader.py`
- Project root = `config_loader.py` → `core/` → `clinical_analytics/` → `src/` → project root
- Config files at `{project_root}/config/*.yaml`

**Schema Validation Approach**:
- Use dataclasses + manual validation (match existing codebase patterns, no Pydantic dependency)
- Create dataclass for each config type with validation methods
- Validate required fields, types, and value ranges

### Phase 1: Create Config Infrastructure

**TDD Workflow**: Red → Green → Refactor

1. **Create `config/` folder and README** (Todo 1)
   - Create `config/README.md` with documentation
   - Document config structure, environment variable precedence, schema validation
   - Document project root detection strategy
   - **No tests needed** (documentation only)

2. **Write failing tests for config_loader** (Todo 2 - TDD Red Phase)
   - Create `tests/core/test_config_loader.py`
   - Test cases:
     - `test_load_nl_query_config_loads_from_yaml_file`
     - `test_load_nl_query_config_env_var_overrides_yaml`
     - `test_load_nl_query_config_missing_file_uses_defaults`
     - `test_load_nl_query_config_invalid_yaml_raises_valueerror`
     - `test_load_nl_query_config_type_coercion_string_to_float`
     - `test_load_nl_query_config_type_coercion_string_to_bool`
     - `test_load_ui_config_loads_from_yaml_file`
     - `test_load_logging_config_loads_from_yaml_file`
     - `test_get_project_root_returns_correct_path`
   - **Run test**: `make test-core PYTEST_ARGS="tests/core/test_config_loader.py -xvs"`
   - **Verify**: All tests fail (Red phase confirmed)

3. **Create config_loader.py module** (Todo 3 - TDD Green Phase)
   - Implement `src/clinical_analytics/core/config_loader.py`
   - Implement all functions to pass tests
   - Use dataclasses for schema validation (match existing patterns)
   - **Run test**: `make test-core PYTEST_ARGS="tests/core/test_config_loader.py -xvs"`
   - **Verify**: All tests pass (Green phase confirmed)

4. **Fix quality issues** (Todo 4 - TDD Refactor Phase)
   - Run: `make format`
   - Run: `make lint-fix`
   - Run: `make type-check`
   - Fix any remaining issues
   - **Run test**: `make test-core`
   - **Verify**: All tests pass, no regressions
   - **Commit**: `feat: Phase 1 - Config loader infrastructure

- Add config_loader.py with YAML loading and env var support
- Add comprehensive test suite (9 tests passing)
- Use dataclasses for schema validation (match existing patterns)

All tests passing: 9/9
Following TDD: Red-Green-Refactor`

### Phase 2: Move Existing YAML Files (Before Creating New Ones)

5. **Move ollama_config.yaml** (Todo 5)
   - Move `ollama_config.yaml` → `config/ollama.yaml`
   - Update any hardcoded paths (if any)
   - **Run test**: `make test-core` (verify no breakage)
   - **Commit**: `refactor: Move ollama_config.yaml to config/ollama.yaml`

6. **Move prompt_learning_config.yaml** (Todo 6)
   - Move `src/clinical_analytics/core/prompt_learning_config.yaml` → `config/prompt_learning.yaml`
   - **Run test**: `make test-core` (verify no breakage)
   - **Commit**: `refactor: Move prompt_learning_config.yaml to config/prompt_learning.yaml`

7. **Move datasets.yaml** (Todo 7)
   - Move `data/configs/datasets.yaml` → `config/datasets.yaml`
   - **Run test**: `make test-core` (verify no breakage)
   - **Commit**: `refactor: Move datasets.yaml to config/datasets.yaml`

### Phase 3: Create New YAML Configuration Files

8. **Create config/nl_query.yaml** (Todo 8)
   - Extract all constants from `nl_query_config.py`
   - Preserve all comments as YAML comments
   - Structure matches API contract return type
   - **No tests needed** (YAML file creation)

9. **Create config/ui.yaml** (Todo 9)
   - Extract from `ui/config.py`
   - Structure matches API contract return type
   - **No tests needed** (YAML file creation)

10. **Create config/logging.yaml** (Todo 10)
    - Extract config values from `logging_config.py`
    - Structure matches API contract return type
    - **No tests needed** (YAML file creation)

### Phase 4: Refactor Python Modules (One at a Time with Tests)

**Sub-Phase 4a: Refactor nl_query_config.py**

11. **Write failing tests for nl_query_config refactor** (Todo 11 - TDD Red Phase)
    - Update `tests/core/test_nl_query_config.py`
    - Test cases:
      - `test_nl_query_config_constants_match_yaml_values`
      - `test_nl_query_config_env_var_overrides_yaml`
      - `test_nl_query_config_backward_compatibility_all_constants_exist`
      - `test_nl_query_config_auto_execute_threshold_matches_semantic_threshold`
    - **Run test**: `make test-core PYTEST_ARGS="tests/core/test_nl_query_config.py -xvs"`
    - **Verify**: Tests fail (Red phase confirmed)

12. **Refactor nl_query_config.py** (Todo 12 - TDD Green Phase)
    - Replace hardcoded constants with `config_loader.load_nl_query_config()`
    - Maintain same public API (same constant names exported)
    - Preserve environment variable precedence
    - Update `_load_ollama_config()` to use new config loader
    - **Run test**: `make test-core PYTEST_ARGS="tests/core/test_nl_query_config.py -xvs"`
    - **Verify**: All tests pass (Green phase confirmed)

13. **Fix quality issues** (Todo 13 - TDD Refactor Phase)
    - Run: `make format`
    - Run: `make lint-fix`
    - Run: `make type-check`
    - **Run test**: `make test-core`
    - **Verify**: All tests pass, no regressions
    - **Commit**: `feat: Phase 4a - Refactor nl_query_config.py to load from YAML

- Replace hardcoded constants with config_loader.load_nl_query_config()
- Maintain backward compatibility (all constants exported)
- Add comprehensive test suite (4 tests passing)

All tests passing: X/Y
Following TDD: Red-Green-Refactor`

**Sub-Phase 4b: Refactor ui/config.py**

14. **Write failing tests for ui/config refactor** (Todo 14 - TDD Red Phase)
    - Create/update `tests/ui/test_config.py`
    - Test cases:
      - `test_ui_config_constants_match_yaml_values`
      - `test_ui_config_env_var_overrides_yaml`
      - `test_ui_config_backward_compatibility_all_constants_exist`
    - **Run test**: `make test-ui PYTEST_ARGS="tests/ui/test_config.py -xvs"`
    - **Verify**: Tests fail (Red phase confirmed)

15. **Refactor ui/config.py** (Todo 15 - TDD Green Phase)
    - Replace constants with `config_loader.load_ui_config()`
    - Maintain same public API
    - Preserve environment variable precedence
    - **Run test**: `make test-ui PYTEST_ARGS="tests/ui/test_config.py -xvs"`
    - **Verify**: All tests pass (Green phase confirmed)

16. **Fix quality issues** (Todo 16 - TDD Refactor Phase)
    - Run: `make format`
    - Run: `make lint-fix`
    - Run: `make type-check`
    - **Run test**: `make test-ui`
    - **Verify**: All tests pass, no regressions
    - **Commit**: `feat: Phase 4b - Refactor ui/config.py to load from YAML

- Replace constants with config_loader.load_ui_config()
- Maintain backward compatibility
- Add comprehensive test suite (3 tests passing)

All tests passing: X/Y
Following TDD: Red-Green-Refactor`

**Sub-Phase 4c: Refactor logging_config.py**

17. **Write failing tests for logging_config refactor** (Todo 17 - TDD Red Phase)
    - Create/update `tests/ui/test_logging_config.py`
    - Test cases:
      - `test_logging_config_loads_from_yaml`
      - `test_configure_logging_is_idempotent`
      - `test_configure_logging_applies_yaml_settings`
    - **Run test**: `make test-ui PYTEST_ARGS="tests/ui/test_logging_config.py -xvs"`
    - **Verify**: Tests fail (Red phase confirmed)

18. **Refactor logging_config.py** (Todo 18 - TDD Green Phase)
    - Extract config values to `config/logging.yaml`
    - Keep `configure_logging()` function
    - Load config from YAML, apply via function
    - Maintain idempotency
    - **Run test**: `make test-ui PYTEST_ARGS="tests/ui/test_logging_config.py -xvs"`
    - **Verify**: All tests pass (Green phase confirmed)

19. **Fix quality issues** (Todo 19 - TDD Refactor Phase)
    - Run: `make format`
    - Run: `make lint-fix`
    - Run: `make type-check`
    - **Run test**: `make test-ui`
    - **Verify**: All tests pass, no regressions
    - **Commit**: `feat: Phase 4c - Refactor logging_config.py to load from YAML

- Extract config values to config/logging.yaml
- Keep configure_logging() function for orchestration
- Maintain idempotency
- Add comprehensive test suite (3 tests passing)

All tests passing: X/Y
Following TDD: Red-Green-Refactor`

**Sub-Phase 4d: Update Path References**

20. **Update prompt_optimizer.py** (Todo 20)
    - Update path to `config/prompt_learning.yaml`
    - **Run test**: `make test-core`
    - **Verify**: No breakage
    - **Commit**: `refactor: Update prompt_optimizer.py to use config/prompt_learning.yaml`

21. **Update mapper.py** (Todo 21)
    - Update path to `config/datasets.yaml`
    - **Run test**: `make test-core`
    - **Verify**: No breakage
    - **Commit**: `refactor: Update mapper.py to use config/datasets.yaml`

### Phase 5: Update Import Statements (By Module with Tests)

**Sub-Phase 5a: Update nl_query_config imports**

22. **Update nl_query_config import statements** (Todo 22)
    - Update 8 files importing from `nl_query_config`:
      - `src/clinical_analytics/core/nl_query_engine.py`
      - `src/clinical_analytics/ui/pages/03_💬_Ask_Questions.py`
      - `src/clinical_analytics/core/filter_extraction.py`
      - `src/clinical_analytics/core/golden_question_generator.py`
      - `src/clinical_analytics/core/error_translation.py`
      - `src/clinical_analytics/core/result_interpretation.py`
      - `src/clinical_analytics/core/llm_feature.py`
      - `src/clinical_analytics/core/ollama_manager.py`
      - `src/clinical_analytics/core/clarifying_questions.py`
      - `src/clinical_analytics/ui/components/question_engine.py`
    - **Run test**: `make test-core` and `make test-ui`
    - **Verify**: All imports work, no breakage
    - **Commit**: `refactor: Update nl_query_config import statements (8 files)

- All imports continue to work without changes
- Backward compatibility maintained`

**Sub-Phase 5b: Update ui.config imports**

23. **Update ui.config import statements** (Todo 23)
    - Update 5 files importing from `ui.config`:
      - `src/clinical_analytics/ui/app.py`
      - `src/clinical_analytics/ui/pages/01_📤_Add_Your_Data.py`
      - `src/clinical_analytics/ui/components/dataset_loader.py`
      - `src/clinical_analytics/ui/storage/user_datasets.py`
      - `src/clinical_analytics/ui/helpers.py`
    - **Run test**: `make test-ui`
    - **Verify**: All imports work, no breakage
    - **Commit**: `refactor: Update ui.config import statements (5 files)

- All imports continue to work without changes
- Backward compatibility maintained`

**Sub-Phase 5c: Update logging_config imports**

24. **Update logging_config import statements** (Todo 24)
    - Update 2 files importing from `logging_config`:
      - `src/clinical_analytics/ui/app.py`
    - **Run test**: `make test-ui`
    - **Verify**: All imports work, no breakage
    - **Commit**: `refactor: Update logging_config import statements

- All imports continue to work without changes
- Backward compatibility maintained`

### Phase 6: Backward Compatibility Verification and Final Testing

25. **Add backward compatibility integration test** (Todo 25)
    - Create `tests/integration/test_config_backward_compatibility.py`
    - Test cases:
      - `test_all_nl_query_constants_importable_and_match_yaml`
      - `test_all_ui_constants_importable_and_match_yaml`
      - `test_all_import_sites_work_without_changes`
      - `test_environment_variables_override_yaml_correctly`
      - `test_missing_yaml_files_fallback_to_defaults`
    - **Run test**: `make test-integration`
    - **Verify**: All backward compatibility tests pass
    - **Commit**: `test: Add backward compatibility integration tests

- Verify all constants importable and match YAML values
- Verify all import sites work without changes
- Verify environment variable precedence
- Add comprehensive test suite (5 tests passing)`

26. **Run full test suite** (Todo 26)
    - Run: `make test`
    - **Verify**: All tests pass
    - Run: `make check` (full quality gate)
    - **Verify**: All quality gates pass
    - **Final commit**: `feat: Complete config migration to YAML

- All configuration migrated to YAML in config/ folder
- Backward compatibility maintained (100%)
- All tests passing: X/Y
- All quality gates passing`

## Key Files to Modify

### New Files

- `config/README.md` - Configuration documentation
- `config/nl_query.yaml` - NL query engine configuration
- `config/ui.yaml` - UI configuration
- `config/logging.yaml` - Logging configuration
- `config/ollama.yaml` - Ollama LLM configuration (moved)
- `config/prompt_learning.yaml` - Prompt learning configuration (moved)
- `config/datasets.yaml` - Dataset configurations (moved)
- `src/clinical_analytics/core/config_loader.py` - Centralized config loader

### Modified Files

- `src/clinical_analytics/core/nl_query_config.py` - Refactor to load from YAML
- `src/clinical_analytics/ui/config.py` - Refactor to load from YAML
- `src/clinical_analytics/ui/logging_config.py` - Extract values to YAML, keep function
- `src/clinical_analytics/core/prompt_optimizer.py` - Update config path
- `src/clinical_analytics/core/mapper.py` - Update config path
- `tests/core/test_nl_query_config.py` - Update tests for YAML loading

### Files with Import Updates (15+ files)

- `src/clinical_analytics/core/nl_query_engine.py`
- `src/clinical_analytics/ui/app.py`
- `src/clinical_analytics/ui/pages/03_💬_Ask_Questions.py`
- `src/clinical_analytics/core/filter_extraction.py`
- `src/clinical_analytics/core/golden_question_generator.py`
- `src/clinical_analytics/core/error_translation.py`
- `src/clinical_analytics/core/result_interpretation.py`
- `src/clinical_analytics/core/llm_feature.py`
- `src/clinical_analytics/ui/pages/01_📤_Add_Your_Data.py`
- `src/clinical_analytics/ui/components/dataset_loader.py`
- `src/clinical_analytics/ui/storage/user_datasets.py`
- `src/clinical_analytics/ui/components/question_engine.py`
- `src/clinical_analytics/core/ollama_manager.py`
- `src/clinical_analytics/core/clarifying_questions.py`
- `src/clinical_analytics/ui/helpers.py`

## Design Principles

1. **Config is Data**: Values that change by environment → YAML
2. **Code is Behavior**: Orchestration, validation, side effects → Python
3. **Environment Variable Precedence**: Env vars override YAML (maintain existing behavior)
4. **Backward Compatibility**: Public API unchanged (same constant names)
5. **Schema Validation**: Validate YAML structure and types
6. **Idempotent Loading**: Safe to call multiple times
7. **Clear Error Messages**: Fail fast with helpful errors

## Guardrails

- YAML must be schema-validated using dataclasses + manual validation (match existing codebase patterns)
- No conditionals in YAML
- Python functions must be idempotent and boring
- No importing app code inside config loading
- Config initializes first or not at all
- Environment variables take precedence over YAML (env var → YAML → defaults)

## Testing Strategy

1. **Unit Tests**: Config loader, schema validation, env var overrides
   - Test loading from YAML file
   - Test environment variable override
   - Test missing file → defaults
   - Test invalid YAML → error handling
   - Test type coercion (string "30.0" → float 30.0, string "true" → bool True)
2. **Integration Tests**: Full config loading, all imports work
   - Test all constants importable and match YAML values
   - Test all import sites work without changes
   - Test environment variable precedence
3. **Regression Tests**: Existing tests still pass
4. **Edge Cases**: Missing files, invalid YAML, missing keys, type coercion failures

## Migration Notes

- Maintain 100% backward compatibility
- All existing imports continue to work
- Environment variables continue to work
- Git branch strategy: Use feature branch for migration
- Rollback plan: If migration fails, revert commits and restore old config files
- Verification: Run backward compatibility integration tests before final commit