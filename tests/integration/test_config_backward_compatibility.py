"""Integration tests for config migration backward compatibility.

Verifies that all imports continue to work and constants match YAML values.
"""

import os
from pathlib import Path
from unittest.mock import patch

import yaml
from clinical_analytics.core.config_loader import (
    load_logging_config,
    load_nl_query_config,
    load_ui_config,
)
from clinical_analytics.core.nl_query_config import (
    AUTO_EXECUTE_CONFIDENCE_THRESHOLD,
    CLARIFYING_QUESTIONS_THRESHOLD,
    ENABLE_CLARIFYING_QUESTIONS,
    ENABLE_PROGRESSIVE_FEEDBACK,
    ENABLE_RESULT_INTERPRETATION,
    FUZZY_MATCH_CUTOFF,
    LLM_TIMEOUT_ERROR_TRANSLATION_S,
    LLM_TIMEOUT_FILTER_EXTRACTION_S,
    LLM_TIMEOUT_FOLLOWUPS_S,
    LLM_TIMEOUT_INTERPRETATION_S,
    LLM_TIMEOUT_MAX_S,
    LLM_TIMEOUT_PARSE_S,
    LLM_TIMEOUT_RESULT_INTERPRETATION_S,
    OLLAMA_BASE_URL,
    OLLAMA_DEFAULT_MODEL,
    OLLAMA_FALLBACK_MODEL,
    OLLAMA_JSON_MODE,
    OLLAMA_MAX_RETRIES,
    OLLAMA_TIMEOUT_SECONDS,
    SEMANTIC_SIMILARITY_THRESHOLD,
    TIER_1_PATTERN_MATCH_THRESHOLD,
    TIER_2_SEMANTIC_MATCH_THRESHOLD,
    TIER_3_EXECUTION_THRESHOLD,
    TIER_3_MIN_CONFIDENCE,
    TIER_TIMEOUT_SECONDS,
)
from clinical_analytics.ui.config import (
    ASK_QUESTIONS_PAGE,
    LOG_LEVEL,
    MAX_UPLOAD_SIZE_MB,
    MULTI_TABLE_ENABLED,
    V1_MVP_MODE,
)


class TestConfigBackwardCompatibility:
    """Test suite for backward compatibility after config migration."""

    def test_all_nl_query_constants_importable_and_match_yaml(self):
        """Test that all NL query constants are importable and match YAML values."""
        # Arrange: Load config from YAML
        project_root = Path(__file__).parent.parent.parent
        config_path = project_root / "config" / "nl_query.yaml"
        yaml_config = load_nl_query_config(config_path=config_path if config_path.exists() else None)

        # Act & Assert: All constants should be importable and match YAML
        assert TIER_1_PATTERN_MATCH_THRESHOLD == yaml_config["tier_1_pattern_match_threshold"]
        assert TIER_2_SEMANTIC_MATCH_THRESHOLD == yaml_config["tier_2_semantic_match_threshold"]
        assert CLARIFYING_QUESTIONS_THRESHOLD == yaml_config["clarifying_questions_threshold"]
        assert AUTO_EXECUTE_CONFIDENCE_THRESHOLD == yaml_config["auto_execute_confidence_threshold"]
        assert TIER_TIMEOUT_SECONDS == yaml_config["tier_timeout_seconds"]
        assert SEMANTIC_SIMILARITY_THRESHOLD == yaml_config["semantic_similarity_threshold"]
        assert FUZZY_MATCH_CUTOFF == yaml_config["fuzzy_match_cutoff"]
        assert ENABLE_CLARIFYING_QUESTIONS == yaml_config["enable_clarifying_questions"]
        assert ENABLE_PROGRESSIVE_FEEDBACK == yaml_config["enable_progressive_feedback"]
        assert OLLAMA_BASE_URL == yaml_config["ollama_base_url"]
        assert OLLAMA_DEFAULT_MODEL == yaml_config["ollama_default_model"]
        assert OLLAMA_FALLBACK_MODEL == yaml_config["ollama_fallback_model"]
        assert OLLAMA_TIMEOUT_SECONDS == yaml_config["ollama_timeout_seconds"]
        assert OLLAMA_MAX_RETRIES == yaml_config["ollama_max_retries"]
        assert OLLAMA_JSON_MODE == yaml_config["ollama_json_mode"]
        assert TIER_3_MIN_CONFIDENCE == yaml_config["tier_3_min_confidence"]
        assert TIER_3_EXECUTION_THRESHOLD == yaml_config["tier_3_execution_threshold"]
        assert LLM_TIMEOUT_PARSE_S == yaml_config["llm_timeout_parse_s"]
        assert LLM_TIMEOUT_FOLLOWUPS_S == yaml_config["llm_timeout_followups_s"]
        assert LLM_TIMEOUT_INTERPRETATION_S == yaml_config["llm_timeout_interpretation_s"]
        assert LLM_TIMEOUT_RESULT_INTERPRETATION_S == yaml_config["llm_timeout_result_interpretation_s"]
        assert LLM_TIMEOUT_ERROR_TRANSLATION_S == yaml_config["llm_timeout_error_translation_s"]
        assert LLM_TIMEOUT_FILTER_EXTRACTION_S == yaml_config["llm_timeout_filter_extraction_s"]
        assert LLM_TIMEOUT_MAX_S == yaml_config["llm_timeout_max_s"]
        assert ENABLE_RESULT_INTERPRETATION == yaml_config["enable_result_interpretation"]

    def test_all_ui_constants_importable_and_match_yaml(self):
        """Test that all UI constants are importable and match YAML values."""
        # Arrange: Load config from YAML
        project_root = Path(__file__).parent.parent.parent
        config_path = project_root / "config" / "ui.yaml"
        yaml_config = load_ui_config(config_path=config_path if config_path.exists() else None)

        # Act & Assert: All constants should be importable and match YAML
        assert MULTI_TABLE_ENABLED == yaml_config["multi_table_enabled"]
        assert V1_MVP_MODE == yaml_config["v1_mvp_mode"]
        assert LOG_LEVEL == yaml_config["log_level"]
        assert MAX_UPLOAD_SIZE_MB == yaml_config["max_upload_size_mb"]
        assert ASK_QUESTIONS_PAGE == yaml_config["ask_questions_page"]

    def test_all_import_sites_work_without_changes(self):
        """Test that all import sites continue to work without code changes."""
        # This test verifies that existing code can import constants without modification
        # Act: Import modules that use config constants (verify no ImportError)
        # We test that modules can be imported, which means their config imports work
        import clinical_analytics.core.error_translation
        import clinical_analytics.core.filter_extraction
        import clinical_analytics.core.nl_query_engine
        import clinical_analytics.core.result_interpretation
        import clinical_analytics.ui.app
        import clinical_analytics.ui.components.dataset_loader

        # Assert: Imports succeed (no ImportError)
        # If we get here, all config imports in these modules worked
        assert clinical_analytics.core.nl_query_engine is not None
        assert clinical_analytics.core.filter_extraction is not None
        assert clinical_analytics.core.error_translation is not None
        assert clinical_analytics.core.result_interpretation is not None
        assert clinical_analytics.ui.app is not None
        assert clinical_analytics.ui.components.dataset_loader is not None

    def test_environment_variables_override_yaml_correctly(self, tmp_path):
        """Test that environment variables override YAML values correctly."""
        # Arrange: Create YAML files
        config_dir = tmp_path / "config"
        config_dir.mkdir()

        nl_query_file = config_dir / "nl_query.yaml"
        nl_query_file.write_text(yaml.dump({"enable_clarifying_questions": True}))

        ui_file = config_dir / "ui.yaml"
        ui_file.write_text(yaml.dump({"multi_table_enabled": False}))

        # Act: Set environment variables and load configs
        with patch.dict(
            os.environ,
            {
                "ENABLE_CLARIFYING_QUESTIONS": "false",
                "MULTI_TABLE_ENABLED": "true",
            },
            clear=False,
        ):
            nl_config = load_nl_query_config(config_path=nl_query_file)
            ui_config = load_ui_config(config_path=ui_file)

        # Assert: Environment variables override YAML
        assert nl_config["enable_clarifying_questions"] is False
        assert ui_config["multi_table_enabled"] is True

    def test_missing_yaml_files_fallback_to_defaults(self):
        """Test that missing YAML files fallback to defaults gracefully."""
        # Arrange: Non-existent config files
        missing_nl_query = Path("/nonexistent/nl_query.yaml")
        missing_ui = Path("/nonexistent/ui.yaml")
        missing_logging = Path("/nonexistent/logging.yaml")

        # Act: Load configs (should use defaults)
        nl_config = load_nl_query_config(config_path=missing_nl_query)
        ui_config = load_ui_config(config_path=missing_ui)
        logging_config = load_logging_config(config_path=missing_logging)

        # Assert: Defaults are used
        assert nl_config["tier_1_pattern_match_threshold"] == 0.9
        assert ui_config["multi_table_enabled"] is False
        assert logging_config["root_level"] == "INFO"
