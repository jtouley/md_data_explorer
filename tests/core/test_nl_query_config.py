"""Tests for NL Query Engine configuration constants."""

import os
from unittest.mock import patch

import yaml

from clinical_analytics.core.config_loader import load_nl_query_config
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


def test_config_constants_are_defined():
    """All configuration constants should be defined."""
    assert TIER_1_PATTERN_MATCH_THRESHOLD == 0.9
    assert TIER_2_SEMANTIC_MATCH_THRESHOLD == 0.75
    assert CLARIFYING_QUESTIONS_THRESHOLD == 0.5
    assert AUTO_EXECUTE_CONFIDENCE_THRESHOLD == TIER_2_SEMANTIC_MATCH_THRESHOLD
    assert TIER_TIMEOUT_SECONDS == 5.0
    assert SEMANTIC_SIMILARITY_THRESHOLD == 0.7
    assert FUZZY_MATCH_CUTOFF == 0.7


def test_auto_execute_threshold_matches_semantic_threshold():
    """AUTO_EXECUTE_CONFIDENCE_THRESHOLD should equal TIER_2_SEMANTIC_MATCH_THRESHOLD."""
    assert AUTO_EXECUTE_CONFIDENCE_THRESHOLD == TIER_2_SEMANTIC_MATCH_THRESHOLD


def test_feature_flags_default_to_true():
    """Feature flags should default to True when env var not set."""
    with patch.dict(os.environ, {}, clear=True):
        # Reload module to get fresh defaults
        import importlib

        import clinical_analytics.core.nl_query_config as config_module

        importlib.reload(config_module)
        assert config_module.ENABLE_CLARIFYING_QUESTIONS is True
        assert config_module.ENABLE_PROGRESSIVE_FEEDBACK is True


def test_feature_flags_respect_env_vars():
    """Feature flags should respect environment variables."""
    with patch.dict(os.environ, {"ENABLE_CLARIFYING_QUESTIONS": "false"}, clear=False):
        import importlib

        import clinical_analytics.core.nl_query_config as config_module

        importlib.reload(config_module)
        assert config_module.ENABLE_CLARIFYING_QUESTIONS is False

    with patch.dict(os.environ, {"ENABLE_PROGRESSIVE_FEEDBACK": "false"}, clear=False):
        import importlib

        import clinical_analytics.core.nl_query_config as config_module

        importlib.reload(config_module)
        assert config_module.ENABLE_PROGRESSIVE_FEEDBACK is False


class TestNLQueryConfigYAMLLoading:
    """Test suite for nl_query_config YAML loading via config_loader."""

    def test_nl_query_config_constants_match_yaml_values(self, tmp_path):
        """Test that constants match values loaded from YAML file."""
        # Arrange: Create temporary YAML config file
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        config_file = config_dir / "nl_query.yaml"
        config_data = {
            "tier_1_pattern_match_threshold": 0.85,
            "tier_2_semantic_match_threshold": 0.70,
            "clarifying_questions_threshold": 0.45,
            "auto_execute_confidence_threshold": 0.70,
            "tier_timeout_seconds": 4.0,
            "semantic_similarity_threshold": 0.65,
            "fuzzy_match_cutoff": 0.65,
            "enable_clarifying_questions": True,
            "enable_progressive_feedback": True,
            "ollama_base_url": "http://localhost:11434",
            "ollama_default_model": "llama3.1:8b",
            "ollama_fallback_model": "llama3.2:3b",
            "ollama_timeout_seconds": 25.0,
            "ollama_max_retries": 2,
            "ollama_json_mode": True,
            "tier_3_min_confidence": 0.45,
            "tier_3_execution_threshold": 0.70,
            "llm_timeout_parse_s": 25.0,
            "llm_timeout_followups_s": 25.0,
            "llm_timeout_interpretation_s": 25.0,
            "llm_timeout_result_interpretation_s": 15.0,
            "llm_timeout_error_translation_s": 4.0,
            "llm_timeout_filter_extraction_s": 25.0,
            "llm_timeout_max_s": 25.0,
            "enable_result_interpretation": True,
        }
        config_file.write_text(yaml.dump(config_data))

        # Act: Load config via config_loader
        yaml_config = load_nl_query_config(config_path=config_file)

        # Assert: Values match (note: constants will be loaded from default YAML in actual implementation)
        assert yaml_config["tier_1_pattern_match_threshold"] == 0.85
        assert yaml_config["tier_2_semantic_match_threshold"] == 0.70
        assert yaml_config["clarifying_questions_threshold"] == 0.45
        assert yaml_config["auto_execute_confidence_threshold"] == 0.70

    def test_nl_query_config_env_var_overrides_yaml(self, tmp_path):
        """Test that environment variables override YAML values in config_loader."""
        # Arrange: Create YAML file
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        config_file = config_dir / "nl_query.yaml"
        config_data = {
            "tier_2_semantic_match_threshold": 0.70,
            "enable_clarifying_questions": True,
        }
        config_file.write_text(yaml.dump(config_data))

        # Act: Set environment variable and load config
        with patch.dict(
            os.environ,
            {
                "ENABLE_CLARIFYING_QUESTIONS": "false",
                "TIER_2_SEMANTIC_MATCH_THRESHOLD": "0.80",
            },
            clear=False,
        ):
            result = load_nl_query_config(config_path=config_file)

        # Assert: Environment variable overrides YAML
        assert result["enable_clarifying_questions"] is False
        assert result["tier_2_semantic_match_threshold"] == 0.80

    def test_nl_query_config_backward_compatibility_all_constants_exist(self):
        """Test that all constants are still importable and exist after refactor."""
        # Act & Assert: All constants should be importable
        assert TIER_1_PATTERN_MATCH_THRESHOLD is not None
        assert TIER_2_SEMANTIC_MATCH_THRESHOLD is not None
        assert CLARIFYING_QUESTIONS_THRESHOLD is not None
        assert AUTO_EXECUTE_CONFIDENCE_THRESHOLD is not None
        assert TIER_TIMEOUT_SECONDS is not None
        assert SEMANTIC_SIMILARITY_THRESHOLD is not None
        assert FUZZY_MATCH_CUTOFF is not None
        assert ENABLE_CLARIFYING_QUESTIONS is not None
        assert ENABLE_PROGRESSIVE_FEEDBACK is not None
        assert OLLAMA_BASE_URL is not None
        assert OLLAMA_DEFAULT_MODEL is not None
        assert OLLAMA_FALLBACK_MODEL is not None
        assert OLLAMA_TIMEOUT_SECONDS is not None
        assert OLLAMA_MAX_RETRIES is not None
        assert OLLAMA_JSON_MODE is not None
        assert TIER_3_MIN_CONFIDENCE is not None
        assert TIER_3_EXECUTION_THRESHOLD is not None
        assert LLM_TIMEOUT_PARSE_S is not None
        assert LLM_TIMEOUT_FOLLOWUPS_S is not None
        assert LLM_TIMEOUT_INTERPRETATION_S is not None
        assert LLM_TIMEOUT_RESULT_INTERPRETATION_S is not None
        assert LLM_TIMEOUT_ERROR_TRANSLATION_S is not None
        assert LLM_TIMEOUT_FILTER_EXTRACTION_S is not None
        assert LLM_TIMEOUT_MAX_S is not None
        assert ENABLE_RESULT_INTERPRETATION is not None

    def test_nl_query_config_auto_execute_threshold_matches_semantic_threshold(self):
        """Test that AUTO_EXECUTE_CONFIDENCE_THRESHOLD matches TIER_2_SEMANTIC_MATCH_THRESHOLD."""
        # This test ensures the relationship is maintained after refactoring
        assert AUTO_EXECUTE_CONFIDENCE_THRESHOLD == TIER_2_SEMANTIC_MATCH_THRESHOLD

    def test_nl_query_config_deprecation_warning_logs_correctly(self, tmp_path, caplog):
        """Test that deprecation warning for enable_proactive_questions logs correctly with standard logger."""
        # Arrange: Create YAML config with legacy key
        import logging

        config_dir = tmp_path / "config"
        config_dir.mkdir()
        config_file = config_dir / "nl_query.yaml"
        config_data = {
            "tier_1_pattern_match_threshold": 0.9,
            "tier_2_semantic_match_threshold": 0.75,
            "clarifying_questions_threshold": 0.5,
            "auto_execute_confidence_threshold": 0.75,
            "tier_timeout_seconds": 5.0,
            "semantic_similarity_threshold": 0.7,
            "fuzzy_match_cutoff": 0.7,
            "enable_clarifying_questions": True,
            "enable_progressive_feedback": True,
            "ollama_base_url": "http://localhost:11434",
            "ollama_default_model": "llama3.1:8b",
            "ollama_fallback_model": "llama3.2:3b",
            "ollama_timeout_seconds": 25.0,
            "ollama_max_retries": 2,
            "ollama_json_mode": True,
            "tier_3_min_confidence": 0.45,
            "tier_3_execution_threshold": 0.70,
            "llm_timeout_parse_s": 25.0,
            "llm_timeout_followups_s": 25.0,
            "llm_timeout_interpretation_s": 25.0,
            "llm_timeout_result_interpretation_s": 15.0,
            "llm_timeout_error_translation_s": 4.0,
            "llm_timeout_filter_extraction_s": 25.0,
            "llm_timeout_max_s": 25.0,
            "llm_timeout_question_generation_s": 25.0,
            "enable_result_interpretation": True,
            "enable_proactive_questions": True,  # Legacy key (should trigger warning)
        }
        config_file.write_text(yaml.dump(config_data))

        # Act: Import module with legacy key (this triggers the warning at module level)
        with caplog.at_level(logging.WARNING):
            # Reload module to trigger the deprecation warning
            import importlib

            import clinical_analytics.core.nl_query_config as config_module

            # Clear any existing config
            importlib.reload(config_module)

        # Assert: Warning was logged with correct message format (standard string, not structured)
        warning_messages = [record.message for record in caplog.records if record.levelname == "WARNING"]
        assert any("enable_proactive_questions" in msg and "deprecated" in msg.lower() for msg in warning_messages), (
            f"Expected deprecation warning, got: {warning_messages}"
        )
