"""Tests for config_loader module.

Following AGENTS.md guidelines:
- AAA pattern (Arrange-Act-Assert)
- Descriptive test names: test_unit_scenario_expectedBehavior
- Test isolation (no shared mutable state)
"""

import os
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

# Import will fail initially (module doesn't exist yet - this is expected in Red phase)
# pylint: disable=import-error
from clinical_analytics.core.config_loader import (
    get_project_root,
    load_logging_config,
    load_nl_query_config,
    load_ui_config,
)


class TestConfigLoaderNLQuery:
    """Test suite for NL query configuration loading."""

    def test_load_nl_query_config_loads_from_yaml_file(self, tmp_path):
        """Test that load_nl_query_config loads values from YAML file."""
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
            "enable_parallel_tier_matching": False,
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

        # Act: Load config
        result = load_nl_query_config(config_path=config_file)

        # Assert: Values match YAML file
        assert result["tier_1_pattern_match_threshold"] == 0.85
        assert result["tier_2_semantic_match_threshold"] == 0.70
        assert result["clarifying_questions_threshold"] == 0.45
        assert result["auto_execute_confidence_threshold"] == 0.70
        assert result["tier_timeout_seconds"] == 4.0
        assert result["enable_parallel_tier_matching"] is False
        assert result["semantic_similarity_threshold"] == 0.65
        assert result["fuzzy_match_cutoff"] == 0.65
        assert result["enable_clarifying_questions"] is True
        assert result["enable_progressive_feedback"] is True
        assert result["ollama_base_url"] == "http://localhost:11434"
        assert result["ollama_default_model"] == "llama3.1:8b"
        assert result["ollama_fallback_model"] == "llama3.2:3b"
        assert result["ollama_timeout_seconds"] == 25.0
        assert result["ollama_max_retries"] == 2
        assert result["ollama_json_mode"] is True
        assert result["tier_3_min_confidence"] == 0.45
        assert result["tier_3_execution_threshold"] == 0.70
        assert result["llm_timeout_parse_s"] == 25.0
        assert result["llm_timeout_followups_s"] == 25.0
        assert result["llm_timeout_interpretation_s"] == 25.0
        assert result["llm_timeout_result_interpretation_s"] == 15.0
        assert result["llm_timeout_error_translation_s"] == 4.0
        assert result["llm_timeout_filter_extraction_s"] == 25.0
        assert result["llm_timeout_max_s"] == 25.0
        assert result["enable_result_interpretation"] is True

    def test_load_nl_query_config_env_var_overrides_yaml(self, tmp_path):
        """Test that environment variables override YAML values."""
        # Arrange: Create YAML file with one value
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

    def test_load_nl_query_config_missing_file_uses_defaults(self):
        """Test that missing YAML file uses default values."""
        # Arrange: Non-existent config file
        config_file = Path("/nonexistent/config/nl_query.yaml")

        # Act: Load config (should use defaults)
        result = load_nl_query_config(config_path=config_file)

        # Assert: Default values are used
        assert result["tier_1_pattern_match_threshold"] == 0.9
        assert result["tier_2_semantic_match_threshold"] == 0.75
        assert result["clarifying_questions_threshold"] == 0.5
        assert result["auto_execute_confidence_threshold"] == 0.75
        assert result["tier_timeout_seconds"] == 5.0
        assert result["enable_parallel_tier_matching"] is False
        assert result["semantic_similarity_threshold"] == 0.7
        assert result["fuzzy_match_cutoff"] == 0.7
        assert result["enable_clarifying_questions"] is True
        assert result["enable_progressive_feedback"] is True
        assert result["ollama_base_url"] == "http://localhost:11434"
        assert result["ollama_default_model"] == "llama3.1:8b"
        assert result["ollama_fallback_model"] == "llama3.2:3b"
        assert result["ollama_timeout_seconds"] == 30.0
        assert result["ollama_max_retries"] == 3
        assert result["ollama_json_mode"] is True
        assert result["tier_3_min_confidence"] == 0.5
        assert result["tier_3_execution_threshold"] == 0.75
        assert result["llm_timeout_parse_s"] == 30.0
        assert result["llm_timeout_followups_s"] == 30.0
        assert result["llm_timeout_interpretation_s"] == 30.0
        assert result["llm_timeout_result_interpretation_s"] == 20.0
        assert result["llm_timeout_error_translation_s"] == 5.0
        assert result["llm_timeout_filter_extraction_s"] == 30.0
        assert result["llm_timeout_max_s"] == 30.0
        assert result["enable_result_interpretation"] is True

    def test_load_nl_query_config_invalid_yaml_raises_valueerror(self, tmp_path):
        """Test that invalid YAML raises ValueError."""
        # Arrange: Create invalid YAML file
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        config_file = config_dir / "nl_query.yaml"
        config_file.write_text("invalid: yaml: content: [unclosed")

        # Act & Assert: Loading invalid YAML should raise ValueError
        with pytest.raises(ValueError, match="Invalid YAML"):
            load_nl_query_config(config_path=config_file)

    def test_load_nl_query_config_type_coercion_string_to_float(self, tmp_path):
        """Test that string values are coerced to float."""
        # Arrange: Create YAML with string values that should be floats
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        config_file = config_dir / "nl_query.yaml"
        config_data = {
            "tier_2_semantic_match_threshold": "0.80",  # String, should become float
            "tier_timeout_seconds": "4.5",  # String, should become float
        }
        config_file.write_text(yaml.dump(config_data))

        # Act: Load config
        result = load_nl_query_config(config_path=config_file)

        # Assert: Values are floats
        assert isinstance(result["tier_2_semantic_match_threshold"], float)
        assert result["tier_2_semantic_match_threshold"] == 0.80
        assert isinstance(result["tier_timeout_seconds"], float)
        assert result["tier_timeout_seconds"] == 4.5

    def test_load_nl_query_config_type_coercion_string_to_bool(self, tmp_path):
        """Test that string values are coerced to bool."""
        # Arrange: Create YAML with string values that should be bools
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        config_file = config_dir / "nl_query.yaml"
        config_data = {
            "enable_clarifying_questions": "true",  # String, should become bool
            "enable_progressive_feedback": "false",  # String, should become bool
            "ollama_json_mode": "True",  # String with capital, should become bool
        }
        config_file.write_text(yaml.dump(config_data))

        # Act: Load config
        result = load_nl_query_config(config_path=config_file)

        # Assert: Values are bools
        assert isinstance(result["enable_clarifying_questions"], bool)
        assert result["enable_clarifying_questions"] is True
        assert isinstance(result["enable_progressive_feedback"], bool)
        assert result["enable_progressive_feedback"] is False
        assert isinstance(result["ollama_json_mode"], bool)
        assert result["ollama_json_mode"] is True


class TestConfigLoaderUI:
    """Test suite for UI configuration loading."""

    def test_load_ui_config_loads_from_yaml_file(self, tmp_path):
        """Test that load_ui_config loads values from YAML file."""
        # Arrange: Create temporary YAML config file
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        config_file = config_dir / "ui.yaml"
        config_data = {
            "multi_table_enabled": True,
            "v1_mvp_mode": False,
            "log_level": "DEBUG",
            "max_upload_size_mb": 200,
            "ask_questions_page": "pages/03_💬_Ask_Questions.py",
        }
        config_file.write_text(yaml.dump(config_data))

        # Act: Load config
        result = load_ui_config(config_path=config_file)

        # Assert: Values match YAML file
        assert result["multi_table_enabled"] is True
        assert result["v1_mvp_mode"] is False
        assert result["log_level"] == "DEBUG"
        assert result["max_upload_size_mb"] == 200
        assert result["ask_questions_page"] == "pages/03_💬_Ask_Questions.py"

    def test_load_ui_config_env_var_overrides_yaml(self, tmp_path):
        """Test that environment variables override YAML values."""
        # Arrange: Create YAML file
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        config_file = config_dir / "ui.yaml"
        config_data = {
            "multi_table_enabled": False,
            "v1_mvp_mode": True,
        }
        config_file.write_text(yaml.dump(config_data))

        # Act: Set environment variable and load config
        with patch.dict(
            os.environ,
            {
                "MULTI_TABLE_ENABLED": "true",
                "V1_MVP_MODE": "false",
            },
            clear=False,
        ):
            result = load_ui_config(config_path=config_file)

        # Assert: Environment variable overrides YAML
        assert result["multi_table_enabled"] is True
        assert result["v1_mvp_mode"] is False


class TestConfigLoaderLogging:
    """Test suite for logging configuration loading."""

    def test_load_logging_config_loads_from_yaml_file(self, tmp_path):
        """Test that load_logging_config loads values from YAML file."""
        # Arrange: Create temporary YAML config file
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        config_file = config_dir / "logging.yaml"
        config_data = {
            "root_level": "DEBUG",
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            "module_levels": {
                "clinical_analytics.core.semantic": "DEBUG",
                "clinical_analytics.core.registry": "DEBUG",
            },
            "reduce_noise": {
                "streamlit": "WARNING",
                "urllib3": "WARNING",
            },
        }
        config_file.write_text(yaml.dump(config_data))

        # Act: Load config
        result = load_logging_config(config_path=config_file)

        # Assert: Values match YAML file
        assert result["root_level"] == "DEBUG"
        assert result["format"] == "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        assert result["module_levels"]["clinical_analytics.core.semantic"] == "DEBUG"
        assert result["module_levels"]["clinical_analytics.core.registry"] == "DEBUG"
        assert result["reduce_noise"]["streamlit"] == "WARNING"
        assert result["reduce_noise"]["urllib3"] == "WARNING"


class TestConfigLoaderProjectRoot:
    """Test suite for project root detection."""

    def test_get_project_root_returns_correct_path(self):
        """Test that get_project_root returns correct path."""
        # Arrange: Expected project root (where config_loader.py is located)
        # config_loader.py → core/ → clinical_analytics/ → src/ → project_root
        # So project_root should be 4 levels up from config_loader.py

        # Act: Get project root
        project_root = get_project_root()

        # Assert: Path exists and contains expected files
        assert project_root.exists()
        assert project_root.is_dir()
        # Should contain config/ directory (after we create it)
        # Should contain src/ directory
        assert (project_root / "src").exists()
        # Should contain pyproject.toml
        assert (project_root / "pyproject.toml").exists()
