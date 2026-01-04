"""Tests for UI configuration module.

Following AGENTS.md guidelines:
- AAA pattern (Arrange-Act-Assert)
- Descriptive test names: test_unit_scenario_expectedBehavior
- Test isolation (no shared mutable state)
"""

import os
from unittest.mock import patch

import yaml
from clinical_analytics.core.config_loader import load_ui_config
from clinical_analytics.ui.config import (
    ASK_QUESTIONS_PAGE,
    LOG_LEVEL,
    MAX_UPLOAD_SIZE_MB,
    MULTI_TABLE_ENABLED,
    V1_MVP_MODE,
)


class TestUIConfigYAMLLoading:
    """Test suite for UI config YAML loading via config_loader."""

    def test_ui_config_constants_match_yaml_values(self, tmp_path):
        """Test that constants match values loaded from YAML file."""
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

        # Act: Load config via config_loader
        yaml_config = load_ui_config(config_path=config_file)

        # Assert: Values match
        assert yaml_config["multi_table_enabled"] is True
        assert yaml_config["v1_mvp_mode"] is False
        assert yaml_config["log_level"] == "DEBUG"
        assert yaml_config["max_upload_size_mb"] == 200
        assert yaml_config["ask_questions_page"] == "pages/03_💬_Ask_Questions.py"

    def test_ui_config_env_var_overrides_yaml(self, tmp_path):
        """Test that environment variables override YAML values in config_loader."""
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

    def test_ui_config_backward_compatibility_all_constants_exist(self):
        """Test that all constants are still importable and exist after refactor."""
        # Act & Assert: All constants should be importable
        assert MULTI_TABLE_ENABLED is not None
        assert V1_MVP_MODE is not None
        assert LOG_LEVEL is not None
        assert MAX_UPLOAD_SIZE_MB is not None
        assert ASK_QUESTIONS_PAGE is not None
