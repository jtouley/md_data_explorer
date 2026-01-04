"""Tests for logging configuration module.

Following AGENTS.md guidelines:
- AAA pattern (Arrange-Act-Assert)
- Descriptive test names: test_unit_scenario_expectedBehavior
- Test isolation (no shared mutable state)
"""

import logging

import yaml
from clinical_analytics.core.config_loader import load_logging_config
from clinical_analytics.ui.logging_config import configure_logging


class TestLoggingConfigYAMLLoading:
    """Test suite for logging config YAML loading via config_loader."""

    def test_logging_config_loads_from_yaml(self, tmp_path):
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

        # Act: Load config via config_loader
        yaml_config = load_logging_config(config_path=config_file)

        # Assert: Values match YAML file
        assert yaml_config["root_level"] == "DEBUG"
        assert yaml_config["format"] == "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        assert yaml_config["module_levels"]["clinical_analytics.core.semantic"] == "DEBUG"
        assert yaml_config["module_levels"]["clinical_analytics.core.registry"] == "DEBUG"
        assert yaml_config["reduce_noise"]["streamlit"] == "WARNING"
        assert yaml_config["reduce_noise"]["urllib3"] == "WARNING"

    def test_configure_logging_is_idempotent(self):
        """Test that configure_logging() is idempotent (safe to call multiple times)."""
        # Arrange: Clear any existing handlers
        root_logger = logging.getLogger()
        root_logger.handlers.clear()

        # Act: Call configure_logging multiple times
        configure_logging()
        handler_count_1 = len(root_logger.handlers)

        configure_logging()
        handler_count_2 = len(root_logger.handlers)

        # Assert: Should only configure once (idempotent)
        assert handler_count_1 == handler_count_2
        assert handler_count_1 > 0  # Should have at least one handler

    def test_configure_logging_applies_yaml_settings(self):
        """Test that configure_logging() applies settings from YAML config."""
        # Arrange: Clear any existing handlers
        root_logger = logging.getLogger()
        root_logger.handlers.clear()

        # Act: Configure logging (should use YAML values via config_loader)
        configure_logging()

        # Assert: Root logger level should be set (from YAML default: INFO)
        assert root_logger.level == logging.INFO or root_logger.level == logging.NOTSET

        # Assert: Module-specific loggers should be set
        semantic_logger = logging.getLogger("clinical_analytics.core.semantic")
        assert semantic_logger.level == logging.INFO

        # Assert: Noise reduction should be applied
        streamlit_logger = logging.getLogger("streamlit")
        assert streamlit_logger.level == logging.WARNING
