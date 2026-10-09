"""Tests for UI configuration module.

Following AGENTS.md guidelines:
- AAA pattern (Arrange-Act-Assert)
- Descriptive test names: test_unit_scenario_expectedBehavior
- Test isolation (no shared mutable state)
"""

from clinical_analytics.ui.config import (
    ASK_QUESTIONS_PAGE,
    LOG_LEVEL,
    MAX_UPLOAD_SIZE_MB,
    MULTI_TABLE_ENABLED,
    V1_MVP_MODE,
)


class TestUIConfigYAMLLoading:
    """UI constants stay importable. YAML parsing is covered in test_config_loader.py."""

    def test_ui_constants_stay_importable_after_yaml_refactor(self):
        """Test that all constants are still importable and exist after refactor."""
        # Act & Assert: All constants should be importable
        assert MULTI_TABLE_ENABLED is not None
        assert V1_MVP_MODE is not None
        assert LOG_LEVEL is not None
        assert MAX_UPLOAD_SIZE_MB is not None
        assert ASK_QUESTIONS_PAGE is not None
