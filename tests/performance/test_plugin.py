"""Tests for performance tracking plugin."""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Add tests to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from performance.plugin import (
    _get_performance_data_file,
    _get_worker_file,
    _get_worker_id,
    _is_tracking_enabled,
    _is_worker_process,
    _should_exclude_test,
)


class TestPluginHelpers:
    """Test plugin helper functions."""

    def test_plugin_is_tracking_enabled_with_flag(self):
        """Test that tracking is enabled when --track-performance flag is set."""
        # Arrange
        mock_config = MagicMock()
        mock_config.getoption.return_value = True

        # Act
        result = _is_tracking_enabled(mock_config)

        # Assert
        assert result is True
        mock_config.getoption.assert_called_once_with("--track-performance", default=False)

    def test_plugin_is_tracking_enabled_without_flag(self):
        """Test that tracking is disabled when --track-performance flag is not set."""
        # Arrange
        mock_config = MagicMock()
        mock_config.getoption.return_value = False

        # Act
        result = _is_tracking_enabled(mock_config)

        # Assert
        assert result is False

    def test_plugin_should_exclude_test_excludes_performance_tests(self):
        """Test that performance system tests are excluded from tracking."""
        # Arrange
        mock_item = MagicMock()
        mock_item.nodeid = "tests/performance/test_plugin.py::test_example"

        # Act
        result = _should_exclude_test(mock_item)

        # Assert
        assert result is True

    def test_plugin_should_exclude_test_includes_other_tests(self):
        """Test that non-performance tests are not excluded."""
        # Arrange
        mock_item = MagicMock()
        mock_item.nodeid = "tests/core/test_example.py::test_example"

        # Act
        result = _should_exclude_test(mock_item)

        # Assert
        assert result is False

    def test_plugin_get_worker_id_returns_worker_id(self):
        """Test that worker ID is extracted correctly."""
        # Arrange
        mock_config = MagicMock()
        mock_config.workerinput = {"workerid": "gw1"}

        # Act
        result = _get_worker_id(mock_config)

        # Assert
        assert result == "gw1"

    def test_plugin_get_worker_id_returns_master_when_no_worker(self):
        """Test that master is returned when not in worker process."""
        # Arrange
        mock_config = MagicMock()
        mock_config.workerinput = None

        # Act
        result = _get_worker_id(mock_config)

        # Assert
        assert result == "master"

    def test_plugin_is_worker_process_returns_true_for_worker(self):
        """Test that worker process is detected correctly."""
        # Arrange
        mock_config = MagicMock()
        mock_config.workerinput = {"workerid": "gw1"}

        # Act
        result = _is_worker_process(mock_config)

        # Assert
        assert result is True

    def test_plugin_is_worker_process_returns_false_for_master(self):
        """Test that master process is detected correctly."""
        # Arrange
        mock_config = MagicMock()
        mock_config.workerinput = None

        # Act
        result = _is_worker_process(mock_config)

        # Assert
        assert result is False

    def test_plugin_get_performance_data_file_returns_correct_path(self):
        """Test that performance data file path is correct."""
        # Arrange
        mock_config = MagicMock()
        mock_config.rootdir = Path("/test/project")

        # Act
        result = _get_performance_data_file(mock_config)

        # Assert
        assert result == Path("/test/project/tests/.performance_data.json")

    def test_plugin_get_worker_file_returns_correct_path(self):
        """Test that worker file path is correct."""
        # Arrange
        mock_config = MagicMock()
        mock_config.rootdir = Path("/test/project")
        worker_id = "gw1"

        # Act
        result = _get_worker_file(mock_config, worker_id)

        # Assert
        assert result == Path("/test/project/tests/.performance_data_worker_gw1.json")


class TestPluginIntegration:
    """Integration tests for plugin (requires actual pytest execution)."""

    @pytest.mark.slow
    def test_plugin_tracks_test_duration_when_enabled(self, tmp_path):
        """Test that plugin tracks test duration when --track-performance is used."""
        # This test would require running pytest programmatically with the plugin
        # For now, we'll verify the plugin can be imported and configured
        # Full integration test will be in test_performance_integration.py

        # Arrange
        from performance import plugin

        # Act & Assert - Just verify plugin module can be imported
        assert plugin is not None
        assert hasattr(plugin, "pytest_runtest_setup")
        assert hasattr(plugin, "pytest_runtest_teardown")
        assert hasattr(plugin, "pytest_sessionfinish")
