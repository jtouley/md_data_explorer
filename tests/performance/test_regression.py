"""Tests for performance regression detection."""

import sys
from pathlib import Path

import pytest

# Add tests to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from performance.regression import (
    RegressionError,
    calculate_percentage_increase,
    check_regressions,
)


class TestRegression:
    """Test regression detection."""

    def test_regression_calculate_percentage_increase(self):
        """Test that percentage increase is calculated correctly."""
        # Arrange
        baseline_duration = 10.0
        current_duration = 12.0

        # Act
        increase = calculate_percentage_increase(baseline_duration, current_duration)

        # Assert
        assert increase == 20.0

    def test_regression_calculate_percentage_increase_with_zero_baseline(self):
        """Test that percentage increase handles zero baseline."""
        # Arrange
        baseline_duration = 0.0
        current_duration = 5.0

        # Act
        increase = calculate_percentage_increase(baseline_duration, current_duration)

        # Assert
        # Should return 0 or handle gracefully
        assert isinstance(increase, (int, float))

    def test_regression_check_regressions_detects_individual_test_regression(self):
        """Test that regression detection finds individual test regressions."""
        # Arrange
        baseline = {
            "baseline_date": "2025-01-15",
            "tests": {
                "tests/core/test_example.py::test_example": {
                    "duration": 10.0,
                    "threshold": 12.0,  # 20% threshold
                }
            },
            "suite_metrics": {},
        }

        current_data = {
            "run_id": "2025-01-16T10:30:00",
            "tests": [
                {
                    "nodeid": "tests/core/test_example.py::test_example",
                    "duration": 15.0,  # 50% increase, exceeds 20% threshold
                    "markers": [],
                    "module": "core",
                    "status": "passed",
                }
            ],
            "summary": {},
        }

        # Act & Assert
        with pytest.raises(RegressionError) as exc_info:
            check_regressions(current_data, baseline, individual_threshold=20.0, suite_threshold=15.0)

        error_msg = str(exc_info.value)
        assert "test_example" in error_msg
        assert "10.0" in error_msg or "baseline" in error_msg.lower()
        assert "15.0" in error_msg or "current" in error_msg.lower()

    def test_regression_check_regressions_passes_when_no_regression(self):
        """Test that regression check passes when no regressions found."""
        # Arrange
        baseline = {
            "baseline_date": "2025-01-15",
            "tests": {
                "tests/core/test_example.py::test_example": {
                    "duration": 10.0,
                    "threshold": 12.0,
                }
            },
            "suite_metrics": {},
        }

        current_data = {
            "run_id": "2025-01-16T10:30:00",
            "tests": [
                {
                    "nodeid": "tests/core/test_example.py::test_example",
                    "duration": 11.0,  # 10% increase, within 20% threshold
                    "markers": [],
                    "module": "core",
                    "status": "passed",
                }
            ],
            "summary": {},
        }

        # Act - Should not raise
        check_regressions(current_data, baseline, individual_threshold=20.0, suite_threshold=15.0)

    def test_regression_check_regressions_handles_missing_baseline_gracefully(self):
        """Test that regression check handles missing baseline gracefully."""
        # Arrange
        baseline = {
            "baseline_date": "",
            "tests": {},
            "suite_metrics": {},
        }

        current_data = {
            "run_id": "2025-01-16T10:30:00",
            "tests": [
                {
                    "nodeid": "tests/core/test_example.py::test_example",
                    "duration": 15.0,
                    "markers": [],
                    "module": "core",
                    "status": "passed",
                }
            ],
            "summary": {},
        }

        # Act - Should not raise when baseline is empty
        check_regressions(current_data, baseline, individual_threshold=20.0, suite_threshold=15.0)

    def test_regression_check_regressions_detects_suite_level_regression(self):
        """Test that regression detection finds suite-level regressions."""
        # Arrange
        baseline = {
            "baseline_date": "2025-01-15",
            "tests": {},
            "suite_metrics": {
                "core": {"total_duration": 100.0, "threshold": 115.0},  # 15% threshold
            },
        }

        current_data = {
            "run_id": "2025-01-16T10:30:00",
            "tests": [
                {
                    "nodeid": "tests/core/test_example.py::test_example",
                    "duration": 60.0,
                    "markers": [],
                    "module": "core",
                    "status": "passed",
                },
                {
                    "nodeid": "tests/core/test_another.py::test_another",
                    "duration": 70.0,  # Total: 130.0, exceeds 115.0 threshold
                    "markers": [],
                    "module": "core",
                    "status": "passed",
                },
            ],
            "summary": {},
        }

        # Act & Assert
        with pytest.raises(RegressionError) as exc_info:
            check_regressions(current_data, baseline, individual_threshold=20.0, suite_threshold=15.0)

        error_msg = str(exc_info.value)
        assert "core" in error_msg.lower() or "suite" in error_msg.lower()
