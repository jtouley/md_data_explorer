"""
Tests for automated test categorization script (Phase 3).

Tests verify that the categorization script correctly identifies
uncategorized slow tests and generates actionable reports.

Test name follows: test_unit_scenario_expectedBehavior
"""

import json
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


class TestCategorizeSlowTests:
    """Test suite for slow test categorization script."""

    def test_identifies_uncategorized_slow_tests(self, tmp_path):
        """
        Test that script identifies tests >30s without @pytest.mark.slow.

        This test creates mock performance data and verifies the script
        correctly identifies uncategorized slow tests.
        """
        # Arrange: Create mock performance data
        mock_data = {
            "run_id": "test_run",
            "tests": [
                {
                    "nodeid": "tests/core/test_fast.py::test_fast_test",
                    "duration": 0.5,
                    "markers": [],
                    "module": "core",
                    "status": "passed",
                },
                {
                    "nodeid": "tests/core/test_slow_categorized.py::test_slow_test",
                    "duration": 45.0,
                    "markers": ["slow"],
                    "module": "core",
                    "status": "passed",
                },
                {
                    "nodeid": "tests/core/test_slow_uncategorized.py::test_uncategorized_slow",
                    "duration": 35.0,
                    "markers": [],  # Missing @pytest.mark.slow
                    "module": "core",
                    "status": "passed",
                },
            ],
        }

        data_file = tmp_path / ".performance_data.json"
        with open(data_file, "w") as f:
            json.dump(mock_data, f)

        # Act: Run categorization script
        from scripts.categorize_slow_tests import categorize_slow_tests

        uncategorized = categorize_slow_tests(data_file, threshold=30.0)

        # Assert: Identifies uncategorized slow test
        assert len(uncategorized) == 1
        assert uncategorized[0]["nodeid"] == "tests/core/test_slow_uncategorized.py::test_uncategorized_slow"
        assert uncategorized[0]["duration"] == 35.0
        assert "slow" not in uncategorized[0]["markers"]

    def test_respects_threshold(self, tmp_path):
        """
        Test that script respects the duration threshold.

        Tests below threshold should not be categorized as slow.
        """
        # Arrange: Create mock performance data with test just below threshold
        mock_data = {
            "run_id": "test_run",
            "tests": [
                {
                    "nodeid": "tests/core/test_below_threshold.py::test_test",
                    "duration": 29.9,  # Just below 30s threshold
                    "markers": [],
                    "module": "core",
                    "status": "passed",
                },
            ],
        }

        data_file = tmp_path / ".performance_data.json"
        with open(data_file, "w") as f:
            json.dump(mock_data, f)

        # Act: Run categorization script
        from scripts.categorize_slow_tests import categorize_slow_tests

        uncategorized = categorize_slow_tests(data_file, threshold=30.0)

        # Assert: No uncategorized slow tests (below threshold)
        assert len(uncategorized) == 0

    def test_ignores_already_categorized_tests(self, tmp_path):
        """
        Test that script ignores tests already marked with @pytest.mark.slow.

        Tests with "slow" marker should not appear in uncategorized list.
        """
        # Arrange: Create mock performance data with categorized slow test
        mock_data = {
            "run_id": "test_run",
            "tests": [
                {
                    "nodeid": "tests/core/test_slow_categorized.py::test_slow_test",
                    "duration": 50.0,
                    "markers": ["slow"],  # Already categorized
                    "module": "core",
                    "status": "passed",
                },
            ],
        }

        data_file = tmp_path / ".performance_data.json"
        with open(data_file, "w") as f:
            json.dump(mock_data, f)

        # Act: Run categorization script
        from scripts.categorize_slow_tests import categorize_slow_tests

        uncategorized = categorize_slow_tests(data_file, threshold=30.0)

        # Assert: No uncategorized tests (already categorized)
        assert len(uncategorized) == 0

    def test_generates_report(self, tmp_path):
        """
        Test that script generates actionable report.

        Report should include test nodeid, duration, and recommendation.
        """
        # Arrange: Create mock performance data
        mock_data = {
            "run_id": "test_run",
            "tests": [
                {
                    "nodeid": "tests/core/test_slow_uncategorized.py::test_uncategorized_slow",
                    "duration": 35.0,
                    "markers": [],
                    "module": "core",
                    "status": "passed",
                },
            ],
        }

        data_file = tmp_path / ".performance_data.json"
        with open(data_file, "w") as f:
            json.dump(mock_data, f)

        # Act: Generate report
        from scripts.categorize_slow_tests import generate_report

        report = generate_report(data_file, threshold=30.0)

        # Assert: Report contains expected information
        assert "Uncategorized Slow Tests" in report
        assert "test_slow_uncategorized.py::test_uncategorized_slow" in report
        assert "35.0" in report
        assert "@pytest.mark.slow" in report or "slow" in report.lower()
