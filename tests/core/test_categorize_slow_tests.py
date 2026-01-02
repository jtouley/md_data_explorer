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

        report = generate_report(data_file, slow_threshold=30.0, fast_threshold=1.0, integration_threshold=10.0)

        # Assert: Report contains expected information
        assert "Uncategorized Slow Tests" in report
        assert "test_slow_uncategorized.py::test_uncategorized_slow" in report
        assert "35.0" in report
        assert "@pytest.mark.slow" in report or "slow" in report.lower()

    def test_identifies_incorrectly_marked_slow_tests(self, tmp_path):
        """
        Test that script identifies fast tests incorrectly marked as slow.

        Tests <1s with @pytest.mark.slow should be flagged.
        """
        # Arrange: Create mock performance data
        mock_data = {
            "run_id": "test_run",
            "tests": [
                {
                    "nodeid": "tests/core/test_fast_incorrectly_marked.py::test_fast_test",
                    "duration": 0.5,  # Fast test
                    "markers": ["slow"],  # Incorrectly marked as slow
                    "module": "core",
                    "status": "passed",
                },
            ],
        }

        data_file = tmp_path / ".performance_data.json"
        with open(data_file, "w") as f:
            json.dump(mock_data, f)

        # Act: Run categorization script
        from scripts.categorize_slow_tests import find_incorrectly_marked_slow_tests

        incorrectly_marked = find_incorrectly_marked_slow_tests(data_file, fast_threshold=1.0)

        # Assert: Identifies incorrectly marked test
        assert len(incorrectly_marked) == 1
        assert incorrectly_marked[0]["nodeid"] == "tests/core/test_fast_incorrectly_marked.py::test_fast_test"
        assert incorrectly_marked[0]["duration"] == 0.5
        assert "slow" in incorrectly_marked[0]["markers"]

    def test_identifies_uncategorized_integration_tests(self, tmp_path):
        """
        Test that script identifies integration tests without @pytest.mark.integration.

        Tests >10s are likely integration tests (heuristic).
        """
        # Arrange: Create mock performance data
        mock_data = {
            "run_id": "test_run",
            "tests": [
                {
                    "nodeid": "tests/core/test_integration_uncategorized.py::test_integration_test",
                    "duration": 15.0,  # Slow test (likely integration)
                    "markers": [],  # Missing @pytest.mark.integration
                    "module": "core",
                    "status": "passed",
                },
            ],
        }

        data_file = tmp_path / ".performance_data.json"
        with open(data_file, "w") as f:
            json.dump(mock_data, f)

        # Act: Run categorization script
        from scripts.categorize_slow_tests import find_uncategorized_integration_tests

        uncategorized = find_uncategorized_integration_tests(data_file, integration_threshold=10.0)

        # Assert: Identifies uncategorized integration test
        assert len(uncategorized) == 1
        assert uncategorized[0]["nodeid"] == "tests/core/test_integration_uncategorized.py::test_integration_test"
        assert uncategorized[0]["duration"] == 15.0
        assert "integration" not in uncategorized[0]["markers"]

    def test_identifies_incorrectly_marked_integration_tests(self, tmp_path):
        """
        Test that script identifies unit tests incorrectly marked as integration.

        Tests <1s with @pytest.mark.integration should be flagged.
        """
        # Arrange: Create mock performance data
        mock_data = {
            "run_id": "test_run",
            "tests": [
                {
                    "nodeid": "tests/core/test_unit_incorrectly_marked.py::test_unit_test",
                    "duration": 0.3,  # Fast test (unit test)
                    "markers": ["integration"],  # Incorrectly marked as integration
                    "module": "core",
                    "status": "passed",
                },
            ],
        }

        data_file = tmp_path / ".performance_data.json"
        with open(data_file, "w") as f:
            json.dump(mock_data, f)

        # Act: Run categorization script
        from scripts.categorize_slow_tests import find_incorrectly_marked_integration_tests

        incorrectly_marked = find_incorrectly_marked_integration_tests(data_file, fast_threshold=1.0)

        # Assert: Identifies incorrectly marked test
        assert len(incorrectly_marked) == 1
        assert incorrectly_marked[0]["nodeid"] == "tests/core/test_unit_incorrectly_marked.py::test_unit_test"
        assert incorrectly_marked[0]["duration"] == 0.3
        assert "integration" in incorrectly_marked[0]["markers"]

    def test_generates_comprehensive_report(self, tmp_path):
        """
        Test that script generates comprehensive report with all categories.

        Report should include all categorization issues.
        """
        # Arrange: Create mock performance data with various issues
        mock_data = {
            "run_id": "test_run",
            "tests": [
                {
                    "nodeid": "tests/core/test_slow_uncategorized.py::test_slow",
                    "duration": 35.0,
                    "markers": [],
                    "module": "core",
                    "status": "passed",
                },
                {
                    "nodeid": "tests/core/test_fast_incorrectly_slow.py::test_fast",
                    "duration": 0.5,
                    "markers": ["slow"],
                    "module": "core",
                    "status": "passed",
                },
                {
                    "nodeid": "tests/core/test_integration_uncategorized.py::test_integration",
                    "duration": 15.0,
                    "markers": [],
                    "module": "core",
                    "status": "passed",
                },
            ],
        }

        data_file = tmp_path / ".performance_data.json"
        with open(data_file, "w") as f:
            json.dump(mock_data, f)

        # Act: Generate comprehensive report
        from scripts.categorize_slow_tests import generate_report

        report = generate_report(data_file, slow_threshold=30.0, fast_threshold=1.0, integration_threshold=10.0)

        # Assert: Report contains all categories
        assert "Uncategorized Slow Tests" in report
        assert "Incorrectly Marked Slow Tests" in report
        assert "Uncategorized Integration Tests" in report
        assert "Summary" in report
        assert "test_slow_uncategorized" in report
        assert "test_fast_incorrectly_slow" in report
        assert "test_integration_uncategorized" in report
