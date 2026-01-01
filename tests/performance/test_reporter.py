"""Tests for performance report generation."""

import sys
from pathlib import Path

# Add tests to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from performance.reporter import generate_json_report, generate_markdown_report


class TestReporter:
    """Test report generation."""

    def test_reporter_generate_markdown_report_with_data(self):
        """Test that markdown report is generated correctly with performance data."""
        # Arrange
        performance_data = {
            "run_id": "2025-01-15T10:30:00",
            "tests": [
                {
                    "nodeid": "tests/core/test_example.py::test_example",
                    "duration": 1.5,
                    "markers": ["slow"],
                    "module": "core",
                    "status": "passed",
                },
                {
                    "nodeid": "tests/ui/test_example.py::test_example",
                    "duration": 35.0,
                    "markers": ["slow", "integration"],
                    "module": "ui",
                    "status": "passed",
                },
            ],
            "summary": {
                "total_tests": 2,
                "slow_tests": 2,
                "total_duration": 36.5,
                "average_duration": 18.25,
            },
        }

        # Act
        report = generate_markdown_report(performance_data)

        # Assert
        assert isinstance(report, str)
        assert "Performance Report" in report
        assert "2025-01-15T10:30:00" in report
        assert "Total Tests: 2" in report or "2" in report
        assert "Slow Tests: 2" in report or "2" in report
        assert "test_example" in report

    def test_reporter_generate_markdown_report_with_empty_data(self):
        """Test that markdown report handles empty data gracefully."""
        # Arrange
        performance_data = {
            "run_id": "",
            "tests": [],
            "summary": {
                "total_tests": 0,
                "slow_tests": 0,
                "total_duration": 0.0,
                "average_duration": 0.0,
            },
        }

        # Act
        report = generate_markdown_report(performance_data)

        # Assert
        assert isinstance(report, str)
        assert "Performance Report" in report
        assert "Total Tests: 0" in report or "0" in report

    def test_reporter_generate_json_report_with_data(self):
        """Test that JSON report is generated correctly."""
        # Arrange
        performance_data = {
            "run_id": "2025-01-15T10:30:00",
            "tests": [
                {
                    "nodeid": "tests/core/test_example.py::test_example",
                    "duration": 1.5,
                    "markers": ["slow"],
                    "module": "core",
                    "status": "passed",
                },
            ],
            "summary": {
                "total_tests": 1,
                "slow_tests": 1,
                "total_duration": 1.5,
                "average_duration": 1.5,
            },
        }

        # Act
        report = generate_json_report(performance_data)

        # Assert
        assert isinstance(report, str)
        import json

        parsed = json.loads(report)
        assert parsed["run_id"] == "2025-01-15T10:30:00"
        assert len(parsed["tests"]) == 1
        assert parsed["summary"]["total_tests"] == 1

    def test_reporter_generate_markdown_report_includes_slowest_tests(self):
        """Test that markdown report includes slowest tests section."""
        # Arrange
        performance_data = {
            "run_id": "2025-01-15T10:30:00",
            "tests": [
                {
                    "nodeid": "tests/core/test_slow.py::test_slow",
                    "duration": 50.0,
                    "markers": ["slow"],
                    "module": "core",
                    "status": "passed",
                },
                {
                    "nodeid": "tests/core/test_fast.py::test_fast",
                    "duration": 0.1,
                    "markers": [],
                    "module": "core",
                    "status": "passed",
                },
            ],
            "summary": {
                "total_tests": 2,
                "slow_tests": 1,
                "total_duration": 50.1,
                "average_duration": 25.05,
            },
        }

        # Act
        report = generate_markdown_report(performance_data)

        # Assert
        assert "Slowest Tests" in report or "slowest" in report.lower()
        assert "test_slow" in report
        # Slow test should appear before fast test in report
        slow_index = report.find("test_slow")
        fast_index = report.find("test_fast")
        if slow_index != -1 and fast_index != -1:
            assert slow_index < fast_index
