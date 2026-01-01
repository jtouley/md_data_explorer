"""Integration tests for performance tracking end-to-end workflow."""

import sys
from pathlib import Path

import pytest

# Add tests to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from performance.regression import RegressionError, check_regressions
from performance.reporter import generate_markdown_report
from performance.storage import load_baseline, load_performance_data, save_baseline, save_performance_data


class TestPerformanceIntegration:
    """Integration tests for complete performance tracking workflow."""

    def test_integration_track_report_baseline_regression_workflow(self, tmp_path):
        """Test complete workflow: track → report → baseline → regression."""
        # Arrange
        data_file = tmp_path / ".performance_data.json"
        baseline_file = tmp_path / ".performance_baseline.json"

        # Step 1: Simulate performance data from test run
        performance_data = {
            "run_id": "2025-01-15T10:30:00",
            "tests": [
                {
                    "nodeid": "tests/core/test_example.py::test_example",
                    "duration": 10.0,
                    "markers": [],
                    "module": "core",
                    "status": "passed",
                },
                {
                    "nodeid": "tests/ui/test_example.py::test_example",
                    "duration": 5.0,
                    "markers": ["slow"],
                    "module": "ui",
                    "status": "passed",
                },
            ],
            "summary": {
                "total_tests": 2,
                "slow_tests": 1,
                "total_duration": 15.0,
                "average_duration": 7.5,
            },
        }

        # Act Step 1: Save performance data (simulating plugin output)
        save_performance_data(performance_data, data_file)

        # Assert Step 1: Verify data saved
        assert data_file.exists()
        loaded_data = load_performance_data(data_file)
        assert loaded_data["run_id"] == "2025-01-15T10:30:00"
        assert len(loaded_data["tests"]) == 2

        # Act Step 2: Generate report
        report = generate_markdown_report(performance_data)

        # Assert Step 2: Verify report generated
        assert isinstance(report, str)
        assert "Performance Report" in report
        assert "Total Tests: 2" in report or "2" in report

        # Act Step 3: Create baseline
        from datetime import date

        baseline = {
            "baseline_date": date.today().isoformat(),
            "tests": {},
            "suite_metrics": {},
        }

        # Convert test results to baseline format
        for test in performance_data.get("tests", []):
            nodeid = test.get("nodeid")
            duration = test.get("duration", 0.0)
            baseline["tests"][nodeid] = {
                "duration": duration,
                "threshold": duration * 1.2,  # 20% threshold
            }

        # Group by module for suite metrics
        module_durations: dict[str, float] = {}
        for test in performance_data.get("tests", []):
            module = test.get("module", "unknown")
            duration = test.get("duration", 0.0)
            module_durations[module] = module_durations.get(module, 0.0) + duration

        for module, total_duration in module_durations.items():
            baseline["suite_metrics"][module] = {
                "total_duration": total_duration,
                "threshold": total_duration * 1.15,  # 15% threshold
            }

        save_baseline(baseline, baseline_file)

        # Assert Step 3: Verify baseline created
        assert baseline_file.exists()
        loaded_baseline = load_baseline(baseline_file)
        assert "baseline_date" in loaded_baseline
        assert len(loaded_baseline["tests"]) == 2

        # Act Step 4: Check for regressions (should pass - no regression)
        check_regressions(performance_data, baseline, individual_threshold=20.0, suite_threshold=15.0)

        # Assert Step 4: No exception raised (no regression)

        # Act Step 5: Simulate regression (test runs slower)
        regressed_data = {
            "run_id": "2025-01-16T10:30:00",
            "tests": [
                {
                    "nodeid": "tests/core/test_example.py::test_example",
                    "duration": 15.0,  # 50% increase, exceeds 20% threshold
                    "markers": [],
                    "module": "core",
                    "status": "passed",
                },
                {
                    "nodeid": "tests/ui/test_example.py::test_example",
                    "duration": 5.0,
                    "markers": ["slow"],
                    "module": "ui",
                    "status": "passed",
                },
            ],
            "summary": {
                "total_tests": 2,
                "slow_tests": 1,
                "total_duration": 20.0,
                "average_duration": 10.0,
            },
        }

        # Assert Step 5: Regression detected
        with pytest.raises(RegressionError) as exc_info:
            check_regressions(regressed_data, baseline, individual_threshold=20.0, suite_threshold=15.0)

        error_msg = str(exc_info.value)
        assert "test_example" in error_msg
        assert "regressed" in error_msg.lower()
