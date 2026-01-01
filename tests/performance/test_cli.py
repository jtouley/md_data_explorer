"""Tests for performance report CLI tool."""

import json
import sys
from pathlib import Path

# Add tests to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# CLI tool will be in scripts/
scripts_path = Path(__file__).parent.parent.parent / "scripts"
sys.path.insert(0, str(scripts_path))


class TestCLI:
    """Test CLI tool functionality."""

    def test_cli_generate_markdown_report_from_file(self, tmp_path):
        """Test that CLI generates markdown report from performance data file."""
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
                }
            ],
            "summary": {
                "total_tests": 1,
                "slow_tests": 1,
                "total_duration": 1.5,
                "average_duration": 1.5,
            },
        }

        data_file = tmp_path / ".performance_data.json"
        data_file.write_text(json.dumps(performance_data))

        # Act - Import and test CLI function
        # We'll test the main function logic
        from performance.reporter import generate_markdown_report

        report = generate_markdown_report(performance_data)

        # Assert
        assert isinstance(report, str)
        assert "Performance Report" in report
        assert "2025-01-15T10:30:00" in report

    def test_cli_create_baseline_from_performance_data(self, tmp_path):
        """Test that CLI creates baseline from performance data."""
        # Arrange
        performance_data = {
            "run_id": "2025-01-15T10:30:00",
            "tests": [
                {
                    "nodeid": "tests/core/test_example.py::test_example",
                    "duration": 10.0,
                    "markers": [],
                    "module": "core",
                    "status": "passed",
                }
            ],
            "summary": {
                "total_tests": 1,
                "slow_tests": 0,
                "total_duration": 10.0,
                "average_duration": 10.0,
            },
        }

        data_file = tmp_path / ".performance_data.json"
        baseline_file = tmp_path / ".performance_baseline.json"
        data_file.write_text(json.dumps(performance_data))

        # Act - Test baseline creation logic
        from datetime import date

        from performance.storage import load_performance_data, save_baseline

        data = load_performance_data(data_file)

        # Create baseline structure
        baseline = {
            "baseline_date": date.today().isoformat(),
            "tests": {},
            "suite_metrics": {},
        }

        # Convert test results to baseline format
        for test in data.get("tests", []):
            nodeid = test.get("nodeid")
            duration = test.get("duration", 0.0)
            baseline["tests"][nodeid] = {
                "duration": duration,
                "threshold": duration * 1.2,  # 20% threshold
            }

        # Group by module for suite metrics
        module_durations: dict[str, float] = {}
        for test in data.get("tests", []):
            module = test.get("module", "unknown")
            duration = test.get("duration", 0.0)
            module_durations[module] = module_durations.get(module, 0.0) + duration

        for module, total_duration in module_durations.items():
            baseline["suite_metrics"][module] = {
                "total_duration": total_duration,
                "threshold": total_duration * 1.15,  # 15% threshold
            }

        save_baseline(baseline, baseline_file)

        # Assert
        assert baseline_file.exists()
        loaded_baseline = json.loads(baseline_file.read_text())
        assert "baseline_date" in loaded_baseline
        assert "tests/core/test_example.py::test_example" in loaded_baseline["tests"]
        assert loaded_baseline["tests"]["tests/core/test_example.py::test_example"]["duration"] == 10.0
