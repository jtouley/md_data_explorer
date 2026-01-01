"""Tests for performance tracking storage utilities."""

import json
import sys
import tempfile
from pathlib import Path

# Add tests to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from performance.storage import (
    load_baseline,
    load_performance_data,
    save_baseline,
    save_performance_data,
)


class TestStorageReadWrite:
    """Test JSON read/write operations."""

    def test_storage_save_and_load_performance_data_roundtrip(self):
        """Test that saving and loading performance data works correctly."""
        # Arrange
        test_data = {
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
                "min_duration": 1.5,
                "max_duration": 1.5,
            },
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            data_file = Path(tmpdir) / ".performance_data.json"

            # Act
            save_performance_data(test_data, data_file)
            loaded_data = load_performance_data(data_file)

            # Assert
            assert loaded_data == test_data
            assert loaded_data["run_id"] == "2025-01-15T10:30:00"
            assert len(loaded_data["tests"]) == 1
            assert loaded_data["summary"]["total_tests"] == 1

    def test_storage_load_missing_file_returns_empty_structure(self):
        """Test that loading missing file returns empty structure."""
        # Arrange
        missing_file = Path("/nonexistent/path/.performance_data.json")

        # Act
        result = load_performance_data(missing_file)

        # Assert
        assert result == {
            "run_id": "",
            "tests": [],
            "summary": {
                "total_tests": 0,
                "slow_tests": 0,
                "total_duration": 0.0,
                "average_duration": 0.0,
                "min_duration": 0.0,
                "max_duration": 0.0,
            },
        }

    def test_storage_summary_includes_min_max_duration(self):
        """Test that performance data summary includes min and max duration."""
        # Arrange
        test_data = {
            "run_id": "2025-01-15T10:30:00",
            "tests": [
                {
                    "nodeid": "tests/core/test_fast.py::test_example",
                    "duration": 0.5,
                    "markers": [],
                    "module": "core",
                    "status": "passed",
                },
                {
                    "nodeid": "tests/core/test_medium.py::test_example",
                    "duration": 2.0,
                    "markers": [],
                    "module": "core",
                    "status": "passed",
                },
                {
                    "nodeid": "tests/core/test_slow.py::test_example",
                    "duration": 5.0,
                    "markers": ["slow"],
                    "module": "core",
                    "status": "passed",
                },
            ],
            "summary": {
                "total_tests": 3,
                "slow_tests": 1,
                "total_duration": 7.5,
                "average_duration": 2.5,
                "min_duration": 0.5,
                "max_duration": 5.0,
            },
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            data_file = Path(tmpdir) / ".performance_data.json"

            # Act
            save_performance_data(test_data, data_file)
            loaded_data = load_performance_data(data_file)

            # Assert: Verify min/max are included and correct
            summary = loaded_data["summary"]
            assert "min_duration" in summary
            assert "max_duration" in summary
            assert summary["min_duration"] == 0.5
            assert summary["max_duration"] == 5.0
            assert summary["average_duration"] == 2.5

    def test_storage_load_corrupted_json_returns_empty_structure(self):
        """Test that loading corrupted JSON returns empty structure."""
        # Arrange
        with tempfile.TemporaryDirectory() as tmpdir:
            corrupted_file = Path(tmpdir) / ".performance_data.json"
            corrupted_file.write_text("{ invalid json }")

            # Act
            result = load_performance_data(corrupted_file)

            # Assert
            assert result == {
                "run_id": "",
                "tests": [],
                "summary": {
                    "total_tests": 0,
                    "slow_tests": 0,
                    "total_duration": 0.0,
                    "average_duration": 0.0,
                    "min_duration": 0.0,
                    "max_duration": 0.0,
                },
            }

    def test_storage_save_and_load_baseline_roundtrip(self):
        """Test that saving and loading baseline works correctly."""
        # Arrange
        baseline_data = {
            "baseline_date": "2025-01-15",
            "tests": {
                "tests/core/test_example.py::test_example": {
                    "duration": 1.0,
                    "threshold": 1.2,
                }
            },
            "suite_metrics": {
                "core": {"total_duration": 100.0, "threshold": 115.0},
            },
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            baseline_file = Path(tmpdir) / ".performance_baseline.json"

            # Act
            save_baseline(baseline_data, baseline_file)
            loaded_baseline = load_baseline(baseline_file)

            # Assert
            assert loaded_baseline == baseline_data
            assert loaded_baseline["baseline_date"] == "2025-01-15"
            assert "tests/core/test_example.py::test_example" in loaded_baseline["tests"]

    def test_storage_load_missing_baseline_returns_empty_structure(self):
        """Test that loading missing baseline returns empty structure."""
        # Arrange
        missing_file = Path("/nonexistent/path/.performance_baseline.json")

        # Act
        result = load_baseline(missing_file)

        # Assert
        assert result == {
            "baseline_date": "",
            "tests": {},
            "suite_metrics": {},
        }

    def test_storage_load_validates_schema_on_load(self):
        """Test that loading validates schema and handles invalid data."""
        # Arrange
        with tempfile.TemporaryDirectory() as tmpdir:
            invalid_file = Path(tmpdir) / ".performance_data.json"
            # Missing required fields
            invalid_data = {"run_id": "test"}
            invalid_file.write_text(json.dumps(invalid_data))

            # Act
            result = load_performance_data(invalid_file)

            # Assert - Should return empty structure for invalid schema
            assert "tests" in result
            assert "summary" in result
            assert result["summary"]["total_tests"] == 0
