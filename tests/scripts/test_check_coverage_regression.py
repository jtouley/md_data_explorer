"""Tests for coverage regression detection script."""

import importlib.util
import json
import sys
from pathlib import Path

# Load the script module directly
script_path = Path(__file__).parent.parent.parent / "scripts" / "check_coverage_regression.py"
spec = importlib.util.spec_from_file_location("check_coverage_regression", script_path)
check_coverage_regression = importlib.util.module_from_spec(spec)
sys.modules["check_coverage_regression"] = check_coverage_regression
spec.loader.exec_module(check_coverage_regression)

main = check_coverage_regression.main


class TestCheckCoverageRegression:
    """Test suite for check_coverage_regression.py."""

    def test_main_missingCoverageJson_returnsOne(self, tmp_path, monkeypatch):
        """Test that missing coverage.json returns exit code 1."""
        monkeypatch.chdir(tmp_path)
        # Need to reload the module with new paths
        check_coverage_regression.CURRENT_FILE = tmp_path / "coverage.json"
        check_coverage_regression.BASELINE_FILE = tmp_path / "tests" / ".coverage_baseline.json"
        assert check_coverage_regression.main() == 1

    def test_main_coverageBelowThreshold_returnsOne(self, tmp_path, monkeypatch):
        """Test that coverage below 67% returns exit code 1."""
        monkeypatch.chdir(tmp_path)
        coverage_data = {"totals": {"percent_covered": 50.0}}
        coverage_file = tmp_path / "coverage.json"
        coverage_file.write_text(json.dumps(coverage_data))

        check_coverage_regression.CURRENT_FILE = coverage_file
        check_coverage_regression.BASELINE_FILE = tmp_path / "tests" / ".coverage_baseline.json"
        assert check_coverage_regression.main() == 1

    def test_main_coverageAboveThreshold_noBaseline_returnsZero(self, tmp_path, monkeypatch):
        """Test that coverage >= 67% with no baseline returns exit code 0."""
        monkeypatch.chdir(tmp_path)
        coverage_data = {"totals": {"percent_covered": 70.0}}
        coverage_file = tmp_path / "coverage.json"
        coverage_file.write_text(json.dumps(coverage_data))

        check_coverage_regression.CURRENT_FILE = coverage_file
        check_coverage_regression.BASELINE_FILE = tmp_path / "tests" / ".coverage_baseline.json"
        assert check_coverage_regression.main() == 0

    def test_main_regressionExceedsTolerance_returnsOne(self, tmp_path, monkeypatch):
        """Test that regression > 0.5% from baseline returns exit code 1."""
        monkeypatch.chdir(tmp_path)
        # Create tests directory for baseline
        tests_dir = tmp_path / "tests"
        tests_dir.mkdir()

        # Baseline: 70%, Current: 68% (regression of 2% > 0.5% tolerance)
        baseline_data = {"totals": {"percent_covered": 70.0}}
        baseline_file = tests_dir / ".coverage_baseline.json"
        baseline_file.write_text(json.dumps(baseline_data))

        current_data = {"totals": {"percent_covered": 68.0}}
        coverage_file = tmp_path / "coverage.json"
        coverage_file.write_text(json.dumps(current_data))

        check_coverage_regression.CURRENT_FILE = coverage_file
        check_coverage_regression.BASELINE_FILE = baseline_file
        assert check_coverage_regression.main() == 1

    def test_main_regressionWithinTolerance_returnsZero(self, tmp_path, monkeypatch):
        """Test that regression <= 0.5% from baseline returns exit code 0."""
        monkeypatch.chdir(tmp_path)
        tests_dir = tmp_path / "tests"
        tests_dir.mkdir()

        # Baseline: 70%, Current: 69.7% (regression of 0.3% <= 0.5% tolerance)
        baseline_data = {"totals": {"percent_covered": 70.0}}
        baseline_file = tests_dir / ".coverage_baseline.json"
        baseline_file.write_text(json.dumps(baseline_data))

        current_data = {"totals": {"percent_covered": 69.7}}
        coverage_file = tmp_path / "coverage.json"
        coverage_file.write_text(json.dumps(current_data))

        check_coverage_regression.CURRENT_FILE = coverage_file
        check_coverage_regression.BASELINE_FILE = baseline_file
        assert check_coverage_regression.main() == 0
