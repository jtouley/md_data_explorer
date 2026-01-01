"""Performance regression test suite.

This test suite validates that test performance hasn't regressed beyond acceptable thresholds.
Requires baseline and performance data files to be present.
"""

import sys
from pathlib import Path

import pytest

# Add tests to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from performance.regression import RegressionError, check_regressions
from performance.storage import load_baseline, load_performance_data


@pytest.mark.slow
@pytest.mark.integration
class TestPerformanceRegression:
    """Test suite for performance regression detection."""

    @pytest.fixture
    def performance_data_file(self):
        """Get path to performance data file."""
        project_root = Path(__file__).parent.parent
        return project_root / "tests" / ".performance_data.json"

    @pytest.fixture
    def baseline_file(self):
        """Get path to baseline file."""
        project_root = Path(__file__).parent.parent
        return project_root / "tests" / ".performance_baseline.json"

    def test_individual_test_performance(self, performance_data_file, baseline_file):
        """Verify individual tests haven't regressed beyond threshold."""
        # Check if files exist
        if not baseline_file.exists():
            pytest.skip("Baseline not found. Run 'make performance-baseline' to create initial baseline.")

        if not performance_data_file.exists():
            pytest.skip("No performance data found. Run tests with '--track-performance' flag first.")

        # Load data
        baseline = load_baseline(baseline_file)
        current_data = load_performance_data(performance_data_file)

        # Skip if baseline is empty
        if not baseline.get("tests"):
            pytest.skip("Baseline is empty. Run 'make performance-baseline' to create initial baseline.")

        # Check for regressions
        # Should raise RegressionError if regressions found
        try:
            check_regressions(current_data, baseline, individual_threshold=20.0, suite_threshold=15.0)
        except RegressionError as e:
            pytest.fail(f"Performance regression detected:\n{e}")

    def test_suite_performance(self, performance_data_file, baseline_file):
        """Verify test suite performance hasn't regressed."""
        # Check if files exist
        if not baseline_file.exists():
            pytest.skip("Baseline not found. Run 'make performance-baseline' to create initial baseline.")

        if not performance_data_file.exists():
            pytest.skip("No performance data found. Run tests with '--track-performance' flag first.")

        # Load data
        baseline = load_baseline(baseline_file)
        current_data = load_performance_data(performance_data_file)

        # Skip if baseline suite metrics are empty
        if not baseline.get("suite_metrics"):
            pytest.skip("Baseline suite metrics are empty. Run 'make performance-baseline' to create initial baseline.")

        # Check for regressions
        try:
            check_regressions(current_data, baseline, individual_threshold=20.0, suite_threshold=15.0)
        except RegressionError as e:
            pytest.fail(f"Suite-level performance regression detected:\n{e}")

    def test_slow_test_count(self, performance_data_file, baseline_file):
        """Verify slow test count hasn't increased unexpectedly."""
        if not performance_data_file.exists():
            pytest.skip("No performance data found. Run tests with '--track-performance' flag first.")

        current_data = load_performance_data(performance_data_file)
        current_slow_count = current_data.get("summary", {}).get("slow_tests", 0)

        # Warning if slow test count is high (informational, not a failure)
        if current_slow_count > 50:
            pytest.fail(
                f"Slow test count is high: {current_slow_count}. "
                "Consider optimizing slow tests or reviewing test categorization."
            )
