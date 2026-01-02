"""Performance regression detection utilities."""

from typing import Any


class RegressionError(Exception):
    """Exception raised when performance regression is detected."""

    pass


def calculate_percentage_increase(baseline: float, current: float) -> float:
    """
    Calculate percentage increase from baseline to current.

    Args:
        baseline: Baseline duration
        current: Current duration

    Returns:
        Percentage increase (0.0 if baseline is 0)
    """
    if baseline == 0.0:
        return 0.0
    return ((current - baseline) / baseline) * 100.0


def check_regressions(
    current_data: dict[str, Any],
    baseline: dict[str, Any],
    individual_threshold: float = 20.0,
    suite_threshold: float = 15.0,
) -> None:
    """
    Check for performance regressions against baseline.

    Args:
        current_data: Current performance data
        baseline: Baseline performance data
        individual_threshold: Percentage threshold for individual tests (default 20%)
        suite_threshold: Percentage threshold for suite-level metrics (default 15%)

    Raises:
        RegressionError: If regressions are detected
    """
    errors = []

    # Check individual test regressions
    baseline_tests = baseline.get("tests", {})
    current_tests = current_data.get("tests", [])

    for test in current_tests:
        nodeid = test.get("nodeid")
        if not nodeid:
            continue

        baseline_test = baseline_tests.get(nodeid)
        if not baseline_test:
            continue  # New test, no baseline to compare

        baseline_duration = baseline_test.get("duration", 0.0)
        current_duration = test.get("duration", 0.0)

        if baseline_duration == 0.0:
            continue  # Skip zero baseline

        percentage_increase = calculate_percentage_increase(baseline_duration, current_duration)

        if percentage_increase > individual_threshold:
            errors.append(
                f"Test '{nodeid}' regressed: "
                f"baseline={baseline_duration:.2f}s, "
                f"current={current_duration:.2f}s, "
                f"increase={percentage_increase:.1f}% "
                f"(threshold={individual_threshold}%)"
            )

    # Check suite-level regressions
    baseline_suites = baseline.get("suite_metrics", {})
    if baseline_suites:
        # Group current tests by module
        module_durations: dict[str, float] = {}
        for test in current_tests:
            module = test.get("module", "unknown")
            duration = test.get("duration", 0.0)
            module_durations[module] = module_durations.get(module, 0.0) + duration

        # Compare against baseline suite metrics
        for module, current_total in module_durations.items():
            baseline_suite = baseline_suites.get(module)
            if not baseline_suite:
                continue

            baseline_duration = baseline_suite.get("total_duration", 0.0)
            if baseline_duration == 0.0:
                continue

            percentage_increase = calculate_percentage_increase(baseline_duration, current_total)

            if percentage_increase > suite_threshold:
                errors.append(
                    f"Suite '{module}' regressed: "
                    f"baseline={baseline_duration:.2f}s, "
                    f"current={current_total:.2f}s, "
                    f"increase={percentage_increase:.1f}% "
                    f"(threshold={suite_threshold}%)"
                )

    # Raise error if regressions found
    if errors:
        error_msg = "Performance regressions detected:\n" + "\n".join(f"  - {e}" for e in errors)
        raise RegressionError(error_msg)
