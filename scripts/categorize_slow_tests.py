#!/usr/bin/env python3
"""
Automated test categorization script (Phase 3).

Verifies test categorization correctness:
1. Slow tests (>30s) without @pytest.mark.slow
2. Fast tests (<1s) incorrectly marked as @pytest.mark.slow
3. Integration tests (>10s) without @pytest.mark.integration
4. Unit tests (<1s) incorrectly marked as @pytest.mark.integration

Generates actionable reports to improve test categorization.

Usage:
    python scripts/categorize_slow_tests.py [--threshold SECONDS] [--report]
"""

import argparse
import json
import sys
from pathlib import Path


def load_performance_data(data_file: Path) -> dict:
    """Load performance data from JSON file."""
    if not data_file.exists():
        print(f"Error: Performance data file not found: {data_file}")
        print("Run 'make test-performance' first to generate performance data.")
        sys.exit(1)

    with open(data_file) as f:
        return json.load(f)


def categorize_slow_tests(data_file: Path, threshold: float = 30.0) -> list[dict]:
    """
    Identify uncategorized slow tests.

    Args:
        data_file: Path to .performance_data.json
        threshold: Duration threshold in seconds (default: 30.0)

    Returns:
        List of test dicts that are slow but not marked with @pytest.mark.slow
    """
    data = load_performance_data(data_file)
    uncategorized = []

    for test in data.get("tests", []):
        duration = test.get("duration", 0)
        markers = test.get("markers", [])

        # Check if test is slow but not categorized
        if duration > threshold and "slow" not in markers:
            uncategorized.append(test)

    # Sort by duration (slowest first)
    uncategorized.sort(key=lambda x: x.get("duration", 0), reverse=True)

    return uncategorized


def find_incorrectly_marked_slow_tests(data_file: Path, fast_threshold: float = 1.0) -> list[dict]:
    """
    Find fast tests incorrectly marked as slow.

    Args:
        data_file: Path to .performance_data.json
        fast_threshold: Duration threshold for "fast" tests in seconds (default: 1.0)

    Returns:
        List of test dicts that are fast but marked with @pytest.mark.slow
    """
    data = load_performance_data(data_file)
    incorrectly_marked = []

    for test in data.get("tests", []):
        duration = test.get("duration", 0)
        markers = test.get("markers", [])

        # Check if test is fast but marked as slow
        if duration < fast_threshold and "slow" in markers:
            incorrectly_marked.append(test)

    # Sort by duration (fastest first)
    incorrectly_marked.sort(key=lambda x: x.get("duration", 0))

    return incorrectly_marked


def find_uncategorized_integration_tests(data_file: Path, integration_threshold: float = 10.0) -> list[dict]:
    """
    Find integration tests without @pytest.mark.integration marker.

    Heuristic: Tests >10s are likely integration tests (using real services).

    Args:
        data_file: Path to .performance_data.json
        integration_threshold: Duration threshold for integration tests in seconds (default: 10.0)

    Returns:
        List of test dicts that are likely integration tests but not marked
    """
    data = load_performance_data(data_file)
    uncategorized = []

    for test in data.get("tests", []):
        duration = test.get("duration", 0)
        markers = test.get("markers", [])

        # Check if test is likely integration but not marked
        if duration > integration_threshold and "integration" not in markers:
            uncategorized.append(test)

    # Sort by duration (slowest first)
    uncategorized.sort(key=lambda x: x.get("duration", 0), reverse=True)

    return uncategorized


def find_incorrectly_marked_integration_tests(data_file: Path, fast_threshold: float = 1.0) -> list[dict]:
    """
    Find unit tests incorrectly marked as integration.

    Args:
        data_file: Path to .performance_data.json
        fast_threshold: Duration threshold for "fast" tests in seconds (default: 1.0)

    Returns:
        List of test dicts that are fast but marked with @pytest.mark.integration
    """
    data = load_performance_data(data_file)
    incorrectly_marked = []

    for test in data.get("tests", []):
        duration = test.get("duration", 0)
        markers = test.get("markers", [])

        # Check if test is fast but marked as integration
        if duration < fast_threshold and "integration" in markers:
            incorrectly_marked.append(test)

    # Sort by duration (fastest first)
    incorrectly_marked.sort(key=lambda x: x.get("duration", 0))

    return incorrectly_marked


def generate_report(
    data_file: Path,
    slow_threshold: float = 30.0,
    fast_threshold: float = 1.0,
    integration_threshold: float = 10.0,
) -> str:
    """
    Generate comprehensive categorization report.

    Args:
        data_file: Path to .performance_data.json
        slow_threshold: Duration threshold for slow tests in seconds (default: 30.0)
        fast_threshold: Duration threshold for fast tests in seconds (default: 1.0)
        integration_threshold: Duration threshold for integration tests in seconds (default: 10.0)

    Returns:
        Formatted report string
    """
    report_lines = []

    # 1. Uncategorized slow tests
    uncategorized_slow = categorize_slow_tests(data_file, slow_threshold)
    if uncategorized_slow:
        report_lines.extend([
            f"Uncategorized Slow Tests (>{slow_threshold}s without @pytest.mark.slow)",
            "=" * 80,
            "",
        ])
        for test in uncategorized_slow:
            nodeid = test.get("nodeid", "unknown")
            duration = test.get("duration", 0)
            module = test.get("module", "unknown")
            report_lines.append(f"Test: {nodeid}")
            report_lines.append(f"  Duration: {duration:.2f}s")
            report_lines.append(f"  Module: {module}")
            report_lines.append(f"  Recommendation: Add @pytest.mark.slow marker")
            report_lines.append("")
        report_lines.append(f"Total: {len(uncategorized_slow)} uncategorized slow test(s)")
        report_lines.append("")
    else:
        report_lines.append(f"✅ No uncategorized slow tests found (threshold: {slow_threshold}s)\n")

    # 2. Fast tests incorrectly marked as slow
    incorrectly_slow = find_incorrectly_marked_slow_tests(data_file, fast_threshold)
    if incorrectly_slow:
        report_lines.extend([
            f"Incorrectly Marked Slow Tests (<{fast_threshold}s with @pytest.mark.slow)",
            "=" * 80,
            "",
        ])
        for test in incorrectly_slow:
            nodeid = test.get("nodeid", "unknown")
            duration = test.get("duration", 0)
            module = test.get("module", "unknown")
            report_lines.append(f"Test: {nodeid}")
            report_lines.append(f"  Duration: {duration:.2f}s")
            report_lines.append(f"  Module: {module}")
            report_lines.append(f"  Recommendation: Remove @pytest.mark.slow marker")
            report_lines.append("")
        report_lines.append(f"Total: {len(incorrectly_slow)} incorrectly marked slow test(s)")
        report_lines.append("")
    else:
        report_lines.append(f"✅ No incorrectly marked slow tests found (threshold: {fast_threshold}s)\n")

    # 3. Uncategorized integration tests
    uncategorized_integration = find_uncategorized_integration_tests(data_file, integration_threshold)
    if uncategorized_integration:
        report_lines.extend([
            f"Uncategorized Integration Tests (>{integration_threshold}s without @pytest.mark.integration)",
            "=" * 80,
            "",
        ])
        for test in uncategorized_integration:
            nodeid = test.get("nodeid", "unknown")
            duration = test.get("duration", 0)
            module = test.get("module", "unknown")
            report_lines.append(f"Test: {nodeid}")
            report_lines.append(f"  Duration: {duration:.2f}s")
            report_lines.append(f"  Module: {module}")
            report_lines.append(f"  Recommendation: Add @pytest.mark.integration marker (if using real services)")
            report_lines.append("")
        report_lines.append(f"Total: {len(uncategorized_integration)} uncategorized integration test(s)")
        report_lines.append("")
    else:
        report_lines.append(f"✅ No uncategorized integration tests found (threshold: {integration_threshold}s)\n")

    # 4. Fast tests incorrectly marked as integration
    incorrectly_integration = find_incorrectly_marked_integration_tests(data_file, fast_threshold)
    if incorrectly_integration:
        report_lines.extend([
            f"Incorrectly Marked Integration Tests (<{fast_threshold}s with @pytest.mark.integration)",
            "=" * 80,
            "",
        ])
        for test in incorrectly_integration:
            nodeid = test.get("nodeid", "unknown")
            duration = test.get("duration", 0)
            module = test.get("module", "unknown")
            report_lines.append(f"Test: {nodeid}")
            report_lines.append(f"  Duration: {duration:.2f}s")
            report_lines.append(f"  Module: {module}")
            report_lines.append(f"  Recommendation: Remove @pytest.mark.integration marker (likely unit test)")
            report_lines.append("")
        report_lines.append(f"Total: {len(incorrectly_integration)} incorrectly marked integration test(s)")
        report_lines.append("")
    else:
        report_lines.append(f"✅ No incorrectly marked integration tests found (threshold: {fast_threshold}s)\n")

    # Summary
    total_issues = (
        len(uncategorized_slow)
        + len(incorrectly_slow)
        + len(uncategorized_integration)
        + len(incorrectly_integration)
    )
    if total_issues > 0:
        report_lines.extend([
            "=" * 80,
            f"Summary: {total_issues} categorization issue(s) found",
            "",
            "To fix:",
            "  - Add @pytest.mark.slow to slow tests (>30s)",
            "  - Remove @pytest.mark.slow from fast tests (<1s)",
            "  - Add @pytest.mark.integration to integration tests (>10s, using real services)",
            "  - Remove @pytest.mark.integration from unit tests (<1s, using mocks)",
        ])
    else:
        report_lines.append("✅ All tests correctly categorized!")

    return "\n".join(report_lines)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Categorize slow tests using performance tracking data"
    )
    parser.add_argument(
        "--data-file",
        type=Path,
        default=Path("tests/.performance_data.json"),
        help="Path to performance data JSON file (default: tests/.performance_data.json)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=30.0,
        help="Duration threshold in seconds (default: 30.0)",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Generate and print report",
    )

    args = parser.parse_args()

    # Generate report
    report = generate_report(args.data_file, args.threshold)
    print(report)

    # Exit with error code if any categorization issues found
    uncategorized_slow = categorize_slow_tests(args.data_file, args.threshold)
    incorrectly_slow = find_incorrectly_marked_slow_tests(args.data_file)
    uncategorized_integration = find_uncategorized_integration_tests(args.data_file)
    incorrectly_integration = find_incorrectly_marked_integration_tests(args.data_file)

    total_issues = (
        len(uncategorized_slow)
        + len(incorrectly_slow)
        + len(uncategorized_integration)
        + len(incorrectly_integration)
    )

    if total_issues > 0:
        sys.exit(1)  # Exit with error to indicate categorization issues found
    else:
        sys.exit(0)  # Exit successfully if all tests correctly categorized


if __name__ == "__main__":
    main()

