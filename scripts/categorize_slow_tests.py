#!/usr/bin/env python3
"""
Automated test categorization script (Phase 3).

Identifies uncategorized slow tests (>30s without @pytest.mark.slow)
and generates actionable reports to improve test-fast effectiveness.

Usage:
    python scripts/categorize_slow_tests.py [--threshold SECONDS] [--auto-fix]
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


def generate_report(data_file: Path, threshold: float = 30.0) -> str:
    """
    Generate report of uncategorized slow tests.

    Args:
        data_file: Path to .performance_data.json
        threshold: Duration threshold in seconds (default: 30.0)

    Returns:
        Formatted report string
    """
    uncategorized = categorize_slow_tests(data_file, threshold)

    if not uncategorized:
        return f"✅ No uncategorized slow tests found (threshold: {threshold}s)\n"

    report_lines = [
        f"Uncategorized Slow Tests (>{threshold}s without @pytest.mark.slow)",
        "=" * 80,
        "",
    ]

    for test in uncategorized:
        nodeid = test.get("nodeid", "unknown")
        duration = test.get("duration", 0)
        module = test.get("module", "unknown")

        report_lines.append(f"Test: {nodeid}")
        report_lines.append(f"  Duration: {duration:.2f}s")
        report_lines.append(f"  Module: {module}")
        report_lines.append(f"  Recommendation: Add @pytest.mark.slow marker")
        report_lines.append("")

    report_lines.append(f"Total: {len(uncategorized)} uncategorized slow test(s)")
    report_lines.append("")
    report_lines.append("To fix, add @pytest.mark.slow to these tests:")
    report_lines.append("  @pytest.mark.slow")
    report_lines.append("  def test_example():")

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

    # Exit with error code if uncategorized tests found
    uncategorized = categorize_slow_tests(args.data_file, args.threshold)
    if uncategorized:
        sys.exit(1)  # Exit with error to indicate uncategorized tests found
    else:
        sys.exit(0)  # Exit successfully if all slow tests are categorized


if __name__ == "__main__":
    main()

