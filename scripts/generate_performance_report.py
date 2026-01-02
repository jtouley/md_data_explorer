#!/usr/bin/env python3
"""CLI tool for generating performance reports and managing baselines."""

import argparse
import json
import sys
from datetime import date
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "tests"))

from performance.reporter import generate_json_report, generate_markdown_report
from performance.storage import load_baseline, load_performance_data, save_baseline

# Import categorization script
sys.path.insert(0, str(project_root))
from scripts.categorize_slow_tests import categorize_slow_tests, generate_report as generate_categorization_report


def get_default_paths() -> tuple[Path, Path]:
    """Get default paths for performance data and baseline files."""
    project_root = Path(__file__).parent.parent
    tests_dir = project_root / "tests"
    data_file = tests_dir / ".performance_data.json"
    baseline_file = tests_dir / ".performance_baseline.json"
    return data_file, baseline_file


def create_baseline(data_file: Path, baseline_file: Path, individual_threshold: float = 20.0, suite_threshold: float = 15.0) -> None:
    """
    Create baseline from performance data.

    Args:
        data_file: Path to performance data file
        baseline_file: Path to save baseline file
        individual_threshold: Percentage threshold for individual tests
        suite_threshold: Percentage threshold for suite-level metrics
    """
    if not data_file.exists():
        print(f"Error: Performance data file not found: {data_file}")
        print("Run tests with '--track-performance' flag first.")
        sys.exit(1)

    data = load_performance_data(data_file)
    if not data.get("tests"):
        print("Error: No test data found in performance file.")
        sys.exit(1)

    # Create baseline structure
    baseline = {
        "baseline_date": date.today().isoformat(),
        "tests": {},
        "suite_metrics": {},
    }

    # Convert test results to baseline format
    for test in data.get("tests", []):
        nodeid = test.get("nodeid")
        if not nodeid:
            continue
        duration = test.get("duration", 0.0)
        baseline["tests"][nodeid] = {
            "duration": duration,
            "threshold": duration * (1 + individual_threshold / 100.0),
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
            "threshold": total_duration * (1 + suite_threshold / 100.0),
        }

    save_baseline(baseline, baseline_file)
    print(f"Baseline created: {baseline_file}")
    print(f"  Tests: {len(baseline['tests'])}")
    print(f"  Suites: {len(baseline['suite_metrics'])}")


def generate_report(
    data_file: Path,
    baseline_file: Path | None = None,
    format: str = "markdown",
    output_file: Path | None = None,
    compare_baseline: bool = False,
) -> None:
    """
    Generate performance report.

    Args:
        data_file: Path to performance data file
        baseline_file: Path to baseline file (optional)
        format: Output format ('markdown' or 'json')
        output_file: Path to save report (optional, prints to stdout if not provided)
        compare_baseline: Whether to compare against baseline
    """
    if not data_file.exists():
        print(f"Error: Performance data file not found: {data_file}")
        sys.exit(1)

    data = load_performance_data(data_file)

    if format == "json":
        report = generate_json_report(data)
    else:
        report = generate_markdown_report(data)

        # Add baseline comparison if requested
        if compare_baseline and baseline_file and baseline_file.exists():
            from performance.regression import check_regressions

            baseline = load_baseline(baseline_file)
            try:
                check_regressions(data, baseline)
                report += "\n## Regression Check\n\n✅ No regressions detected.\n"
            except Exception as e:
                report += f"\n## Regression Check\n\n❌ Regressions detected:\n\n```\n{e}\n```\n"

        # Add comprehensive categorization section
        report += "\n## Test Categorization Verification\n\n"
        categorization_report = generate_categorization_report(
            data_file, slow_threshold=30.0, fast_threshold=1.0, integration_threshold=10.0
        )
        report += categorization_report
        report += "\n"

    if output_file:
        output_file.write_text(report, encoding="utf-8")
        print(f"Report saved to: {output_file}")
    else:
        print(report)


def update_docs(data_file: Path, baseline_file: Path | None = None) -> None:
    """
    Update PERFORMANCE.md with current benchmarks.

    Args:
        data_file: Path to performance data file
        baseline_file: Path to baseline file (optional)
    """
    project_root = Path(__file__).parent.parent
    docs_file = project_root / "tests" / "PERFORMANCE.md"

    if not data_file.exists():
        print(f"Error: Performance data file not found: {data_file}")
        sys.exit(1)

    data = load_performance_data(data_file)
    report = generate_markdown_report(data)

    # Read existing docs
    if docs_file.exists():
        content = docs_file.read_text(encoding="utf-8")
    else:
        content = "# Test Performance Documentation\n\n"

    # Find or create automated tracking section
    if "## Automated Performance Tracking" in content:
        # Replace existing section
        start_marker = "## Automated Performance Tracking"
        end_marker = "## "
        start_idx = content.find(start_marker)
        if start_idx != -1:
            # Find next section
            next_section_idx = content.find(end_marker, start_idx + len(start_marker))
            if next_section_idx != -1:
                content = content[:start_idx] + f"## Automated Performance Tracking\n\n{report}\n\n" + content[next_section_idx:]
            else:
                content = content[:start_idx] + f"## Automated Performance Tracking\n\n{report}\n"
    else:
        # Append new section
        content += f"\n## Automated Performance Tracking\n\n{report}\n"

    docs_file.write_text(content, encoding="utf-8")
    print(f"Updated: {docs_file}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Generate performance reports and manage baselines")
    parser.add_argument(
        "--data-file",
        type=Path,
        help="Path to performance data file (default: tests/.performance_data.json)",
    )
    parser.add_argument(
        "--baseline-file",
        type=Path,
        help="Path to baseline file (default: tests/.performance_baseline.json)",
    )
    parser.add_argument("--format", choices=["markdown", "json"], default="markdown", help="Output format")
    parser.add_argument("--output", type=Path, help="Output file path (default: stdout)")
    parser.add_argument("--compare-baseline", action="store_true", help="Compare against baseline")
    parser.add_argument("--create-baseline", action="store_true", help="Create baseline from current performance data")
    parser.add_argument("--update-docs", action="store_true", help="Update PERFORMANCE.md with current benchmarks")
    parser.add_argument("--individual-threshold", type=float, default=20.0, help="Individual test threshold percentage")
    parser.add_argument("--suite-threshold", type=float, default=15.0, help="Suite threshold percentage")

    args = parser.parse_args()

    # Get default paths
    default_data_file, default_baseline_file = get_default_paths()
    data_file = args.data_file or default_data_file
    baseline_file = args.baseline_file or default_baseline_file

    if args.create_baseline:
        create_baseline(data_file, baseline_file, args.individual_threshold, args.suite_threshold)
    elif args.update_docs:
        update_docs(data_file, baseline_file)
    else:
        generate_report(data_file, baseline_file, args.format, args.output, args.compare_baseline)


if __name__ == "__main__":
    main()

