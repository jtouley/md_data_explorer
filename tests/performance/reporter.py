"""Report generation for performance tracking data."""

import json
from typing import Any


def generate_markdown_report(performance_data: dict[str, Any]) -> str:
    """
    Generate markdown report from performance data.

    Args:
        performance_data: Performance data dictionary

    Returns:
        Markdown-formatted report string
    """
    report_lines = []
    report_lines.append("# Performance Report\n")
    report_lines.append(f"**Run ID**: {performance_data.get('run_id', 'N/A')}\n")

    # Summary section
    summary = performance_data.get("summary", {})
    report_lines.append("\n## Summary\n")
    report_lines.append(f"- **Total Tests**: {summary.get('total_tests', 0)}")
    report_lines.append(f"- **Slow Tests** (>30s): {summary.get('slow_tests', 0)}")
    report_lines.append(f"- **Total Duration**: {summary.get('total_duration', 0.0):.2f}s")
    report_lines.append(f"- **Mean Duration**: {summary.get('average_duration', 0.0):.2f}s")
    report_lines.append(f"- **Min Duration**: {summary.get('min_duration', 0.0):.2f}s")
    report_lines.append(f"- **Max Duration**: {summary.get('max_duration', 0.0):.2f}s\n")

    # Slowest tests section
    tests = performance_data.get("tests", [])
    if tests:
        # Sort by duration (descending)
        sorted_tests = sorted(tests, key=lambda x: x.get("duration", 0), reverse=True)

        # Show slowest tests (>30s)
        slow_tests = [t for t in sorted_tests if t.get("duration", 0) > 30.0]
        if slow_tests:
            report_lines.append("## Slowest Tests (>30s)\n")
            report_lines.append("| Test | Duration (s) | Module | Markers |")
            report_lines.append("|------|--------------|--------|---------|")

            for test in slow_tests[:20]:  # Top 20 slowest
                nodeid = test.get("nodeid", "N/A")
                duration = test.get("duration", 0)
                module = test.get("module", "N/A")
                markers = ", ".join(test.get("markers", []))
                report_lines.append(f"| {nodeid} | {duration:.2f} | {module} | {markers} |")

            report_lines.append("")

        # Show top 10 slowest overall
        report_lines.append("## Top 10 Slowest Tests\n")
        report_lines.append("| Test | Duration (s) | Module | Markers |")
        report_lines.append("|------|--------------|--------|---------|")

        for test in sorted_tests[:10]:
            nodeid = test.get("nodeid", "N/A")
            duration = test.get("duration", 0)
            module = test.get("module", "N/A")
            markers = ", ".join(test.get("markers", []))
            report_lines.append(f"| {nodeid} | {duration:.2f} | {module} | {markers} |")

        report_lines.append("")

    return "\n".join(report_lines)


def generate_json_report(performance_data: dict[str, Any]) -> str:
    """
    Generate JSON report from performance data.

    Args:
        performance_data: Performance data dictionary

    Returns:
        JSON-formatted report string
    """
    return json.dumps(performance_data, indent=2)
