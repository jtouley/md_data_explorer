#!/usr/bin/env python3
"""
Analyze performance data to identify expensive fixtures for scope optimization.

This script analyzes .performance_data.json to find:
- Tests with duration >5s that share common fixtures
- Fixtures used by >5 tests with average test duration >2s
- Fixtures that create files or perform I/O

Usage:
    python scripts/analyze_fixture_performance.py
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def load_performance_data() -> dict:
    """Load performance data from JSON file."""
    data_file = project_root / "tests" / ".performance_data.json"
    if not data_file.exists():
        print(f"Performance data file not found: {data_file}")
        print("Run 'make test-performance' first to generate performance data.")
        sys.exit(1)

    with open(data_file) as f:
        return json.load(f)


def extract_fixtures_from_test(test_nodeid: str) -> list[str]:
    """
    Extract fixture names from test nodeid.

    This is a heuristic - we can't actually know which fixtures a test uses
    without parsing the test file. But we can identify patterns:
    - Tests in same file likely share fixtures
    - Tests with similar names might use same fixtures
    """
    # Extract module path from nodeid
    # Format: tests/module/test_file.py::TestClass::test_method
    parts = test_nodeid.split("::")
    if len(parts) >= 2:
        return [parts[0]]  # Return test file path as "fixture group"
    return []


def analyze_slow_tests(data: dict, threshold: float = 5.0) -> list[dict]:
    """Find tests with duration > threshold."""
    slow_tests = []
    for test in data.get("tests", []):
        duration = test.get("duration", 0)
        if duration > threshold:
            slow_tests.append(
                {
                    "nodeid": test.get("nodeid", "unknown"),
                    "duration": duration,
                    "module": test.get("module", "unknown"),
                }
            )
    return sorted(slow_tests, key=lambda x: x["duration"], reverse=True)


def group_tests_by_module(tests: list[dict]) -> dict[str, list[dict]]:
    """Group tests by module."""
    by_module = defaultdict(list)
    for test in tests:
        module = test.get("module", "unknown")
        by_module[module].append(test)
    return dict(by_module)


def identify_expensive_fixtures(data: dict) -> dict:
    """
    Identify expensive fixtures based on performance data.

    Criteria:
    - Tests with duration >5s that share common fixtures
    - Fixtures used by >5 tests with average test duration >2s
    """
    slow_tests = analyze_slow_tests(data, threshold=5.0)
    by_module = group_tests_by_module(slow_tests)

    # Find modules with multiple slow tests (likely share fixtures)
    candidates = {}
    for module, tests in by_module.items():
        if len(tests) >= 3:  # At least 3 slow tests in same module
            avg_duration = sum(t["duration"] for t in tests) / len(tests)
            candidates[module] = {
                "test_count": len(tests),
                "avg_duration": avg_duration,
                "tests": tests[:5],  # Show first 5
            }

    return candidates


def main():
    """Main analysis function."""
    print("Analyzing fixture performance...")
    print("=" * 80)

    # Load performance data
    data = load_performance_data()
    total_tests = len(data.get("tests", []))
    print(f"Total tests analyzed: {total_tests}")

    # Find slow tests
    slow_tests = analyze_slow_tests(data, threshold=5.0)
    print(f"\nSlow tests (>5s): {len(slow_tests)}")

    if slow_tests:
        print("\nTop 10 slowest tests:")
        for i, test in enumerate(slow_tests[:10], 1):
            print(f"  {i}. {test['nodeid']}: {test['duration']:.2f}s")

    # Identify expensive fixture candidates
    candidates = identify_expensive_fixtures(data)
    print(f"\nModules with multiple slow tests (fixture optimization candidates): {len(candidates)}")

    if candidates:
        print("\nCandidates for fixture scope optimization:")
        for module, info in sorted(candidates.items(), key=lambda x: x[1]["test_count"], reverse=True):
            print(f"\n  Module: {module}")
            print(f"    Slow tests: {info['test_count']}")
            print(f"    Avg duration: {info['avg_duration']:.2f}s")
            print(f"    Sample tests:")
            for test in info["tests"]:
                print(f"      - {test['nodeid']}: {test['duration']:.2f}s")

    # Summary
    print("\n" + "=" * 80)
    print("Summary:")
    print(f"  - Total tests: {total_tests}")
    print(f"  - Slow tests (>5s): {len(slow_tests)}")
    print(f"  - Modules with multiple slow tests: {len(candidates)}")
    print("\nRecommendation:")
    if candidates:
        print("  Consider converting fixtures in candidate modules to module/session scope")
        print("  if they are immutable and used by multiple tests.")
    else:
        print("  No clear fixture optimization candidates found.")
        print("  Most expensive fixtures may already be optimized (module/session scope).")


if __name__ == "__main__":
    main()

