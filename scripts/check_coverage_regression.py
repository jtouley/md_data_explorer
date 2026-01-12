#!/usr/bin/env python3
"""Check coverage regression against baseline.

Exit codes:
    0 - Coverage passes all checks
    1 - Coverage below threshold or regression detected
"""

import json
import sys
from pathlib import Path

BASELINE_FILE = Path("tests/.coverage_baseline.json")
CURRENT_FILE = Path("coverage.json")  # Matches --cov-report=json:coverage.json
THRESHOLD = 67.0  # Minimum coverage percentage (current baseline, target 95%)
REGRESSION_TOLERANCE = 0.5  # Max allowed drop from baseline


def main() -> int:
    if not CURRENT_FILE.exists():
        print(f"ERROR: {CURRENT_FILE} not found. Run: make test-cov-diff")
        return 1

    with open(CURRENT_FILE) as f:
        current = json.load(f)
    current_pct = current["totals"]["percent_covered"]

    # Check absolute threshold
    if current_pct < THRESHOLD:
        print(f"FAIL: Coverage {current_pct:.1f}% is below {THRESHOLD}% threshold")
        return 1

    # Check regression if baseline exists
    if BASELINE_FILE.exists():
        with open(BASELINE_FILE) as f:
            baseline = json.load(f)
        baseline_pct = baseline["totals"]["percent_covered"]

        if current_pct < baseline_pct - REGRESSION_TOLERANCE:
            print(f"FAIL: Coverage dropped {baseline_pct:.1f}% -> {current_pct:.1f}%")
            print(
                f"      Regression of {baseline_pct - current_pct:.1f}% " f"exceeds {REGRESSION_TOLERANCE}% tolerance"
            )
            return 1
        print(f"OK: Coverage {current_pct:.1f}% (baseline: {baseline_pct:.1f}%)")
    else:
        print(f"OK: Coverage {current_pct:.1f}% (no baseline)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
