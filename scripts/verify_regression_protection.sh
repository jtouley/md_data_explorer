#!/bin/bash
# Regression Protection Verification Script
# Compares current test results against baseline to detect regressions

set -e

BASELINE_FILE="tests/docs/baseline_test_results.txt"
CURRENT_FILE="tests/docs/current_test_results.txt"

if [ ! -f "$BASELINE_FILE" ]; then
    echo "ERROR: Baseline file not found: $BASELINE_FILE"
    echo "Run 'make test 2>&1 | tee tests/docs/baseline_test_results.txt' first to create baseline"
    exit 1
fi

echo "Running tests to compare against baseline..."
make test 2>&1 | tee "$CURRENT_FILE"

BASELINE_PASSED=$(grep -c "PASSED" "$BASELINE_FILE" 2>/dev/null || echo "0")
BASELINE_FAILED=$(grep -c "FAILED" "$BASELINE_FILE" 2>/dev/null || echo "0")
CURRENT_PASSED=$(grep -c "PASSED" "$CURRENT_FILE" 2>/dev/null || echo "0")
CURRENT_FAILED=$(grep -c "FAILED" "$CURRENT_FILE" 2>/dev/null || echo "0")

echo ""
echo "Baseline: $BASELINE_PASSED passed, $BASELINE_FAILED failed"
echo "Current:  $CURRENT_PASSED passed, $CURRENT_FAILED failed"

if [ "$CURRENT_FAILED" -gt "$BASELINE_FAILED" ]; then
    echo "ERROR: New test failures detected!"
    echo "Baseline failures: $BASELINE_FAILED"
    echo "Current failures:  $CURRENT_FAILED"
    exit 1
fi

if [ "$CURRENT_PASSED" -lt "$BASELINE_PASSED" ]; then
    echo "WARNING: Fewer passing tests than baseline"
    echo "Baseline passed: $BASELINE_PASSED"
    echo "Current passed:  $CURRENT_PASSED"
    # Don't fail, but warn
fi

echo "✓ Regression protection check passed"
exit 0
