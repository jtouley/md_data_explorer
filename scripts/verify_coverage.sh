#!/bin/bash
# Coverage Verification Script
# Compares current coverage against baseline to ensure no significant decrease

set -e

BASELINE_FILE="tests/docs/baseline_coverage.txt"
CURRENT_FILE="tests/docs/current_coverage.txt"
THRESHOLD=2  # Maximum allowed coverage decrease (percentage)

if [ ! -f "$BASELINE_FILE" ]; then
    echo "ERROR: Baseline coverage file not found: $BASELINE_FILE"
    echo "Run 'make test-cov 2>&1 | tee tests/docs/baseline_coverage.txt' first to create baseline"
    exit 1
fi

echo "Running tests with coverage to compare against baseline..."
make test-cov 2>&1 | tee "$CURRENT_FILE"

# Extract coverage percentage from output
# Format: "TOTAL    XX%    YYY    ZZZ"
BASELINE_COVERAGE=$(grep -oP 'TOTAL\s+\K\d+' "$BASELINE_FILE" | head -1 || echo "0")
CURRENT_COVERAGE=$(grep -oP 'TOTAL\s+\K\d+' "$CURRENT_FILE" | head -1 || echo "0")

if [ "$BASELINE_COVERAGE" = "0" ] || [ -z "$BASELINE_COVERAGE" ]; then
    echo "ERROR: Could not extract baseline coverage from $BASELINE_FILE"
    exit 1
fi

if [ "$CURRENT_COVERAGE" = "0" ] || [ -z "$CURRENT_COVERAGE" ]; then
    echo "ERROR: Could not extract current coverage from $CURRENT_FILE"
    exit 1
fi

echo ""
echo "Baseline coverage: ${BASELINE_COVERAGE}%"
echo "Current coverage:   ${CURRENT_COVERAGE}%"

COVERAGE_DIFF=$((BASELINE_COVERAGE - CURRENT_COVERAGE))

if [ "$COVERAGE_DIFF" -gt "$THRESHOLD" ]; then
    echo "ERROR: Coverage decreased by ${COVERAGE_DIFF}% (threshold: ${THRESHOLD}%)"
    exit 1
fi

if [ "$COVERAGE_DIFF" -gt 0 ]; then
    echo "WARNING: Coverage decreased by ${COVERAGE_DIFF}% (within threshold)"
fi

echo "✓ Coverage verification passed (within ${THRESHOLD}% threshold)"
exit 0
