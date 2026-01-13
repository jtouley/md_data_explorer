#!/usr/bin/env bash
# Enforce that tests are updated when src/clinical_analytics changes
# Blocks edits if source code changes without corresponding test updates

set -euo pipefail

# Check if we have uncommitted changes in src/clinical_analytics
SRC_CHANGES=$(git diff --name-only HEAD 2>/dev/null | grep -c "^src/clinical_analytics/" || echo "0")

# Check if we have uncommitted changes in tests/
TEST_CHANGES=$(git diff --name-only HEAD 2>/dev/null | grep -c "^tests/" || echo "0")

# Allowlist: docs, config files, comments-only changes shouldn't trigger
ALLOWLIST_PATTERN="(docs/|config/|mkdocs|\.md$|\.yaml$|\.toml$|\.json$)"
ALLOWLISTED_ONLY=$(git diff --name-only HEAD 2>/dev/null | grep -vE "$ALLOWLIST_PATTERN" | wc -l || echo "0")

# If no changes at all, allow
if [[ "$SRC_CHANGES" -eq 0 ]]; then
    echo '{"block": false}'
    exit 0
fi

# If only allowlisted files changed, allow
if [[ "$ALLOWLISTED_ONLY" -eq 0 ]]; then
    echo '{"block": false, "message": "ℹ️  Only docs/config changed, skipping test enforcement"}'
    exit 0
fi

# If src changed but no test changes, block
if [[ "$SRC_CHANGES" -gt 0 ]] && [[ "$TEST_CHANGES" -eq 0 ]]; then
    cat <<EOF
{
  "block": true,
  "message": "🚫 Source code changed without test updates!\n\n📂 Files changed in src/clinical_analytics/: $SRC_CHANGES\n📝 Files changed in tests/: $TEST_CHANGES\n\n✅ To proceed:\n  1. Add/update tests for the behavior change\n  2. Run: make test-fast\n  3. Or justify in commit message if tests aren't needed\n\n💡 Use factory fixtures from tests/conftest.py:\n   - make_semantic_layer\n   - make_cohort_with_categorical\n   - make_multi_table_setup"
}
EOF
    exit 0
fi

# If tests were updated, run fast tests to verify
echo '{"block": false, "message": "✅ Tests updated - running fast test suite..."}'

# Run fast tests (non-blocking, just informative)
if command -v make &> /dev/null && grep -q "test-fast:" Makefile; then
    TEST_OUTPUT=$(make test-fast 2>&1 || echo "FAILED")
    if echo "$TEST_OUTPUT" | grep -q "FAILED"; then
        # Extract last 30 lines of output
        LAST_LINES=$(echo "$TEST_OUTPUT" | tail -30)
        cat <<EOF
{
  "block": true,
  "message": "❌ Fast tests failed!\n\nLast 30 lines:\n$LAST_LINES\n\nFix tests before proceeding."
}
EOF
    else
        echo '{"block": false, "message": "✅ Fast tests passed"}'
    fi
else
    echo '{"block": false, "message": "⚠️  Could not run tests (make test-fast not found)"}'
fi
