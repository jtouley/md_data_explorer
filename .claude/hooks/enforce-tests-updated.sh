#!/usr/bin/env bash
# Enforce that tests are updated when src/clinical_analytics changes
# PostToolUse hook: the edit already happened, so exit 2 can't undo it —
# it surfaces the message to Claude as feedback to act on.
set -euo pipefail

INPUT=$(cat)
FILE_PATH=$(python3 -c "import json,sys; print(json.load(sys.stdin).get('tool_input', {}).get('file_path', ''))" <<<"$INPUT" 2>/dev/null || echo "")

case "$FILE_PATH" in
    *src/clinical_analytics/*.py) ;;
    *) exit 0 ;;
esac

# Check if we have uncommitted changes in src/clinical_analytics
SRC_CHANGES=$(git diff --name-only HEAD 2>/dev/null | grep -c "^src/clinical_analytics/" || echo "0")

# Check if we have uncommitted changes in tests/
TEST_CHANGES=$(git diff --name-only HEAD 2>/dev/null | grep -c "^tests/" || echo "0")

# Allowlist: docs, config files, comments-only changes shouldn't trigger
ALLOWLIST_PATTERN="(docs/|config/|mkdocs|\.md$|\.yaml$|\.toml$|\.json$)"
ALLOWLISTED_ONLY=$(git diff --name-only HEAD 2>/dev/null | grep -vE "$ALLOWLIST_PATTERN" | wc -l || echo "0")

# If no changes at all, allow
if [[ "$SRC_CHANGES" -eq 0 ]]; then
    exit 0
fi

# If only allowlisted files changed, allow
if [[ "$ALLOWLISTED_ONLY" -eq 0 ]]; then
    exit 0
fi

# If src changed but no test changes, warn (feedback, not a hard block)
if [[ "$SRC_CHANGES" -gt 0 ]] && [[ "$TEST_CHANGES" -eq 0 ]]; then
    cat >&2 <<EOF
🚫 Source code changed without test updates!

📂 Files changed in src/clinical_analytics/: $SRC_CHANGES
📝 Files changed in tests/: $TEST_CHANGES

✅ To proceed:
  1. Add/update tests for the behavior change
  2. Run: make test-fast
  3. Or justify in commit message if tests aren't needed

💡 Use factory fixtures from tests/conftest.py:
   - make_semantic_layer
   - make_cohort_with_categorical
   - make_multi_table_setup
EOF
    exit 2
fi

# Tests were updated - run fast tests to verify, feed failures back as feedback
if command -v make &> /dev/null && grep -q "test-fast:" Makefile; then
    if ! TEST_OUTPUT=$(make test-fast 2>&1); then
        echo "❌ Fast tests failed!" >&2
        echo "$TEST_OUTPUT" | tail -30 >&2
        exit 2
    fi
fi

exit 0
