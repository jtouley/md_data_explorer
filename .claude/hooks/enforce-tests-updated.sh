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

# Changed paths vs HEAD, including untracked files (a brand-new test file counts).
# `grep -c` prints 0 and exits 1 on no match, so `|| true` (not `|| echo 0`) keeps one number.
CHANGED=$(git status --porcelain --untracked-files=all 2>/dev/null | cut -c4- || true)
SRC_CHANGES=$(grep -c "^src/clinical_analytics/" <<<"$CHANGED" || true)
TEST_CHANGES=$(grep -c "^tests/" <<<"$CHANGED" || true)

if [[ "$SRC_CHANGES" -eq 0 ]] || [[ "$TEST_CHANGES" -gt 0 ]]; then
    exit 0
fi

# Feedback only: running the test suite here would cost minutes on every edit.
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
