# Pull Request Review Command

Review current branch changes against main using the code-reviewer agent checklist.

## What This Does

1. **Shows diff**: Displays all changes in current branch vs main
2. **Applies checklist**: Uses `.claude/agents/code-reviewer.md` criteria
3. **Flags missing tests**: Explicitly calls out test coverage gaps
4. **Suggests improvements**: Provides actionable feedback

## Usage

Run this command before creating a pull request to self-review changes.

## Implementation

```bash
#!/usr/bin/env bash
set -euo pipefail

echo "📋 Pull Request Review"
echo ""

# Get current branch
CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
BASE_BRANCH=${1:-main}

echo "Branch: $CURRENT_BRANCH"
echo "Base: $BASE_BRANCH"
echo ""

# Check if branch exists
if ! git rev-parse --verify "$BASE_BRANCH" &> /dev/null; then
    echo "❌ Base branch '$BASE_BRANCH' not found"
    exit 1
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📊 Change Summary"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Files changed
echo "Files changed:"
git diff --name-status "$BASE_BRANCH"...HEAD
echo ""

# Stats
COMMITS=$(git rev-list --count "$BASE_BRANCH"..HEAD)
FILES_CHANGED=$(git diff --name-only "$BASE_BRANCH"...HEAD | wc -l)
SRC_FILES=$(git diff --name-only "$BASE_BRANCH"...HEAD | grep -c "^src/" || echo "0")
TEST_FILES=$(git diff --name-only "$BASE_BRANCH"...HEAD | grep -c "^tests/" || echo "0")

echo "📈 Stats:"
echo "  - Commits: $COMMITS"
echo "  - Files changed: $FILES_CHANGED"
echo "  - Source files: $SRC_FILES"
echo "  - Test files: $TEST_FILES"
echo ""

# Check for missing tests
if [[ $SRC_FILES -gt 0 ]] && [[ $TEST_FILES -eq 0 ]]; then
    echo "⚠️  WARNING: Source files changed but no test files modified!"
    echo "   Consider adding tests for new/modified behavior."
    echo ""
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🔍 Code Review Checklist"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Show the diff for Claude to review
echo "Full diff (for review):"
echo ""
git diff "$BASE_BRANCH"...HEAD
echo ""

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📝 Review Against Checklist"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Apply the checklist from .claude/agents/code-reviewer.md:"
echo ""
echo "1. ✓ Error handling (specific exceptions, fail fast)"
echo "2. ✓ Contract validation (input checks, type hints)"
echo "3. ✓ Idempotency (safe to run twice)"
echo "4. ✓ Tests updated (factory fixtures, AAA pattern)"
echo "5. ✓ Polars best practices (lazy, no map_elements)"
echo "6. ✓ Style compliance (ruff, mypy)"
echo "7. ✓ Makefile commands used (make test-fast, etc.)"
echo ""

echo "💡 Run code quality checks:"
echo "   make check-fast"
echo ""
echo "💡 Create PR when ready:"
echo "   gh pr create --title \"Your PR title\" --body \"Description\""
```

## Manual Review Points

After running this command, manually verify:

1. **Tests cover new behavior**: Every new code path has a test
2. **No map_elements used**: Scan diff for this anti-pattern
3. **Factory fixtures used**: No inline setup in tests
4. **Error messages helpful**: Errors provide context for debugging
5. **Commit messages clear**: Describe the "why", not just the "what"

## Next Steps

1. Review the diff and checklist output
2. Fix any issues found
3. Run `make check-fast` to verify
4. Create PR with: `gh pr create`
