# Code Quality Check Command

Run comprehensive quality checks on the codebase.

## What This Does

1. **Pre-commit hooks**: Runs all pre-commit checks (ruff, mypy, yaml/json validation)
2. **Fast tests**: Runs the fast test suite (skips slow integration tests)
3. **Coverage regression**: Checks if coverage decreased (if script exists)

## Usage

Invoke this command to get a quality report before committing or pushing.

## Implementation

```bash
#!/usr/bin/env bash
set -euo pipefail

echo "🔍 Running Code Quality Checks..."
echo ""

# Track failures
FAILURES=0

# 1. Pre-commit checks
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "1️⃣  Pre-commit Hooks (ruff, mypy, etc.)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if command -v pre-commit &> /dev/null; then
    if pre-commit run --all-files; then
        echo "✅ Pre-commit checks passed"
    else
        echo "❌ Pre-commit checks failed"
        FAILURES=$((FAILURES + 1))
    fi
else
    echo "⚠️  pre-commit not installed (run: make install-pre-commit)"
fi
echo ""

# 2. Fast tests
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "2️⃣  Fast Test Suite"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if make test-fast; then
    echo "✅ Fast tests passed"
else
    echo "❌ Fast tests failed"
    FAILURES=$((FAILURES + 1))
fi
echo ""

# 3. Coverage regression (if script exists)
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "3️⃣  Coverage Regression Check"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if [[ -f "scripts/check_coverage_regression.py" ]]; then
    if uv run python scripts/check_coverage_regression.py; then
        echo "✅ Coverage maintained or improved"
    else
        echo "❌ Coverage regression detected"
        FAILURES=$((FAILURES + 1))
    fi
else
    echo "ℹ️  Coverage regression script not found (skipping)"
fi
echo ""

# Summary
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📊 Quality Check Summary"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if [[ $FAILURES -eq 0 ]]; then
    echo "✅ All quality checks passed!"
    exit 0
else
    echo "❌ $FAILURES check(s) failed"
    echo ""
    echo "💡 Next steps:"
    echo "  - Fix failing checks above"
    echo "  - Run 'make format && make lint-fix' to auto-fix style issues"
    echo "  - Run 'make test-fast' to verify fixes"
    exit 1
fi
```

## Quick Fixes

If checks fail:
- **Style issues**: `make format && make lint-fix`
- **Type errors**: Check mypy output, add type hints
- **Test failures**: `make test-fast` for details, fix broken tests
- **Coverage drop**: Add tests for new code paths
