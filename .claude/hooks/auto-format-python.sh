#!/usr/bin/env bash
# Auto-format Python files after Edit/Write operations
# Uses ruff for fast formatting and linting (matches pre-commit config)

set -euo pipefail

# Get the file path from environment variable
FILE_PATH="${CLAUDE_TOOL_INPUT_FILE_PATH:-}"

# Exit gracefully if no file path (shouldn't happen, but be safe)
if [[ -z "$FILE_PATH" ]]; then
    echo '{"block": false, "message": "⚠️  No file path provided to auto-format hook"}'
    exit 0
fi

# Only process if file exists
if [[ ! -f "$FILE_PATH" ]]; then
    echo '{"block": false}'
    exit 0
fi

# Run ruff format (auto-fix formatting)
if command -v uv &> /dev/null; then
    uv run ruff format "$FILE_PATH" &> /dev/null || true
    uv run ruff check --fix --quiet "$FILE_PATH" &> /dev/null || true
else
    ruff format "$FILE_PATH" &> /dev/null || true
    ruff check --fix --quiet "$FILE_PATH" &> /dev/null || true
fi

# Non-blocking: just inform the user
echo '{"block": false, "message": "✅ Auto-formatted with ruff"}'
