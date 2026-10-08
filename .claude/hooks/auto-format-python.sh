#!/usr/bin/env bash
# Auto-format Python files after Edit/Write operations
# Uses ruff for fast formatting and linting (matches pre-commit config)
# PostToolUse hook: reads the tool call payload from stdin; settings.json has
# no native file-glob filter, so the .py check happens here.
set -euo pipefail

INPUT=$(cat)
FILE_PATH=$(python3 -c "import json,sys; print(json.load(sys.stdin).get('tool_input', {}).get('file_path', ''))" <<<"$INPUT" 2>/dev/null || echo "")

if [[ -z "$FILE_PATH" ]] || [[ "$FILE_PATH" != *.py ]] || [[ ! -f "$FILE_PATH" ]]; then
    exit 0
fi

if command -v uv &> /dev/null; then
    uv run ruff format "$FILE_PATH" &> /dev/null || true
    uv run ruff check --fix --quiet "$FILE_PATH" &> /dev/null || true
else
    ruff format "$FILE_PATH" &> /dev/null || true
    ruff check --fix --quiet "$FILE_PATH" &> /dev/null || true
fi

exit 0
