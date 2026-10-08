#!/usr/bin/env bash
# Block edits on main/master branch
# PreToolUse hook: exit 2 blocks the tool call, message goes to stderr.
set -euo pipefail

cat >/dev/null # drain stdin JSON payload (unused by this hook)

CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "")

if [[ "$CURRENT_BRANCH" == "main" ]] || [[ "$CURRENT_BRANCH" == "master" ]]; then
    cat >&2 <<EOF
🚫 Cannot edit files on '$CURRENT_BRANCH' branch.

Create a feature branch first:
  git checkout -b feat/your-feature-name

Or switch to an existing branch:
  git checkout <branch-name>
EOF
    exit 2
fi

exit 0
