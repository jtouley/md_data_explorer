#!/usr/bin/env bash
# Block edits on main/master branch
# This hook prevents accidentally editing files while on the main branch

set -euo pipefail

# Get current branch
CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "")

# If we're on main or master, block the edit
if [[ "$CURRENT_BRANCH" == "main" ]] || [[ "$CURRENT_BRANCH" == "master" ]]; then
    cat <<EOF
{
  "block": true,
  "message": "🚫 Cannot edit files on '$CURRENT_BRANCH' branch.\n\nCreate a feature branch first:\n  git checkout -b feat/your-feature-name\n\nOr switch to an existing branch:\n  git checkout <branch-name>"
}
EOF
    exit 0
fi

# Allow edit on non-main branches
echo '{"block": false}'
