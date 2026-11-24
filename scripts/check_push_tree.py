#!/usr/bin/env python3
"""Pre-push guard: the gate may only pass a push when the tree it tests is the commit being pushed.

The pre-push gate (``.pre-commit-config.yaml``) runs ``make check-fast`` and ``make test-fast`` on
the working tree, not on the pushed commit. pre-commit stashes unstaged edits before a pre-push
hook, but staged edits and untracked files stay, and ``git push origin other-branch`` pushes a
commit that is not checked out at all. In either case the gate would test something other than
what CI gets, so this guard fails first:

- the pushed commit (``PRE_COMMIT_TO_REF``, set by pre-commit) is not ``HEAD``;
- ``git status --porcelain`` lists anything: a staged edit, an untracked file not ignored (an
  unstaged edit too, when run outside pre-commit).

Run without ``PRE_COMMIT_TO_REF`` (by hand) only the second check applies.

    python scripts/check_push_tree.py      # make check-push-tree

Exit codes: 0 the tree is the pushed commit, 1 it is not, 2 the check could not run.
"""

from __future__ import annotations

import os
import sys
from typing import Final

if __package__:  # imported as scripts.check_push_tree (tests)
    from scripts.git_tracked import GitQueryError, commit_of, repo_root, uncommitted_paths
else:  # run as a file: scripts/ is sys.path[0]
    from git_tracked import GitQueryError, commit_of, repo_root, uncommitted_paths

TO_REF_ENV: Final = "PRE_COMMIT_TO_REF"
SHOWN: Final = 10


def problems(head: str, pushed: str | None, dirty: list[str]) -> list[str]:
    """Why the tree at ``head`` with ``dirty`` status entries is not the ``pushed`` commit."""
    found: list[str] = []
    if pushed is not None and pushed != head:
        found.append(f"the pushed commit {pushed[:12]} is not the checked-out HEAD {head[:12]}; check it out first")
    if dirty:
        listed = "\n".join(f"    {entry}" for entry in dirty[:SHOWN])
        more = f"\n    ... and {len(dirty) - SHOWN} more" if len(dirty) > SHOWN else ""
        found.append(
            "staged changes or untracked files would be tested but not pushed; commit them, "
            f"`git stash --include-untracked`, or list scratch files in .git/info/exclude:\n{listed}{more}"
        )
    return found


def main() -> int:
    try:
        root = repo_root()
        head = commit_of(root, "HEAD")
        to_ref = os.environ.get(TO_REF_ENV)
        pushed = commit_of(root, to_ref) if to_ref else None
        dirty = uncommitted_paths(root)
    except GitQueryError as exc:
        print(f"check_push_tree: cannot run: {exc}", file=sys.stderr)
        return 2
    found = problems(head, pushed, dirty)
    for problem in found:
        print(f"check-push-tree: FAIL: {problem}", file=sys.stderr)
    if found:
        print(
            "check-push-tree: the pre-push gate tests the working tree, so it must be the pushed commit",
            file=sys.stderr,
        )
        return 1
    print("check-push-tree: PASS: the working tree is the pushed commit")
    return 0


if __name__ == "__main__":
    sys.exit(main())
