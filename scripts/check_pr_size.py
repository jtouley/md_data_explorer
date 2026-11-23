#!/usr/bin/env python3
"""Warn above 1,500 and fail above 3,000 added non-test lines against the base branch.

Large pull requests are reviewed badly: review turns into the test suite and rounds multiply.
Counted: lines added (``git diff --numstat``) against the merge base with ``--base``, excluding
tests (``tests/``, ``electron/tests/``, ``*.test.js``/``*.spec.js``, ``test_*.py``) and
generated lock files. Binary files count 0. ``--approved`` (CI sets it when the PR
carries ``large-pr-approved``) turns the failure into a warning.

    python scripts/check_pr_size.py                    # make pr-size
    python scripts/check_pr_size.py --base origin/main --approved

Exit codes: 0 within the limit (or approved), 1 over FAIL_LINES, 2 the check could not run.
"""

from __future__ import annotations

import argparse
import fnmatch
import os
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Final, Literal, NamedTuple

if __package__:  # imported as scripts.check_pr_size (tests)
    from scripts.git_tracked import GitQueryError, diff_against_base, repo_root
else:  # run as a file: scripts/ is sys.path[0]
    from git_tracked import GitQueryError, diff_against_base, repo_root

DEFAULT_BASE: Final = "origin/main"
WARN_LINES: Final = 1500
FAIL_LINES: Final = 3000
APPROVAL_LABEL: Final = "large-pr-approved"
EXEMPT_PATTERNS: Final[tuple[str, ...]] = (
    "tests/*",
    "electron/tests/*",
    "*.test.js",
    "*.spec.js",
    "*/test_*.py",
    "test_*.py",
    "uv.lock",
    "*package-lock.json",
)

Verdict = Literal["ok", "warn", "fail", "approved"]


class FileDelta(NamedTuple):
    path: str
    added: int


def is_exempt(path: str) -> bool:
    return any(fnmatch.fnmatch(path, pattern) for pattern in EXEMPT_PATTERNS)


def parse_numstat(text: str) -> list[FileDelta]:
    """Parse ``git diff --numstat -z --no-renames`` output ("added\\tdeleted\\tpath\\0" records)."""
    deltas = []
    for record in text.split("\0"):
        if not record:
            continue
        added, _deleted, path = record.split("\t", 2)
        deltas.append(FileDelta(path, 0 if added == "-" else int(added)))  # "-" marks a binary file
    return deltas


def counted_lines(deltas: Iterable[FileDelta]) -> int:
    return sum(d.added for d in deltas if not is_exempt(d.path))


def verdict(lines: int, *, approved: bool) -> Verdict:
    if lines > FAIL_LINES:
        return "approved" if approved else "fail"
    return "warn" if lines > WARN_LINES else "ok"


def message(lines: int, result: Verdict) -> str:
    head = f"{lines} added non-test lines"
    return {
        "ok": f"{head} (warn above {WARN_LINES}, fail above {FAIL_LINES})",
        "warn": f"{head}: above {WARN_LINES}; split the PR unless it cannot be split",
        "fail": f"{head}: above {FAIL_LINES}; split the PR or have it marked '{APPROVAL_LABEL}'",
        "approved": f"{head}: above {FAIL_LINES}, allowed by '{APPROVAL_LABEL}'",
    }[result]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base", default=DEFAULT_BASE, help=f"ref to diff against (default {DEFAULT_BASE})")
    parser.add_argument("--approved", action="store_true", help=f"the PR is marked '{APPROVAL_LABEL}'")
    args = parser.parse_args(argv)

    try:
        root: Path = repo_root()
        numstat = diff_against_base(root, args.base, ["--numstat", "-z", "--no-renames"])
    except GitQueryError as exc:
        print(f"check_pr_size: cannot run: {exc}", file=sys.stderr)
        return 2

    lines = counted_lines(parse_numstat(numstat))
    result = verdict(lines, approved=args.approved)
    text = f"pr-size: {message(lines, result)}"
    if os.environ.get("GITHUB_ACTIONS") == "true" and result != "ok":
        level = "error" if result == "fail" else "warning"
        print(f"::{level} title=PR size::{text}")
    print(text)
    return 1 if result == "fail" else 0


if __name__ == "__main__":
    sys.exit(main())
