#!/usr/bin/env python3
"""Fail if git tracks data files that may hold patient-level data.

Runtime data (uploads, DuckDB store, session DB) lives in the per-user data home, never
in the repo. Only synthetic or public reference files on the allowlist may be tracked.
"""

from __future__ import annotations

import fnmatch
import subprocess
import sys

DATA_EXTENSIONS = (
    ".duckdb", ".db", ".sqlite", ".sqlite3", ".parquet", ".feather", ".arrow",
    ".sav", ".dta", ".sas7bdat", ".xlsx", ".xls", ".csv", ".tsv",
)
# Synthetic fixtures only. Data dictionaries come with the user's upload and stay local.
ALLOWLIST = (
    "data/demo/*.csv",
    "tests/fixtures/synthetic/*",
)


def tracked_files() -> list[str]:
    out = subprocess.run(["git", "ls-files", "-z"], capture_output=True, check=True).stdout
    return [p for p in out.decode().split("\0") if p]


def violations(paths: list[str]) -> list[str]:
    bad = []
    for path in paths:
        if any(fnmatch.fnmatch(path, pattern) for pattern in ALLOWLIST):
            continue
        if path.lower().endswith(DATA_EXTENSIONS) or path.startswith("data/"):
            bad.append(path)
    return bad


def main() -> int:
    bad = violations(tracked_files())
    if bad:
        print("Tracked data files are not allowed (patient data must stay out of git):", file=sys.stderr)
        for path in bad:
            print(f"  {path}", file=sys.stderr)
        print("Move them out of the repo, or add synthetic fixtures to the allowlist in this script.", file=sys.stderr)
        return 1
    print("No tracked data files.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
