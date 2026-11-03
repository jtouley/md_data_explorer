#!/usr/bin/env python3
"""Fail if git tracks data files that may hold patient-level data.

Runtime data (uploads, DuckDB store, session DB, data contracts, quality reports) lives in the
per-user data home, never in the repo. Only synthetic or public reference files on the allowlist
may be tracked.

Data contracts describe a real upload (column names, codebooks, a hash of the source file), so
``*.odcs.yaml`` / ``*.odps.yaml`` files, ``dq/`` and ``lineage/`` directories and eval golden sets
are allowed only under ``tests/fixtures/synthetic/``. Any tracked YAML that records a
``sourceFileSha256`` custom property must also be tagged ``synthetic``.
"""

from __future__ import annotations

import fnmatch
import re
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import yaml

DATA_EXTENSIONS = (
    ".duckdb", ".db", ".sqlite", ".sqlite3", ".parquet", ".feather", ".arrow",
    ".sav", ".dta", ".sas7bdat", ".xlsx", ".xls", ".csv", ".tsv",
)  # fmt: skip
CONTRACT_SUFFIXES = (".odcs.yaml", ".odcs.yml", ".odps.yaml", ".odps.yml")
DERIVED_DIRS = ("dq", "lineage")
EVAL_GOLDEN = re.compile(r"(^|/)eval/golden[^/]*")
YAML_SUFFIXES = (".yaml", ".yml")
SOURCE_HASH_PROPERTY = "sourceFileSha256"
SYNTHETIC_TAG = "synthetic"
# Synthetic fixtures and public data dictionaries only.
ALLOWLIST = (
    "data/demo/*.csv",
    "data/dictionaries/*.pdf",
    "tests/fixtures/synthetic/*",
    # Hand-written NL questions and expected intents; no rows, no column profiles.
    "tests/eval/golden_questions.yaml",
)


def tracked_files() -> list[str]:
    out = subprocess.run(["git", "ls-files", "-z"], capture_output=True, check=True).stdout
    return [p for p in out.decode().split("\0") if p]


def _derived_artifact(path: str) -> bool:
    lower = path.lower()
    parts = lower.split("/")
    return (
        lower.endswith(CONTRACT_SUFFIXES)
        or any(d in parts[:-1] for d in DERIVED_DIRS)
        or bool(EVAL_GOLDEN.search(lower))
    )


def violations(paths: list[str]) -> list[str]:
    """Paths that look like data or data-derived artifacts and are not allowlisted."""
    bad = []
    for path in paths:
        if any(fnmatch.fnmatch(path, pattern) for pattern in ALLOWLIST):
            continue
        if path.lower().endswith(DATA_EXTENSIONS) or path.startswith("data/") or _derived_artifact(path):
            bad.append(path)
    return bad


def _records_source_hash(document: object) -> bool:
    if not isinstance(document, dict):
        return False
    props = document.get("customProperties") or []
    return any(isinstance(p, dict) and p.get("property") == SOURCE_HASH_PROPERTY for p in props)


def _tagged_synthetic(document: object) -> bool:
    return isinstance(document, dict) and SYNTHETIC_TAG in (document.get("tags") or [])


def content_violations(paths: list[str], read_text: Callable[[str], str | None]) -> list[str]:
    """YAML files recording a source-file hash without a ``synthetic`` tag (anywhere, allowlist too).

    A file that mentions the property but cannot be parsed is flagged: it cannot be shown safe.
    """
    bad = []
    for path in paths:
        if not path.lower().endswith(YAML_SUFFIXES):
            continue
        text = read_text(path)
        if text is None or SOURCE_HASH_PROPERTY not in text:
            continue
        try:
            documents = list(yaml.safe_load_all(text))
        except yaml.YAMLError:
            bad.append(path)
            continue
        if any(_records_source_hash(d) and not _tagged_synthetic(d) for d in documents):
            bad.append(path)
    return bad


def _read_working_tree(path: str) -> str | None:
    try:
        return Path(path).read_text(encoding="utf-8")
    except FileNotFoundError:
        return None  # tracked but deleted in the working tree
    except UnicodeDecodeError:
        return None


def main() -> int:
    paths = tracked_files()
    bad = sorted(set(violations(paths)) | set(content_violations(paths, _read_working_tree)))
    if bad:
        print("Tracked data files are not allowed (patient data must stay out of git):", file=sys.stderr)
        for path in bad:
            print(f"  {path}", file=sys.stderr)
        print(
            "Move them out of the repo, or add synthetic fixtures under tests/fixtures/synthetic/ "
            "(contracts there must be tagged 'synthetic').",
            file=sys.stderr,
        )
        return 1
    print("No tracked data files.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
