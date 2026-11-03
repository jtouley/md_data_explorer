#!/usr/bin/env python3
"""Fail if git tracks data files that may hold patient-level data.

Runtime data (uploads, DuckDB store, session DB, data contracts, quality reports) lives in the
per-user data home, never in the repo. Only synthetic or public reference files on the allowlist
may be tracked.

Data contracts describe a real upload (column names, codebooks, a hash of the source file), so
``*.odcs.yaml`` / ``*.odcs.json`` (and ODPS) files, data files (JSON, YAML, Parquet, ...) under a
``dq/`` or ``lineage/`` directory and eval golden sets are allowed only under
``tests/fixtures/synthetic/``. Source code and docs in such directories (a ``lineage`` package)
are fine. By content: any tracked YAML or JSON shaped like an ODCS contract or a quality report (``apiVersion`` v3.x
with ``kind: DataContract``, or a ``schema`` list with ``physicalName``) is allowed only under
``tests/fixtures/synthetic/`` and tagged ``synthetic``, whatever its file name; any tracked YAML
that records a ``sourceFileSha256`` custom property must also be tagged ``synthetic``.
"""

from __future__ import annotations

import fnmatch
import json
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
CONTRACT_SUFFIXES = (".odcs.yaml", ".odcs.yml", ".odcs.json", ".odps.yaml", ".odps.yml", ".odps.json")
DERIVED_DIRS = ("dq", "lineage")
# Files under dq/ or lineage/ that hold results; .py, .md and other code or docs are not data.
DERIVED_DATA_SUFFIXES = (".json", ".jsonl", ".ndjson", ".yaml", ".yml", ".txt", *DATA_EXTENSIONS)
SYNTHETIC_DIR = "tests/fixtures/synthetic/"
ODCS_KIND = "DataContract"
# Keys every stored quality report has (contracts.quality.QualityReport); counts are suppressed,
# but column names, codebooks and missingness still describe a real upload.
REPORT_KEYS = frozenset({"row_count", "summary", "checks"})
ODCS_API_PREFIX = "v3."
EVAL_GOLDEN = re.compile(r"(^|/)eval/golden[^/]*")
YAML_SUFFIXES = (".yaml", ".yml")
JSON_SUFFIXES = (".json",)
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
        or (any(d in parts[:-1] for d in DERIVED_DIRS) and lower.endswith(DERIVED_DATA_SUFFIXES))
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


def _odcs_shaped(document: object) -> bool:
    """An ODCS contract by content, whatever the file is called."""
    if not isinstance(document, dict):
        return False
    api = document.get("apiVersion")
    if document.get("kind") == ODCS_KIND and isinstance(api, str) and api.startswith(ODCS_API_PREFIX):
        return True
    schema = document.get("schema")
    return isinstance(schema, list) and any(isinstance(o, dict) and "physicalName" in o for o in schema)


def _documents(path: str, text: str) -> list[object] | None:
    """Parsed documents, or None if the text cannot be parsed."""
    try:
        if path.lower().endswith(JSON_SUFFIXES):
            return [json.loads(text)]
        return list(yaml.safe_load_all(text))
    except (yaml.YAMLError, json.JSONDecodeError):
        return None


def _report_shaped(document: object) -> bool:
    """A stored data-quality report, wherever it was copied to."""
    return (
        isinstance(document, dict)
        and REPORT_KEYS <= document.keys()
        and isinstance(document.get("summary"), list)
        and isinstance(document.get("checks"), list)
    )


def _flagged(path: str, document: object) -> bool:
    if _records_source_hash(document) and not _tagged_synthetic(document):
        return True
    in_fixtures = path.startswith(SYNTHETIC_DIR)
    if _report_shaped(document) and not in_fixtures:
        return True
    return _odcs_shaped(document) and not (in_fixtures and _tagged_synthetic(document))


def content_violations(paths: list[str], read_text: Callable[[str], str | None]) -> list[str]:
    """YAML/JSON contracts and source-hash records that are not tagged synthetic fixtures.

    * any document recording a source-file hash without a ``synthetic`` tag (anywhere);
    * any ODCS-shaped document outside ``tests/fixtures/synthetic/`` or without the tag;
* any data-quality report (``row_count`` + ``summary`` + ``checks`` lists) outside the fixtures.

    A YAML file that mentions the hash property but cannot be parsed is flagged: it cannot be
    shown safe. Unparseable JSON is left alone (it cannot be loaded as a contract either).
    """
    bad = []
    for path in paths:
        lower = path.lower()
        if not lower.endswith(YAML_SUFFIXES + JSON_SUFFIXES):
            continue
        text = read_text(path)
        markers = (SOURCE_HASH_PROPERTY, "physicalName", ODCS_KIND, "row_count")
        if text is None or not any(marker in text for marker in markers):
            continue
        documents = _documents(path, text)
        if documents is None:
            if lower.endswith(YAML_SUFFIXES) and SOURCE_HASH_PROPERTY in text:
                bad.append(path)
            continue
        if any(_flagged(path, d) for d in documents):
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
