#!/usr/bin/env python3
"""Fail if git tracks data files that may hold patient-level data.

Runtime data (uploads, DuckDB store, session DB, data contracts, quality reports) lives in the
per-user data home, never in the repo. Only synthetic or public reference files on the allowlist
may be tracked.

Data contracts describe a real upload (column names, codebooks, a hash of the source file), so
contract files (``*.odcs.yaml`` / ``*.odcs.json`` and ODPS) and data files under a ``dq/`` or
``lineage/`` directory are allowed only under ``tests/fixtures/synthetic/`` (reason
``contract_or_report``). Source code and docs in such directories (a ``lineage`` package) are
fine. By content, any tracked YAML or JSON shaped like an ODCS contract (``apiVersion`` v3.x with
``kind: DataContract``, or a ``schema`` list with ``physicalName``) or a quality report
(``row_count`` + ``summary`` + ``checks``) is allowed only there too, contracts also tagged
``synthetic``; any document recording a ``sourceFileSha256`` must be tagged ``synthetic``.
"""

from __future__ import annotations

import fnmatch
import json
import re
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Literal, TypeAlias

import yaml

DATA_EXTENSIONS = (
    ".duckdb",
    ".db",
    ".sqlite",
    ".sqlite3",
    ".parquet",
    ".feather",
    ".arrow",
    ".sav",
    ".dta",
    ".sas7bdat",
    ".xlsx",
    ".xls",
    ".csv",
    ".tsv",
)
# Synthetic fixtures only. Data dictionaries come with the user's upload and stay local.
ALLOWLIST = (
    "src/clinical_analytics/resources/demo/*.csv",  # synthetic demo cohort shipped with the app
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


# --- Local-only artifacts -------------------------------------------------------------------
# .gitignore keeps these out of `git add .`, but not out of `git add -f`. Each rule below names
# what the app or a developer produces locally; see the "Local-only" section of .gitignore for
# the code path that writes each one.

Reason: TypeAlias = Literal[
    "not_in_synthetic_kit",
    "golden_outside_kit",
    "overlay_or_patch",
    "settings_or_session",
    "env_file",
    "local_override",
    "store_sidecar",
    "log_or_cache",
    "data_home_path",
    "contract_or_report",
]

# The synthetic kit: exactly the files scripts/make_synthetic_data.py writes. A data file in a
# kit directory that is not listed here was not generated, so it may be real.
SYNTHETIC_KIT = frozenset(
    {
        "src/clinical_analytics/resources/demo/sample_cohort.csv",
        "tests/fixtures/synthetic/cohort.csv",
        "tests/fixtures/synthetic/separation.csv",
        "tests/fixtures/synthetic/sparse_cells.csv",
        "tests/fixtures/synthetic/unequal_groups.csv",
    }
)
KIT_DIRS = ("src/clinical_analytics/resources/demo/", "tests/fixtures/synthetic/")
# Hand-written golden sets: questions and expected intents, no rows and no aggregates.
GOLDEN_ALLOWLIST = frozenset({"tests/eval/golden_questions.yaml", "src/clinical_analytics/resources/config/golden_examples.yaml"})
GOLDEN_DATA_SUFFIXES = (".yaml", ".yml", ".json", ".jsonl", ".csv", ".parquet")
# Project-wide Claude Code permissions; personal overrides go in settings.local.json.
SETTINGS_ALLOWLIST = frozenset({".claude/settings.json"})
ENV_ALLOWLIST = frozenset({".env.example"})
# Overlay store (core/overlay_store.py), NL prompt overlay (core/nl_query_engine.py), the
# per-install log key (core/log_redaction.py) and golden sets mined from query logs.
OVERLAY_NAMES = frozenset({"pending.json", "resolved.json", "patches.jsonl"})
LOCAL_NAMES: dict[str, Reason] = {
    **dict.fromkeys(OVERLAY_NAMES, "overlay_or_patch"),
    "prompt_overlay.txt": "log_or_cache",
    ".log-key": "log_or_cache",
    "golden_questions.json": "golden_outside_kit",
}
SESSION_DB_NAMES = frozenset({"clinical_analytics.db", "analytics.duckdb"})
# WAL, engine marker, backups and compaction marker of the DuckDB store (storage/store_format.py,
# storage/dataset_deletion.py) and SQLite journals of the session DB.
STORE_SIDECAR = re.compile(r"\.(duckdb|db|sqlite3?)[.-]")
LOG_CACHE_SUFFIXES = (".log", ".jsonl", ".ndjson", ".npy", ".npz", ".pkl", ".pickle", ".joblib", ".faiss")
# Directories of the per-user data home (core/data_home.py) and its platform folder names, in
# case the data home was pointed inside the checkout.
DATA_HOME_SEGMENTS = (
    "/data/uploads/",
    "/data/parquet/",
    "/data/nl_query_learning/",
    "/data/query_logs/",
    "/data/contracts/",
    "/data/dq/",
    "/overlays/",
    "/clinicalanalytics/",
    "/clinical-analytics/",
)


# Data contracts and quality reports (contracts/store.py writes them under the data home).
SYNTHETIC_DIR = "tests/fixtures/synthetic/"
CONTRACT_SUFFIXES = (".odcs.yaml", ".odcs.yml", ".odcs.json", ".odps.yaml", ".odps.yml", ".odps.json")
DERIVED_DIRS = ("dq", "lineage")
# Files under dq/ or lineage/ that hold results; .py, .md and other code or docs are not data.
DERIVED_DATA_SUFFIXES = (".json", ".jsonl", ".ndjson", ".yaml", ".yml", ".txt", *DATA_EXTENSIONS)
ODCS_KIND = "DataContract"
ODCS_API_PREFIX = "v3."
# Keys every stored quality report has (contracts.quality.QualityReport); counts are suppressed,
# but column names, codebooks and missingness still describe a real upload.
REPORT_KEYS = frozenset({"row_count", "summary", "checks"})
YAML_SUFFIXES = (".yaml", ".yml")
JSON_SUFFIXES = (".json",)
SOURCE_HASH_PROPERTY = "sourceFileSha256"
SYNTHETIC_TAG = "synthetic"


def _contract_path(lower: str) -> bool:
    if lower.startswith(SYNTHETIC_DIR):
        return False
    parts = lower.split("/")
    in_derived_dir = any(d in parts[:-1] for d in DERIVED_DIRS)
    return lower.endswith(CONTRACT_SUFFIXES) or (in_derived_dir and lower.endswith(DERIVED_DATA_SUFFIXES))


def _local_only_reason(path: str) -> Reason | None:
    lower = path.lower()
    name = lower.rsplit("/", 1)[-1]
    if path in SYNTHETIC_KIT:
        return None
    if lower.startswith(KIT_DIRS) and lower.endswith(DATA_EXTENSIONS):
        return "not_in_synthetic_kit"
    if _contract_path(lower):
        return "contract_or_report"
    in_golden_dir = "/golden/" in f"/{lower}"
    if (
        ("golden" in name or in_golden_dir)
        and lower.endswith(GOLDEN_DATA_SUFFIXES)
        and path not in GOLDEN_ALLOWLIST
        and not lower.startswith("tests/fixtures/synthetic/")
    ):
        return "golden_outside_kit"
    if name in LOCAL_NAMES:
        return LOCAL_NAMES[name]
    if lower.endswith(".json") and ("patch" in name or "overlay" in name):
        return "overlay_or_patch"
    if name in SESSION_DB_NAMES or (name == "settings.json" and path not in SETTINGS_ALLOWLIST):
        return "settings_or_session"
    if (name in {".env", ".envrc"} or name.startswith(".env.")) and path not in ENV_ALLOWLIST:
        return "env_file"
    if ".local." in name:
        return "local_override"
    if STORE_SIDECAR.search(name):
        return "store_sidecar"
    if lower.endswith(LOG_CACHE_SUFFIXES):
        return "log_or_cache"
    if any(segment in f"/{lower}" for segment in DATA_HOME_SEGMENTS):
        return "data_home_path"
    return None


def local_only_violations(paths: list[str]) -> list[tuple[str, Reason]]:
    """Tracked paths that only a local install or a developer machine should hold, with the reason."""
    found = ((path, _local_only_reason(path)) for path in paths)
    return [(path, reason) for path, reason in found if reason is not None]


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


def _report_shaped(document: object) -> bool:
    """A stored data-quality report, wherever it was copied to."""
    return (
        isinstance(document, dict)
        and REPORT_KEYS <= document.keys()
        and isinstance(document.get("summary"), list)
        and isinstance(document.get("checks"), list)
    )


def _documents(path: str, text: str) -> list[object] | None:
    """Parsed documents, or None if the text cannot be parsed."""
    try:
        if path.lower().endswith(JSON_SUFFIXES):
            return [json.loads(text)]
        return list(yaml.safe_load_all(text))
    except (yaml.YAMLError, json.JSONDecodeError):
        return None


def _flagged(path: str, document: object) -> bool:
    if _records_source_hash(document) and not _tagged_synthetic(document):
        return True
    in_fixtures = path.startswith(SYNTHETIC_DIR)
    if _report_shaped(document) and not in_fixtures:
        return True
    return _odcs_shaped(document) and not (in_fixtures and _tagged_synthetic(document))


def content_violations(paths: list[str], read_text: Callable[[str], str | None]) -> list[tuple[str, Reason]]:
    """YAML/JSON files whose content is a contract, a quality report or a source-file hash record.

    * any document recording a source-file hash without a ``synthetic`` tag (anywhere);
    * any ODCS-shaped document outside ``tests/fixtures/synthetic/`` or without the tag;
    * any data-quality report (``row_count`` + ``summary`` + ``checks`` lists) outside the fixtures.

    A YAML file that mentions the hash property but cannot be parsed is flagged: it cannot be
    shown safe. Unparseable JSON is left alone (it cannot be loaded as a contract either).
    """
    bad: list[tuple[str, Reason]] = []
    markers = (SOURCE_HASH_PROPERTY, "physicalName", ODCS_KIND, "row_count")
    for path in paths:
        lower = path.lower()
        if not lower.endswith(YAML_SUFFIXES + JSON_SUFFIXES):
            continue
        text = read_text(path)
        if text is None or not any(marker in text for marker in markers):
            continue
        documents = _documents(path, text)
        if documents is None:
            if lower.endswith(YAML_SUFFIXES) and SOURCE_HASH_PROPERTY in text:
                bad.append((path, "contract_or_report"))
            continue
        if any(_flagged(path, d) for d in documents):
            bad.append((path, "contract_or_report"))
    return bad


def _read_working_tree(path: str) -> str | None:
    try:
        return Path(path).read_text(encoding="utf-8")
    except (FileNotFoundError, IsADirectoryError, UnicodeDecodeError):
        return None  # deleted in the working tree, a directory entry, or binary


def main() -> int:
    paths = tracked_files()
    reasons: dict[str, str] = {path: "data_file" for path in violations(paths)}
    reasons.update(dict(local_only_violations(paths)))
    reasons.update(dict(content_violations(paths, _read_working_tree)))
    if reasons:
        print("Tracked data or local-only files are not allowed (patient data must stay out of git):", file=sys.stderr)
        for path in sorted(reasons):
            print(f"  {path}  [{reasons[path]}]", file=sys.stderr)
        print(
            "Remove them with `git rm --cached <path>`. Synthetic fixtures belong in the kit written by "
            "scripts/make_synthetic_data.py (see tests/fixtures/synthetic/README.md).",
            file=sys.stderr,
        )
        return 1
    print("No tracked data or local-only files.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
