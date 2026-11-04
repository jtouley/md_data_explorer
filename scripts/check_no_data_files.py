#!/usr/bin/env python3
"""Fail if git tracks data files that may hold patient-level data.

Runtime data (uploads, DuckDB store, session DB) lives in the per-user data home, never
in the repo. Only synthetic or public reference files on the allowlist may be tracked.
"""

from __future__ import annotations

import fnmatch
import re
import subprocess
import sys
from typing import Literal, TypeAlias

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
]

# The synthetic kit: exactly the files scripts/make_synthetic_data.py writes. A data file in a
# kit directory that is not listed here was not generated, so it may be real.
SYNTHETIC_KIT = frozenset(
    {
        "data/demo/sample_cohort.csv",
        "tests/fixtures/synthetic/cohort.csv",
        "tests/fixtures/synthetic/separation.csv",
        "tests/fixtures/synthetic/sparse_cells.csv",
        "tests/fixtures/synthetic/unequal_groups.csv",
    }
)
KIT_DIRS = ("data/demo/", "tests/fixtures/synthetic/")
# Hand-written golden sets: questions and expected intents, no rows and no aggregates.
GOLDEN_ALLOWLIST = frozenset({"tests/eval/golden_questions.yaml", "config/golden_examples.yaml"})
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
    "/overlays/",
    "/clinicalanalytics/",
    "/clinical-analytics/",
)


def _local_only_reason(path: str) -> Reason | None:
    lower = path.lower()
    name = lower.rsplit("/", 1)[-1]
    if path in SYNTHETIC_KIT:
        return None
    if lower.startswith(KIT_DIRS) and lower.endswith(DATA_EXTENSIONS):
        return "not_in_synthetic_kit"
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


def main() -> int:
    paths = tracked_files()
    reasons: dict[str, str] = {path: "data_file" for path in violations(paths)}
    reasons.update(dict(local_only_violations(paths)))
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
