"""Where runtime data lives: outside the source tree, per user.

Uploads, the DuckDB analytics store, Parquet exports and the session database can hold
patient-level data, so they must never sit inside a git checkout. Everything is stored under
one data home, laid out as ``<data home>/data/...`` so code that treats the data home as its
workspace root (for example ``SemanticLayer``) resolves the same relative paths as before.

Resolution order:
1. ``CLINICAL_ANALYTICS_DATA_HOME`` (tests and custom installs)
2. The OS per-user application data directory
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import structlog

logger = structlog.get_logger()

DATA_HOME_ENV = "CLINICAL_ANALYTICS_DATA_HOME"
APP_DIR_NAME = "ClinicalAnalytics"


def _platform_data_dir() -> Path:
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / APP_DIR_NAME
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
        return Path(base) / APP_DIR_NAME
    base = os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
    return Path(base) / "clinical-analytics"


def data_home() -> Path:
    """Root for runtime data. Resolved on every call so the env override applies at runtime."""
    override = os.environ.get(DATA_HOME_ENV, "").strip()
    return Path(override).expanduser().resolve() if override else _platform_data_dir()


def uploads_dir() -> Path:
    """Directory for uploaded datasets (raw files, metadata, overlays)."""
    return data_home() / "data" / "uploads"


def session_db_path() -> Path:
    """SQLite file backing API sessions."""
    return data_home() / "data" / "clinical_analytics.db"


def warn_if_legacy_repo_data(project_root: Path) -> None:
    """Point users at uploads left inside the checkout by versions that stored data there."""
    legacy = project_root / "data" / "uploads" / "metadata"
    if legacy.is_dir() and any(legacy.glob("*.json")) and legacy.resolve() != (uploads_dir() / "metadata").resolve():
        logger.warning(
            "legacy_repo_uploads_found",
            legacy_dir=str(legacy.parent),
            data_home=str(data_home()),
            action="Move data/uploads (and data/analytics.duckdb) into the data home, then delete them from the repo",
        )
