# SPDX-License-Identifier: MIT
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
import stat
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


DATA_HOME_MODE = 0o700


class DataHomeOwnershipError(PermissionError):
    """The data home belongs to another account, which could read every dataset in it."""


def ensure_data_home() -> Path:
    """Create the data home owner-only (0700) and tighten one this user owns that is wider.

    Everything below inherits the protection: other local accounts cannot traverse into it,
    whatever the mode of the user's home directory. ``mkdir(mode=...)`` applies only to the
    last path component, so parents keep the default mode; the data home itself is the guard.

    Raises DataHomeOwnershipError when the folder is owned by another account (it can read the
    data and this app cannot fix the mode). A world-writable parent without the sticky bit is
    logged as a warning: another account could swap the folder, but refusing would stop
    ordinary shared setups.
    """
    home = data_home()
    home.mkdir(mode=DATA_HOME_MODE, parents=True, exist_ok=True)
    if not hasattr(os, "getuid"):  # Windows: the per-user LOCALAPPDATA ACLs apply
        return home
    st = home.stat()
    if st.st_uid != os.getuid():
        raise DataHomeOwnershipError(
            f"The data folder {home} is owned by another user account, which could read the data in "
            f"it. Use a folder you own: set {DATA_HOME_ENV} or fix the folder's owner."
        )
    if st.st_mode & 0o077:
        home.chmod(DATA_HOME_MODE)
        logger.info("data_home_permissions_tightened", mode=oct(DATA_HOME_MODE))
    parent_mode = home.parent.stat().st_mode
    if parent_mode & stat.S_IWOTH and not parent_mode & stat.S_ISVTX:
        logger.warning(
            "data_home_parent_world_writable",
            action="Any account can rename the data folder's parent entries; move the data home",
        )
    return home


def uploads_dir() -> Path:
    """Directory for uploaded datasets (raw files, metadata, overlays)."""
    return data_home() / "data" / "uploads"


def session_db_path() -> Path:
    """SQLite file backing API sessions."""
    return data_home() / "data" / "clinical_analytics.db"


def nl_learning_dir() -> Path:
    """NL prompt-learning files (overlay read into the LLM prompt, iteration logs with query text).

    Kept in the per-user data home rather than a shared temp directory: the logs can contain the
    questions users typed, and the overlay is appended to the system prompt, so other local
    accounts must not be able to read or plant it.
    """
    return data_home() / "data" / "nl_query_learning"


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
