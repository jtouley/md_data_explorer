# SPDX-License-Identifier: MIT
"""Keep analytics.duckdb readable across DuckDB upgrades and rollbacks.

DuckDB reads files written in older storage formats and keeps their format when it writes to
them, but a file *created* by a newer DuckDB uses that release's format by default, and older
releases cannot open it (DuckDB 2.0 migration guide: "DuckDB versions before 2.0.0 cannot read
database files that use the DuckDB 2.0.0 storage format"). The store is created on first upload
and recreated by every dataset delete (``COPY FROM DATABASE`` into a fresh file), so without a
pin a library upgrade would silently make users' stores one-way. Two guards:

* New and rewritten store files use ``STORE_STORAGE_VERSION`` (see ``new_store_config`` and
  ``attach_storage_option``), so the previous DuckDB line can still open them. The DuckDB version
  that created the store is recorded next to it (``analytics.duckdb.engine``).
* ``prepare_store_for_engine`` runs at startup. When the DuckDB line that last opened the store
  differs from the running one, it copies the store (and WAL) to a backup, opens the store
  read-write with the running DuckDB and reads its catalog, records the new version, then
  deletes the backup. The backup is a full copy of patient data and the on-disk format does not
  change, so it is kept only while that check runs. If the check fails, the backup is kept, the
  marker is left alone and startup stops with ``StoreFormatError`` naming the backup. A store
  held open by another program raises ``StoreLockedError`` and leaves no backup.

Backups that are still on disk are removed by the next successful start and by a dataset delete
(see ``store_backups``).
"""

from __future__ import annotations

import os
import re
import secrets
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final, TypeAlias

import duckdb
import structlog

logger = structlog.get_logger()

# Storage format for store files this app creates. v1.0.0 is readable by DuckDB 1.x and 2.x.
# Raise it only once a rollback to the older DuckDB line is no longer supported.
STORE_STORAGE_VERSION: Final = "v1.0.0"
STORE_FILENAME: Final = "analytics.duckdb"
ENGINE_MARKER_SUFFIX: Final = ".engine"
BACKUP_INFIX: Final = ".backup-"
# The value type duckdb.connect(config=...) accepts.
DuckDBConfig: TypeAlias = dict[str, str | bool | int | float | list[str]]
_STORAGE_VERSION_PATTERN = re.compile(r"^v\d+\.\d+\.\d+$")


class StoreFormatError(RuntimeError):
    """The store cannot be opened by the running DuckDB (for example, written by a newer one)."""


class StoreLockedError(StoreFormatError):
    """Another program holds DuckDB's lock on the store file."""


@dataclass(frozen=True)
class EnginePreparation:
    """What ``prepare_store_for_engine`` did (safe to log: no paths)."""

    previous_engine: str | None
    current_engine: str
    verified: bool  # the engine line changed and the running DuckDB opened the store
    storage_version: str | None


def new_store_config() -> DuckDBConfig:
    """``duckdb.connect`` config for creating a store file in the pinned storage format."""
    return {"storage_compatibility_version": STORE_STORAGE_VERSION}


def attach_storage_option(storage_version: str) -> str:
    """The ``ATTACH`` option that writes the attached file in ``storage_version`` format."""
    if not _STORAGE_VERSION_PATTERN.fullmatch(storage_version):
        raise StoreFormatError(f"Unexpected DuckDB storage version {storage_version!r}")
    return f"(STORAGE_VERSION '{storage_version}')"


def current_storage_version(con: duckdb.DuckDBPyConnection) -> str:
    """Storage format of the connection's main database, e.g. ``v1.0.0`` (tag ``v1.0.0+``)."""
    rows = con.execute(
        "SELECT tags['storage_version'] FROM duckdb_databases() WHERE database_name = current_database()"
    ).fetchall()
    if not rows or not rows[0][0]:
        raise StoreFormatError("DuckDB did not report a storage version for the store")
    return str(rows[0][0]).rstrip("+")


def engine_marker(db_path: Path) -> Path:
    return db_path.with_name(f"{db_path.name}{ENGINE_MARKER_SUFFIX}")


def store_backups(db_path: Path) -> list[Path]:
    """Backup copies of the store (and their WAL files) made before an engine change."""
    return sorted(p for p in db_path.parent.glob(f"{db_path.name}{BACKUP_INFIX}*") if p.is_file())


def _engine_line(version: str) -> str:
    """Major.minor of a DuckDB version: patch releases share a storage format."""
    return ".".join(version.split(".")[:2])


def _fsync_dir(directory: Path) -> None:
    """Make file creations and deletions in ``directory`` durable (no-op where unsupported)."""
    if os.name == "nt":  # Windows cannot open a directory for fsync
        return
    fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _remove_backups(paths: list[Path], db_path: Path) -> None:
    for path in paths:
        path.unlink(missing_ok=True)
    if paths:
        _fsync_dir(db_path.parent)


def _backup_store(db_path: Path, previous_engine: str | None) -> list[Path]:
    """Copy store and WAL to backups; on any failure remove what was written and re-raise."""
    # Random suffix: two starts in the same second must never write (or delete) the same name.
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    label = f"{previous_engine or 'unknown'}-{stamp}-{secrets.token_hex(4)}"
    written: list[Path] = []
    try:
        for source in (db_path, db_path.with_name(f"{db_path.name}.wal")):
            if source.exists():
                target = source.with_name(f"{db_path.name}{BACKUP_INFIX}{label}{source.name[len(db_path.name) :]}")
                written.append(target)
                shutil.copy2(source, target)
                with target.open("rb") as f:
                    os.fsync(f.fileno())
        _fsync_dir(db_path.parent)
    except OSError:
        _remove_backups(written, db_path)
        raise
    return written


def _write_marker(marker: Path, engine: str) -> None:
    tmp = marker.with_name(f"{marker.name}.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        f.write(f"{engine}\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, marker)


def record_store_engine(db_path: Path, engine: str | None = None) -> None:
    """Record the DuckDB version that created the store, so its first restart needs no check."""
    _write_marker(engine_marker(db_path), engine or duckdb.__version__)


def _is_lock_error(error: duckdb.Error) -> bool:
    return isinstance(error, duckdb.IOException) and "lock" in str(error).lower()


def _locked_error() -> StoreLockedError:
    return StoreLockedError("The data store is open in another program. Close that program and start the app again.")


def _probe_lock(db_path: Path) -> None:
    """Raise StoreLockedError if another program holds the store, before anything is copied.

    Read-only, so it never changes the file. Other open errors are left to ``_verify_store``,
    which runs with the copy in place.
    """
    try:
        con = duckdb.connect(str(db_path), read_only=True, config=new_store_config())
    except duckdb.Error as e:
        if _is_lock_error(e):
            raise _locked_error() from e
        return
    con.close()


def _verify_store(db_path: Path, current: str) -> str:
    """Open the store read-write with the running DuckDB, read its catalog; return its format."""
    try:
        con = duckdb.connect(str(db_path), read_only=False, config=new_store_config())
    except duckdb.Error as e:  # IOException on 1.x; 2.0 raises some header errors as plain Error
        if _is_lock_error(e):
            raise _locked_error() from e
        raise StoreFormatError(
            f"DuckDB {current} cannot open the data store (written by a newer DuckDB, or damaged) ({type(e).__name__})"
        ) from e
    try:
        con.execute("SELECT count(*) FROM duckdb_tables()").fetchall()
        return current_storage_version(con)
    except duckdb.Error as e:
        raise StoreFormatError(f"DuckDB {current} cannot read the data store's catalog ({type(e).__name__})") from e
    finally:
        con.close()


def prepare_store_for_engine(data_dir: Path, engine: str | None = None) -> EnginePreparation | None:
    """Check the store opens on the running DuckDB, with a backup only during the check. Run at startup.

    Returns None when there is no store yet. Safe to run twice: a second run with the same
    engine finds the marker current, removes any leftover backup and does nothing else.
    """
    from clinical_analytics.storage.datastore import exclusive_store  # datastore imports this module

    current = engine or duckdb.__version__
    # Resolved like open_store_connection, so the marker it writes is the one read here.
    db_path = (data_dir / STORE_FILENAME).resolve()
    if not db_path.exists():
        # No store to restore into: a copy left behind is only a stray copy of patient data.
        with exclusive_store(db_path):
            _remove_backups(store_backups(db_path), db_path)
        return None
    marker = engine_marker(db_path)
    previous = marker.read_text(encoding="utf-8").strip() if marker.exists() else ""
    with exclusive_store(db_path):
        return _prepare_locked(db_path, marker, previous or None, current)


def _prepare_locked(db_path: Path, marker: Path, previous: str | None, current: str) -> EnginePreparation:
    leftover = store_backups(db_path)
    if previous is not None and _engine_line(previous) == _engine_line(current):
        if previous != current:
            _write_marker(marker, current)
        # Left by a run that verified the store but stopped before deleting its backup.
        _remove_backups(leftover, db_path)
        return EnginePreparation(previous, current, verified=False, storage_version=None)

    log = logger.bind(previous_engine=previous, current_engine=current)
    _probe_lock(db_path)
    backups = _backup_store(db_path, previous)
    # A backup kept by an earlier failed check may predate later writes (for example, by the
    # previous app release): the fresh copy replaces it, so there is never more than one.
    _remove_backups([p for p in leftover if p not in backups], db_path)
    log.info("store_backed_up_for_engine_check", replaced_backups=len(leftover))
    try:
        storage_version = _verify_store(db_path, current)
    except StoreLockedError:
        _remove_backups(backups, db_path)
        raise
    except StoreFormatError as e:
        names = ", ".join(str(p) for p in backups)
        raise StoreFormatError(
            f"{e} A copy taken just before the check is kept at: {names}. The check normally leaves "
            "the store as it was, so start the app version you used before. Copy it back over "
            "analytics.duckdb only if that version cannot open the store either. Then delete the "
            "copy: it holds patient data."
        ) from e
    _write_marker(marker, current)
    _remove_backups(backups, db_path)
    log.info("store_engine_recorded", storage_version=storage_version)
    return EnginePreparation(previous, current, verified=True, storage_version=storage_version)
