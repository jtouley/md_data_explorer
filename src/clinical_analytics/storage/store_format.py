"""Keep analytics.duckdb readable across DuckDB upgrades and rollbacks.

DuckDB reads files written in older storage formats and keeps their format when it writes to
them, but a file *created* by a newer DuckDB uses that release's format by default, and older
releases cannot open it (DuckDB 2.0 migration guide: "DuckDB versions before 2.0.0 cannot read
database files that use the DuckDB 2.0.0 storage format"). The store is created on first upload
and recreated by every dataset delete (``COPY FROM DATABASE`` into a fresh file), so without a
pin a library upgrade would silently make users' stores one-way. Two guards:

* New and rewritten store files use ``STORE_STORAGE_VERSION`` (see ``new_store_config`` and
  ``attach_storage_option``), so the previous DuckDB line can still open them.
* ``prepare_store_for_engine`` runs at startup. When the DuckDB version that last opened the
  store differs from the running one, it copies the store (and WAL) to a backup first, then
  checks the running DuckDB can open it, then records the new version and drops older backups
  (only the newest is kept). A store the running DuckDB cannot read fails loudly with
  ``StoreFormatError`` and is left untouched.

Backups hold patient rows, so a dataset delete removes them too (see ``store_backups``).
"""

from __future__ import annotations

import os
import re
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


@dataclass(frozen=True)
class EnginePreparation:
    """What ``prepare_store_for_engine`` did (safe to log: no paths)."""

    previous_engine: str | None
    current_engine: str
    backed_up: bool
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


def _backup_store(db_path: Path, previous_engine: str | None) -> list[Path]:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    label = f"{previous_engine or 'unknown'}-{stamp}"
    written: list[Path] = []
    for source in (db_path, db_path.with_name(f"{db_path.name}.wal")):
        if source.exists():
            target = source.with_name(f"{db_path.name}{BACKUP_INFIX}{label}{source.name[len(db_path.name) :]}")
            shutil.copy2(source, target)
            with target.open("rb") as f:
                os.fsync(f.fileno())
            written.append(target)
    return written


def _write_marker(marker: Path, engine: str) -> None:
    tmp = marker.with_name(f"{marker.name}.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        f.write(f"{engine}\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, marker)


def prepare_store_for_engine(data_dir: Path, engine: str | None = None) -> EnginePreparation | None:
    """Back up and verify the store before the running DuckDB first opens it. Run at startup.

    Returns None when there is no store yet. Safe to run twice: a second run with the same
    engine finds the marker current and does nothing.
    """
    from clinical_analytics.storage.datastore import exclusive_store  # datastore imports this module

    current = engine or duckdb.__version__
    db_path = data_dir / STORE_FILENAME
    if not db_path.exists():
        return None
    marker = engine_marker(db_path)
    previous = marker.read_text(encoding="utf-8").strip() if marker.exists() else ""
    with exclusive_store(db_path):
        return _prepare_locked(db_path, marker, previous or None, current)


def _prepare_locked(db_path: Path, marker: Path, previous: str | None, current: str) -> EnginePreparation:
    if previous is not None and _engine_line(previous) == _engine_line(current):
        if previous != current:
            _write_marker(marker, current)
        return EnginePreparation(previous, current, backed_up=False, storage_version=None)

    log = logger.bind(previous_engine=previous, current_engine=current)
    written = _backup_store(db_path, previous)
    log.info("store_backed_up_before_engine_change")
    try:
        con = duckdb.connect(str(db_path), read_only=True, config=new_store_config())
    except duckdb.Error as e:  # IOException on 1.x; 2.0 raises some header errors as plain Error
        raise StoreFormatError(
            f"DuckDB {current} cannot open the data store (written by a newer DuckDB, or damaged); "
            f"restore a backup or run the DuckDB release that wrote it ({type(e).__name__})"
        ) from e
    try:
        storage_version = current_storage_version(con)
    finally:
        con.close()
    _write_marker(marker, current)
    # Keep only the newest backup: each one is a full copy of the patient data.
    stale = [p for p in store_backups(db_path) if p not in written]
    for path in stale:
        path.unlink()
    log.info("store_engine_recorded", storage_version=storage_version, stale_backups_removed=len(stale))
    return EnginePreparation(previous, current, backed_up=True, storage_version=storage_version)
