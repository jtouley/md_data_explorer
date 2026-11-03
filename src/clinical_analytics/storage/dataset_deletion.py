"""Remove every on-disk trace of one uploaded dataset.

An upload leaves data in several places under the data home (``<data home>/data``):

* ``uploads/raw/{id}.csv``, ``uploads/raw/{id}_tables/``, ``uploads/raw/{id}/`` and the stored
  original file named in metadata (``stored_relpath`` / ``stored_filename``)
* ``uploads/metadata/{id}.json`` and the enrichment overlays ``uploads/metadata/overlays/{id}/``
* Parquet exports ``parquet/{id}_<table>_<version>.parquet``
* DuckDB tables ``{id}_<table>_<version>`` in ``analytics.duckdb``

Dropping a DuckDB table frees its blocks but leaves the bytes in the file, so after dropping,
the store is rewritten into a fresh file (``COPY FROM DATABASE``) and swapped in. A durable
``analytics.duckdb.needs-compaction`` marker covers the window between DROP and swap. Metadata
is removed last, only after the swap, so a failure part-way leaves the dataset listed and the
delete can be retried; running it twice is safe. The purge takes the store lock before it
touches anything and refuses to run while any registered connection to the store is open (see
``storage.datastore``); on a busy store nothing is removed. Order under the lock: DuckDB tables
(and the rewrite), then files, then metadata.

Sessions and in-memory query results are API state and are removed by the API route.
"""

from __future__ import annotations

import json
import os
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import duckdb
import structlog

from clinical_analytics.storage.datastore import StoreInUseError, exclusive_store

logger = structlog.get_logger()

DATASET_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_\-]{0,127}$")
DUCKDB_FILENAME = "analytics.duckdb"
PARQUET_DIRNAME = "parquet"


class DatasetDeletionError(RuntimeError):
    """Deleting a dataset failed part-way; the message says what was left."""


class DatasetStoreBusyError(DatasetDeletionError):
    """The data store has open connections; nothing was dropped. Retry shortly."""


class InvalidDatasetIdError(ValueError):
    """The dataset id is not a plain identifier (rejects path tricks such as ``../``)."""


@dataclass
class DeletionReport:
    """What a purge removed (counts are safe to log; paths stay out of logs)."""

    dataset_id: str
    removed_paths: list[Path] = field(default_factory=list)
    dropped_tables: list[str] = field(default_factory=list)

    @property
    def found_anything(self) -> bool:
        return bool(self.removed_paths or self.dropped_tables)


def validate_dataset_id(dataset_id: str) -> str:
    """Return the id if it is a plain identifier, else raise InvalidDatasetIdError."""
    if not DATASET_ID_PATTERN.fullmatch(dataset_id):
        raise InvalidDatasetIdError(f"Invalid dataset id: {dataset_id!r}")
    return dataset_id


def _owned_by(name: str, dataset_id: str, other_ids: set[str]) -> bool:
    """``{id}_...`` names belong to ``dataset_id`` unless a longer id also prefixes them."""
    if not name.startswith(f"{dataset_id}_"):
        return False
    return not any(len(o) > len(dataset_id) and name.startswith(f"{o}_") for o in other_ids)


def _inside(path: Path, root: Path) -> bool:
    return path.resolve().is_relative_to(root.resolve())


def _remove(path: Path, root: Path, report: DeletionReport) -> None:
    if not path.exists() and not path.is_symlink():
        return
    if not _inside(path, root):
        # Name only: error text reaches the API response and logs.
        raise DatasetDeletionError(f"Refusing to delete '{path.name}': it is outside the data directory")
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink()
    report.removed_paths.append(path)


def _known_dataset_ids(metadata_dir: Path) -> set[str]:
    return {p.stem for p in metadata_dir.glob("*.json")} if metadata_dir.is_dir() else set()


def _stored_original(metadata_path: Path, raw_dir: Path) -> list[Path]:
    """Original upload named in metadata; it must live inside raw/."""
    if not metadata_path.exists():
        return []
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        raise DatasetDeletionError(f"Cannot read metadata {metadata_path.name}: {type(e).__name__}") from e
    names = [metadata.get("stored_relpath"), metadata.get("stored_filename")]
    paths = [raw_dir / n for n in names if isinstance(n, str) and n]
    for path in paths:
        if not _inside(path, raw_dir):
            raise DatasetDeletionError(f"Refusing to delete '{path.name}': stored file is outside raw/")
    return paths


def compaction_marker(db_path: Path) -> Path:
    """Exists while dropped tables may still have bytes in the store file."""
    return db_path.with_name(f"{db_path.name}.needs-compaction")


def _write_marker(marker: Path) -> None:
    with marker.open("w", encoding="utf-8") as f:
        f.write("dropped tables not yet compacted out of the store\n")
        f.flush()
        os.fsync(f.fileno())


def _purge_duckdb(db_path: Path, dataset_id: str | None, other_ids: set[str], report: DeletionReport) -> None:
    """Drop the dataset's tables, then rewrite the store so their bytes are gone.

    A durable marker is written before the first DROP and removed only after the rewritten file
    has replaced the old one. A retry that finds no tables but finds the marker still rewrites,
    so a failed swap can never be reported as a successful delete.
    """
    if not db_path.exists():
        return
    marker = compaction_marker(db_path)
    compacted = db_path.with_name(f"{db_path.name}.compact")
    try:
        with exclusive_store(db_path):
            compacted.unlink(missing_ok=True)
            con = duckdb.connect(str(db_path))
            try:
                tables = [row[0] for row in con.execute("SHOW TABLES").fetchall()]
                owned = [t for t in tables if dataset_id is not None and _owned_by(t, dataset_id, other_ids)]
                if not owned and not marker.exists():
                    return
                _write_marker(marker)
                for table in owned:
                    con.execute(f'DROP TABLE IF EXISTS "{table}"')
                report.dropped_tables.extend(owned)
                # Freed blocks keep the old bytes: copy what is left into a fresh file.
                source = con.execute("SELECT current_database()").fetchone()
                if source is None:
                    raise DatasetDeletionError("DuckDB returned no current database")
                attach_path = str(compacted).replace("'", "''")
                con.execute(f"ATTACH '{attach_path}' AS compacted_store")
                con.execute(f'COPY FROM DATABASE "{source[0]}" TO compacted_store')
                con.execute("DETACH compacted_store")
            finally:
                con.close()
            os.replace(compacted, db_path)
            db_path.with_name(f"{db_path.name}.wal").unlink(missing_ok=True)
            marker.unlink()
    except StoreInUseError as e:
        raise DatasetStoreBusyError(f"The data store is in use ({e}); try again in a moment") from e
    except duckdb.Error as e:
        compacted.unlink(missing_ok=True)
        raise DatasetDeletionError(f"Could not remove DuckDB tables ({type(e).__name__})") from e
    except OSError:
        compacted.unlink(missing_ok=True)
        raise


def resume_pending_compaction(data_dir: Path) -> bool:
    """Finish a rewrite a crash or failed swap left behind (marker present). Run at startup.

    Returns True when a rewrite ran. Raises DatasetDeletionError if it still cannot finish.
    """
    db_path = data_dir / DUCKDB_FILENAME
    marker = compaction_marker(db_path)
    if not marker.exists():
        return False
    if not db_path.exists():
        marker.unlink()
        return False
    try:
        _purge_duckdb(db_path, None, set(), DeletionReport(dataset_id="(startup)"))
    except OSError as e:
        raise DatasetDeletionError(
            f"Could not finish the pending store rewrite: {e.strerror or type(e).__name__}"
        ) from e
    logger.info("store_compaction_resumed")
    return True


def purge_dataset_files(dataset_id: str, uploads_root: Path) -> DeletionReport:
    """Delete all files and DuckDB tables for ``dataset_id`` under the uploads root's parent.

    Args:
        dataset_id: Upload id.
        uploads_root: The ``uploads`` directory (``UserDatasetStorage.upload_dir``).

    Raises:
        InvalidDatasetIdError: id is not a plain identifier.
        DatasetDeletionError: something could not be removed (retry is safe).
    """
    validate_dataset_id(dataset_id)
    data_dir = uploads_root.parent
    raw_dir = uploads_root / "raw"
    metadata_dir = uploads_root / "metadata"
    metadata_path = metadata_dir / f"{dataset_id}.json"
    other_ids = _known_dataset_ids(metadata_dir) - {dataset_id}
    report = DeletionReport(dataset_id=dataset_id)
    log = logger.bind(dataset_id=dataset_id)

    db_path = data_dir / DUCKDB_FILENAME
    try:
        # The store lock comes first: a busy store must refuse before anything is removed, so
        # DatasetStoreBusyError really means nothing changed. The lock is re-entrant, so
        # _purge_duckdb takes it again without waiting.
        with exclusive_store(db_path):
            # Validated before anything is dropped: a bad stored path refuses the whole delete.
            stored_originals = _stored_original(metadata_path, raw_dir)
            _purge_duckdb(db_path, dataset_id, other_ids, report)

            for path in [
                *stored_originals,
                raw_dir / f"{dataset_id}.csv",
                raw_dir / f"{dataset_id}_tables",
                raw_dir / dataset_id,
                metadata_dir / "overlays" / dataset_id,
            ]:
                _remove(path, uploads_root, report)

            parquet_dir = data_dir / PARQUET_DIRNAME
            if parquet_dir.is_dir():
                for parquet in sorted(parquet_dir.glob(f"{dataset_id}_*")):
                    if _owned_by(parquet.name, dataset_id, other_ids):
                        _remove(parquet, data_dir, report)

            # Last: while metadata exists the dataset stays listed, so a failed delete can be retried.
            _remove(metadata_path, uploads_root, report)
    except StoreInUseError as e:
        log.warning("dataset_purge_store_busy")
        raise DatasetStoreBusyError(f"The data store is in use ({e}); try again in a moment") from e
    except OSError as e:
        log.error("dataset_purge_failed", error_type=type(e).__name__, removed=len(report.removed_paths))
        reason = e.strerror or type(e).__name__
        raise DatasetDeletionError(f"Could not delete all files for {dataset_id}: {reason}") from e

    log.info(
        "dataset_purged",
        removed_paths=len(report.removed_paths),
        dropped_tables=len(report.dropped_tables),
    )
    return report
