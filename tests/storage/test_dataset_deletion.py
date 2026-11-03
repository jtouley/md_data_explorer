"""purge_dataset_files: every file and DuckDB table for one dataset goes, nothing else does."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import duckdb
import pytest

from clinical_analytics.storage import dataset_deletion as deletion_mod
from clinical_analytics.storage.dataset_deletion import (
    DatasetDeletionError,
    InvalidDatasetIdError,
    compaction_marker,
    purge_dataset_files,
    resume_pending_compaction,
)
from clinical_analytics.storage.datastore import (
    DataStore,
    StoreBusyError,
    close_store_connection,
    exclusive_store,
    open_connection_count,
    open_store_connection,
)
from clinical_analytics.storage.store_format import (
    STORE_STORAGE_VERSION,
    current_storage_version,
    store_backups,
)


def _seed(data_dir: Path, dataset_id: str) -> None:
    uploads = data_dir / "uploads"
    (uploads / "raw" / f"{dataset_id}_tables").mkdir(parents=True, exist_ok=True)
    (uploads / "raw" / f"{dataset_id}.csv").write_text("id\n1\n")
    (uploads / "raw" / f"{dataset_id}_tables" / "patients.csv").write_text("id\n1\n")
    (uploads / "raw" / f"{dataset_id}.xlsx").write_bytes(b"original")
    overlay = uploads / "metadata" / "overlays" / dataset_id / "v1"
    overlay.mkdir(parents=True, exist_ok=True)
    (overlay / "patches.jsonl").write_text("{}\n")
    (uploads / "metadata" / f"{dataset_id}.json").write_text(json.dumps({"stored_relpath": f"{dataset_id}.xlsx"}))
    (data_dir / "contracts" / dataset_id).mkdir(parents=True, exist_ok=True)
    (data_dir / "contracts" / dataset_id / "v1.odcs.yaml").write_text(f"id: {dataset_id}\n")
    (data_dir / "dq" / dataset_id / "v1").mkdir(parents=True, exist_ok=True)
    (data_dir / "dq" / dataset_id / "v1" / "20261103T120000000000Z.json").write_text("{}")
    (data_dir / "parquet").mkdir(exist_ok=True)
    (data_dir / "parquet" / f"{dataset_id}_patients_v1.parquet").write_bytes(b"PAR1")
    con = duckdb.connect(str(data_dir / "analytics.duckdb"))
    con.execute(f'CREATE OR REPLACE TABLE "{dataset_id}_patients_v1" AS SELECT 1 AS id')
    con.close()


def _duckdb_tables(data_dir: Path) -> set[str]:
    con = duckdb.connect(str(data_dir / "analytics.duckdb"))
    try:
        return {row[0] for row in con.execute("SHOW TABLES").fetchall()}
    finally:
        con.close()


def _seed_sentinel(data_dir: Path, sentinel: str) -> None:
    """A dsx table whose text is stored literally (DuckDB compresses strings by default)."""
    con = duckdb.connect(str(data_dir / "analytics.duckdb"))
    con.execute("PRAGMA force_compression='uncompressed'")
    con.execute(f"CREATE OR REPLACE TABLE dsx_notes_v1 AS SELECT '{sentinel}' || range AS note FROM range(200000)")
    # A later table that stays, so the freed blocks sit inside the file instead of being truncated.
    con.execute("CREATE OR REPLACE TABLE keepme_notes_v1 AS SELECT 'neighbour-row-' || range AS note FROM range(50000)")
    con.execute("CHECKPOINT")
    con.close()


def _store_has(data_dir: Path, needle: bytes) -> bool:
    """Needle in the store file or any sidecar (WAL, leftover compaction file)."""
    return any(needle in p.read_bytes() for p in data_dir.glob("analytics.duckdb*") if p.is_file())


def _names_under(root: Path) -> list[str]:
    return [str(p.relative_to(root)) for p in root.rglob("*")]


def test_purgeDatasetFiles_seededDataset_removesAllArtifactsAndKeepsLongerPrefixedNeighbour(tmp_path: Path) -> None:
    # Arrange: "dsx" is a prefix of its neighbour "dsx_2"; the neighbour must survive.
    _seed(tmp_path, "dsx")
    _seed(tmp_path, "dsx_2")

    # Act
    report = purge_dataset_files("dsx", tmp_path / "uploads")

    # Assert
    remaining = _names_under(tmp_path)
    assert report.dropped_tables == ["dsx_patients_v1"]
    assert not [n for n in remaining if "dsx" in n.replace("dsx_2", "")]
    assert _duckdb_tables(tmp_path) == {"dsx_2_patients_v1"}
    assert (tmp_path / "uploads" / "metadata" / "dsx_2.json").exists()
    assert (tmp_path / "parquet" / "dsx_2_patients_v1.parquet").exists()
    assert (tmp_path / "contracts" / "dsx_2" / "v1.odcs.yaml").exists()
    assert (tmp_path / "dq" / "dsx_2" / "v1").is_dir()


@pytest.mark.parametrize("dirname", ["contracts", "dq"])
def test_purgeDatasetFiles_contractAndQualityDirs_removedAndCounted(tmp_path: Path, dirname: str) -> None:
    # Arrange
    _seed(tmp_path, "dsx")

    # Act
    report = purge_dataset_files("dsx", tmp_path / "uploads")

    # Assert
    assert not (tmp_path / dirname / "dsx").exists()
    assert (tmp_path / dirname).is_dir()  # other datasets' contracts/reports stay
    assert tmp_path / dirname / "dsx" in report.removed_paths


def test_purgeDatasetFiles_droppedTable_leavesNoBytesInDuckdbFile(tmp_path: Path) -> None:
    # Arrange
    _seed(tmp_path, "dsx")
    _seed_sentinel(tmp_path, "okapi-marker-")
    assert _store_has(tmp_path, b"okapi-marker-"), "seed must store the sentinel literally"

    # Act
    purge_dataset_files("dsx", tmp_path / "uploads")

    # Assert
    assert not _store_has(tmp_path, b"okapi-marker-")


def test_purgeDatasetFiles_runTwice_secondRunFindsNothing(tmp_path: Path) -> None:
    # Arrange
    _seed(tmp_path, "dsx")
    purge_dataset_files("dsx", tmp_path / "uploads")

    # Act
    report = purge_dataset_files("dsx", tmp_path / "uploads")

    # Assert
    assert report.found_anything is False


@pytest.mark.parametrize("bad_id", ["", "../uploads", "a/b", "a\\b", ".hidden", "x" * 200, "id with space"])
def test_purgeDatasetFiles_unsafeId_raisesInvalidDatasetIdError(tmp_path: Path, bad_id: str) -> None:
    # Arrange
    (tmp_path / "uploads").mkdir()

    # Act / Assert
    with pytest.raises(InvalidDatasetIdError):
        purge_dataset_files(bad_id, tmp_path / "uploads")


def test_purgeDatasetFiles_storedPathEscapesUploads_refusesAndKeepsMetadata(tmp_path: Path) -> None:
    # Arrange
    _seed(tmp_path, "dsx")
    outside = tmp_path / "keep.txt"
    outside.write_text("not ours")
    (tmp_path / "uploads" / "metadata" / "dsx.json").write_text(json.dumps({"stored_relpath": "../../keep.txt"}))

    # Act / Assert
    with pytest.raises(DatasetDeletionError, match="outside"):
        purge_dataset_files("dsx", tmp_path / "uploads")
    assert outside.exists()
    assert (tmp_path / "uploads" / "metadata" / "dsx.json").exists()


def test_purgeDatasetFiles_swapFailsThenRetry_sentinelGoneAndMetadataKeptUntilSwap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange
    _seed(tmp_path, "dsx")
    _seed_sentinel(tmp_path, "walrus-sentinel-")
    assert _store_has(tmp_path, b"walrus-sentinel-"), "seed must store the sentinel literally"
    real_replace = deletion_mod.os.replace
    calls = {"n": 0}

    def flaky_replace(src: Path, dst: Path) -> None:
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError(28, "No space left on device")
        real_replace(src, dst)

    monkeypatch.setattr(deletion_mod.os, "replace", flaky_replace)
    with pytest.raises(DatasetDeletionError):
        purge_dataset_files("dsx", tmp_path / "uploads")
    assert (tmp_path / "uploads" / "metadata" / "dsx.json").exists()

    # Act
    purge_dataset_files("dsx", tmp_path / "uploads")

    # Assert
    assert not _store_has(tmp_path, b"walrus-sentinel-")
    assert not compaction_marker(tmp_path / "analytics.duckdb").exists()
    assert not (tmp_path / "uploads" / "metadata" / "dsx.json").exists()


def test_purgeDatasetFiles_storeConnectionOpen_refusesAndKeepsData(tmp_path: Path) -> None:
    # Arrange
    _seed(tmp_path, "dsx")
    store = DataStore(tmp_path / "analytics.duckdb")

    # Act / Assert
    try:
        with pytest.raises(DatasetDeletionError, match="in use"):
            purge_dataset_files("dsx", tmp_path / "uploads")
    finally:
        store.close()
    assert "dsx_patients_v1" in _duckdb_tables(tmp_path)
    assert (tmp_path / "uploads" / "metadata" / "dsx.json").exists()


def test_purgeDatasetFiles_storedPathInsideUploadsButOutsideRaw_refuses(tmp_path: Path) -> None:
    # Arrange
    _seed(tmp_path, "dsx")
    _seed(tmp_path, "other")
    neighbour = tmp_path / "uploads" / "metadata" / "other.json"
    (tmp_path / "uploads" / "metadata" / "dsx.json").write_text(
        json.dumps({"stored_relpath": "../metadata/other.json"})
    )

    # Act / Assert
    with pytest.raises(DatasetDeletionError, match="outside raw"):
        purge_dataset_files("dsx", tmp_path / "uploads")
    assert neighbour.exists()


def test_purgeDatasetFiles_dataStoreLeakedWithoutClose_releasedByFinalizer(tmp_path: Path) -> None:
    # Arrange: a DataStore dropped without close() (an exception path that skipped it).
    import gc

    _seed(tmp_path, "dsx")
    leaked = DataStore(tmp_path / "analytics.duckdb")
    del leaked
    gc.collect()

    # Act
    report = purge_dataset_files("dsx", tmp_path / "uploads")

    # Assert
    assert report.dropped_tables == ["dsx_patients_v1"]
    assert open_connection_count(tmp_path / "analytics.duckdb") == 0


def test_dataStore_contextManagerAndDoubleClose_registrationReleasedOnce(tmp_path: Path) -> None:
    # Arrange
    db = tmp_path / "analytics.duckdb"
    other = DataStore(db)

    # Act
    with DataStore(db) as store:
        inside = open_connection_count(db)
    store.close()
    after = open_connection_count(db)
    other.close()

    # Assert
    assert inside == 2
    assert after == 1
    assert open_connection_count(db) == 0


def test_openStoreConnection_purgeHoldsLock_raisesStoreBusyAfterTimeout(tmp_path: Path) -> None:
    # Arrange
    import threading
    import time

    db = tmp_path / "analytics.duckdb"
    duckdb.connect(str(db)).close()
    holding, release = threading.Event(), threading.Event()

    def long_purge() -> None:
        with exclusive_store(db):
            holding.set()
            release.wait(5)

    worker = threading.Thread(target=long_purge)
    worker.start()
    holding.wait(5)

    # Act
    started = time.monotonic()
    try:
        with pytest.raises(StoreBusyError, match="Try again"):
            open_store_connection(db, timeout_s=0.2)
        waited = time.monotonic() - started
    finally:
        release.set()
        worker.join()

    # Assert
    assert waited < 2


def test_resumePendingCompaction_markerLeftByCrash_rewritesAndClearsMarker(tmp_path: Path) -> None:
    # Arrange: rows of a dropped table still in the file, as after a crash before the swap.
    _seed_sentinel(tmp_path, "crash-sentinel-")
    con = duckdb.connect(str(tmp_path / "analytics.duckdb"))
    con.execute("DROP TABLE dsx_notes_v1")
    con.close()
    compaction_marker(tmp_path / "analytics.duckdb").write_text("pending\n")
    assert _store_has(tmp_path, b"crash-sentinel-")

    # Act
    resumed = resume_pending_compaction(tmp_path)
    again = resume_pending_compaction(tmp_path)

    # Assert
    assert resumed is True
    assert again is False
    assert not _store_has(tmp_path, b"crash-sentinel-")
    assert "keepme_notes_v1" in _duckdb_tables(tmp_path)


def _storage_version(data_dir: Path) -> str:
    con = duckdb.connect(str(data_dir / "analytics.duckdb"), read_only=True)
    try:
        return current_storage_version(con)
    finally:
        con.close()


def test_purgeDatasetFiles_compaction_keepsStoreStorageVersion(tmp_path: Path) -> None:
    # Arrange: a store in the pinned format; the rewrite must not move it to a newer format.
    close_store_connection(tmp_path / "analytics.duckdb", open_store_connection(tmp_path / "analytics.duckdb"))
    _seed(tmp_path, "dsx")
    _seed(tmp_path, "dsx_2")
    before = _storage_version(tmp_path)

    # Act
    purge_dataset_files("dsx", tmp_path / "uploads")

    # Assert
    assert before == STORE_STORAGE_VERSION
    assert _storage_version(tmp_path) == before


def test_purgeDatasetFiles_preUpgradeBackupPresent_backupRemoved(tmp_path: Path) -> None:
    # Arrange: a backup kept by a failed engine check holds the dataset's rows too.
    _seed(tmp_path, "dsx")
    _seed_sentinel(tmp_path, "backup-sentinel-")
    db_path = tmp_path / "analytics.duckdb"
    shutil.copy2(db_path, db_path.with_name("analytics.duckdb.backup-1.4.2-20261101T000000Z"))
    backups = store_backups(tmp_path / "analytics.duckdb")
    assert backups and any(b"backup-sentinel-" in p.read_bytes() for p in backups)

    # Act
    report = purge_dataset_files("dsx", tmp_path / "uploads")

    # Assert
    assert store_backups(tmp_path / "analytics.duckdb") == []
    assert set(backups) <= set(report.removed_paths)
    assert not _store_has(tmp_path, b"backup-sentinel-")
