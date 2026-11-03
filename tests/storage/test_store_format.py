"""store_format: store files stay readable across DuckDB upgrades, and are backed up first."""

from __future__ import annotations

import shutil
from pathlib import Path

import duckdb
import pytest

from clinical_analytics.storage.datastore import close_store_connection, open_store_connection
from clinical_analytics.storage.store_format import (
    STORE_STORAGE_VERSION,
    StoreFormatError,
    attach_storage_option,
    current_storage_version,
    engine_marker,
    prepare_store_for_engine,
    store_backups,
)


def _make_store(data_dir: Path) -> Path:
    db_path = data_dir / "analytics.duckdb"
    con = open_store_connection(db_path)
    try:
        con.execute("CREATE TABLE ds_patients_v1 AS SELECT range AS id FROM range(10)")
    finally:
        close_store_connection(db_path, con)
    return db_path


def test_openStoreConnection_newStore_writesPinnedStorageVersion(tmp_path: Path) -> None:
    # Arrange
    db_path = _make_store(tmp_path)

    # Act
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        version = current_storage_version(con)
    finally:
        con.close()

    # Assert
    assert version == STORE_STORAGE_VERSION


def test_prepareStoreForEngine_noStore_returnsNoneAndWritesNothing(tmp_path: Path) -> None:
    # Arrange: empty data dir

    # Act
    result = prepare_store_for_engine(tmp_path, engine="2.0.0")

    # Assert
    assert result is None
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("previous", [None, "", "1.4.2"], ids=["no-marker", "empty-marker", "older-line"])
def test_prepareStoreForEngine_engineLineChanged_backsUpThenRecordsEngine(tmp_path: Path, previous: str | None) -> None:
    # Arrange
    db_path = _make_store(tmp_path)
    if previous is not None:
        engine_marker(db_path).write_text(f"{previous}\n")

    # Act
    result = prepare_store_for_engine(tmp_path, engine="2.0.0")

    # Assert
    assert result is not None and result.backed_up is True
    assert result.storage_version == STORE_STORAGE_VERSION
    backups = store_backups(db_path)
    assert [p.name.startswith(f"analytics.duckdb.backup-{previous or 'unknown'}-") for p in backups] == [True]
    assert backups[0].read_bytes() == db_path.read_bytes()
    assert engine_marker(db_path).read_text().strip() == "2.0.0"


@pytest.mark.parametrize("previous", ["2.0.0", "2.0.1"], ids=["same-version", "patch-differs"])
def test_prepareStoreForEngine_sameEngineLine_noBackup(tmp_path: Path, previous: str) -> None:
    # Arrange
    db_path = _make_store(tmp_path)
    engine_marker(db_path).write_text(f"{previous}\n")

    # Act
    result = prepare_store_for_engine(tmp_path, engine="2.0.3")

    # Assert
    assert result is not None and result.backed_up is False
    assert store_backups(db_path) == []
    assert engine_marker(db_path).read_text().strip() == "2.0.3"


def test_prepareStoreForEngine_runTwice_backsUpOnce(tmp_path: Path) -> None:
    # Arrange
    db_path = _make_store(tmp_path)
    prepare_store_for_engine(tmp_path, engine="2.0.0")

    # Act
    second = prepare_store_for_engine(tmp_path, engine="2.0.0")

    # Assert
    assert second is not None and second.backed_up is False
    assert len(store_backups(db_path)) == 1


def test_prepareStoreForEngine_walPresent_backsUpWalToo(tmp_path: Path) -> None:
    # Arrange: snapshot a store and its WAL while a write is un-checkpointed, as a crash leaves it.
    live = tmp_path / "live"
    live.mkdir()
    db_path = _make_store(live)
    con = duckdb.connect(str(db_path), config={"checkpoint_threshold": "1GB"})
    con.execute("INSERT INTO ds_patients_v1 VALUES (99)")
    data_dir = tmp_path / "crashed"
    data_dir.mkdir()
    for name in ("analytics.duckdb", "analytics.duckdb.wal"):
        shutil.copy2(live / name, data_dir / name)
    con.close()
    wal_bytes = (data_dir / "analytics.duckdb.wal").read_bytes()

    # Act
    result = prepare_store_for_engine(data_dir, engine="2.0.0")

    # Assert
    assert result is not None and result.backed_up is True
    backups = store_backups(data_dir / "analytics.duckdb")
    wal_backups = [p for p in backups if p.name.endswith(".wal")]
    assert len(backups) == 2
    assert [p.read_bytes() for p in wal_backups] == [wal_bytes]


def test_prepareStoreForEngine_unreadableStore_raisesAndLeavesMarker(tmp_path: Path) -> None:
    # Arrange: a store the running DuckDB cannot open (as after a rollback from a newer one).
    db_path = tmp_path / "analytics.duckdb"
    db_path.write_bytes(b"\0" * 8 + b"DUCK" + b"\xff" * 8000)
    engine_marker(db_path).write_text("2.0.0\n")

    # Act / Assert
    with pytest.raises(StoreFormatError, match="cannot open the data store"):
        prepare_store_for_engine(tmp_path, engine="1.4.5")
    assert engine_marker(db_path).read_text().strip() == "2.0.0"
    assert len(store_backups(db_path)) == 1


@pytest.mark.parametrize("version", ["", "latest", "v1.0", "v1.0.0'; DROP TABLE x; --", None])
def test_attachStorageOption_notAVersion_raises(version: str | None) -> None:
    # Act / Assert
    with pytest.raises((StoreFormatError, TypeError)):
        attach_storage_option(version)  # type: ignore[arg-type]


def test_attachStorageOption_version_formatsOption() -> None:
    # Act
    option = attach_storage_option("v1.0.0")

    # Assert
    assert option == "(STORAGE_VERSION 'v1.0.0')"


def test_prepareStoreForEngine_secondEngineChange_keepsOnlyNewestBackup(tmp_path: Path) -> None:
    # Arrange
    db_path = _make_store(tmp_path)
    prepare_store_for_engine(tmp_path, engine="1.4.5")
    first = store_backups(db_path)

    # Act
    prepare_store_for_engine(tmp_path, engine="2.0.0")

    # Assert
    remaining = store_backups(db_path)
    assert len(first) == 1 and len(remaining) == 1
    assert remaining[0].name.startswith("analytics.duckdb.backup-1.4.5-")
