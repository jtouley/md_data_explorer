"""store_format: store files stay readable across DuckDB upgrades; backups exist only while verifying."""

from __future__ import annotations

import shutil
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import duckdb
import pytest

from clinical_analytics.storage import store_format
from clinical_analytics.storage.datastore import close_store_connection, open_store_connection
from clinical_analytics.storage.store_format import (
    STORE_STORAGE_VERSION,
    StoreFormatError,
    StoreLockedError,
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


def _leave_backup(db_path: Path, label: str) -> Path:
    backup = db_path.with_name(f"{db_path.name}.backup-{label}")
    shutil.copy2(db_path, backup)
    return backup


@pytest.fixture
def backups_seen_while_verifying(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[list[Path]]]:
    """Records the backup files present each time the store is verified."""
    seen: list[list[Path]] = []
    verify = store_format._verify_store

    def recording(db_path: Path, current: str) -> str:
        seen.append(store_backups(db_path))
        return verify(db_path, current)

    monkeypatch.setattr(store_format, "_verify_store", recording)
    yield seen


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


def test_openStoreConnection_newStore_recordsEngineSoFirstRestartTakesNoBackup(
    tmp_path: Path, backups_seen_while_verifying: list[list[Path]]
) -> None:
    # Arrange
    db_path = _make_store(tmp_path)

    # Act
    result = prepare_store_for_engine(tmp_path)

    # Assert
    assert engine_marker(db_path).read_text().strip() == duckdb.__version__
    assert result is not None and result.verified is False
    assert backups_seen_while_verifying == []
    assert store_backups(db_path) == []


def test_prepareStoreForEngine_noStore_returnsNoneAndWritesNothing(tmp_path: Path) -> None:
    # Arrange: empty data dir

    # Act
    result = prepare_store_for_engine(tmp_path, engine="2.0.0")

    # Assert
    assert result is None
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("previous", [None, "", "1.4.2"], ids=["no-marker", "empty-marker", "older-line"])
def test_prepareStoreForEngine_engineLineChanged_backupDuringVerifyThenRemoved(
    tmp_path: Path, previous: str | None, backups_seen_while_verifying: list[list[Path]]
) -> None:
    # Arrange
    db_path = _make_store(tmp_path)
    if previous is None:
        engine_marker(db_path).unlink()
    else:
        engine_marker(db_path).write_text(f"{previous}\n")

    # Act
    result = prepare_store_for_engine(tmp_path, engine="2.0.0")

    # Assert
    assert result is not None and result.verified is True
    assert result.storage_version == STORE_STORAGE_VERSION
    [during] = backups_seen_while_verifying
    assert [p.name.startswith(f"analytics.duckdb.backup-{previous or 'unknown'}-") for p in during] == [True]
    assert store_backups(db_path) == []
    assert engine_marker(db_path).read_text().strip() == "2.0.0"


@pytest.mark.parametrize("previous", ["2.0.0", "2.0.1"], ids=["same-version", "patch-differs"])
def test_prepareStoreForEngine_sameEngineLine_noBackup(
    tmp_path: Path, previous: str, backups_seen_while_verifying: list[list[Path]]
) -> None:
    # Arrange
    db_path = _make_store(tmp_path)
    engine_marker(db_path).write_text(f"{previous}\n")

    # Act
    result = prepare_store_for_engine(tmp_path, engine="2.0.3")

    # Assert
    assert result is not None and result.verified is False
    assert backups_seen_while_verifying == []
    assert store_backups(db_path) == []
    assert engine_marker(db_path).read_text().strip() == "2.0.3"


def test_prepareStoreForEngine_runTwice_verifiesOnceAndLeavesNoBackup(tmp_path: Path) -> None:
    # Arrange
    db_path = _make_store(tmp_path)
    engine_marker(db_path).write_text("1.4.2\n")
    prepare_store_for_engine(tmp_path, engine="2.0.0")

    # Act
    second = prepare_store_for_engine(tmp_path, engine="2.0.0")

    # Assert
    assert second is not None and second.verified is False
    assert store_backups(db_path) == []


def test_prepareStoreForEngine_leftoverBackupAfterSuccessfulRun_removedAtStartup(tmp_path: Path) -> None:
    # Arrange: a crash after the marker was written but before the backup was deleted.
    db_path = _make_store(tmp_path)
    _leave_backup(db_path, "1.4.2-20261101T000000Z")

    # Act
    result = prepare_store_for_engine(tmp_path)

    # Assert
    assert result is not None and result.verified is False
    assert store_backups(db_path) == []


def test_prepareStoreForEngine_backupFromFailedUpgrade_replacedByFreshCopyThenRemoved(
    tmp_path: Path, backups_seen_while_verifying: list[list[Path]]
) -> None:
    # Arrange: an earlier start on this engine failed verification and kept its backup.
    db_path = _make_store(tmp_path)
    engine_marker(db_path).write_text("1.4.2\n")
    kept = _leave_backup(db_path, "1.4.2-20261101T000000Z")

    # Act
    result = prepare_store_for_engine(tmp_path, engine="2.0.0")

    # Assert
    assert result is not None and result.verified is True
    [during] = backups_seen_while_verifying
    assert len(during) == 1 and during != [kept]
    assert during[0].name.startswith("analytics.duckdb.backup-1.4.2-")
    assert store_backups(db_path) == []


def test_prepareStoreForEngine_walPresent_backsUpWalForTheWindow(
    tmp_path: Path, backups_seen_while_verifying: list[list[Path]]
) -> None:
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

    # Act
    result = prepare_store_for_engine(data_dir, engine="9.9.0")

    # Assert
    assert result is not None and result.verified is True
    [during] = backups_seen_while_verifying
    assert len(during) == 2 and sum(p.name.endswith(".wal") for p in during) == 1
    assert store_backups(data_dir / "analytics.duckdb") == []


def test_prepareStoreForEngine_unreadableStore_raisesNamingKeptBackup(tmp_path: Path) -> None:
    # Arrange: a store the running DuckDB cannot open (as after a rollback from a newer one).
    db_path = tmp_path / "analytics.duckdb"
    db_path.write_bytes(b"\0" * 8 + b"DUCK" + b"\xff" * 8000)
    engine_marker(db_path).write_text("2.0.0\n")

    # Act
    with pytest.raises(StoreFormatError, match="cannot open the data store") as raised:
        prepare_store_for_engine(tmp_path, engine="1.4.5")

    # Assert
    [backup] = store_backups(db_path)
    assert str(backup) in str(raised.value)
    assert "copy taken just before the check" in str(raised.value)
    assert not isinstance(raised.value, StoreLockedError)
    assert engine_marker(db_path).read_text().strip() == "2.0.0"


def test_prepareStoreForEngine_storeLockedByAnotherProcess_plainMessageAndNoCopyMade(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange: another program holds DuckDB's file lock.
    db_path = _make_store(tmp_path)
    engine_marker(db_path).write_text("1.4.2\n")
    copies: list[Path] = []
    backup = store_format._backup_store
    monkeypatch.setattr(store_format, "_backup_store", lambda p, prev: copies.extend(backup(p, prev)) or copies)
    holder = subprocess.Popen(
        [
            sys.executable,
            "-I",
            "-c",
            "import duckdb,sys; c=duckdb.connect(sys.argv[1]); print('ok', flush=True); sys.stdin.read()",
            str(db_path),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert holder.stdout is not None and holder.stdout.readline().strip() == "ok"

        # Act
        with pytest.raises(StoreLockedError, match="open in another program"):
            prepare_store_for_engine(tmp_path, engine="2.0.0")
    finally:
        holder.communicate("", timeout=30)

    # Assert: the lock is probed before the (possibly large) store is copied.
    assert copies == []
    assert store_backups(db_path) == []
    assert engine_marker(db_path).read_text().strip() == "1.4.2"


def test_prepareStoreForEngine_copyFailsPartWay_partialBackupRemoved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange: the disk fills up while the backup is copied.
    db_path = _make_store(tmp_path)
    engine_marker(db_path).write_text("1.4.2\n")

    def partial_copy(src: Path, dst: Path) -> None:
        Path(dst).write_bytes(Path(src).read_bytes()[:100])
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(store_format.shutil, "copy2", partial_copy)

    # Act
    with pytest.raises(OSError, match="No space left"):
        prepare_store_for_engine(tmp_path, engine="2.0.0")

    # Assert
    assert store_backups(db_path) == []
    assert engine_marker(db_path).read_text().strip() == "1.4.2"


def test_backupStore_twiceInOneSecond_distinctNames(tmp_path: Path) -> None:
    # Arrange
    db_path = _make_store(tmp_path)

    # Act
    first = store_format._backup_store(db_path, "1.4.2")
    second = store_format._backup_store(db_path, "1.4.2")

    # Assert
    assert set(first).isdisjoint(second)
    assert len(store_backups(db_path)) == 2


def test_prepareStoreForEngine_noStoreButLeftoverBackup_backupRemoved(tmp_path: Path) -> None:
    # Arrange: the store was deleted, a copy from a failed check was not.
    db_path = _make_store(tmp_path)
    _leave_backup(db_path, "1.4.2-20261101T000000Z")
    db_path.unlink()

    # Act
    result = prepare_store_for_engine(tmp_path, engine="2.0.0")

    # Assert
    assert result is None
    assert store_backups(db_path) == []


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
