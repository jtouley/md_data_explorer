"""serve.main checks the store before uvicorn starts and reports failures as one plain line."""

from pathlib import Path
from unittest.mock import patch

import pytest

from clinical_analytics.api import serve
from clinical_analytics.core.data_home import DATA_HOME_ENV
from clinical_analytics.storage.dataset_deletion import DUCKDB_FILENAME, compaction_marker
from clinical_analytics.storage.datastore import StoreInUseError
from clinical_analytics.storage.store_format import StoreFormatError, StoreLockedError


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (StoreLockedError("The data store is open in another program."), "open in another program"),
        (StoreFormatError("DuckDB 2.0.0 cannot open the data store."), "cannot open the data store"),
        (OSError(28, "No space left on device"), "No space left on device"),
        (StoreInUseError("The data store is in use."), "in use"),
    ],
    ids=["locked", "unreadable", "disk-full", "in-use"],
)
def test_main_storeCheckFails_plainMessageNonZeroExitNoServer(
    capsys: pytest.CaptureFixture[str], error: Exception, expected: str
) -> None:
    # Arrange
    with (
        patch.object(serve, "prepare_store_for_engine", side_effect=error),
        patch.object(serve.uvicorn, "run") as run,
        pytest.raises(SystemExit) as exited,
    ):
        # Act
        serve.main([])

    # Assert
    err = capsys.readouterr().err
    assert exited.value.code == serve.STORE_CHECK_FAILED_EXIT
    assert exited.value.code != 0
    assert expected in err
    assert "Traceback" not in err and len(err.strip().splitlines()) == 1
    run.assert_not_called()


def test_main_storeCheckPasses_startsUvicorn() -> None:
    # Arrange
    with (
        patch.object(serve, "prepare_store_for_engine", return_value=None) as prepare,
        patch.object(serve.uvicorn, "run") as run,
    ):
        # Act
        serve.main(["--port", "8123"])

    # Assert
    prepare.assert_called_once()
    assert run.call_args.kwargs["port"] == 8123


@pytest.mark.parametrize("store_bytes", [b"not a duckdb file", b""], ids=["unreadable_store", "empty_store"])
def test_main_pendingCompactionCannotFinish_oneLineExit2NoServer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], store_bytes: bytes
) -> None:
    # Arrange: a marker from a failed delete and a store the rewrite cannot open.
    monkeypatch.setenv(DATA_HOME_ENV, str(tmp_path))
    store = tmp_path / "data" / DUCKDB_FILENAME
    store.parent.mkdir(parents=True)
    store.write_bytes(store_bytes)
    compaction_marker(store).write_text("pending\n")

    # Act
    with (
        patch.object(serve, "prepare_store_for_engine", return_value=None),
        patch.object(serve.uvicorn, "run") as run,
        pytest.raises(SystemExit) as exited,
    ):
        serve.main([])

    # Assert
    stderr_lines = [line for line in capsys.readouterr().err.splitlines() if line.startswith("Clinical Analytics")]
    assert exited.value.code == serve.STORE_CHECK_FAILED_EXIT
    assert len(stderr_lines) == 1
    assert "needs-compaction" in stderr_lines[0]
    assert compaction_marker(store).exists()
    run.assert_not_called()
