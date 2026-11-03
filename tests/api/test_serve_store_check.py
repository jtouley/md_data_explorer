"""serve.main checks the store before uvicorn starts and reports failures as one plain line."""

from unittest.mock import patch

import pytest

from clinical_analytics.api import serve
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
