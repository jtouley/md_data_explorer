"""API startup backs up the store before a DuckDB change and refuses a store it cannot read."""

from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from clinical_analytics.api import main
from clinical_analytics.storage.store_format import EnginePreparation, StoreFormatError


@pytest.mark.parametrize(
    "result",
    [None, EnginePreparation("1.4.2", "2.0.0", backed_up=True, storage_version="v1.0.0")],
    ids=["no-store", "engine-changed"],
)
def test_lifespan_storePrepared_startsAndRunsCheckOnDataDir(result: EnginePreparation | None) -> None:
    # Act
    with (
        patch.object(main, "create_tables"),
        patch.object(main, "prepare_store_for_engine", return_value=result) as prepare,
        TestClient(main.app),
    ):
        pass

    # Assert
    prepare.assert_called_once()
    assert isinstance(prepare.call_args.args[0], Path)


@pytest.mark.parametrize("error", [StoreFormatError("unreadable"), OSError(28, "No space left")], ids=["format", "os"])
def test_lifespan_storeCheckFails_startupFails(error: Exception) -> None:
    # Act / Assert
    with (
        patch.object(main, "create_tables"),
        patch.object(main, "prepare_store_for_engine", side_effect=error),
        pytest.raises(type(error)),
        TestClient(main.app),
    ):
        pass
