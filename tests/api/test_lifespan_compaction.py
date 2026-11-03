"""A store rewrite left pending by a failed delete must finish before the API serves."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from clinical_analytics.api import main
from clinical_analytics.core.data_home import DATA_HOME_ENV
from clinical_analytics.storage.dataset_deletion import DUCKDB_FILENAME, compaction_marker


@pytest.mark.parametrize(
    "store_bytes",
    [b"not a duckdb file", b""],
    ids=["unreadable_store", "empty_store"],
)
def test_lifespan_pendingCompactionCannotFinish_refusesToStartAndKeepsMarker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, store_bytes: bytes
) -> None:
    # Arrange: a marker from a failed delete, and a store the rewrite cannot open.
    monkeypatch.setenv(DATA_HOME_ENV, str(tmp_path))
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    store = data_dir / DUCKDB_FILENAME
    store.write_bytes(store_bytes)
    compaction_marker(store).write_text("pending\n")

    # Act
    # (RuntimeError: the base of StoreCompactionPendingError, so the check also runs on old code.)
    with patch.object(main, "create_tables"), pytest.raises(RuntimeError, match="needs-compaction") as raised:
        with TestClient(main.app) as client:
            client.get("/health")

    # Assert
    assert type(raised.value).__name__ == "StoreCompactionPendingError"
    assert compaction_marker(store).exists()


def test_lifespan_noPendingCompaction_serves(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange
    monkeypatch.setenv(DATA_HOME_ENV, str(tmp_path))

    # Act
    with patch.object(main, "create_tables"), TestClient(main.app) as client:
        response = client.get("/health")

    # Assert
    assert response.status_code == 200
