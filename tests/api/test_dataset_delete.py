"""DELETE /api/datasets/{id}: nothing with that dataset id (or its values) is left in the data home (S5)."""

from __future__ import annotations

import secrets
from pathlib import Path
from typing import Any

import polars as pl
import pytest
from fastapi.testclient import TestClient

from clinical_analytics.api.db.database import SessionLocal, engine
from clinical_analytics.api.models import database as db_models
from clinical_analytics.api.routes import queries
from clinical_analytics.api.routes.enrichments import default_overlay_dir
from clinical_analytics.core.data_home import data_home

pytestmark = pytest.mark.usefixtures("isolated_data_home")


def _session_db_bytes() -> bytes:
    database = engine.url.database
    return Path(database).read_bytes() if database and Path(database).exists() else b""


def _upload(test_client: TestClient, marker: str) -> str:
    df = pl.DataFrame(
        {
            "patient_id": [f"P{i:04d}" for i in range(120)],
            "age": [30 + i % 40 for i in range(120)],
            "diagnosis": [marker if i % 3 == 0 else "flu" for i in range(120)],
        }
    )
    response = test_client.post(
        "/api/datasets/upload",
        # Upload ids hash the filename + second; a unique name keeps parallel workers apart.
        files={"file": (f"cohort_{marker}.csv", df.write_csv().encode(), "text/csv")},
        data={"dataset_name": f"cohort {marker}"},
    )
    assert response.status_code == 200, response.text
    return str(response.json()["upload_id"])


def _traces(root: Path, needles: list[bytes]) -> list[str]:
    """Paths under root whose name or contents mention any needle."""
    found = []
    for path in root.rglob("*"):
        if any(n.decode() in str(path.relative_to(root)) for n in needles):
            found.append(str(path))
        elif path.is_file() and any(n in path.read_bytes() for n in needles):
            found.append(f"{path} (contents)")
    return found


def test_deleteDataset_uploadWithSessionsOverlaysAndResults_leavesNoTraceInDataHome(test_client: TestClient) -> None:
    # Arrange
    marker = f"zebrafinchitis{secrets.token_hex(4)}"
    dataset_id = _upload(test_client, marker)
    session_id = test_client.post("/api/sessions", json={"dataset_id": dataset_id}).json()["session_id"]
    with SessionLocal() as db:
        db.add(db_models.Message(message_id=f"m_{marker}", session_id=session_id, role="user", content=marker))
        db.commit()
    overlay = default_overlay_dir() / "overlays" / dataset_id / "v1"
    overlay.mkdir(parents=True)
    (overlay / "patches.jsonl").write_text(f'{{"column": "diagnosis", "value": "{marker}"}}\n')
    queries.get_query_service_for_dataset(dataset_id)
    needles = [dataset_id.encode(), marker.encode()]
    assert _traces(data_home(), needles), "the upload should have left data to delete"
    store = data_home() / "data" / "analytics.duckdb"
    assert dataset_id.encode() in store.read_bytes(), "the DuckDB catalog should name the dataset's tables"
    assert marker.encode() in _session_db_bytes()

    # Act
    response = test_client.delete(f"/api/datasets/{dataset_id}")

    # Assert
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["deleted_sessions"] == 1
    assert body["dropped_tables"] >= 1
    assert _traces(data_home(), needles) == []
    assert dataset_id.encode() not in store.read_bytes()
    assert marker.encode() not in _session_db_bytes()
    assert dataset_id.encode() not in _session_db_bytes()
    assert dataset_id not in queries._query_services
    assert test_client.get(f"/api/datasets/{dataset_id}").status_code == 404
    assert dataset_id not in [d["dataset_id"] for d in test_client.get("/api/datasets").json()["datasets"]]


def test_deleteDataset_secondDelete_returns404(test_client: TestClient) -> None:
    # Arrange
    dataset_id = _upload(test_client, f"okapi{secrets.token_hex(4)}")
    test_client.delete(f"/api/datasets/{dataset_id}")

    # Act
    response = test_client.delete(f"/api/datasets/{dataset_id}")

    # Assert
    assert response.status_code == 404


@pytest.mark.parametrize(("dataset_id", "status"), [("does_not_exist_123", 404), ("bad%20id", 400), ("..", 404)])
def test_deleteDataset_unknownOrUnsafeId_rejected(test_client: TestClient, dataset_id: str, status: int) -> None:
    # Arrange / Act
    response = test_client.delete(f"/api/datasets/{dataset_id}")

    # Assert
    assert response.status_code == status


def test_deleteDataset_queryAndPreviewArriveDuringPurge_rejected409AndNothingCachedAfter(
    test_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange
    from clinical_analytics.api import dependencies
    from clinical_analytics.api.routes import datasets as datasets_routes

    dataset_id = _upload(test_client, f"tapir{secrets.token_hex(4)}")
    real_purge = datasets_routes.purge_dataset_files
    seen: dict[str, int] = {}

    def purge_with_concurrent_requests(ds_id: str, uploads_root: Path) -> Any:
        # Real HTTP requests through the app while the delete is in progress.
        seen["query"] = test_client.post(
            "/api/queries", json={"session_id": "s1", "dataset_id": ds_id, "query_text": "how many"}
        ).status_code
        seen["preview"] = test_client.get(f"/api/datasets/{ds_id}/preview").status_code
        seen["detail"] = test_client.get(f"/api/datasets/{ds_id}").status_code
        return real_purge(ds_id, uploads_root)

    monkeypatch.setattr(datasets_routes, "purge_dataset_files", purge_with_concurrent_requests)

    # Act
    response = test_client.delete(f"/api/datasets/{dataset_id}")

    # Assert
    assert response.status_code == 200, response.text
    assert seen == {"query": 409, "preview": 409, "detail": 409}
    assert dataset_id not in queries._query_services
    assert dataset_id not in dependencies._semantic_layer_cache
    assert dataset_id not in dependencies._deleting_datasets


def test_submitQuery_storeLockHeld_eventLoopStaysResponsiveAnd503(
    test_client: TestClient, real_server: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange: a dataset whose semantic layer must be built from the store, while a long
    # purge holds the store lock in another thread.
    import threading
    import time

    import requests

    from clinical_analytics.storage import datastore

    monkeypatch.setattr(datastore, "STORE_LOCK_TIMEOUT_S", 2.0)
    dataset_id = _upload(test_client, f"quoll{secrets.token_hex(4)}")
    store = data_home() / "data" / "analytics.duckdb"
    holding, release = threading.Event(), threading.Event()

    def long_purge() -> None:
        with datastore.exclusive_store(store):
            holding.set()
            release.wait(10)

    purge = threading.Thread(target=long_purge)
    purge.start()
    holding.wait(5)
    result: dict[str, Any] = {}

    def submit() -> None:
        result["submit"] = requests.post(
            f"{real_server}/api/queries",
            json={"session_id": "s1", "dataset_id": dataset_id, "query_text": "how many"},
            timeout=20,
        )

    submitter = threading.Thread(target=submit)

    # Act
    try:
        submitter.start()
        time.sleep(0.3)  # the submit is now waiting on the store lock
        started = time.monotonic()
        health = requests.get(f"{real_server}/health", timeout=10)
        health_latency = time.monotonic() - started
        submitter.join(15)
    finally:
        release.set()
        purge.join()

    # Assert
    assert health.status_code == 200
    assert health_latency < 1.0
    assert result["submit"].status_code == 503
    assert "Try again" in result["submit"].json()["detail"]


def test_generateUploadId_sameFilenameTwice_distinctIdsMatchingDeletePattern(upload_storage: Any) -> None:
    # Arrange
    from clinical_analytics.storage.dataset_deletion import DATASET_ID_PATTERN

    # Act
    first = upload_storage.generate_upload_id("cohort.csv")
    second = upload_storage.generate_upload_id("cohort.csv")

    # Assert
    assert first != second
    assert DATASET_ID_PATTERN.fullmatch(first)
    assert "cohort" not in first


def test_vacuumIfFreePages_rowsDeletedBeforeSecureDelete_bytesGone(tmp_path: Path) -> None:
    # Arrange: a DB written without secure_delete, like files from before this change.
    from sqlalchemy import create_engine as make_engine

    from clinical_analytics.api.db.database import vacuum_if_free_pages

    plain = make_engine(f"sqlite:///{tmp_path / 'old.db'}")
    with plain.begin() as conn:
        conn.exec_driver_sql("CREATE TABLE m (content TEXT)")
        for i in range(300):
            conn.exec_driver_sql(f"INSERT INTO m VALUES ('pangolin-question-{i:04d}' || hex(randomblob(64)))")
    with plain.begin() as conn:
        # Some SQLite builds default secure_delete on; old files were written where it was off.
        conn.exec_driver_sql("PRAGMA secure_delete=OFF")
        conn.exec_driver_sql("DELETE FROM m")
    plain.dispose()
    assert b"pangolin-question-" in (tmp_path / "old.db").read_bytes()

    # Act
    vacuumed = vacuum_if_free_pages(plain)
    again = vacuum_if_free_pages(plain)
    plain.dispose()

    # Assert
    assert vacuumed is True
    assert again is False
    assert b"pangolin-question-" not in (tmp_path / "old.db").read_bytes()
