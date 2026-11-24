"""Upload writes a contract + quality report; GET /api/datasets/{id}/quality serves (or backfills) it."""

from __future__ import annotations

import shutil
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import yaml
from fastapi.testclient import TestClient

from clinical_analytics.contracts.declarations import Declarations
from clinical_analytics.contracts.errors import ContractStoreError
from clinical_analytics.contracts.store import contract_path
from clinical_analytics.storage.dataset_deletion import DeletionReport
from clinical_analytics.ui.storage.user_datasets import UserDatasetStorage

pytestmark = pytest.mark.usefixtures("isolated_data_home")

_COHORT = Path(__file__).resolve().parents[1] / "fixtures" / "synthetic" / "cohort.csv"


def _strip_semantics(data_dir: Path, upload_id: str, version: str) -> None:
    """Store the version's contract as before ADR017: without its semantics section."""
    path = contract_path(data_dir, upload_id, version)
    document = yaml.safe_load(path.read_text())
    document["customProperties"] = [p for p in document["customProperties"] if p["property"] != "semantics"]
    path.write_text(yaml.safe_dump(document))


@pytest.fixture
def data_home_storage(isolated_data_home: Path) -> UserDatasetStorage:
    """The upload store of the test's own data home (the one the API writes to)."""
    return UserDatasetStorage()


def _upload_cohort(test_client: TestClient) -> dict:
    response = test_client.post(
        "/api/datasets/upload",
        files={"file": ("cohort.csv", _COHORT.read_bytes(), "text/csv")},
        data={"dataset_name": "synthetic cohort"},
    )
    assert response.status_code == 200, response.text
    return response.json()


def test_uploadDataset_syntheticCohort_responseCarriesQualitySummary(test_client: TestClient, isolated_data_home):
    # Act
    body = _upload_cohort(test_client)

    # Assert
    quality = body["quality"]
    assert quality["summary"] == ["ldl: 5.8% missing"]
    assert (quality["failed"], quality["row_count"]) == (0, 400)
    version = quality["dataset_version"]
    assert contract_path(isolated_data_home / "data", body["upload_id"], version).is_file()


def test_getDatasetQuality_afterUpload_returnsUploadRun(test_client: TestClient):
    # Arrange
    body = _upload_cohort(test_client)

    # Act
    response = test_client.get(f"/api/datasets/{body['upload_id']}/quality")

    # Assert
    assert response.status_code == 200
    assert response.json() == body["quality"]


def test_getDatasetQuality_legacyUploadWithoutReport_backfillsContractAndReport(
    test_client: TestClient, isolated_data_home
):
    # Arrange: an upload made before contracts existed has neither directory.
    body = _upload_cohort(test_client)
    data_dir = isolated_data_home / "data"
    shutil.rmtree(data_dir / "contracts")
    shutil.rmtree(data_dir / "dq")

    # Act
    response = test_client.get(f"/api/datasets/{body['upload_id']}/quality")

    # Assert
    assert response.status_code == 200
    assert response.json()["summary"] == ["ldl: 5.8% missing"]
    assert contract_path(data_dir, body["upload_id"], body["quality"]["dataset_version"]).is_file()


@pytest.mark.parametrize(
    ("dataset_id", "status"),
    [("no_such_upload", 404), ("bad..id", 400)],
    ids=["unknown", "unsafe_id"],
)
def test_getDatasetQuality_badDatasetId_rejected(test_client: TestClient, dataset_id: str, status: int):
    # Act
    response = test_client.get(f"/api/datasets/{dataset_id}/quality")

    # Assert
    assert response.status_code == status


def test_getDatasetQuality_corruptContract_returns500WithoutDetailLeak(test_client: TestClient, isolated_data_home):
    # Arrange
    body = _upload_cohort(test_client)
    data_dir = isolated_data_home / "data"
    shutil.rmtree(data_dir / "dq")
    contract_path(data_dir, body["upload_id"], body["quality"]["dataset_version"]).write_text("{bad: [yaml")

    # Act
    response = test_client.get(f"/api/datasets/{body['upload_id']}/quality")

    # Assert
    assert response.status_code == 500
    assert response.json()["detail"] == "The saved data contract for this cohort is damaged."


def test_uploadDataset_qualityCheckFails_responseSaysNotRun(test_client: TestClient, monkeypatch):
    # Arrange
    def broken(*_args, **_kwargs):
        raise ContractStoreError("disk full")

    monkeypatch.setattr("clinical_analytics.contracts.service.check_upload", broken)

    # Act
    body = _upload_cohort(test_client)

    # Assert: the upload stands, and the failed check is visible rather than silently absent.
    assert body["status"] == "ready"
    quality = {k: v for k, v in body["quality"].items() if k != "facts"}
    assert quality == {"status": "not_run", "reason": "the check failed (ContractStoreError)", "summary": []}
    assert body["quality"]["facts"] == body["facts"]  # what the file holds is still said


def test_uploadDataset_runFailsAfterReportWritten_olderReportNotReturned(test_client: TestClient, monkeypatch):
    # Arrange: a report lands on disk, then the run fails; that report is not this upload's result.
    from clinical_analytics.contracts import service

    real = service.check_upload

    def report_then_fail(*args, **kwargs):
        real(*args, **kwargs)
        raise ContractStoreError("failed after the report")

    monkeypatch.setattr(service, "check_upload", report_then_fail)

    # Act
    body = _upload_cohort(test_client)

    # Assert
    assert body["quality"]["status"] == "not_run"


def test_getDatasetQuality_deleteDuringBackfill_leavesNoContractOrReport(
    test_client: TestClient, isolated_data_home, monkeypatch
):
    # Arrange: a legacy upload; the backfill blocks inside check_upload until released.
    body = _upload_cohort(test_client)
    upload_id = body["upload_id"]
    data_dir = isolated_data_home / "data"
    shutil.rmtree(data_dir / "contracts")
    shutil.rmtree(data_dir / "dq")
    from clinical_analytics.api.routes import datasets

    real = datasets.check_upload
    started, release = threading.Event(), threading.Event()

    def slow_check(*args, **kwargs):
        started.set()
        assert release.wait(10)
        return real(*args, **kwargs)

    monkeypatch.setattr(datasets, "check_upload", slow_check)
    results: dict[str, int] = {}

    def get_quality() -> None:
        results["get"] = test_client.get(f"/api/datasets/{upload_id}/quality").status_code

    def delete() -> None:
        results["delete"] = test_client.delete(f"/api/datasets/{upload_id}").status_code

    # Act
    getter = threading.Thread(target=get_quality)
    getter.start()
    assert started.wait(10)
    deleter = threading.Thread(target=delete)
    deleter.start()
    deleter.join(0.5)  # a correct delete waits for the in-flight backfill
    release.set()
    getter.join(30)
    deleter.join(30)

    # Assert
    assert results["delete"] == 200
    assert results["get"] == 200
    assert not (data_dir / "contracts" / upload_id).exists()
    assert not (data_dir / "dq" / upload_id).exists()


@pytest.mark.slow
@pytest.mark.integration
def test_getDatasetQuality_deletePurgesBetweenBackfillAndSemantics_waitsForTheResponse(
    test_client: TestClient, isolated_data_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange: a legacy upload. Once the backfill is done, and before the request reads its
    # declarations and the contract's semantics, a delete gets up to 0.3 s to purge the files; it
    # then pauses before removing the metadata. That is the interleaving behind CI run 45316573902
    # (metadata still there, contract and parquet gone: the semantics read raised), forced here
    # instead of left to the scheduler.
    body = _upload_cohort(test_client)
    upload_id = body["upload_id"]
    data_dir = isolated_data_home / "data"
    shutil.rmtree(data_dir / "contracts")
    shutil.rmtree(data_dir / "dq")
    from clinical_analytics.api.routes import datasets

    real_declarations, real_purge = datasets._upload_declarations, datasets.purge_dataset_files
    purged, release = threading.Event(), threading.Event()
    results: dict[str, int] = {}

    def purge_then_pause(*args: Any, before_metadata: Callable[[], None], **kwargs: Any) -> DeletionReport:
        def pause_before_metadata() -> None:
            purged.set()
            assert release.wait(10)
            before_metadata()

        return real_purge(*args, before_metadata=pause_before_metadata, **kwargs)

    def delete() -> None:
        results["delete"] = test_client.delete(f"/api/datasets/{upload_id}").status_code

    deleter = threading.Thread(target=delete)

    def declarations_after_purge_chance(*args: Any, **kwargs: Any) -> Declarations:
        deleter.start()
        purged.wait(0.3)  # a correct request still holds the files lock, so the purge cannot start
        return real_declarations(*args, **kwargs)

    monkeypatch.setattr(datasets, "purge_dataset_files", purge_then_pause)
    monkeypatch.setattr(datasets, "_upload_declarations", declarations_after_purge_chance)

    # Act
    try:
        response = test_client.get(f"/api/datasets/{upload_id}/quality")
    finally:
        release.set()
        deleter.join(30)

    # Assert
    assert not deleter.is_alive(), "the delete did not finish"
    assert response.status_code == 200, response.text
    assert response.json()["dataset_version"] == body["quality"]["dataset_version"]
    assert results.get("delete") == 200
    assert not (data_dir / "contracts" / upload_id).exists()
    assert not (data_dir / "dq" / upload_id).exists()


def test_getDatasetQuality_deletionMarkedWhileWaitingForLock_rejected409(test_client: TestClient, isolated_data_home):
    # Arrange: a legacy upload, with a delete holding the files lock and the deleting mark.
    from clinical_analytics.api.dependencies import _deleting_datasets, dataset_files_lock

    body = _upload_cohort(test_client)
    upload_id = body["upload_id"]
    shutil.rmtree(isolated_data_home / "data" / "dq")
    result: dict[str, int] = {}

    def get_quality() -> None:
        result["get"] = test_client.get(f"/api/datasets/{upload_id}/quality").status_code

    # Act: the request passes the first check, then waits on the lock while the delete marks the id.
    with dataset_files_lock(upload_id):
        getter = threading.Thread(target=get_quality)
        getter.start()
        getter.join(0.5)
        _deleting_datasets.add(upload_id)
    try:
        getter.join(30)
    finally:
        _deleting_datasets.discard(upload_id)

    # Assert
    assert result["get"] == 409
    assert not (isolated_data_home / "data" / "dq" / upload_id).exists()


def test_datasetFilesLock_released_entryNotKeptForever():
    # Arrange
    import gc

    from clinical_analytics.api.dependencies import _dataset_files_locks, dataset_files_lock

    # Act
    with dataset_files_lock("ds_lock_probe"):
        held = "ds_lock_probe" in _dataset_files_locks
    gc.collect()

    # Assert: a desktop session touching many cohorts does not grow the lock map without bound.
    assert held is True
    assert "ds_lock_probe" not in _dataset_files_locks


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.parametrize("route", ["quality", "semantics"])
def test_datasetRoutes_versionStoredBeforeAdr017_infersAfterReleasingFilesLock(
    test_client: TestClient, isolated_data_home: Path, monkeypatch: pytest.MonkeyPatch, route: str
) -> None:
    # Arrange: a contract without semantics, so the request infers them from the stored frame
    # (4 to 6 s on a 1.8M-row cohort). While it infers, another thread tries the files lock.
    from clinical_analytics.api.dependencies import dataset_files_lock
    from clinical_analytics.contracts import service

    body = _upload_cohort(test_client)
    upload_id = body["upload_id"]
    _strip_semantics(isolated_data_home / "data", upload_id, body["quality"]["dataset_version"])
    real = service.stored_semantics
    lock_free_while_inferring: list[bool] = []

    def infer_while_probing_lock(*args: Any, **kwargs: Any) -> Any:
        taken = threading.Event()

        def take_lock() -> None:
            with dataset_files_lock(upload_id):
                taken.set()

        threading.Thread(target=take_lock, daemon=True).start()
        lock_free_while_inferring.append(taken.wait(5))
        return real(*args, **kwargs)

    monkeypatch.setattr(service, "stored_semantics", infer_while_probing_lock)

    # Act
    response = test_client.get(f"/api/datasets/{upload_id}/{route}")

    # Assert: a delete would not have waited for the inference
    assert response.status_code == 200, response.text
    assert lock_free_while_inferring == [True]
    assert response.json()["facts"]


@pytest.mark.slow
@pytest.mark.integration
def test_getDatasetQuality_legacyVersionKeyStale_reportAndFactsForActiveVersion(
    test_client: TestClient, isolated_data_home: Path, data_home_storage: UserDatasetStorage
) -> None:
    # Arrange: version_history names the active version; the legacy dataset_version key is stale.
    import json

    body = _upload_cohort(test_client)
    upload_id = body["upload_id"]
    active = body["quality"]["dataset_version"]
    metadata_path = data_home_storage._metadata_path(upload_id)
    metadata = json.loads(metadata_path.read_text())
    assert any(v.get("is_active") and v.get("version") == active for v in metadata["version_history"])
    metadata["dataset_version"] = "stale_version"
    metadata_path.write_text(json.dumps(metadata))

    # Act
    response = test_client.get(f"/api/datasets/{upload_id}/quality")

    # Assert: the stored report of the active version, with its facts; nothing built for the stale key
    assert response.status_code == 200, response.text
    assert response.json() == body["quality"]
    assert not contract_path(isolated_data_home / "data", upload_id, "stale_version").exists()


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.parametrize("route", ["quality", "semantics"])
@pytest.mark.parametrize("damage", ["missing", "corrupt"])
def test_datasetRoutes_storedParquetUnreadable_logged500(
    test_client: TestClient, isolated_data_home: Path, data_home_storage: UserDatasetStorage, route: str, damage: str
) -> None:
    # Arrange: a version stored before ADR017 (semantics come from the stored frame), whose
    # stored cohort file is then lost or damaged.
    from clinical_analytics.api.routes.datasets import STORED_DATA_UNREADABLE

    body = _upload_cohort(test_client)
    upload_id = body["upload_id"]
    _strip_semantics(isolated_data_home / "data", upload_id, body["quality"]["dataset_version"])
    cohort = data_home_storage.stored_dataset(upload_id).cohort_path
    assert cohort is not None
    if damage == "missing":
        cohort.unlink()
    else:
        cohort.write_bytes(b"not a parquet file")

    # Act
    response = test_client.get(f"/api/datasets/{upload_id}/{route}")

    # Assert: a 500 that says what is wrong, not an unhandled exception
    assert response.status_code == 500
    assert response.json()["detail"] == STORED_DATA_UNREADABLE


@pytest.mark.slow
@pytest.mark.integration
def test_getDatasetSemantics_deletionMarkedWhileWaitingForLock_rejected409(
    test_client: TestClient, isolated_data_home: Path
) -> None:
    # Arrange: a delete holding the files lock and the deleting mark.
    from clinical_analytics.api.dependencies import _deleting_datasets, dataset_files_lock

    upload_id = _upload_cohort(test_client)["upload_id"]
    result: dict[str, int] = {}

    def get_semantics() -> None:
        result["get"] = test_client.get(f"/api/datasets/{upload_id}/semantics").status_code

    # Act: the request passes the first check, then waits on the lock while the delete marks the id.
    with dataset_files_lock(upload_id):
        getter = threading.Thread(target=get_semantics)
        getter.start()
        getter.join(0.5)
        waited = getter.is_alive()
        _deleting_datasets.add(upload_id)
    try:
        getter.join(30)
    finally:
        _deleting_datasets.discard(upload_id)

    # Assert
    assert waited is True
    assert result.get("get") == 409
