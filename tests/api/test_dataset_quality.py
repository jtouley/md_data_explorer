"""Upload writes a contract + quality report; GET /api/datasets/{id}/quality serves (or backfills) it."""

from __future__ import annotations

import shutil
import threading
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from clinical_analytics.contracts.errors import ContractStoreError
from clinical_analytics.contracts.store import contract_path

pytestmark = pytest.mark.usefixtures("isolated_data_home")

_COHORT = Path(__file__).resolve().parents[1] / "fixtures" / "synthetic" / "cohort.csv"


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
    assert response.json()["detail"] == "Could not check this dataset's quality (ContractValidationError)."


def test_uploadDataset_qualityCheckFails_responseSaysNotRun(test_client: TestClient, monkeypatch):
    # Arrange
    def broken(*_args, **_kwargs):
        raise ContractStoreError("disk full")

    monkeypatch.setattr("clinical_analytics.contracts.service.check_upload", broken)

    # Act
    body = _upload_cohort(test_client)

    # Assert: the upload stands, and the failed check is visible rather than silently absent.
    assert body["status"] == "ready"
    assert body["quality"] == {"status": "not_run", "reason": "the check failed (ContractStoreError)", "summary": []}


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
