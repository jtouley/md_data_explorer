"""Upload writes a contract + quality report; GET /api/datasets/{id}/quality serves (or backfills) it."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

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
