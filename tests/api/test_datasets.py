"""Tests for Dataset API routes.

Phase 1 of Electron UI Migration: Expose dataset listing and metadata via API.

Tests cover:
- GET /api/datasets - List available datasets
- GET /api/datasets/{dataset_id} - Get dataset metadata
- GET /api/datasets/{dataset_id}/preview - Get sample rows
"""

from unittest.mock import patch

import polars as pl
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def test_client():
    """Create FastAPI test client."""
    from clinical_analytics.api.main import app

    return TestClient(app)


@pytest.fixture
def mock_empty_uploads():
    """Mock UploadedDatasetFactory to return empty list."""
    with patch("clinical_analytics.api.routes.datasets.UploadedDatasetFactory") as mock_factory:
        mock_factory.list_available_uploads.return_value = []
        yield mock_factory


@pytest.fixture
def mock_uploads_with_data():
    """Mock UploadedDatasetFactory to return test datasets."""
    with patch("clinical_analytics.api.routes.datasets.UploadedDatasetFactory") as mock_factory:
        mock_factory.list_available_uploads.return_value = [
            {
                "upload_id": "test_upload_001",
                "dataset_name": "Test Dataset 1",
                "upload_timestamp": "2026-01-20T12:00:00+00:00",
                "row_count": 100,
                "column_count": 5,
                "columns": ["patient_id", "age", "sex", "outcome", "treatment"],
            },
            {
                "upload_id": "test_upload_002",
                "dataset_name": "Test Dataset 2",
                "upload_timestamp": "2026-01-20T13:00:00+00:00",
                "row_count": 50,
                "column_count": 3,
                "columns": ["patient_id", "mortality", "los"],
            },
        ]
        yield mock_factory


class TestDatasetListEndpoint:
    """Tests for GET /api/datasets endpoint."""

    def test_datasets_list_empty_returns_empty_list(self, test_client, mock_empty_uploads):
        """When no datasets exist, return empty list."""
        # Act
        response = test_client.get("/api/datasets")

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["datasets"] == []
        assert data["total"] == 0

    def test_datasets_list_with_uploads_returns_datasets(self, test_client, mock_uploads_with_data):
        """When datasets exist, return dataset summaries."""
        # Act
        response = test_client.get("/api/datasets")

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert len(data["datasets"]) == 2
        assert data["total"] == 2

        # Verify first dataset fields
        dataset = data["datasets"][0]
        assert dataset["dataset_id"] == "test_upload_001"
        assert dataset["name"] == "Test Dataset 1"
        assert dataset["source"] == "uploaded"
        assert dataset["row_count"] == 100


class TestDatasetDetailEndpoint:
    """Tests for GET /api/datasets/{dataset_id} endpoint."""

    def test_datasets_get_existing_returns_detail(self, test_client, mock_uploads_with_data):
        """When dataset exists, return full metadata."""
        # Arrange
        mock_uploads_with_data.create_dataset.return_value.get_info.return_value = {
            "upload_id": "test_upload_001",
            "name": "Test Dataset 1",
            "uploaded_at": "2026-01-20T12:00:00+00:00",
            "original_filename": "test.csv",
            "row_count": 100,
            "column_count": 5,
            "columns": ["patient_id", "age", "sex", "outcome", "treatment"],
        }

        # Act
        response = test_client.get("/api/datasets/test_upload_001")

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["dataset_id"] == "test_upload_001"
        assert data["name"] == "Test Dataset 1"
        assert data["source"] == "uploaded"
        # schema contains column:dtype mappings
        assert "patient_id" in data["schema"]
        assert len(data["tables"]) >= 1

    def test_datasets_get_missing_returns_404(self, test_client, mock_empty_uploads):
        """When dataset doesn't exist, return 404."""
        # Arrange
        mock_empty_uploads.create_dataset.side_effect = ValueError("Upload not_found_id not found")

        # Act
        response = test_client.get("/api/datasets/not_found_id")

        # Assert
        assert response.status_code == 404
        data = response.json()
        assert "not found" in data["detail"].lower()


class TestDatasetPreviewEndpoint:
    """Tests for GET /api/datasets/{dataset_id}/preview endpoint."""

    def test_datasets_preview_returns_rows(self, test_client, tmp_path):
        """Preview endpoint returns sample rows."""
        # Arrange: Create mock storage with CSV file
        upload_id = "preview_test_001"
        upload_dir = tmp_path / "uploads"
        upload_dir.mkdir(parents=True)

        # Create unified cohort CSV directly
        df = pl.DataFrame(
            {
                "patient_id": ["P001", "P002", "P003"],
                "age": [25, 35, 45],
                "outcome": [0, 1, 0],
            }
        )
        csv_path = upload_dir / f"{upload_id}_unified_cohort.csv"
        df.write_csv(csv_path)

        # Mock storage to return metadata and use our upload_dir
        with patch("clinical_analytics.api.routes.datasets.UserDatasetStorage") as mock_storage_cls:
            mock_storage = mock_storage_cls.return_value
            mock_storage.get_upload_metadata.return_value = {
                "upload_id": upload_id,
                "dataset_name": "Preview Test",
            }
            mock_storage.upload_dir = upload_dir

            # Act
            response = test_client.get(f"/api/datasets/{upload_id}/preview")

            # Assert
            assert response.status_code == 200
            data = response.json()
            assert data["dataset_id"] == upload_id
            assert len(data["rows"]) == 3
            assert data["total_rows"] == 3
            assert "patient_id" in data["columns"]

    def test_datasets_preview_respects_limit(self, test_client, tmp_path):
        """Preview endpoint respects limit parameter."""
        # Arrange: Create dataset with many rows
        upload_id = "limit_test_001"
        upload_dir = tmp_path / "uploads"
        upload_dir.mkdir(parents=True)

        df = pl.DataFrame(
            {
                "patient_id": [f"P{i:03d}" for i in range(100)],
                "age": [25 + i for i in range(100)],
            }
        )
        csv_path = upload_dir / f"{upload_id}_unified_cohort.csv"
        df.write_csv(csv_path)

        with patch("clinical_analytics.api.routes.datasets.UserDatasetStorage") as mock_storage_cls:
            mock_storage = mock_storage_cls.return_value
            mock_storage.get_upload_metadata.return_value = {
                "upload_id": upload_id,
                "dataset_name": "Limit Test",
            }
            mock_storage.upload_dir = upload_dir

            # Act
            response = test_client.get(f"/api/datasets/{upload_id}/preview?limit=5")

            # Assert
            assert response.status_code == 200
            data = response.json()
            assert len(data["rows"]) == 5
            assert data["total_rows"] == 100

    def test_datasets_preview_missing_returns_404(self, test_client):
        """Preview endpoint returns 404 for missing dataset."""
        # Arrange
        with patch("clinical_analytics.api.routes.datasets.UserDatasetStorage") as mock_storage_cls:
            mock_storage = mock_storage_cls.return_value
            mock_storage.get_upload_metadata.return_value = None

            # Act
            response = test_client.get("/api/datasets/not_found/preview")

            # Assert
            assert response.status_code == 404
