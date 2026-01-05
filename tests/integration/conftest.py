"""Shared fixtures for integration tests."""

from pathlib import Path

import polars as pl
import pytest
from clinical_analytics.ui.storage.user_datasets import UserDatasetStorage


@pytest.fixture
def integration_tmp_dir(tmp_path: Path) -> Path:
    """Temp directory with diagnostic logging."""
    return tmp_path / "integration_tests"


@pytest.fixture
def real_storage(integration_tmp_dir: Path) -> UserDatasetStorage:
    """Real UserDatasetStorage (not mocked)."""
    upload_dir = integration_tmp_dir / "uploads"
    return UserDatasetStorage(upload_dir=upload_dir)


@pytest.fixture
def sample_csv_file(integration_tmp_dir: Path) -> Path:
    """Real CSV file for upload testing (must be at least 1KB)."""
    csv_path = integration_tmp_dir / "sample.csv"
    # Create larger dataset to meet 1KB minimum size requirement
    df = pl.DataFrame(
        {
            "patient_id": [f"P{i:04d}" for i in range(200)],
            "age": [25 + (i % 50) for i in range(200)],
            "outcome": [i % 2 for i in range(200)],
            "category": [f"cat_{i % 5}" for i in range(200)],
            "value": [10.0 + (i % 100) for i in range(200)],
        }
    )
    csv_path.write_text(df.write_csv())
    return csv_path
