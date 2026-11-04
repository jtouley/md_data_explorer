"""Tests for one-click sample clinical cohort demo dataset."""

import polars as pl
import pytest

from clinical_analytics.datasets.demo import (
    DEMO_DATASET_NAME,
    build_demo_metadata,
    get_demo_csv_path,
    load_sample_dataset,
)
from clinical_analytics.datasets.uploaded.definition import UploadedDataset


class TestDemoDataset:
    """One-click demo CSV registration into UserDatasetStorage."""

    def test_demo_csv_path_points_to_existing_sample_cohort(self):
        """Bundled demo CSV exists with required clinical columns."""
        # Arrange / Act
        path = get_demo_csv_path()

        # Assert
        assert path.exists()
        assert path.name == "sample_cohort.csv"
        df = pl.read_csv(path)
        for col in ("patient_id", "outcome", "age", "sex", "diabetes", "hypertension"):
            assert col in df.columns
        assert df.height >= 50
        assert set(df["outcome"].unique().to_list()) <= {0, 1}

    def test_build_demo_metadata_includes_complete_variable_mapping(self):
        """Metadata skips the mapping wizard with patient_id, outcome, predictors."""
        # Arrange / Act
        metadata = build_demo_metadata()

        # Assert
        assert metadata["dataset_name"] == DEMO_DATASET_NAME
        mapping = metadata["variable_mapping"]
        assert mapping["patient_id"] == "patient_id"
        assert mapping["outcome"] == "outcome"
        assert "age" in mapping["predictors"]
        assert "sex" in mapping["predictors"]
        assert "diabetes" in mapping["predictors"]
        assert "hypertension" in mapping["predictors"]

    def test_load_sample_dataset_registers_ready_semantic_layer(self, upload_storage):
        """load_sample_dataset saves via UserDatasetStorage without mapping wizard."""
        # Arrange / Act
        success, message, upload_id = load_sample_dataset(upload_storage, overwrite=True)

        # Assert
        assert success is True, message
        assert upload_id is not None
        metadata = upload_storage.get_upload_metadata(upload_id)
        assert metadata is not None
        assert metadata["dataset_name"] == DEMO_DATASET_NAME
        assert metadata["variable_mapping"]["outcome"] == "outcome"

        dataset = UploadedDataset(upload_id, storage=upload_storage)
        semantic = dataset.get_semantic_layer()
        assert semantic is not None
        assert "outcome" in semantic.config.get("outcomes", {}) or semantic.get_base_view() is not None

    def test_load_sample_dataset_missing_csv_raises_file_not_found(self, upload_storage, tmp_path):
        """Missing demo CSV fails loud with FileNotFoundError."""
        # Arrange
        missing = tmp_path / "missing.csv"

        # Act / Assert
        with pytest.raises(FileNotFoundError, match="sample cohort"):
            load_sample_dataset(upload_storage, csv_path=missing)

    def test_load_sample_dataset_overwrite_idempotent(self, upload_storage):
        """Re-loading the sample dataset overwrites cleanly."""
        # Arrange
        success1, _, id1 = load_sample_dataset(upload_storage, overwrite=True)
        assert success1 is True

        # Act
        success2, _, id2 = load_sample_dataset(upload_storage, overwrite=True)

        # Assert
        assert success2 is True
        assert id2 is not None
        # Same dataset_name; overwrite may preserve upload_id
        meta = upload_storage.get_upload_metadata(id2)
        assert meta is not None
        assert meta["dataset_name"] == DEMO_DATASET_NAME


def _user_csv() -> bytes:
    """Synthetic user cohort, distinct from the bundled sample."""
    return (
        pl.DataFrame(
            {
                "patient_id": [f"U{i:03d}" for i in range(150)],
                "ldl": [100 + i for i in range(150)],
                "arm": ["a" if i % 2 else "b" for i in range(150)],
            }
        )
        .write_csv()
        .encode()
    )


@pytest.mark.parametrize("sample_loads", [1, 2], ids=["first_load", "reload"])
def test_loadSampleDataset_userDatasetHasDemoName_userDatasetUntouchedAndDemoSeparate(
    upload_storage, sample_loads: int
):
    """Only the demo is created or replaced; a clinician's dataset with the same name is not."""
    # Arrange
    ok, message, user_id = upload_storage.save_upload(
        file_bytes=_user_csv(),
        original_filename="mine.csv",
        metadata={"dataset_name": DEMO_DATASET_NAME},
    )
    assert ok is True, message
    user_meta_before = upload_storage.get_upload_metadata(user_id)

    # Act
    demo_ids = [load_sample_dataset(upload_storage, overwrite=True) for _ in range(sample_loads)]

    # Assert
    assert all(success for success, _, _ in demo_ids)
    ids = {upload_id for _, _, upload_id in demo_ids}
    assert len(ids) == 1, "reloading the sample reuses the demo's own id"
    demo_id = ids.pop()
    assert demo_id != user_id
    assert upload_storage.get_upload_metadata(user_id) == user_meta_before
    assert upload_storage.get_upload_metadata(demo_id)["is_demo"] is True
    assert upload_storage.get_upload_metadata(demo_id)["dataset_name"] == f"{DEMO_DATASET_NAME} (sample)"
    assert set(upload_storage.get_upload_data(user_id).collect_schema().names()) == {"patient_id", "ldl", "arm"}


def test_loadSampleDataset_noNameClash_keepsPlainSampleName(upload_storage):
    """Without a user dataset of that name, the sample keeps its plain display name."""
    # Arrange / Act
    ok, message, demo_id = load_sample_dataset(upload_storage, overwrite=True)

    # Assert
    assert ok is True, message
    assert upload_storage.get_upload_metadata(demo_id)["dataset_name"] == DEMO_DATASET_NAME
