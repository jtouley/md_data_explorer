# SPDX-License-Identifier: MIT
"""Every file UserDatasetStorage opens or writes stays inside its data directory.

Ids are checked before they name a file, paths read from metadata are refused when they resolve
elsewhere, and a legacy upload's conversion writes owner-only Parquet inside the data directory.
"""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path

import polars as pl
import polars.testing as plt
import pytest

from clinical_analytics.storage.dataset_deletion import InvalidDatasetIdError
from clinical_analytics.ui.storage.user_datasets import (
    COHORT_PARQUET_KEY,
    STORAGE_LAYOUT,
    STORAGE_LAYOUT_KEY,
    StoredPathError,
    UserDatasetStorage,
    cohort_parquet_path,
    table_parquet_path,
)

BAD_IDS = ["../../../../outside_datahome/evil", "../evil", "a/b", "..", "", ".hidden", "ok\n", "CON", "nul", "Com1"]


@pytest.fixture
def nested_storage(tmp_path: Path) -> UserDatasetStorage:
    """Storage whose data directory (``home/data``) leaves room for files outside it in tmp_path.

    ``upload_storage`` (conftest) puts uploads at tmp_path, so its data directory is tmp_path's
    parent and nothing in tmp_path counts as outside; these tests need an outside.
    """
    return UserDatasetStorage(upload_dir=tmp_path / "home" / "data" / "uploads")


def _write_legacy_upload(storage: UserDatasetStorage, upload_id: str, df: pl.DataFrame, metadata: dict) -> None:
    df.write_csv(storage.raw_dir / f"{upload_id}.csv")
    (storage.metadata_dir / f"{upload_id}.json").write_text(json.dumps({**metadata, "upload_id": upload_id}))


class TestIdCheckedBeforePath:
    @pytest.mark.parametrize("upload_id", BAD_IDS)
    def test_metadata_path_invalidId_raises(self, upload_storage, upload_id):
        # Arrange / Act / Assert
        with pytest.raises(InvalidDatasetIdError):
            upload_storage._metadata_path(upload_id)

    @pytest.mark.parametrize("upload_id", BAD_IDS)
    def test_parquet_paths_invalidId_raise(self, upload_storage, upload_id):
        # Arrange / Act / Assert
        with pytest.raises(InvalidDatasetIdError):
            table_parquet_path(upload_storage, upload_id, "t", "v1")
        with pytest.raises(InvalidDatasetIdError):
            cohort_parquet_path(upload_storage, upload_id, "v1", None)

    def test_metadata_path_plainId_isInsideMetadataDir(self, upload_storage):
        # Arrange
        upload_id = "user_upload_" + "a" * 32

        # Act
        path = upload_storage._metadata_path(upload_id)

        # Assert
        assert path == upload_storage.metadata_dir / f"{upload_id}.json"

    def test_get_upload_metadata_traversalId_neverReadsOutsideFile(self, nested_storage, tmp_path):
        # Arrange: a metadata-shaped file outside the data directory
        outside = tmp_path / "outside_datahome"
        outside.mkdir()
        (outside / "evil.json").write_text(json.dumps({"upload_id": "evil"}))
        traversal = os.path.relpath(outside / "evil", nested_storage.metadata_dir)

        # Act / Assert
        with pytest.raises(InvalidDatasetIdError):
            nested_storage.get_upload_metadata(traversal)

    def test_stored_dataset_traversalLegacyId_writesNothingOutside(
        self, nested_storage, tmp_path, sample_upload_df, sample_upload_metadata
    ):
        # Arrange: a legacy-layout upload planted outside the data directory
        outside = tmp_path / "outside_datahome"
        (outside / "raw").mkdir(parents=True)
        sample_upload_df.write_csv(outside / "raw" / "evil.csv")
        (outside / "evil.json").write_text(json.dumps(sample_upload_metadata))
        before = sorted(p.relative_to(outside) for p in outside.rglob("*"))
        traversal = os.path.relpath(outside / "evil", nested_storage.metadata_dir)

        # Act
        with pytest.raises(InvalidDatasetIdError):
            nested_storage.stored_dataset(traversal)

        # Assert
        assert sorted(p.relative_to(outside) for p in outside.rglob("*")) == before


class TestStoredPathsFromMetadata:
    @pytest.mark.parametrize("key", ["cohort", "table"])
    def test_stored_dataset_metadataPathOutsideDataDir_refused(self, nested_storage, tmp_path, sample_upload_df, key):
        # Arrange: typed-layout metadata whose Parquet path points outside the data directory
        outside = tmp_path / "outside.parquet"
        sample_upload_df.write_parquet(outside)
        inside = nested_storage.upload_dir.parent / "parquet" / "inside.parquet"
        inside.parent.mkdir(parents=True)
        sample_upload_df.write_parquet(inside)
        metadata = {
            "upload_id": "tampered",
            STORAGE_LAYOUT_KEY: STORAGE_LAYOUT,
            COHORT_PARQUET_KEY: str(outside if key == "cohort" else inside),
            "parquet_paths": {"t": str(outside if key == "table" else inside)},
        }
        (nested_storage.metadata_dir / "tampered.json").write_text(json.dumps(metadata))

        # Act / Assert
        with pytest.raises(StoredPathError):
            nested_storage.stored_dataset("tampered")

    @pytest.mark.parametrize(
        "value",
        ["../../../outside.parquet", "relative.parquet", ""],
        ids=["dotdot", "relative", "empty"],
    )
    def test_stored_dataset_unsafeCohortValue_refused(self, nested_storage, value):
        # Arrange
        if value.startswith(".."):
            value = str(nested_storage.upload_dir.parent / "parquet" / value)
        metadata = {"upload_id": "tampered", STORAGE_LAYOUT_KEY: STORAGE_LAYOUT, COHORT_PARQUET_KEY: value}
        (nested_storage.metadata_dir / "tampered.json").write_text(json.dumps(metadata))

        # Act / Assert
        with pytest.raises(StoredPathError):
            nested_storage.stored_dataset("tampered")

    def test_stored_dataset_symlinkLeavingDataDir_refused(self, nested_storage, tmp_path, sample_upload_df):
        # Arrange: a path inside the data directory that is a symlink to a file outside it
        outside = tmp_path / "outside.parquet"
        sample_upload_df.write_parquet(outside)
        link = nested_storage.upload_dir.parent / "parquet" / "link.parquet"
        link.parent.mkdir(parents=True)
        try:
            link.symlink_to(outside)
        except OSError:
            pytest.skip("symlinks not available on this file system")
        metadata = {"upload_id": "linked", STORAGE_LAYOUT_KEY: STORAGE_LAYOUT, COHORT_PARQUET_KEY: str(link)}
        (nested_storage.metadata_dir / "linked.json").write_text(json.dumps(metadata))

        # Act / Assert
        with pytest.raises(StoredPathError):
            nested_storage.stored_dataset("linked")

    def test_legacy_conversion_storedRelpathEscapingRawDir_refused(
        self, nested_storage, tmp_path, sample_upload_df, sample_upload_metadata
    ):
        # Arrange: a legacy upload whose stored_relpath climbs out to a file outside
        sample_upload_df.write_csv(tmp_path / "outside.csv")
        relpath = os.path.relpath(tmp_path / "outside.csv", nested_storage.raw_dir)
        metadata = {**sample_upload_metadata, "upload_id": "legacy_rel", "stored_relpath": relpath}
        (nested_storage.metadata_dir / "legacy_rel.json").write_text(json.dumps(metadata))

        # Act / Assert
        with pytest.raises(StoredPathError):
            nested_storage.stored_dataset("legacy_rel")


class TestLegacyConversionOutput:
    def test_stored_dataset_legacyUpload_writesOwnerOnlyParquetInsideDataDir(
        self, nested_storage, sample_upload_df, sample_upload_metadata
    ):
        # Arrange
        _write_legacy_upload(nested_storage, "legacy_one", sample_upload_df, sample_upload_metadata)
        data_dir = nested_storage.upload_dir.parent.resolve()

        # Act
        stored = nested_storage.stored_dataset("legacy_one")

        # Assert
        assert stored.cohort_path is not None
        assert stored.cohort_path.resolve().is_relative_to(data_dir)
        assert Path(stored.metadata[COHORT_PARQUET_KEY]).resolve().is_relative_to(data_dir)
        if os.name != "nt":  # Windows has no POSIX mode bits; the data home's ACLs apply
            assert stat.S_IMODE(stored.cohort_path.stat().st_mode) == 0o600
        plt.assert_frame_equal(stored.scan_cohort().collect(), sample_upload_df, check_dtypes=False)
