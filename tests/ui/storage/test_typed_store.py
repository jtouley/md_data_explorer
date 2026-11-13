# SPDX-License-Identifier: MIT
"""One typed source of truth per dataset version: typed Parquet in, the same frame out.

Covers the storage review findings: the semantic layer, the contract checks and the preview all
read the one typed Parquet written at upload (DQ H1, dbt H1, DuckDB M5); no pandas round trip
(DuckDB H2); every write is atomic and versioned (DuckDB H1); no read-only/read-write clash on
analytics.duckdb (DuckDB H3); the overwrite drift check compares stored types (dbt M1).
"""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from clinical_analytics.datasets.uploaded.definition import UploadedDataset
from clinical_analytics.ui.storage import user_datasets
from clinical_analytics.ui.storage.user_datasets import UserDatasetStorage

MAPPING = {"patient_id": "patient_id", "outcome": "outcome", "predictors": ["age"]}


def _cohort_csv(n: int = 200, *, age_offset: int = 40, with_date: bool = True) -> bytes:
    """Synthetic cohort over the 1 KB upload floor: int with nulls, leading-zero zip, ISO date."""
    header = "patient_id,age,zip,outcome" + (",visit_date" if with_date else "")
    rows = []
    for i in range(n):
        age = "" if i % 7 == 0 else str(age_offset + i % 30)
        row = f"P{i:04d},{age},0213{i % 10},{i % 2}"
        if with_date:
            row += f",2024-01-{1 + i % 28:02d}"
        rows.append(row)
    return ("\n".join([header, *rows]) + "\n").encode()


def _upload(storage: UserDatasetStorage, data: bytes, name: str = "cohort", **kwargs) -> str:
    ok, message, upload_id = storage.save_upload(
        data, f"{name}.csv", {"dataset_name": name, "variable_mapping": dict(MAPPING)}, **kwargs
    )
    assert ok, message
    assert upload_id is not None
    return upload_id


def _layer_frame(storage: UserDatasetStorage, upload_id: str) -> pl.DataFrame:
    dataset = UploadedDataset(upload_id, storage=storage)
    dataset.load()
    return dataset.get_semantic_layer().raw.to_polars()


@pytest.fixture
def store(tmp_path: Path) -> UserDatasetStorage:
    """Storage in its own data dir (parquet/, contracts/ sit next to uploads/)."""
    return UserDatasetStorage(upload_dir=tmp_path / "data" / "uploads")


# --- one typed source of truth -------------------------------------------------------------


def test_upload_writesTypedCohortParquet_withStagedTypes(store):
    # Arrange
    data = _cohort_csv()

    # Act
    upload_id = _upload(store, data)

    # Assert
    metadata = store.get_upload_metadata(upload_id)
    stored = pl.read_parquet(metadata["cohort_parquet"])
    assert stored.schema["age"] == pl.Int64
    assert stored.schema["zip"] == pl.String
    assert stored.schema["visit_date"] == pl.Date
    assert stored["zip"].head(2).to_list() == ["02130", "02131"]
    assert stored["age"].null_count() > 0


def test_upload_semanticLayer_readsTheStoredParquet(store):
    # Arrange
    upload_id = _upload(store, _cohort_csv())
    stored = pl.read_parquet(store.get_upload_metadata(upload_id)["cohort_parquet"])

    # Act
    layer_frame = _layer_frame(store, upload_id)

    # Assert
    assert_frame_equal(layer_frame, stored)


def test_upload_preview_readsTheStoredParquet(store):
    # Arrange
    upload_id = _upload(store, _cohort_csv())
    stored = pl.read_parquet(store.get_upload_metadata(upload_id)["cohort_parquet"])

    # Act
    preview = store.get_upload_data(upload_id).collect()

    # Assert
    assert_frame_equal(preview, stored)


def test_upload_qualityChecks_readTheStoredFrame(store, monkeypatch):
    # Arrange
    seen: list[pl.DataFrame] = []
    from clinical_analytics.contracts import service

    real = service.check_upload_or_log

    def spy(data_dir, df, **kwargs):
        seen.append(df)
        return real(data_dir, df, **kwargs)

    monkeypatch.setattr(service, "check_upload_or_log", spy)

    # Act
    upload_id = _upload(store, _cohort_csv())

    # Assert
    stored = pl.read_parquet(store.get_upload_metadata(upload_id)["cohort_parquet"])
    assert_frame_equal(seen[0], stored)


def test_upload_keepsOriginalBytes_andNoCsvCopies(store):
    # Arrange
    data = _cohort_csv()

    # Act
    upload_id = _upload(store, data)

    # Assert
    metadata = store.get_upload_metadata(upload_id)
    assert Path(metadata["raw_landing"]).read_bytes() == data
    assert not (store.raw_dir / f"{upload_id}.csv").exists()
    assert not (store.raw_dir / f"{upload_id}_tables").exists()
    assert not (store.upload_dir.parent / "analytics.duckdb").exists()


def test_upload_viewCannotBeBuilt_failsWithReason_andNothingIsListed(store, monkeypatch):
    # Arrange
    import duckdb

    from clinical_analytics.core.semantic import SemanticLayer

    def broken(self):
        raise duckdb.ConversionException("Could not convert string 'x' to INT64")

    monkeypatch.setattr(SemanticLayer, "get_base_view", broken)

    # Act
    ok, message, upload_id = store.save_upload(
        _cohort_csv(), "cohort.csv", {"dataset_name": "cohort", "variable_mapping": dict(MAPPING)}
    )

    # Assert
    assert not ok
    assert upload_id is None
    assert "could not be prepared for questions" in message
    assert store.list_uploads() == []


def test_upload_typeNotes_reportFallbackColumn(store):
    # Arrange: creat has one censored value, so only creat stays text
    rows = ["patient_id,age,creat,outcome"] + [f"P{i},{40 + i % 30},{80 + i % 9},{i % 2}" for i in range(199)]
    rows.append("P199,50,<5,1")
    data = ("\n".join(rows) + "\n").encode()

    # Act
    upload_id = _upload(store, data)

    # Assert
    notes = store.get_upload_metadata(upload_id)["column_types"]
    creat = next(n for n in notes if n["column"] == "creat")
    assert creat["reason"] == "not_numeric"
    assert creat["stored_type"] == "String"


def test_upload_excelUnitsRow_isNotCountedAsPatient(store):
    # Arrange
    import io

    import openpyxl

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(["patient_id", "age", "ldl", "outcome"])
    ws.append([None, "years", "mg/dL", None])
    for i in range(200):
        ws.append([f"P{i:03d}", 30 + i % 40, 100 + i, i % 2])
    buf = io.BytesIO()
    wb.save(buf)

    # Act
    ok, message, upload_id = store.save_upload(
        buf.getvalue(), "units.xlsx", {"dataset_name": "units", "variable_mapping": dict(MAPPING)}
    )

    # Assert
    assert ok, message
    metadata = store.get_upload_metadata(upload_id)
    assert metadata["row_count"] == 200
    assert metadata["column_units"] == {"age": "years", "ldl": "mg/dL"}


# --- legacy uploads (CSV only, no cohort Parquet) -------------------------------------------


def _make_legacy(store: UserDatasetStorage, upload_id: str = "user_upload_legacy0001") -> str:
    """The layout before typed storage: raw/{id}.csv and metadata with no cohort_parquet."""
    (store.raw_dir / f"{upload_id}.csv").write_bytes(_cohort_csv())
    metadata = {
        "upload_id": upload_id,
        "dataset_name": "legacy",
        "upload_timestamp": "2025-01-01T00:00:00",
        "original_filename": "legacy.csv",
        "variable_mapping": dict(MAPPING),
    }
    (store.metadata_dir / f"{upload_id}.json").write_text(json.dumps(metadata))
    return upload_id


def test_legacy_firstLoad_convertsToTypedParquet(store):
    # Arrange
    upload_id = _make_legacy(store)

    # Act
    layer_frame = _layer_frame(store, upload_id)

    # Assert
    metadata = store.get_upload_metadata(upload_id)
    stored = pl.read_parquet(metadata["cohort_parquet"])
    assert_frame_equal(layer_frame, stored)
    assert stored.schema["zip"] == pl.String
    assert stored.schema["visit_date"] == pl.Date


def test_legacy_conversionCannotBeSaved_fallsBackToStagedCsv(store, monkeypatch):
    # Arrange
    upload_id = _make_legacy(store)

    def refuse(*args, **kwargs):
        raise OSError(30, "Read-only file system")

    monkeypatch.setattr(user_datasets, "atomic_write", refuse)

    # Act
    layer_frame = _layer_frame(store, upload_id)

    # Assert
    assert "cohort_parquet" not in store.get_upload_metadata(upload_id)
    assert layer_frame.schema["zip"] == pl.String
    assert layer_frame.height == 200


# --- atomic, versioned writes ---------------------------------------------------------------


@pytest.mark.parametrize("crash_at", ["parquet", "raw_landing", "metadata"])
def test_overwrite_crashMidSave_previousVersionLoadsIntact(store, monkeypatch, crash_at):
    # Arrange
    upload_id = _upload(store, _cohort_csv(), name="trial")
    before = _layer_frame(store, upload_id)
    real = user_datasets.atomic_write

    def crash(path, writer, **kwargs):
        name = Path(path).name
        hit = {
            "parquet": name.endswith(".parquet"),
            "raw_landing": Path(path).parent.parent == store.raw_dir,
            "metadata": name.endswith(".json"),
        }[crash_at]
        if hit:
            writer(Path(path).with_name(f".{name}.crash.part"))  # half-done temp file
            raise OSError(5, "simulated crash")
        return real(path, writer, **kwargs)

    monkeypatch.setattr(user_datasets, "atomic_write", crash)

    # Act
    ok, _, _ = store.save_upload(
        _cohort_csv(300, age_offset=60),
        "trial.csv",
        {"dataset_name": "trial", "variable_mapping": dict(MAPPING)},
        overwrite=True,
    )

    # Assert
    assert not ok
    assert_frame_equal(_layer_frame(store, upload_id), before)


def test_overwrite_keepsPreviousVersionFiles(store):
    # Arrange
    upload_id = _upload(store, _cohort_csv(), name="trial")
    v1_path = Path(store.get_upload_metadata(upload_id)["cohort_parquet"])
    v1_bytes = v1_path.read_bytes()

    # Act
    _upload(store, _cohort_csv(300, age_offset=60), name="trial", overwrite=True)

    # Assert
    v2_path = Path(store.get_upload_metadata(upload_id)["cohort_parquet"])
    assert v2_path != v1_path
    assert v1_path.read_bytes() == v1_bytes
    assert pl.read_parquet(v2_path).height == 300


def test_metadataWrite_leavesNoPartFiles(store):
    # Arrange / Act
    _upload(store, _cohort_csv())

    # Assert
    data_dir = store.upload_dir.parent
    assert list(data_dir.rglob("*.part")) == []


# --- overwrite drift check -------------------------------------------------------------------


def test_overwrite_identicalReupload_isAllowed(store):
    # Arrange
    upload_id = _upload(store, _cohort_csv(), name="same")

    # Act
    ok, message, again = store.save_upload(
        _cohort_csv(), "same.csv", {"dataset_name": "same", "variable_mapping": dict(MAPPING)}, overwrite=True
    )

    # Assert
    assert ok, message
    assert again == upload_id


def test_overwrite_realTypeChange_isBlocked(store):
    # Arrange: every visit_date becomes free text
    _upload(store, _cohort_csv(), name="drift")
    changed = _cohort_csv().replace(b"2024-01-", b"January day ")

    # Act
    ok, message, _ = store.save_upload(
        changed, "drift.csv", {"dataset_name": "drift", "variable_mapping": dict(MAPPING)}, overwrite=True
    )

    # Assert
    assert not ok
    assert "visit_date" in message


def test_overwrite_storedSchemaUnreadable_isRefused_notSkipped(store):
    # Arrange
    upload_id = _upload(store, _cohort_csv(), name="gone")
    Path(store.get_upload_metadata(upload_id)["cohort_parquet"]).unlink()

    # Act
    ok, message, _ = store.save_upload(
        _cohort_csv(), "gone.csv", {"dataset_name": "gone", "variable_mapping": dict(MAPPING)}, overwrite=True
    )

    # Assert
    assert not ok
    assert "stored version" in message


# --- DuckDB connection clash and partial layers --------------------------------------------


def test_layerBuild_whileStoreOpenReadWrite_registersEveryTable(store):
    # Arrange: an upload in flight holds a read-write connection to analytics.duckdb
    from clinical_analytics.storage.datastore import DataStore

    upload_id = _upload(store, _cohort_csv())
    held = DataStore(store.upload_dir.parent / "analytics.duckdb")

    # Act
    try:
        dataset = UploadedDataset(upload_id, storage=store)
        dataset.load()
        layer = dataset.get_semantic_layer()
        tables = {row[0] for row in layer.con.con.execute("SHOW TABLES").fetchall()}
    finally:
        held.close()

    # Assert: the cohort view plus the one stored table
    assert len(tables) == 2
    assert any(t.endswith("_raw") for t in tables)


def test_layerBuild_failsPartWay_leavesNoPartialLayer(store, monkeypatch):
    # Arrange
    from clinical_analytics.datasets.uploaded import definition
    from clinical_analytics.datasets.uploaded.definition import SemanticLayerBuildError

    upload_id = _upload(store, _cohort_csv())

    def broken(*args, **kwargs):
        raise OSError(5, "disk went away")

    monkeypatch.setattr(definition, "_register_stored_table", broken)
    dataset = UploadedDataset(upload_id, storage=store)
    dataset.load()

    # Act / Assert
    with pytest.raises(SemanticLayerBuildError):
        dataset.get_semantic_layer()
    assert dataset._semantic is None
