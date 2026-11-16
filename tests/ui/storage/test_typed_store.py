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
from clinical_analytics.storage import atomic
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

    monkeypatch.setattr(atomic, "atomic_write", refuse)

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
    real = atomic.atomic_write

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

    monkeypatch.setattr(atomic, "atomic_write", crash)

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


# --- multi-table cohort build (#92 report) and cleanup on failure ---------------------------


def _owned_files(store: UserDatasetStorage, upload_id: str) -> list[str]:
    data_dir = store.upload_dir.parent
    return sorted(str(p.relative_to(data_dir)) for p in data_dir.rglob("*") if upload_id in str(p) and p.is_file())


def test_saveTableList_tableLeftOutOfCohort_reportedInMetadataAndNotes(store, make_cohort_with_categorical):
    # Arrange: a small repeated-key table classifies as reference and is not joined
    from clinical_analytics.ui.storage.user_datasets import describe_cohort_build, save_table_list

    patients = make_cohort_with_categorical(n_patients=30)
    visits = pl.DataFrame({"patient_id": ["P001", "P001", "P002"], "sbp": [120, 130, 140]})
    tables = [{"name": "patients", "data": patients}, {"name": "visits", "data": visits}]

    # Act
    ok, message = save_table_list(store, tables, "user_upload_multi0001", {"original_filename": "m.zip"})

    # Assert
    assert ok, message
    report = store.get_upload_metadata("user_upload_multi0001")["cohort_build"]
    assert [t["table"] for t in report["excluded_tables"]] == ["visits"]
    lines = describe_cohort_build(report)
    assert lines[0].startswith("visits: not in the cohort")


def test_saveTableList_duplicateAnchorKey_failsClearly_andLeavesNoFiles(store, make_cohort_with_categorical):
    # Arrange
    from clinical_analytics.ui.storage.user_datasets import save_table_list

    cohort = make_cohort_with_categorical(n_patients=300)
    patients = pl.concat([cohort, cohort.head(11)])
    visits = pl.DataFrame({"patient_id": ["P001", "P001", "P002"], "sbp": [120, 130, 140]})
    tables = [{"name": "patients", "data": patients}, {"name": "visits", "data": visits}]

    # Act
    ok, message = save_table_list(
        store, tables, "user_upload_dup0001", {"original_filename": "d.zip"}, raw_bytes=b"zip bytes"
    )

    # Assert
    assert not ok
    assert "duplicate rows on key 'patient_id'" in message
    assert _owned_files(store, "user_upload_dup0001") == []


def test_saveUpload_failsAfterDataWritten_removesOnlyFilesItWrote(store, monkeypatch):
    # Arrange: v1 committed; v2's query view will fail after its files are written
    import duckdb

    from clinical_analytics.core.semantic import SemanticLayer

    upload_id = _upload(store, _cohort_csv(), name="trial")
    v1_files = _owned_files(store, upload_id)

    def broken(self):
        raise duckdb.ConversionException("bad")

    monkeypatch.setattr(SemanticLayer, "get_base_view", broken)

    # Act
    ok, _, _ = store.save_upload(
        _cohort_csv(300, age_offset=60),
        "trial.csv",
        {"dataset_name": "trial", "variable_mapping": dict(MAPPING)},
        overwrite=True,
    )

    # Assert
    assert not ok
    assert _owned_files(store, upload_id) == v1_files


# --- review round: privacy ------------------------------------------------------------------


def _library_errors() -> list[BaseException]:
    import pyarrow as pa

    return [
        ValueError("invalid literal for int() with base 10: 'Jane Doe'"),
        pa.ArrowInvalid("Could not convert 'Jane Doe' with type str"),
        TypeError("'Jane Doe' is not a number"),
        KeyError("Jane Doe"),
    ]


@pytest.mark.parametrize("error", _library_errors(), ids=["ValueError", "ArrowInvalid", "TypeError", "KeyError"])
def test_uploadFailureReason_libraryError_namedByTypeOnly(error):
    # Act
    reason = user_datasets._upload_failure_reason(error)

    # Assert
    assert reason == type(error).__name__
    assert "Jane" not in reason


@pytest.mark.parametrize(
    "error",
    [
        user_datasets.StagingError("The CSV file could not be read"),
        user_datasets.UploadError("No CSV files in ZIP"),
        user_datasets.SecurityError("Invalid path"),
        user_datasets.DataQualityError("duplicate rows on key 'patient_id'"),
    ],
    ids=["staging", "upload", "security", "data_quality"],
)
def test_uploadFailureReason_projectError_keepsItsMessage(error):
    # Act / Assert
    assert user_datasets._upload_failure_reason(error) == str(error)


def test_saveTableList_unexpectedErrorType_stillRemovesUncommittedFiles(store, monkeypatch):
    # Arrange: a TypeError is not in any except tuple; the files must go anyway
    from clinical_analytics.contracts import service

    def broken(*args, **kwargs):
        raise TypeError("unexpected")

    monkeypatch.setattr(service, "check_upload_or_log", broken)

    # Act / Assert
    with pytest.raises(TypeError):
        store.save_upload(_cohort_csv(), "c.csv", {"dataset_name": "c", "variable_mapping": dict(MAPPING)})
    data_dir = store.upload_dir.parent
    assert [p for p in (data_dir / "parquet").glob("*") if p.is_file()] == []
    assert [p for p in store.raw_dir.rglob("*") if p.is_file()] == []


def test_deleteUpload_removesLeftoverPartFiles(store):
    # Arrange: a crash left hidden temp files that hold patient rows
    upload_id = _upload(store, _cohort_csv())
    parquet_part = store.upload_dir.parent / "parquet" / f".{upload_id}_cohort_abc.parquet.0123.part"
    metadata_part = store.metadata_dir / f".{upload_id}.json.0123.part"
    parquet_part.write_bytes(b"rows")
    metadata_part.write_bytes(b"{}")

    # Act
    ok, message = store.delete_upload(upload_id)

    # Assert
    assert ok, message
    assert not parquet_part.exists()
    assert not metadata_part.exists()


# --- review round: wrong or lost data -------------------------------------------------------


@pytest.mark.parametrize(
    "names", [("Lab Results", "lab-results"), ("labs", "Labs"), ("a b", "a_b")], ids=["space_dash", "case", "space"]
)
def test_saveTableList_namesSanitizingAlike_failFast_noFiles(store, names):
    # Arrange
    from clinical_analytics.ui.storage.user_datasets import save_table_list

    df = pl.DataFrame({"patient_id": ["P1", "P2"], "v": [1, 2]})
    tables = [{"name": names[0], "data": df}, {"name": names[1], "data": df.with_columns(pl.col("v") * 10)}]

    # Act
    ok, message = save_table_list(store, tables, "user_upload_collide01", {"original_filename": "c.zip"})

    # Assert
    assert not ok
    assert names[0] in message and names[1] in message
    assert _owned_files(store, "user_upload_collide01") == []


def test_saveTableList_existingFileWithOtherContent_isRefused(store):
    # Arrange: a file already holds the version's name but different rows
    from clinical_analytics.storage.versioning import compute_dataset_version
    from clinical_analytics.ui.storage.user_datasets import save_table_list, table_parquet_path

    df = pl.DataFrame({"patient_id": ["P1", "P2"], "v": [1, 2]})
    version = compute_dataset_version([df])
    path = table_parquet_path(store, "user_upload_squat01", "t", version)
    path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"x": ["other"]}).write_parquet(path)

    # Act
    ok, message = save_table_list(store, [{"name": "t", "data": df}], "user_upload_squat01", {"original_filename": "t"})

    # Assert
    assert not ok
    assert "does not match" in message


def test_saveTableList_cohortLogicChange_neverReusesStaleCohort(store, make_cohort_with_categorical, monkeypatch):
    # Arrange
    from clinical_analytics.core import multi_table_handler
    from clinical_analytics.ui.storage.user_datasets import save_table_list

    patients = make_cohort_with_categorical(n_patients=30)
    visits = pl.DataFrame({"patient_id": ["P001", "P001", "P002"], "sbp": [120, 130, 140]})

    def tables():
        return [{"name": "patients", "data": patients}, {"name": "visits", "data": visits}]

    ok, message = save_table_list(store, tables(), "user_upload_logic01", {"original_filename": "m.zip"})
    assert ok, message
    first = store.get_upload_metadata("user_upload_logic01")["cohort_parquet"]

    # Act
    monkeypatch.setattr(multi_table_handler, "COHORT_LOGIC_VERSION", "test-next")
    meta = {
        "original_filename": "m.zip",
        "_base_active_version": store.get_upload_metadata("user_upload_logic01")["dataset_version"],
        "_is_overwrite": True,
    }
    ok, message = save_table_list(store, tables(), "user_upload_logic01", meta)

    # Assert
    assert ok, message
    second = store.get_upload_metadata("user_upload_logic01")["cohort_parquet"]
    assert second != first
    assert Path(first).exists() and Path(second).exists()


def test_extractZipTables_sharedKey_hasOneTypeInEveryTable():
    # Arrange: ward has a leading zero only in patients
    import io
    import zipfile

    from clinical_analytics.ui.storage.user_datasets import extract_zip_tables

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("patients.csv", "patient_id,ward\nP1,01\nP2,12\n")
        z.writestr("visits.csv", "patient_id,ward,sbp\nP1,12,120\nP2,13,130\n")

    # Act
    tables = {t["name"]: t for t in extract_zip_tables(buf.getvalue())}

    # Assert
    assert tables["patients"]["data"].schema["ward"] == pl.String
    assert tables["visits"]["data"].schema["ward"] == pl.String
    assert [n.reason for n in tables["visits"]["notes"] if n.column == "ward"] == ["shared_column_type"]


def _make_legacy_multi(store: UserDatasetStorage, upload_id: str = "user_upload_legacymt01") -> str:
    """A legacy multi-table upload: cohort CSV, per-table CSVs, and old pandas-typed Parquet."""
    _make_legacy(store, upload_id)
    tables_dir = store.raw_dir / f"{upload_id}_tables"
    tables_dir.mkdir()
    (tables_dir / "patients.csv").write_bytes(_cohort_csv())
    old_parquet = store.upload_dir.parent / "parquet" / f"{upload_id}_patients_oldver.parquet"
    old_parquet.parent.mkdir(parents=True, exist_ok=True)
    # The old DataStore round trip: zip lost its leading zeros and became Int64
    pl.read_csv(io_bytes(_cohort_csv())).write_parquet(old_parquet)
    only_parquet = store.upload_dir.parent / "parquet" / f"{upload_id}_extra_oldver.parquet"
    pl.DataFrame({"patient_id": ["P0000"], "n": [1]}).write_parquet(only_parquet)
    path = store.metadata_dir / f"{upload_id}.json"
    metadata = json.loads(path.read_text())
    metadata["tables"] = ["patients", "extra"]
    metadata["parquet_paths"] = {"patients": str(old_parquet), "extra": str(only_parquet)}
    path.write_text(json.dumps(metadata))
    return upload_id


def io_bytes(data: bytes):
    import io

    return io.BytesIO(data)


def test_legacy_conversion_restagesEveryTableWithCsv_dtypesMatchCohort(store):
    # Arrange
    upload_id = _make_legacy_multi(store)

    # Act
    stored = store.stored_dataset(upload_id)

    # Assert
    table = pl.read_parquet(stored.table_paths["patients"])
    cohort = pl.read_parquet(stored.cohort_path)
    assert table.schema == cohort.schema
    assert table.schema["zip"] == pl.String
    assert stored.metadata["legacy_tables_untyped"] == ["extra"]


def test_overwrite_renamedNumericId_identicalReupload_isAllowed(store):
    # Arrange: record_number is not ID-named; ensure_patient_id renames it to patient_id
    rows = ["record_number,age,outcome"] + [f"{1000 + i},{40 + i % 30},{i % 2}" for i in range(200)]
    data = ("\n".join(rows) + "\n").encode()
    meta = {"patient_id": "patient_id", "outcome": "outcome", "predictors": ["age"]}
    ok, message, upload_id = store.save_upload(data, "rn.csv", {"dataset_name": "rn", "variable_mapping": dict(meta)})
    assert ok, message

    # Act
    ok, message, again = store.save_upload(
        data, "rn.csv", {"dataset_name": "rn", "variable_mapping": dict(meta)}, overwrite=True
    )

    # Assert
    assert ok, message
    assert again == upload_id


def test_addDocumentation_dictionaryPdf_addsCodebooks(store):
    # Arrange: a real PDF data dictionary
    import pymupdf

    meta = {
        "dataset_name": "doc",
        "variable_mapping": dict(MAPPING),
        "variable_types": {"outcome": {"type": "binary"}},
    }
    ok, message, upload_id = store.save_upload(_cohort_csv(), "doc.csv", meta)
    assert ok, message
    doc = pymupdf.open()
    doc.new_page().insert_text((72, 72), "outcome: 0: Alive, 1: Dead")
    pdf = doc.tobytes()

    # Act
    ok, message = store.add_documentation_to_upload(upload_id, pdf)

    # Assert
    assert ok, message
    codebook = store.get_upload_metadata(upload_id)["variable_types"]["outcome"]["codebook"]
    assert codebook == {"0": "Alive", "1": "Dead"}


# --- review round: concurrency --------------------------------------------------------------


def test_overwrite_datasetChangedDuringSave_failsRetryably_andKeepsOtherVersion(store, monkeypatch):
    # Arrange: while v2 is being checked, another save commits v3
    from clinical_analytics.contracts import service

    upload_id = _upload(store, _cohort_csv(), name="race")
    real = service.check_upload_or_log
    path = store.metadata_dir / f"{upload_id}.json"

    def other_save_commits(*args, **kwargs):
        def mutate(current):
            for v in current["version_history"]:
                v["is_active"] = False
            current["version_history"].append({"version": "other", "is_active": True, "tables": {}})
            current["events"].append({"event_id": "other-event"})
            return True

        user_datasets._update_metadata(path, mutate)
        return real(*args, **kwargs)

    monkeypatch.setattr(service, "check_upload_or_log", other_save_commits)

    # Act
    ok, message, _ = store.save_upload(
        _cohort_csv(300, age_offset=60),
        "race.csv",
        {"dataset_name": "race", "variable_mapping": dict(MAPPING)},
        overwrite=True,
    )

    # Assert
    assert not ok
    assert "upload again" in message.lower()
    after = store.get_upload_metadata(upload_id)
    assert [v["version"] for v in after["version_history"] if v["is_active"]] == ["other"]
    assert after["events"][-1]["event_id"] == "other-event"


def test_overwrite_commitRereadsHistory_keepsConcurrentEvents(store, monkeypatch):
    # Arrange: an event appended (not a new version) while the save runs must survive the commit
    from clinical_analytics.contracts import service

    upload_id = _upload(store, _cohort_csv(), name="merge")
    real = service.check_upload_or_log
    path = store.metadata_dir / f"{upload_id}.json"

    def concurrent_event(*args, **kwargs):
        def mutate(current):
            current["events"].append({"event_id": "concurrent"})
            return True

        user_datasets._update_metadata(path, mutate)
        return real(*args, **kwargs)

    monkeypatch.setattr(service, "check_upload_or_log", concurrent_event)

    # Act
    ok, message, _ = store.save_upload(
        _cohort_csv(300, age_offset=60),
        "merge.csv",
        {"dataset_name": "merge", "variable_mapping": dict(MAPPING)},
        overwrite=True,
    )

    # Assert
    assert ok, message
    events = [e["event_id"] for e in store.get_upload_metadata(upload_id)["events"]]
    assert "concurrent" in events
    assert len(store.get_upload_metadata(upload_id)["version_history"]) == 2


def test_aliasAdd_readModifyWrite_keepsConcurrentMetadataChange(store, monkeypatch):
    # Arrange: the layer holds a stale copy; another writer changes the file meanwhile
    upload_id = _upload(store, _cohort_csv())
    dataset = UploadedDataset(upload_id, storage=store)
    dataset.load()
    layer = dataset.get_semantic_layer()
    stale = store.get_upload_metadata(upload_id)
    monkeypatch.setattr(layer, "_load_metadata", lambda _uid: json.loads(json.dumps(stale)))
    ok, message = store.update_metadata(upload_id, {"concurrent_key": 1})
    assert ok, message

    # Act
    layer.add_user_alias("years", "age", upload_id, stale["dataset_version"])

    # Assert
    after = store.get_upload_metadata(upload_id)
    assert after["concurrent_key"] == 1
    assert after["alias_mappings"]["user_aliases"]["years"] == "age"


# --- review round: smaller items -------------------------------------------------------------


def test_probe_viewSchemaDiffersFromParquet_raisesBuildError(store, monkeypatch):
    # Arrange
    from clinical_analytics.datasets.uploaded import definition

    upload_id = _upload(store, _cohort_csv())
    metadata = store.get_upload_metadata(upload_id)
    real = definition._parquet_schema

    def wrong(path):
        return {**real(path), "age": pl.String()}

    monkeypatch.setattr(definition, "_parquet_schema", wrong)

    # Act / Assert
    with pytest.raises(definition.SemanticLayerBuildError, match="types"):
        definition.probe_query_view(store, upload_id, metadata, expected_rows=metadata["row_count"])


def test_upload_progressCallbackError_isLogged_uploadSucceeds(store):
    # Arrange
    def failing_cb(progress: int, message: str) -> None:
        raise RuntimeError("UI went away")

    # Act
    ok, message, _ = store.save_upload(
        _cohort_csv(), "cb.csv", {"dataset_name": "cb", "variable_mapping": dict(MAPPING)}, progress_cb=failing_cb
    )

    # Assert
    assert ok, message


@pytest.mark.parametrize(
    ("suffix", "source_format", "error"),
    [(".pq", None, "Unknown source file type"), (".parquet", "csv", "does not match")],
    ids=["unknown_suffix", "format_mismatch"],
)
def test_semanticLayer_sourceFormat_isExplicit_neverFallsBackToCsv(
    tmp_path, make_semantic_layer, suffix, source_format, error
):
    # Arrange
    source = tmp_path / f"cohort{suffix}"
    pl.DataFrame({"patient_id": ["P1"], "age": [40]}).write_parquet(source)
    init_params: dict[str, str] = {"source_path": str(source)}
    if source_format:
        init_params["source_format"] = source_format

    # Act / Assert
    with pytest.raises(ValueError, match=error):
        make_semantic_layer(config_overrides={"init_params": init_params})
