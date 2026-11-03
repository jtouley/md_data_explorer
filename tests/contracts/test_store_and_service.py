"""Contract/report storage in the data home and the upload-time orchestration."""

from __future__ import annotations

import errno
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from clinical_analytics.contracts.errors import ContractStoreError, ContractValidationError
from clinical_analytics.contracts.quality import QualityNotRun, run_quality
from clinical_analytics.contracts.service import (
    accepted_patch_ids,
    check_upload,
    check_upload_or_log,
    id_column_from_metadata,
)
from clinical_analytics.contracts.store import (
    contract_path,
    dq_dir,
    latest_report,
    read_contract,
    write_contract,
    write_report,
)
from clinical_analytics.core.metadata_patch import MetadataPatch, PatchOperation, PatchStatus
from clinical_analytics.core.overlay_store import OverlayStore
from clinical_analytics.storage.dataset_deletion import InvalidDatasetIdError

_META = {"dataset_name": "cohort", "inferred_schema": {"column_mapping": {"patient_id": "patient_id"}}}


def test_writeContract_secondWrite_keepsFirstFileUnchanged(tmp_path: Path, load_synthetic_dataset, make_contract):
    # Arrange
    df = load_synthetic_dataset("cohort")
    first = make_contract(df)
    second = make_contract(df, dataset_name="renamed")

    # Act
    wrote_first = write_contract(tmp_path, "u1", "v1", first)
    wrote_second = write_contract(tmp_path, "u1", "v1", second)

    # Assert
    path = contract_path(tmp_path, "u1", "v1")
    assert (wrote_first, wrote_second) == (True, False)
    assert path == tmp_path / "contracts" / "u1" / "v1.odcs.yaml"
    assert read_contract(path).name == "test cohort"
    assert [p.name for p in path.parent.iterdir()] == ["v1.odcs.yaml"]  # no temp files left


@pytest.mark.parametrize(
    ("text", "error"),
    [
        ("{not: [yaml", ContractValidationError),
        ("- a\n- b\n", ContractValidationError),
        ("bogus: 1\n", ContractValidationError),
    ],
    ids=["bad_yaml", "not_mapping", "not_odcs"],
)
def test_readContract_invalidFile_raisesContractValidationError(tmp_path: Path, text, error):
    # Arrange
    path = tmp_path / "c.odcs.yaml"
    path.write_text(text)

    # Act / Assert
    with pytest.raises(error):
        read_contract(path)


def test_readContract_missingFile_raisesContractStoreError(tmp_path: Path):
    # Act / Assert
    with pytest.raises(ContractStoreError, match="Could not read contract"):
        read_contract(tmp_path / "missing.odcs.yaml")


@pytest.mark.parametrize(
    ("upload_id", "version", "error"),
    [("../x", "v1", InvalidDatasetIdError), ("u1", "../v1", ContractStoreError), ("u1", "", ContractStoreError)],
    ids=["traversal_id", "traversal_version", "empty_version"],
)
def test_contractPath_unsafeParts_rejected(tmp_path: Path, upload_id, version, error):
    # Act / Assert
    with pytest.raises(error):
        contract_path(tmp_path, upload_id, version)


def test_latestReport_twoRuns_returnsNewest(tmp_path: Path, load_synthetic_dataset, make_contract):
    # Arrange
    df = load_synthetic_dataset("cohort")
    contract = make_contract(df)
    t0 = datetime(2026, 11, 3, 12, tzinfo=UTC)
    older = run_quality(df.lazy(), contract, run_at=t0)
    newer = run_quality(df.head(50).lazy(), contract, run_at=t0 + timedelta(seconds=1))

    # Act
    write_report(tmp_path, newer)
    write_report(tmp_path, older)
    latest = latest_report(tmp_path, "u1", "v1")

    # Assert
    assert latest == newer


def test_latestReport_noRuns_returnsNone(tmp_path: Path):
    # Act / Assert
    assert latest_report(tmp_path, "u1", "v1") is None


def test_latestReport_corruptNewest_raisesContractStoreError(tmp_path: Path):
    # Arrange
    directory = dq_dir(tmp_path, "u1", "v1")
    directory.mkdir(parents=True)
    (directory / "20261103T120000000000Z.json").write_text("{}")

    # Act / Assert
    with pytest.raises(ContractStoreError, match="invalid"):
        latest_report(tmp_path, "u1", "v1")


@pytest.mark.parametrize(
    ("metadata", "expected"),
    [
        (_META, "patient_id"),
        ({"inferred_schema": {"column_mapping": {"Subject": "patient_id", "x": "other"}}}, "Subject"),
        ({"inferred_schema": {}}, None),
        ({}, None),
    ],
    ids=["mapped", "renamed", "no_mapping", "no_schema"],
)
def test_idColumnFromMetadata_inferredSchema_returnsPatientIdColumn(metadata, expected):
    # Act / Assert
    assert id_column_from_metadata(metadata) == expected


def _patch(patch_id: str, status: PatchStatus, minute: int) -> MetadataPatch:
    return MetadataPatch(
        patch_id=patch_id,
        operation=PatchOperation.SET_LABEL,
        column="ldl",
        value="LDL cholesterol",
        status=status,
        created_at=datetime(2026, 11, 3, 12, minute, tzinfo=UTC),
        provenance="user",
    )


def test_acceptedPatchIds_appendOnlyLog_latestRowPerIdWins():
    # Arrange
    patches = [
        _patch("a", PatchStatus.ACCEPTED, 0),
        _patch("b", PatchStatus.ACCEPTED, 1),
        _patch("b", PatchStatus.REVERTED, 2),
        _patch("c", PatchStatus.PENDING, 3),
    ]

    # Act / Assert
    assert accepted_patch_ids(patches) == ("a",)


def test_checkUpload_runTwice_contractReusedAndTwoReports(tmp_path: Path, load_synthetic_dataset):
    # Arrange
    df = load_synthetic_dataset("cohort")
    t0 = datetime(2026, 11, 3, 12, tzinfo=UTC)

    # Act
    first = check_upload(tmp_path, df, upload_id="u1", dataset_version="v1", metadata=_META, run_at=t0)
    second = check_upload(
        tmp_path, df, upload_id="u1", dataset_version="v1", metadata=_META, run_at=t0 + timedelta(seconds=1)
    )

    # Assert
    assert first.summary == second.summary == ["ldl: 5.8% missing"]
    assert len(list(dq_dir(tmp_path, "u1", "v1").glob("*.json"))) == 2
    assert len(list((tmp_path / "contracts" / "u1").iterdir())) == 1


def test_checkUpload_acceptedPatchInOverlay_labelInContract(tmp_path: Path, load_synthetic_dataset):
    # Arrange
    df = load_synthetic_dataset("cohort")
    overlay_dir = tmp_path / "uploads" / "metadata"
    OverlayStore(base_dir=overlay_dir).append_patch("u1", "v1", _patch("p1", PatchStatus.ACCEPTED, 0))

    # Act
    report = check_upload(tmp_path, df, upload_id="u1", dataset_version="v1", metadata=_META, overlay_dir=overlay_dir)

    # Assert
    contract = read_contract(contract_path(tmp_path, "u1", "v1"))
    ldl = next(p for p in contract.schema_[0].properties if p.name == "ldl")
    assert ldl.businessName == "LDL cholesterol"
    assert report.summary == ["LDL cholesterol: 5.8% missing"]
    assert {c.property: c.value for c in contract.customProperties}["metadataPatchIds"] == ["p1"]


def test_checkUploadOrLog_checkFails_returnsVisibleNotRunStatus(tmp_path: Path):
    # Arrange: the mapped ID column is missing, so building the contract fails.
    df = pl.DataFrame({"a": [1, 2]})

    # Act
    result = check_upload_or_log(tmp_path, df, upload_id="u1", dataset_version="v1", metadata=_META)

    # Assert
    assert isinstance(result, QualityNotRun)
    assert (result.status, result.reason) == ("not_run", "the check failed (ValueError)")
    assert not (tmp_path / "contracts").exists()


@pytest.mark.parametrize("errno_code", [errno.EPERM, errno.ENOTSUP, errno.EXDEV], ids=["eperm", "enotsup", "exdev"])
def test_writeContract_hardLinkUnsupported_fallsBackToExclusiveCreate(
    tmp_path: Path, monkeypatch, load_synthetic_dataset, make_contract, errno_code
):
    # Arrange: exFAT and SMB shares refuse hard links.
    def no_link(src, dst):
        raise OSError(errno_code, os.strerror(errno_code))

    monkeypatch.setattr("clinical_analytics.contracts.store.os.link", no_link)
    contract = make_contract(load_synthetic_dataset("cohort"))
    path = contract_path(tmp_path, "u1", "v1")

    # Act
    first = write_contract(tmp_path, "u1", "v1", contract)
    second = write_contract(tmp_path, "u1", "v1", make_contract(load_synthetic_dataset("cohort").head(3)))

    # Assert
    assert (first, second) == (True, False)
    assert read_contract(path) == contract
    assert [p.name for p in path.parent.iterdir()] == [path.name]  # no temp file left


def test_writeContract_fallbackTargetAppearsConcurrently_keepsOtherWritersFile(
    tmp_path: Path, monkeypatch, load_synthetic_dataset, make_contract
):
    # Arrange: link unsupported, and another writer creates the file after our exists() check.
    path = contract_path(tmp_path, "u1", "v1")

    def no_link(src, dst):
        Path(dst).write_text("other writer", encoding="utf-8")
        raise OSError(errno.EPERM, "Operation not permitted")

    monkeypatch.setattr("clinical_analytics.contracts.store.os.link", no_link)

    # Act
    written = write_contract(tmp_path, "u1", "v1", make_contract(load_synthetic_dataset("cohort")))

    # Assert
    assert written is False
    assert path.read_text(encoding="utf-8") == "other writer"
