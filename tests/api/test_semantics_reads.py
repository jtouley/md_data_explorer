"""The dataset routes' semantics reads and backfill, without loading data (fakes only).

``_read_semantics`` does the file reads a route holds the files lock around and hands back the
inference step; ``_contract_semantics`` runs both; ``_backfill_quality`` builds the report. The
API tests in ``test_dataset_quality`` drive them through real uploads (slow); these pin what each
passes on and how each failure is reported.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import polars as pl
import polars.testing
import pytest
from fastapi import HTTPException

from clinical_analytics.api.routes import datasets
from clinical_analytics.contracts.declarations import MemoryDeclarationStore
from clinical_analytics.contracts.errors import ContractStoreError
from clinical_analytics.core.log_redaction import error_fields

METADATA: dict[str, Any] = {"version_history": [{"version": "v2", "is_active": True}], "dataset_version": "v1"}


@pytest.fixture
def fake_storage(tmp_path: Path) -> MagicMock:
    """A storage stand-in: real directories, every read a mock."""
    storage = MagicMock()
    storage.upload_dir = tmp_path / "uploads"
    storage.metadata_dir = tmp_path / "metadata"
    return storage


@pytest.fixture
def visits() -> pl.DataFrame:
    """A two-row stored cohort."""
    return pl.DataFrame({"visit": [1, 2]})


@pytest.fixture
def recorded_reads(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """``read_version_semantics`` replaced by a recorder whose step returns a marker."""
    recorded: dict[str, Any] = {"step_result": object()}

    def record(*args: Any, **kwargs: Any) -> Any:
        recorded["args"], recorded["kwargs"] = args, kwargs
        return lambda: recorded["step_result"]

    monkeypatch.setattr("clinical_analytics.contracts.service.read_version_semantics", record)
    return recorded


@pytest.mark.parametrize("metadata", [None, {}, {"version_history": [{"version": "v1", "is_active": False}]}])
def test_readSemantics_noActiveVersion_noneNothingRead(
    fake_storage: MagicMock, recorded_reads: dict[str, Any], metadata: dict[str, Any] | None
) -> None:
    # Act
    result = datasets._read_semantics(fake_storage, "ds1", metadata)

    # Assert
    assert result is None
    assert "args" not in recorded_reads


@pytest.mark.parametrize(("declared", "store_type"), [(False, MemoryDeclarationStore), (True, type(None))])
def test_readSemantics_activeVersion_passesItsReadsOn(
    fake_storage: MagicMock, recorded_reads: dict[str, Any], declared: bool, store_type: type
) -> None:
    # Act
    step = datasets._read_semantics(fake_storage, "ds1", METADATA, declared=declared)

    # Assert: the active version (not the stale legacy key), and the user's declarations only if asked
    assert step is not None
    assert step() is recorded_reads["step_result"]
    assert recorded_reads["args"] == (fake_storage.upload_dir.parent, "ds1", "v2")
    kwargs = recorded_reads["kwargs"]
    assert kwargs["metadata"] is METADATA
    assert type(kwargs["declarations"]) is store_type


def test_readSemantics_declaredNotGiven_asInferred(fake_storage: MagicMock, recorded_reads: dict[str, Any]) -> None:
    # Act
    datasets._read_semantics(fake_storage, "ds1", METADATA)

    # Assert
    assert isinstance(recorded_reads["kwargs"]["declarations"], MemoryDeclarationStore)


def test_readSemantics_storedFrame_readsThisDatasetsCohort(
    fake_storage: MagicMock, recorded_reads: dict[str, Any], visits: pl.DataFrame
) -> None:
    # Arrange
    fake_storage.stored_dataset.return_value.scan_cohort.return_value.collect.return_value = visits
    datasets._read_semantics(fake_storage, "ds1", METADATA)

    # Act
    result = recorded_reads["kwargs"]["stored_frame"]()

    # Assert
    assert result is visits
    fake_storage.stored_dataset.assert_called_once_with("ds1", METADATA)


@pytest.mark.parametrize(
    ("error", "event", "detail"),
    [
        (
            ContractStoreError("contract unreadable"),
            "dataset_semantics_unreadable",
            "This cohort's data contract can't be read.",
        ),
        (FileNotFoundError("parquet gone"), "dataset_stored_data_unreadable", datasets.STORED_DATA_UNREADABLE),
        (
            pl.exceptions.ComputeError("parquet damaged"),
            "dataset_stored_data_unreadable",
            datasets.STORED_DATA_UNREADABLE,
        ),
    ],
    ids=["contract", "missing_parquet", "corrupt_parquet"],
)
def test_readSemantics_readFails_logged500(
    fake_storage: MagicMock, monkeypatch: pytest.MonkeyPatch, error: Exception, event: str, detail: str
) -> None:
    # Arrange
    def fail(*_args: Any, **_kwargs: Any) -> Any:
        raise error

    monkeypatch.setattr("clinical_analytics.contracts.service.read_version_semantics", fail)
    log = MagicMock()
    monkeypatch.setattr(datasets, "logger", log)

    # Act
    with pytest.raises(HTTPException) as raised:
        datasets._read_semantics(fake_storage, "ds1", METADATA)

    # Assert
    assert (raised.value.status_code, raised.value.detail) == (500, detail)
    assert raised.value.__cause__ is error
    log.error.assert_called_once_with(event, upload_id="ds1", **error_fields(error))


@pytest.mark.parametrize("declared", [False, True])
def test_contractSemantics_activeVersion_readsThenInfers(
    fake_storage: MagicMock, monkeypatch: pytest.MonkeyPatch, declared: bool
) -> None:
    # Arrange
    fake_storage.get_upload_metadata.return_value = METADATA
    marker = object()
    calls: list[tuple[Any, ...]] = []

    def read(storage: Any, upload_id: str, metadata: Any, *, declared: bool) -> Any:
        calls.append((storage, upload_id, metadata, declared))
        return lambda: marker

    monkeypatch.setattr(datasets, "_read_semantics", read)

    # Act
    result = datasets._contract_semantics(fake_storage, "ds1", declared=declared)

    # Assert
    assert result is marker
    assert calls == [(fake_storage, "ds1", METADATA, declared)]
    fake_storage.get_upload_metadata.assert_called_once_with("ds1")


def test_contractSemantics_declaredNotGiven_asInferred(
    fake_storage: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange
    seen: list[bool] = []

    def read(*_args: Any, declared: bool) -> Any:
        seen.append(declared)
        return lambda: None

    monkeypatch.setattr(datasets, "_read_semantics", read)

    # Act
    datasets._contract_semantics(fake_storage, "ds1")

    # Assert
    assert seen == [False]


def test_contractSemantics_noActiveVersion_none(fake_storage: MagicMock, monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange
    monkeypatch.setattr(datasets, "_read_semantics", lambda *_a, **_k: None)

    # Act
    result = datasets._contract_semantics(fake_storage, "ds1")

    # Assert
    assert result is None


def test_backfillQuality_dataGone_404AndLogged(fake_storage: MagicMock) -> None:
    # Arrange
    fake_storage.get_upload_data.return_value = None
    log = MagicMock()

    # Act
    with pytest.raises(HTTPException) as raised:
        datasets._backfill_quality(fake_storage, "ds1", "v2", METADATA, log)

    # Assert
    assert (raised.value.status_code, raised.value.detail) == (404, "Dataset data not found: ds1")
    log.info.assert_called_once_with("dataset_quality_backfill")
    fake_storage.get_upload_data.assert_called_once_with("ds1", lazy=True)


def test_backfillQuality_data_checkedAndReported(
    fake_storage: MagicMock, monkeypatch: pytest.MonkeyPatch, visits: pl.DataFrame
) -> None:
    # Arrange
    fake_storage.get_upload_data.return_value = visits.lazy()
    report = object()
    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def check(*args: Any, **kwargs: Any) -> Any:
        calls.append((args, kwargs))
        return report

    monkeypatch.setattr(datasets, "check_upload", check)

    # Act
    result = datasets._backfill_quality(fake_storage, "ds1", "v2", METADATA, MagicMock())

    # Assert
    assert result is report
    [(args, kwargs)] = calls
    assert args[0] == fake_storage.upload_dir.parent
    pl.testing.assert_frame_equal(args[1], visits)
    assert kwargs == {
        "upload_id": "ds1",
        "dataset_version": "v2",
        "metadata": METADATA,
        "overlay_dir": fake_storage.metadata_dir,
    }
