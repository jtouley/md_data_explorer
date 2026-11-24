# SPDX-License-Identifier: MIT
"""``read_version_semantics``: every file read happens in the first step, none in the second.

A route holds the dataset's files lock around the first step only, and runs the second (the
inference pass for a version stored before ADR017, seconds on a large file) after releasing it.
So the second step must read nothing a delete could purge in between. Invented domain only:
a hearing-aid fitting clinic.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, call

import polars as pl
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from clinical_analytics.contracts import service
from clinical_analytics.contracts.declarations import Declarations, MemoryDeclarationStore
from clinical_analytics.contracts.semantics import DatasetSemantics
from clinical_analytics.contracts.service import check_upload, read_version_semantics, version_semantics

PROPERTY = settings(
    max_examples=25,
    deadline=None,
    derandomize=True,
    database=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow],
)


class CountingStore(MemoryDeclarationStore):
    """Declarations in memory, counting each load (a file read in the real store)."""

    def __init__(self) -> None:
        super().__init__()
        self.loads = 0

    def load(self, dataset_id: str) -> Declarations:
        self.loads += 1
        return super().load(dataset_id)


def _fittings(gains: list[int], ears: list[str]) -> pl.DataFrame:
    n = min(len(gains), len(ears))
    return pl.DataFrame({"fitting_id": [f"F{i:03d}" for i in range(n)], "gain_db": gains[:n], "ear": ears[:n]})


def _counted(frame: pl.DataFrame) -> tuple[Callable[[], pl.DataFrame], list[int]]:
    calls = [0]

    def stored_frame() -> pl.DataFrame:
        calls[0] += 1
        return frame

    return stored_frame, calls


@PROPERTY
@given(
    gains=st.lists(st.integers(min_value=0, max_value=120), min_size=1, max_size=30),
    ears=st.lists(st.sampled_from(["left", "right", "both"]), min_size=1, max_size=30),
)
def test_readVersionSemantics_versionWithoutContract_secondStepReadsNothingSameResult(
    tmp_path: Path, gains: list[int], ears: list[str]
) -> None:
    # Arrange: no contract for the version (stored before ADR017)
    frame = _fittings(gains, ears)
    stored_frame, calls = _counted(frame)
    store = CountingStore()

    # Act
    infer = read_version_semantics(tmp_path, "fit1", "v1", metadata={}, stored_frame=stored_frame, declarations=store)
    read_counts = (calls[0], store.loads)
    semantics = infer()

    # Assert: the reads all happened in the first step, and the result is version_semantics'
    assert read_counts == (1, 1)
    assert (calls[0], store.loads) == (1, 1)
    expected = version_semantics(
        tmp_path, "fit1", "v1", metadata={}, stored_frame=lambda: frame, declarations=MemoryDeclarationStore()
    )
    assert semantics == expected
    assert semantics.dataset_version == "v1"


def test_readVersionSemantics_contractHoldsSemantics_storedFrameNeverRead(tmp_path: Path) -> None:
    # Arrange: a version whose contract holds its semantics
    frame = _fittings([20, 35, 50, 65], ["left", "right", "both", "left"])
    check_upload(tmp_path, frame, upload_id="fit1", dataset_version="v1", metadata={})
    stored_frame, calls = _counted(frame)
    store = CountingStore()

    # Act
    infer = read_version_semantics(tmp_path, "fit1", "v1", metadata={}, stored_frame=stored_frame, declarations=store)
    read_counts = (calls[0], store.loads)
    semantics = infer()

    # Assert
    assert read_counts == (0, 1)
    assert (calls[0], store.loads) == (0, 1)
    assert isinstance(semantics, DatasetSemantics)
    assert semantics.dataset_version == "v1"


def test_versionSemantics_anyCall_readsThenInfersWithTheSameArguments(monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange
    marker, store, metadata = object(), MemoryDeclarationStore(), {"source": "fitting log"}
    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def stored_frame() -> pl.DataFrame:
        return pl.DataFrame()

    def record(*args: Any, **kwargs: Any) -> Callable[[], Any]:
        calls.append((args, kwargs))
        return lambda: marker

    monkeypatch.setattr(service, "read_version_semantics", record)

    # Act
    result = version_semantics(
        Path("data"), "fit1", "v1", metadata=metadata, stored_frame=stored_frame, declarations=store
    )

    # Assert
    assert result is marker
    assert calls == [
        (
            (Path("data"), "fit1", "v1"),
            {"metadata": metadata, "stored_frame": stored_frame, "declarations": store},
        )
    ]


def test_readVersionSemantics_noStoreGiven_inferredWithTheFileStoresDeclarations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange: no semantics in the contract; every collaborator recorded
    from clinical_analytics.contracts import declarations as declarations_module
    from clinical_analytics.contracts import semantics as semantics_module

    declared = Declarations()
    file_store = MagicMock()
    file_store.load.return_value = declared
    store_class = MagicMock(return_value=file_store)
    monkeypatch.setattr(declarations_module, "FileDeclarationStore", store_class)

    def no_semantics(*_args: Any, **_kwargs: Any) -> Any:
        raise semantics_module.SemanticsMissingError("no semantics")

    monkeypatch.setattr(semantics_module, "semantics_for", no_semantics)
    inferred = MagicMock()
    stored = MagicMock(return_value=inferred)
    monkeypatch.setattr(service, "stored_semantics", stored)
    applied = object()
    with_declarations = MagicMock(return_value=applied)
    monkeypatch.setattr(semantics_module, "with_declarations", with_declarations)
    log = MagicMock()
    monkeypatch.setattr(service, "logger", log)
    frame, metadata = pl.DataFrame({"gain_db": [30]}), {"source": "fitting log"}

    # Act
    infer = read_version_semantics(Path("data"), "fit1", "v1", metadata=metadata, stored_frame=lambda: frame)
    loaded_before_infer = file_store.load.call_args_list[:]
    result = infer()

    # Assert
    assert result is applied
    store_class.assert_called_once_with(Path("data"))
    assert loaded_before_infer == [call("fit1")]
    stored.assert_called_once_with(frame, metadata)
    inferred.model_copy.assert_called_once_with(update={"dataset_version": "v1"})
    with_declarations.assert_called_once_with(inferred.model_copy.return_value, declared)
    log.info.assert_called_once_with("semantics_inferred_for_stored_version", dataset_id="fit1", dataset_version="v1")


def test_readVersionSemantics_contractHoldsSemantics_readWithTheGivenStore(monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange
    from clinical_analytics.contracts import semantics as semantics_module

    marker, store = object(), MemoryDeclarationStore()
    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def semantics_for(*args: Any, **kwargs: Any) -> Any:
        calls.append((args, kwargs))
        return marker

    monkeypatch.setattr(semantics_module, "semantics_for", semantics_for)

    # Act
    result = read_version_semantics(
        Path("data"), "fit1", "v1", metadata={}, stored_frame=pl.DataFrame, declarations=store
    )()

    # Assert
    assert result is marker
    assert calls == [((Path("data"), "fit1", "v1"), {"declarations": store})]
