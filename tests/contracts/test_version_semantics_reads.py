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

import polars as pl
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

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
