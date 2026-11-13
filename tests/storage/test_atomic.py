# SPDX-License-Identifier: MIT
"""The one crash-safe write path: temp file, fsync, rename; versioned files never replaced."""

from __future__ import annotations

from pathlib import Path

import pytest

from clinical_analytics.storage.atomic import atomic_write, atomic_write_bytes, atomic_write_json


def test_atomicWrite_newFile_isPublished(tmp_path: Path):
    # Arrange
    target = tmp_path / "sub" / "a.bin"

    # Act
    wrote = atomic_write_bytes(target, b"hello")

    # Assert
    assert wrote is True
    assert target.read_bytes() == b"hello"
    assert list(target.parent.glob("*.part")) == []


def test_atomicWrite_writerFails_targetUnchanged_noPartLeft(tmp_path: Path):
    # Arrange
    target = tmp_path / "meta.json"
    atomic_write_json(target, {"version": 1})

    def crash(part: Path) -> None:
        part.write_text('{"version": 2, "trunc')
        raise OSError(5, "simulated crash mid-write")

    # Act / Assert
    with pytest.raises(OSError, match="simulated"):
        atomic_write(target, crash)
    assert target.read_text(encoding="utf-8").replace(" ", "").replace("\n", "") == '{"version":1}'
    assert [p.name for p in tmp_path.iterdir()] == ["meta.json"]


@pytest.mark.parametrize("payload", [b"", b"x" * 10, b"\x00\xff" * 1000], ids=["empty", "small", "binary"])
def test_atomicWrite_exclusive_neverReplacesExisting(tmp_path: Path, payload: bytes):
    # Arrange
    target = tmp_path / "v1.parquet"
    atomic_write_bytes(target, b"original", exclusive=True)

    # Act
    wrote = atomic_write_bytes(target, payload, exclusive=True)

    # Assert
    assert wrote is False
    assert target.read_bytes() == b"original"


def test_atomicWrite_nonExclusive_replaces(tmp_path: Path):
    # Arrange
    target = tmp_path / "m.json"
    atomic_write_json(target, {"a": 1})

    # Act
    atomic_write_json(target, {"a": 2})

    # Assert
    assert '"a": 2' in target.read_text(encoding="utf-8")
