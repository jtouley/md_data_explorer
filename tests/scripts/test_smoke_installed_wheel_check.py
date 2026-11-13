"""The installed-wheel smoke test refuses a wheel that does not carry the embedding model."""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from clinical_analytics.core.embeddings import RUNTIME_FILES
from scripts.smoke_installed import MODEL_DIR_IN_WHEEL, missing_model_files


def _wheel(path: Path, names: list[str]) -> Path:
    with zipfile.ZipFile(path, "w") as zf:
        for name in names:
            zf.writestr(name, b"x")
    return path


@pytest.mark.parametrize(
    ("present", "expected_missing"),
    [
        (list(RUNTIME_FILES), []),
        ([], list(RUNTIME_FILES)),
        ([RUNTIME_FILES[0]], list(RUNTIME_FILES[1:])),
    ],
    ids=["model_shipped", "no_model", "partial_model"],
)
def test_missing_model_files_wheelContents_reportsAbsentRuntimeFiles(tmp_path, present, expected_missing):
    # Arrange: a wheel with the package and some of the model's runtime files
    names = ["clinical_analytics/__init__.py", *(f"{MODEL_DIR_IN_WHEEL}/{name}" for name in present)]
    wheel = _wheel(tmp_path / "clinical_analytics-0.0.0-py3-none-any.whl", names)

    # Act
    missing = missing_model_files(wheel)

    # Assert
    assert missing == expected_missing


def test_missing_model_files_modelOutsidePackagePath_stillMissing(tmp_path):
    # Arrange: the files exist in the archive but not where the app looks for them
    wheel = _wheel(tmp_path / "w.whl", [f"models/{name}" for name in RUNTIME_FILES])

    # Act / Assert
    assert missing_model_files(wheel) == list(RUNTIME_FILES)
