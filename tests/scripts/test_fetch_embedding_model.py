"""scripts/fetch_embedding_model.py: pinned files, SHA-256 checked, safe to run twice."""

from __future__ import annotations

import hashlib

import pytest

from clinical_analytics.core.embeddings import MODEL_REPO, MODEL_REVISION
from scripts import fetch_embedding_model as fetch

NAME = "tokenizer.json"


@pytest.fixture
def mirror(tmp_path):
    """A file:// mirror laid out like huggingface.co/<repo>/resolve/<revision>/<file>."""
    root = tmp_path / "mirror"
    (root / MODEL_REPO / "resolve" / MODEL_REVISION).mkdir(parents=True)

    def _put(content: bytes) -> str:
        (root / MODEL_REPO / "resolve" / MODEL_REVISION / NAME).write_bytes(content)
        return hashlib.sha256(content).hexdigest()

    return root.as_uri(), _put


def test_fetch_file_hashMatches_writesFile(mirror, tmp_path):
    # Arrange
    base, put = mirror
    expected = put(b'{"ok": true}')
    dest = tmp_path / "out"

    # Act
    status = fetch.fetch_file(base, NAME, expected, dest)

    # Assert
    assert status == "fetched"
    assert (dest / NAME).read_bytes() == b'{"ok": true}'


def test_fetch_file_hashMismatch_raisesAndLeavesNothing(mirror, tmp_path):
    # Arrange
    base, put = mirror
    put(b"tampered")
    dest = tmp_path / "out"

    # Act / Assert
    with pytest.raises(fetch.ModelIntegrityError, match="does not match pinned"):
        fetch.fetch_file(base, NAME, "0" * 64, dest)
    assert list(dest.rglob("*")) == []


def test_fetch_file_runTwice_keepsVerifiedFileWithoutDownloading(mirror, tmp_path):
    # Arrange
    base, put = mirror
    expected = put(b"model bytes")
    dest = tmp_path / "out"
    fetch.fetch_file(base, NAME, expected, dest)
    put(b"mirror changed")  # a second download would now fail the hash

    # Act
    status = fetch.fetch_file(base, NAME, expected, dest)

    # Assert
    assert status == "kept"
    assert (dest / NAME).read_bytes() == b"model bytes"


@pytest.mark.parametrize("corrupt", [False, True], ids=["missing", "modified"])
def test_verify_badFiles_listed(tmp_path, corrupt):
    # Arrange
    if corrupt:
        (tmp_path / NAME).write_bytes(b"not the pinned tokenizer")

    # Act
    bad = fetch.verify(tmp_path)

    # Assert
    assert NAME in bad
