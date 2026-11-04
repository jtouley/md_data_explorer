"""ONNX sentence encoder: equivalence with sentence-transformers, bundling, and no network."""

from __future__ import annotations

import json
import os
import socket
from pathlib import Path

import numpy as np
import pytest

from clinical_analytics.core import embeddings
from clinical_analytics.core.config_loader import load_golden_examples_config
from clinical_analytics.core.embeddings import (
    EMBEDDING_DIM,
    EmbeddingModelMissingError,
    check_embedding_model,
    mean_pool_normalize,
)

REFERENCE = json.loads(
    (Path(__file__).resolve().parents[1] / "fixtures" / "embeddings" / "minilm_reference.json").read_text()
)
MIN_COSINE = 0.99999  # measured minimum 0.9999999 (1.23.2: 0.99999988)


def _phrase_id(text: str) -> str:
    if not text:
        return "empty"
    return f"long_{len(text)}chars" if len(text) > 80 else text[:40]


def test_reference_fixture_sameModelRevision_matchesCode():
    # Assert: the recorded torch outputs belong to the model the app ships
    assert REFERENCE["repo"] == embeddings.MODEL_REPO
    assert REFERENCE["revision"] == embeddings.MODEL_REVISION


@pytest.mark.parametrize(
    ("index", "phrase"), list(enumerate(REFERENCE["phrases"])), ids=[_phrase_id(p) for p in REFERENCE["phrases"]]
)
def test_encode_fixedClinicalPhrase_matchesTorchCosineAtLeast0999(cached_sentence_transformer, index, phrase):
    # Arrange
    torch_vec = np.asarray(REFERENCE["embeddings"][index], dtype=np.float32)

    # Act
    onnx_vec = cached_sentence_transformer.encode([phrase])[0]

    # Assert
    cosine = float(onnx_vec @ torch_vec / (np.linalg.norm(onnx_vec) * np.linalg.norm(torch_vec)))
    assert cosine >= MIN_COSINE, f"cosine {cosine:.7f} for phrase #{index}"
    assert np.isclose(np.linalg.norm(onnx_vec), 1.0, atol=1e-5)


def test_encode_goldenExamples_topKIdenticalToTorch(cached_sentence_transformer):
    # Arrange
    golden = [q["query"] for q in load_golden_examples_config()["questions"]]
    assert len(golden) == REFERENCE["golden_example_count"]

    # Act
    phrases = cached_sentence_transformer.encode(REFERENCE["phrases"])
    golden_emb = cached_sentence_transformer.encode(golden)
    top_k = np.argsort(-(phrases @ golden_emb.T), axis=1)[:, : REFERENCE["top_k"]]

    # Assert
    assert top_k.tolist() == REFERENCE["golden_top_k"]


@pytest.mark.parametrize(
    ("texts", "expected_rows"),
    [([], 0), ("one string", 1), (["a", "b", "c"], 3)],
    ids=["empty_list", "single_string", "list"],
)
def test_encode_inputShapes_returnsRowsBy384(cached_sentence_transformer, texts, expected_rows):
    # Act
    out = cached_sentence_transformer.encode(texts)

    # Assert
    assert out.shape == (expected_rows, EMBEDDING_DIM)
    assert out.dtype == np.float32


@pytest.mark.parametrize("batch_size", [1, 2, 64], ids=["one", "uneven", "larger_than_input"])
def test_encode_batchSize_doesNotChangeEmbeddings(cached_sentence_transformer, batch_size):
    # Arrange: padding to the longest in a batch must not change any row
    texts = REFERENCE["phrases"][:5]
    encoder = embeddings.OnnxSentenceEncoder(batch_size=batch_size)

    # Act
    out = encoder.encode(texts)

    # Assert
    np.testing.assert_allclose(out, cached_sentence_transformer.encode(texts), atol=1e-5)


def test_mean_pool_normalize_paddingTokens_ignored():
    # Arrange: second token is padding with a huge value that must not leak into the mean
    hidden = np.array([[[3.0, 4.0], [1000.0, -1000.0]]], dtype=np.float32)
    mask = np.array([[1, 0]], dtype=np.int64)

    # Act
    out = mean_pool_normalize(hidden, mask)

    # Assert
    np.testing.assert_allclose(out, [[0.6, 0.8]], atol=1e-6)


@pytest.mark.parametrize(
    ("hidden", "mask"),
    [
        (np.zeros((1, 2, 3), dtype=np.float32), np.ones((1, 2), dtype=np.int64)),
        (np.ones((1, 2, 3), dtype=np.float32), np.zeros((1, 2), dtype=np.int64)),
    ],
    ids=["zero_vector", "all_masked"],
)
def test_mean_pool_normalize_degenerateInput_noNaN(hidden, mask):
    # Act
    out = mean_pool_normalize(hidden, mask)

    # Assert
    assert np.isfinite(out).all()


@pytest.mark.parametrize("missing", embeddings.RUNTIME_FILES)
def test_check_embedding_model_fileMissing_raisesWithFetchHint(tmp_path, missing):
    # Arrange: a model dir with every runtime file except one
    for name in embeddings.RUNTIME_FILES:
        if name != missing:
            (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
            (tmp_path / name).write_bytes(b"x")

    # Act / Assert
    with pytest.raises(EmbeddingModelMissingError, match="make fetch-embedding-model") as excinfo:
        check_embedding_model(tmp_path)
    assert missing in str(excinfo.value)


def test_embedding_model_dir_envOverride_used(monkeypatch, tmp_path):
    # Arrange
    monkeypatch.setenv(embeddings.EMBEDDING_MODEL_DIR_ENV, str(tmp_path))

    # Act / Assert
    assert embeddings.embedding_model_dir() == tmp_path
    with pytest.raises(EmbeddingModelMissingError):
        embeddings.OnnxSentenceEncoder()


def test_bundled_model_dir_default_insidePackage(monkeypatch):
    # Arrange
    monkeypatch.delenv(embeddings.EMBEDDING_MODEL_DIR_ENV, raising=False)

    # Act
    root = embeddings.embedding_model_dir()

    # Assert
    assert root == embeddings.bundled_model_dir()
    assert root.parts[-3:] == ("resources", "models", embeddings.MODEL_NAME)


@pytest.fixture
def sockets_blocked(monkeypatch):
    """Any attempt to open a network connection or resolve a name fails the test."""
    attempts: list[str] = []

    def _blocked(*args, **kwargs):
        attempts.append(repr(args[:2]))
        raise OSError("network access blocked by test")

    monkeypatch.setattr(socket.socket, "connect", _blocked)
    monkeypatch.setattr(socket.socket, "connect_ex", _blocked)
    monkeypatch.setattr(socket.socket, "sendto", _blocked)
    monkeypatch.setattr(socket, "getaddrinfo", _blocked)
    monkeypatch.setattr(socket, "create_connection", _blocked)
    return attempts


def test_embedding_path_socketsBlocked_loadsAndMatchesWithoutNetwork(sockets_blocked, make_semantic_layer):
    # Arrange
    from clinical_analytics.core.nl_query_engine import NLQueryEngine

    semantic = make_semantic_layer(
        dataset_name="offline_check",
        data={"patient_id": [1, 2, 3], "age": [40, 50, 60], "status": ["a", "b", "a"]},
    )

    # Act: fresh model load, golden-example embedding at init, then a Tier 2 match
    engine = NLQueryEngine(semantic)
    intent = engine._semantic_match("how many patients?")

    # Assert
    assert engine._golden_embeddings is not None
    assert engine._golden_embeddings.shape[1] == EMBEDDING_DIM
    assert intent is not None and intent.intent_type == "COUNT"
    assert sockets_blocked == [], f"embedding path tried the network: {sockets_blocked}"


def test_nl_query_engine_unknownEmbeddingModel_raisesValueError(make_semantic_layer):
    # Arrange
    from clinical_analytics.core.nl_query_engine import NLQueryEngine

    semantic = make_semantic_layer(dataset_name="model_name_check", data={"patient_id": [1], "age": [1]})

    # Act / Assert
    with pytest.raises(ValueError, match="bundled embedding model"):
        NLQueryEngine(semantic, embedding_model="all-mpnet-base-v2")


@pytest.fixture
def pinned_model_copy(tmp_path):
    """The runtime model files copied into a temp dir (so they can be damaged)."""
    import shutil

    dest = tmp_path / "model"
    for name in embeddings.RUNTIME_FILES:
        (dest / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(embeddings.bundled_model_dir() / name, dest / name)
    return dest


@pytest.mark.parametrize("damaged", embeddings.RUNTIME_FILES)
def test_check_embedding_model_verifyDamagedFile_raisesIntegrityError(pinned_model_copy, damaged):
    # Arrange: same size, one flipped byte
    path = pinned_model_copy / damaged
    data = bytearray(path.read_bytes())
    data[len(data) // 2] ^= 0xFF
    path.write_bytes(bytes(data))

    # Act / Assert
    with pytest.raises(embeddings.EmbeddingModelIntegrityError, match="pinned SHA-256") as excinfo:
        check_embedding_model(pinned_model_copy, verify=True)
    assert damaged in str(excinfo.value)


def test_check_embedding_model_verifiedCache_skipsHashUntilFileChanges(pinned_model_copy, tmp_path, monkeypatch):
    # Arrange
    cache = tmp_path / "cache" / "check.json"
    check_embedding_model(pinned_model_copy, verify=True, cache_path=cache)
    hashed: list[str] = []
    real_sha = embeddings._sha256
    monkeypatch.setattr(embeddings, "_sha256", lambda p: hashed.append(p.name) or real_sha(p))

    # Act: unchanged files -> no hashing; a touched file -> hashed (and still verified)
    check_embedding_model(pinned_model_copy, verify=True, cache_path=cache)
    hashed_when_unchanged = list(hashed)
    tok = pinned_model_copy / "tokenizer.json"
    os.utime(tok, ns=(tok.stat().st_atime_ns, tok.stat().st_mtime_ns + 1_000_000_000))
    check_embedding_model(pinned_model_copy, verify=True, cache_path=cache)

    # Assert
    assert hashed_when_unchanged == []
    assert hashed == ["tokenizer.json"]


def test_check_embedding_model_cacheSaysVerifiedButBytesReplaced_stillCaught(pinned_model_copy, tmp_path):
    # Arrange: verified once, then replaced with different bytes of a different size
    cache = tmp_path / "check.json"
    check_embedding_model(pinned_model_copy, verify=True, cache_path=cache)
    (pinned_model_copy / "tokenizer.json").write_text("{}")

    # Act / Assert
    with pytest.raises(embeddings.EmbeddingModelIntegrityError):
        check_embedding_model(pinned_model_copy, verify=True, cache_path=cache)
