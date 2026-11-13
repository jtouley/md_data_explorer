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
    rank_by_similarity,
)

REFERENCE = json.loads(
    (Path(__file__).resolve().parents[1] / "fixtures" / "embeddings" / "minilm_reference.json").read_text()
)
MIN_COSINE = 0.99999  # measured minimum 0.9999999 (1.23.2: 0.99999988)
# Retrieval may differ from the torch reference only between examples whose scores are closer than
# this. Measured: ONNX and torch scores differ by at most 3.6e-7; the closest pair of distinct
# golden examples at a top-3 boundary is 1.0e-4 apart; duplicate golden queries are 0.0 apart.
BOUNDARY_TOLERANCE = 1e-5


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


def test_encode_goldenExamples_topKMatchesTorchWithinBoundaryTolerance(cached_sentence_transformer):
    # Arrange
    golden = [q["query"] for q in load_golden_examples_config()["questions"]]
    assert len(golden) == REFERENCE["golden_example_count"]
    top_k = REFERENCE["top_k"]
    torch_phrases = np.asarray(REFERENCE["embeddings"], dtype=np.float32)

    # Act
    phrases = cached_sentence_transformer.encode(REFERENCE["phrases"])
    golden_emb = cached_sentence_transformer.encode(golden)
    onnx_scores = phrases @ golden_emb.T
    torch_scores = torch_phrases @ golden_emb.T  # recorded torch phrase vectors against the same examples

    # Assert: every embedding matches torch, and the retrieved set matches except across a near-tie
    cosines = (phrases * torch_phrases).sum(axis=1)
    assert cosines.min() >= MIN_COSINE
    assert np.abs(onnx_scores - torch_scores).max() < BOUNDARY_TOLERANCE
    for index, torch_top in enumerate(REFERENCE["golden_top_k"]):
        onnx_top = rank_by_similarity(onnx_scores[index], top_k=top_k)
        boundary = np.sort(torch_scores[index])[-top_k]  # torch score of the last retrieved example
        swapped = set(onnx_top) ^ set(torch_top)
        off_boundary = {i: float(torch_scores[index][i] - boundary) for i in swapped}
        assert all(abs(gap) < BOUNDARY_TOLERANCE for gap in off_boundary.values()), (
            f"phrase #{index}: ONNX top-{top_k} {onnx_top} vs torch {torch_top}; "
            f"swapped examples are not tied at the boundary (score - boundary): {off_boundary}"
        )


def test_rank_by_similarity_noiseAtRankBoundary_sameTopKOnBothMachines():
    # Arrange: examples 2 and 4 are duplicates; two CPUs disagree by ~1e-7 on which scores higher
    machine_a = np.array([0.29260436, 0.27214563, 0.26063693, 0.25978377, 0.26063683], dtype=np.float32)
    machine_b = np.array([0.29260436, 0.27214563, 0.26063683, 0.25978377, 0.26063693], dtype=np.float32)
    assert np.argsort(-machine_a)[:3].tolist() != np.argsort(-machine_b)[:3].tolist()  # plain ranking differs

    # Act
    top_a = rank_by_similarity(machine_a, top_k=3)
    top_b = rank_by_similarity(machine_b, top_k=3)

    # Assert
    assert top_a == top_b == [0, 1, 2]


@pytest.mark.parametrize(
    ("scores", "top_k", "expected"),
    [
        ([], 3, []),
        ([0.5], 3, [0]),
        ([0.1, 0.9, 0.5], 0, []),
        ([0.1, 0.9, 0.5], 2, [1, 2]),
        ([0.7, 0.7, 0.7], 2, [0, 1]),
        ([0.5, 0.5 + 5e-7, 0.5 - 5e-7], 1, [0]),
        ([0.5, 0.5 + 1e-4], 1, [1]),
        ([-0.2, -0.1, -0.3], 3, [1, 0, 2]),
    ],
    ids=[
        "empty",
        "fewer_than_k",
        "k_zero",
        "distinct",
        "exact_ties",
        "within_tolerance",
        "above_tolerance",
        "negative",
    ],
)
def test_rank_by_similarity_edgeCases_lowestIndexWinsTies(scores, top_k, expected):
    # Act
    ranked = rank_by_similarity(scores, top_k=top_k)

    # Assert
    assert ranked == expected


@pytest.mark.parametrize(
    ("scores", "top_k"),
    [([0.1, float("nan")], 1), ([0.1, float("inf")], 1), ([[0.1, 0.2]], 1), ([0.1, 0.2], -1)],
    ids=["nan", "inf", "two_dimensional", "negative_k"],
)
def test_rank_by_similarity_invalidInput_raisesValueError(scores, top_k):
    # Act / Assert
    with pytest.raises(ValueError, match="scores|top_k"):
        rank_by_similarity(scores, top_k=top_k)


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
