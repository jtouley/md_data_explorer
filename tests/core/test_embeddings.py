"""ONNX sentence encoder: equivalence with sentence-transformers, bundling, and no network."""

from __future__ import annotations

import json
import os
import socket
import threading
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


def test_nl_query_engine_embeddingModel_isTheBundledOneAndNotAParameter(make_semantic_layer):
    # Arrange
    import inspect

    from clinical_analytics.core.nl_query_engine import NLQueryEngine

    semantic = make_semantic_layer(dataset_name="model_name_check", data={"patient_id": [1], "age": [1]})

    # Act
    engine = NLQueryEngine(semantic)

    # Assert: one model ships, so there is nothing to choose
    assert engine.embedding_model_name == embeddings.MODEL_NAME
    assert list(inspect.signature(NLQueryEngine).parameters) == ["semantic_layer"]


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
    real_sha = embeddings.sha256_file
    monkeypatch.setattr(embeddings, "sha256_file", lambda p: hashed.append(p.name) or real_sha(p))

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


@pytest.fixture
def fresh_encoder_cache(monkeypatch):
    """Empty process-wide encoder and embedding caches for one test (the real ones are put back)."""
    monkeypatch.setattr(embeddings, "_ENCODERS", {})
    monkeypatch.setattr(embeddings, "_TEXT_EMBEDDINGS", {})
    monkeypatch.setattr(embeddings, "_TEXT_EMBEDDINGS_PENDING", {})


@pytest.fixture
def counted_encoder_loads(monkeypatch):
    """Counts how often the encoder is constructed (each is a model lookup and, if found, a load)."""
    real = embeddings.OnnxSentenceEncoder
    loads: list[Path | None] = []

    class CountingEncoder(real):  # type: ignore[misc, valid-type]
        def __init__(self, model_dir: Path | None = None, *, batch_size: int = 32) -> None:
            loads.append(model_dir)
            super().__init__(model_dir, batch_size=batch_size)

    monkeypatch.setattr(embeddings, "OnnxSentenceEncoder", CountingEncoder)
    return loads


@pytest.mark.parametrize("calls", [1, 2, 5])
def test_get_encoder_repeatedCalls_loadsModelOncePerProcess(fresh_encoder_cache, counted_encoder_loads, calls):
    # Act
    encoders = [embeddings.get_encoder() for _ in range(calls)]

    # Assert
    assert all(encoder is encoders[0] for encoder in encoders)
    assert len(counted_encoder_loads) == 1


@pytest.mark.parametrize("calls", [1, 3])
def test_get_encoder_modelMissing_remembersFailureWithoutLookingAgain(
    fresh_encoder_cache, counted_encoder_loads, monkeypatch, tmp_path, calls
):
    # Arrange: a model directory with nothing in it
    monkeypatch.setenv(embeddings.EMBEDDING_MODEL_DIR_ENV, str(tmp_path))

    # Act / Assert: the same error every time, with the command that fixes it
    for _ in range(calls):
        with pytest.raises(EmbeddingModelMissingError, match="make fetch-embedding-model"):
            embeddings.get_encoder()
    assert len(counted_encoder_loads) == 1


def test_get_encoder_modelDirChanges_usesEncoderForThatDirectory(
    fresh_encoder_cache, counted_encoder_loads, monkeypatch, tmp_path
):
    # Arrange: the bundled model first, then an override pointing at an empty directory
    bundled = embeddings.get_encoder()
    monkeypatch.setenv(embeddings.EMBEDDING_MODEL_DIR_ENV, str(tmp_path))

    # Act / Assert: the missing override is not answered with the bundled encoder
    with pytest.raises(EmbeddingModelMissingError):
        embeddings.get_encoder()
    monkeypatch.delenv(embeddings.EMBEDDING_MODEL_DIR_ENV)
    assert embeddings.get_encoder() is bundled
    assert len(counted_encoder_loads) == 2


@pytest.mark.parametrize("queries", [0, 1, 4], ids=["init_only", "one_query", "four_queries"])
def test_nl_query_engine_modelMissing_semanticMatchOffAndModelNotRetriedPerQuery(
    fresh_encoder_cache, counted_encoder_loads, monkeypatch, tmp_path, make_semantic_layer, queries
):
    # Arrange: library use without the bundled model (serve.py's start check does not run here)
    from clinical_analytics.core.nl_query_engine import NLQueryEngine

    monkeypatch.setenv(embeddings.EMBEDDING_MODEL_DIR_ENV, str(tmp_path))
    semantic = make_semantic_layer(dataset_name="no_model", data={"patient_id": [1, 2], "age": [40, 50]})

    # Act: init embeds the golden examples (first and only lookup), then every query is Tier 2
    engine = NLQueryEngine(semantic)
    intents = [engine._semantic_match("how many patients?") for _ in range(queries)]

    # Assert
    assert intents == [None] * queries
    assert engine.encoder is None
    assert engine._golden_embeddings is None
    assert len(counted_encoder_loads) == 1


@pytest.mark.parametrize("engine_count", [2, 3])
def test_nl_query_engine_severalDatasets_shareOneEncoderAndGoldenEmbeddings(
    fresh_encoder_cache, counted_encoder_loads, monkeypatch, make_semantic_layer, engine_count
):
    # Arrange: one engine per dataset, as get_query_service_for_dataset builds them
    from clinical_analytics.core.nl_query_engine import NLQueryEngine

    layers = [
        make_semantic_layer(
            dataset_name=f"shared_{i}",
            data={"patient_id": [1, 2, 3], "age": [40 + i, 50, 60]},
            workspace_name=f"workspace_{i}",
        )
        for i in range(engine_count)
    ]

    # Act
    engines = [NLQueryEngine(layer) for layer in layers]
    intents = [engine._semantic_match("how many patients?") for engine in engines]

    # Assert: one ONNX session and one golden-example encoding for the whole process
    assert len(counted_encoder_loads) == 1
    assert all(engine.encoder is engines[0].encoder for engine in engines)
    assert engines[0]._golden_embeddings is not None
    assert all(engine._golden_embeddings is engines[0]._golden_embeddings for engine in engines)
    assert not engines[0]._golden_embeddings.flags.writeable
    assert [intent.intent_type for intent in intents] == ["COUNT"] * engine_count


def test_golden_embedding_cache_fileEdited_embedsAgain(fresh_encoder_cache, monkeypatch, make_semantic_layer, tmp_path):
    # Arrange: an engine pointed at its own golden examples file
    import yaml

    from clinical_analytics.core.nl_query_engine import NLQueryEngine

    config = tmp_path / "golden_examples.yaml"
    config.write_text(yaml.safe_dump({"questions": [{"id": "a", "query": "how many patients?"}]}))
    semantic = make_semantic_layer(dataset_name="edited_golden", data={"patient_id": [1, 2], "age": [40, 50]})

    def load(engine: NLQueryEngine) -> np.ndarray:
        engine._golden_embeddings = engine._golden_examples = None
        engine._golden_config_path = config
        engine._load_golden_examples()
        return engine._golden_embeddings

    first = load(NLQueryEngine(semantic))

    # Act: same file again, then the file with one more example
    again = load(NLQueryEngine(semantic))
    config.write_text(
        yaml.safe_dump({"questions": [{"id": "a", "query": "how many patients?"}, {"id": "b", "query": "mean age"}]})
    )
    edited = load(NLQueryEngine(semantic))

    # Assert
    assert again is first
    assert first.shape == (1, EMBEDDING_DIM)
    assert edited.shape == (2, EMBEDDING_DIM)


def _damage(model_dir: Path, damaged: str) -> None:
    target = model_dir / damaged
    if damaged.endswith(".onnx"):
        target.write_bytes(target.read_bytes()[:4096])  # truncated download or disk error
    else:
        target.write_text("{ not a tokenizer")


@pytest.mark.parametrize("damaged", embeddings.RUNTIME_FILES)
@pytest.mark.parametrize("calls", [1, 3])
def test_get_encoder_modelDamaged_raisesLoadErrorAndIsNotLoadedAgain(
    fresh_encoder_cache, counted_encoder_loads, pinned_model_copy, monkeypatch, damaged, calls
):
    # Arrange: every runtime file is present, one is unusable (library use: no serve.py hash check)
    _damage(pinned_model_copy, damaged)
    monkeypatch.setenv(embeddings.EMBEDDING_MODEL_DIR_ENV, str(pinned_model_copy))

    # Act / Assert: a clear domain error each time, one load attempt in total
    for _ in range(calls):
        with pytest.raises(embeddings.EmbeddingModelLoadError, match="could not be loaded.*reinstall the app"):
            embeddings.get_encoder()
    assert len(counted_encoder_loads) == 1


def test_get_encoder_runtimeNotImportable_raisesLoadErrorOnce(fresh_encoder_cache, counted_encoder_loads, monkeypatch):
    # Arrange: onnxruntime cannot be imported (a broken install)
    import sys

    monkeypatch.setitem(sys.modules, "onnxruntime", None)

    # Act / Assert
    for _ in range(2):
        with pytest.raises(embeddings.EmbeddingModelLoadError, match="could not be imported"):
            embeddings.get_encoder()
    assert len(counted_encoder_loads) == 1


@pytest.mark.parametrize("queries", [1, 4], ids=["one_query", "four_queries"])
def test_nl_query_engine_modelDamaged_semanticMatchOffAndModelNotReloadedPerQuery(
    fresh_encoder_cache, counted_encoder_loads, pinned_model_copy, monkeypatch, make_semantic_layer, queries
):
    # Arrange
    from clinical_analytics.core.nl_query_engine import NLQueryEngine

    _damage(pinned_model_copy, "onnx/model.onnx")
    monkeypatch.setenv(embeddings.EMBEDDING_MODEL_DIR_ENV, str(pinned_model_copy))
    semantic = make_semantic_layer(dataset_name="damaged_model", data={"patient_id": [1, 2], "age": [40, 50]})

    # Act
    engine = NLQueryEngine(semantic)
    intents = [engine._semantic_match("how many patients?") for _ in range(queries)]

    # Assert: Tier 2 is off; the ~90 MB session is not built again for every query
    assert intents == [None] * queries
    assert engine.encoder is None
    assert len(counted_encoder_loads) == 1


@pytest.mark.parametrize(
    ("target", "error"),
    [
        ("onnxruntime.InferenceSession", TypeError("unexpected keyword argument 'sess_options'")),
        ("onnxruntime.InferenceSession", AttributeError("'module' has no attribute 'SessionOptions'")),
        ("tokenizers.Tokenizer", TypeError("from_file() takes 1 positional argument")),
        ("tokenizers.Tokenizer", AttributeError("no attribute 'from_file'")),
    ],
    ids=["session_typeerror", "session_attributeerror", "tokenizer_typeerror", "tokenizer_attributeerror"],
)
def test_get_encoder_programmingErrorInLibraryCall_raisedAsIsAndNotRemembered(
    fresh_encoder_cache, counted_encoder_loads, monkeypatch, target, error
):
    # Arrange: a changed library API (a bug), not a damaged model file
    def broken(*_args, **_kwargs):
        raise error

    replacement = type("BrokenTokenizer", (), {"from_file": staticmethod(broken)}) if "Tokenizer" in target else broken
    monkeypatch.setattr(target, replacement)

    # Act / Assert: the real error each time, never "model unavailable, reinstall"
    for _ in range(2):
        with pytest.raises(type(error)):
            embeddings.get_encoder()
    assert len(counted_encoder_loads) == 2
    assert embeddings._ENCODERS == {}


@pytest.mark.parametrize("engine_count", [2, 3])
def test_nl_query_engine_templateFallback_templateEmbeddingsSharedAcrossEngines(
    fresh_encoder_cache, make_semantic_layer, engine_count
):
    # Arrange: a question no golden example matches, so every engine falls back to the templates
    from clinical_analytics.core.nl_query_engine import NLQueryEngine

    engines = [
        NLQueryEngine(
            make_semantic_layer(
                dataset_name=f"templates_{i}",
                data={"patient_id": [1, 2], "age": [40 + i, 50]},
                workspace_name=f"templates_ws_{i}",
            )
        )
        for i in range(engine_count)
    ]

    # Act
    for engine in engines:
        engine._semantic_match("xylophone tuning schedule for the orchestra")

    # Assert: one encoding of the fixed template list serves every engine
    first = engines[0].template_embeddings
    assert first is not None
    assert all(engine.template_embeddings is first for engine in engines)
    assert not first.flags.writeable


class _CountingEncoder:
    """A stand-in encoder: counts encode calls and returns one row per text."""

    def __init__(self) -> None:
        self.calls = 0

    def encode(self, texts):
        self.calls += 1
        return np.ones((len(texts), EMBEDDING_DIM), dtype=np.float32)


@pytest.mark.parametrize(
    ("second_key", "second_texts", "same_encoder", "expected_calls"),
    [
        ("k", ["a", "b"], True, 1),
        ("other", ["a", "b"], True, 2),
        ("k", ["a", "b", "c"], True, 2),
        ("k", ["a", "b"], False, 2),
    ],
    ids=["same_key_reused", "different_key", "text_count_changed", "different_encoder"],
)
def test_encode_cached_secondCall_reusesOnlyForSameKeyTextsAndEncoder(
    fresh_encoder_cache, second_key, second_texts, same_encoder, expected_calls
):
    # Arrange
    encoder = _CountingEncoder()
    other = encoder if same_encoder else _CountingEncoder()

    # Act
    first = embeddings.encode_cached(encoder, "k", ["a", "b"])
    second = embeddings.encode_cached(other, second_key, second_texts)

    # Assert
    assert encoder.calls + (0 if same_encoder else other.calls) == expected_calls
    assert (second is first) is (expected_calls == 1)
    assert not first.flags.writeable


def test_encode_cached_manyThreadsManyKeys_noErrorsAndBounded(fresh_encoder_cache):
    """Engines are built concurrently (one per dataset, in API worker threads)."""
    # Arrange

    encoder = _CountingEncoder()
    errors: list[BaseException] = []
    start = threading.Barrier(16)

    def worker(n: int) -> None:
        try:
            start.wait(timeout=10)
            for i in range(200):
                out = embeddings.encode_cached(encoder, (n, i % 20), ["x"])
                assert out.shape == (1, EMBEDDING_DIM)
        except BaseException as e:  # noqa: BLE001 - collected and asserted below
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(16)]

    # Act
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)

    # Assert
    assert errors == []
    assert 0 < len(embeddings._TEXT_EMBEDDINGS) <= embeddings.TEXT_EMBEDDINGS_MAX_ENTRIES


def test_reset_encoder_cache_afterUse_dropsEncodersAndTheEmbeddingsThatHoldThem(fresh_encoder_cache):
    # Arrange: a cached encoder and embeddings made with it (the cache entry references the encoder)
    encoder = embeddings.get_encoder()
    embeddings.encode_cached(encoder, "golden", ["how many patients?"])
    assert embeddings._ENCODERS and embeddings._TEXT_EMBEDDINGS

    # Act
    embeddings.reset_encoder_cache()

    # Assert: nothing keeps the old ~90 MB session alive
    assert embeddings._ENCODERS == {}
    assert embeddings._TEXT_EMBEDDINGS == {}


class _BlockingEncoder:
    """A stand-in encoder whose encode waits until released, to hold one key "in progress"."""

    def __init__(self, fail_first: bool = False) -> None:
        self.calls = 0
        self.started = threading.Event()
        self.release = threading.Event()
        self._fail_first = fail_first

    def encode(self, texts):
        self.calls += 1
        call = self.calls
        self.started.set()
        assert self.release.wait(timeout=10)
        if self._fail_first and call == 1:
            raise RuntimeError("encode failed")
        return np.ones((len(texts), EMBEDDING_DIM), dtype=np.float32)


def test_encode_cached_slowEncodeOfOneKey_otherKeyNotHeldUp(fresh_encoder_cache):
    # Arrange: one thread is encoding key "slow" and has not finished
    slow = _BlockingEncoder()
    worker = threading.Thread(target=embeddings.encode_cached, args=(slow, "slow", ["a"]))
    worker.start()
    assert slow.started.wait(timeout=10)
    other = _CountingEncoder()

    # Act: another engine asks for a different key while that encode is still running
    finished = threading.Event()
    threading.Thread(target=lambda: (embeddings.encode_cached(other, "fast", ["b"]), finished.set())).start()
    answered_while_slow_running = finished.wait(timeout=5)
    slow.release.set()
    worker.join(timeout=10)

    # Assert
    assert answered_while_slow_running
    assert other.calls == 1


@pytest.mark.parametrize("first_fails", [False, True], ids=["first_succeeds", "first_fails"])
def test_encode_cached_sameKeyWhileEncoding_waitsThenReusesOrRetries(fresh_encoder_cache, first_fails):
    # Arrange: thread A is encoding "k"; thread B asks for "k" meanwhile
    encoder = _BlockingEncoder(fail_first=first_fails)
    results: dict[str, object] = {}

    def ask(name: str) -> None:
        try:
            results[name] = embeddings.encode_cached(encoder, "k", ["a", "b"])
        except RuntimeError as e:
            results[name] = e

    a = threading.Thread(target=ask, args=("a",))
    a.start()
    assert encoder.started.wait(timeout=10)
    b = threading.Thread(target=ask, args=("b",))
    b.start()

    # Act
    encoder.release.set()
    a.join(timeout=10)
    b.join(timeout=10)

    # Assert: success is shared (one encode); a failure is A's alone and B encodes for itself
    if first_fails:
        assert isinstance(results["a"], RuntimeError)
        assert isinstance(results["b"], np.ndarray)
        assert encoder.calls == 2
    else:
        assert results["b"] is results["a"]
        assert encoder.calls == 1
    assert embeddings._TEXT_EMBEDDINGS_PENDING == {}
