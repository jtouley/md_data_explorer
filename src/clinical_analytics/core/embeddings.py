# SPDX-License-Identifier: MIT
"""Sentence embeddings with ONNX Runtime and HF ``tokenizers`` (no torch, no network).

Reproduces ``SentenceTransformer("all-MiniLM-L6-v2").encode()`` exactly in structure:
BERT tokenisation (max 256 tokens, longest-first truncation, pad to the longest in the batch),
the official ONNX export of the same weights, mean pooling over the attention mask, then L2
normalisation (the model's Normalize module). tests/core/test_embeddings.py checks
cosine >= 0.999 against recorded torch outputs and identical golden-example top-k.

The model is shipped with the app (``clinical_analytics/resources/models``), fetched once at
build time by ``scripts/fetch_embedding_model.py`` (``make fetch-embedding-model``) at a pinned
revision with SHA-256 checks. It is never downloaded at runtime: a missing model raises
``EmbeddingModelMissingError`` naming the fix.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from collections.abc import Hashable, Sequence
from pathlib import Path
from typing import Final

import numpy as np
import numpy.typing as npt
import structlog

from clinical_analytics.core.resources import resources_root

logger = structlog.get_logger()

MODEL_NAME: Final = "all-MiniLM-L6-v2"
MODEL_REPO: Final = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_REVISION: Final = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
MODEL_LICENSE: Final = "Apache-2.0"
MAX_SEQ_LENGTH: Final = 256  # sentence_bert_config.json; tokenizer.json alone says 128
EMBEDDING_DIM: Final = 384
EMBEDDING_MODEL_DIR_ENV: Final = "CLINICAL_ANALYTICS_EMBEDDING_MODEL_DIR"
FETCH_HINT: Final = "Run `make fetch-embedding-model` when building the app"

# Pinned files of MODEL_REPO at MODEL_REVISION -> SHA-256 (onnx/model.onnx matches the HF LFS oid).
MODEL_FILES: Final[dict[str, str]] = {
    "onnx/model.onnx": "6fd5d72fe4589f189f8ebc006442dbb529bb7ce38f8082112682524616046452",
    "tokenizer.json": "be50c3628f2bf5bb5e3a7f17b1f74611b2561a3a27eeab05e5aa30f411572037",
    "config.json": "953f9c0d463486b10a6871cc2fd59f223b2c70184f49815e7efbcab5d8908b41",
    "modules.json": "84e40c8e006c9b1d6c122e02cba9b02458120b5fb0c87b746c41e0207cf642cf",
    "sentence_bert_config.json": "fc1993fde0a95c24ec6c022539d41cf6e2f7c9721e5415d6fb6897472a9cd4b7",
    "1_Pooling/config.json": "4be450dde3b0273bb9787637cfbd28fe04a7ba6ab9d36ac48e92b11e350ffc23",
    "tokenizer_config.json": "acb92769e8195aabd29b7b2137a9e6d6e25c476a4f15aa4355c233426c61576b",
    "special_tokens_map.json": "303df45a03609e4ead04bc3dc1536d0ab19b5358db685b6f3da123d05ec200e3",
    "README.md": "dcd602d2fd35c203a247304a06fec6654a12f7941b739f9221a064fe8dc3b7f0",
}
RUNTIME_FILES: Final = ("onnx/model.onnx", "tokenizer.json")

# Similarity scores closer than this are one tie. float32 cosine scores of the same text differ by
# up to ~4e-7 between CPUs and runtimes (measured), so an order decided by less than this is noise.
SCORE_TIE_TOLERANCE: Final = 1e-6

Embeddings = npt.NDArray[np.float32]


class EmbeddingModelMissingError(FileNotFoundError):
    """The bundled embedding model is not where the app expects it."""


class EmbeddingModelIntegrityError(EmbeddingModelMissingError):
    """A bundled model file is present but its SHA-256 is not the pinned one (corrupt or replaced)."""


class EmbeddingModelLoadError(RuntimeError):
    """The model files are there but the encoder could not be built from them.

    A truncated ``model.onnx``, an unreadable ``tokenizer.json``, or ``onnxruntime``/``tokenizers``
    failing to import. The original error is the ``__cause__``.
    """


# Everything get_encoder() can raise; callers that degrade (semantic matching off) catch this.
ENCODER_LOAD_ERRORS: Final = (EmbeddingModelMissingError, EmbeddingModelLoadError)


def bundled_model_dir() -> Path:
    """Where the build puts the model inside the package."""
    return resources_root() / "models" / MODEL_NAME


def embedding_model_dir() -> Path:
    """Model directory: ``CLINICAL_ANALYTICS_EMBEDDING_MODEL_DIR`` (bundlers, tests) or the bundled copy."""
    override = os.environ.get(EMBEDDING_MODEL_DIR_ENV, "").strip()
    return Path(override).expanduser() if override else bundled_model_dir()


def sha256_file(path: Path) -> str:
    """SHA-256 of a file, read in 1 MiB blocks (the one helper; the fetch script uses it too)."""
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def _stat_key(path: Path) -> list[int]:
    st = path.stat()
    return [st.st_size, st.st_mtime_ns]


def _read_cache(cache_path: Path | None) -> dict[str, object]:
    if cache_path is None:
        return {}
    try:
        data = json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}  # no cache yet, or unreadable: hash again
    return data if isinstance(data, dict) else {}


def _write_cache(cache_path: Path | None, entries: dict[str, object]) -> None:
    if cache_path is None:
        return
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache_path.with_name(cache_path.name + ".tmp")
        tmp.write_text(json.dumps(entries), encoding="utf-8")
        tmp.replace(cache_path)
    except OSError as e:
        logger.warning("embedding_model_check_cache_not_written", error_type=type(e).__name__)


def check_embedding_model(
    model_dir: Path | None = None, *, verify: bool = False, cache_path: Path | None = None
) -> Path:
    """Return the model directory, or raise naming the missing (or, with verify, corrupt) files.

    With ``verify`` the runtime files' SHA-256 must equal the pinned ``MODEL_FILES`` values. A
    verified (path, size, mtime) is remembered in ``cache_path`` so later starts skip the ~0.3 s hash.
    """
    root = model_dir if model_dir is not None else embedding_model_dir()
    missing = [name for name in RUNTIME_FILES if not (root / name).is_file()]
    if missing:
        raise EmbeddingModelMissingError(
            f"The embedding model {MODEL_NAME} is missing from {root} ({', '.join(missing)}). "
            f"{FETCH_HINT}; the app never downloads it at runtime."
        )
    if not verify:
        return root
    cache = _read_cache(cache_path)
    verified: dict[str, object] = {}
    for name in RUNTIME_FILES:
        path = root / name
        key = [str(path.resolve()), *_stat_key(path), MODEL_FILES[name]]
        if cache.get(name) != key:
            if sha256_file(path) != MODEL_FILES[name]:
                raise EmbeddingModelIntegrityError(
                    f"The embedding model file {name} in {root} does not match its pinned SHA-256 "
                    f"(damaged or replaced). {FETCH_HINT} again, or reinstall the app."
                )
        verified[name] = key
    if verified != cache:
        _write_cache(cache_path, verified)
    return root


# onnxruntime's own error classes for a model it cannot load (a truncated file raises
# InvalidProtobuf). They derive from Exception directly, so they are named one by one.
_ONNXRUNTIME_LOAD_ERROR_NAMES: Final = (
    "EPFail",
    "EngineError",
    "Fail",
    "InvalidArgument",
    "InvalidGraph",
    "InvalidProtobuf",
    "NoModel",
    "NoSuchFile",
    "NotImplemented",
    "RuntimeException",
)


def _onnxruntime_load_errors(ort: object) -> tuple[type[Exception], ...]:
    """The onnxruntime exception classes that mean "this model file cannot be used"."""
    state = getattr(getattr(ort, "capi", None), "onnxruntime_pybind11_state", None)
    found = [getattr(state, name, None) for name in _ONNXRUNTIME_LOAD_ERROR_NAMES]
    return tuple(cls for cls in found if isinstance(cls, type) and issubclass(cls, Exception))


class OnnxSentenceEncoder:
    """Drop-in for ``SentenceTransformer.encode`` returning normalised float32 embeddings."""

    def __init__(self, model_dir: Path | None = None, *, batch_size: int = 32) -> None:
        root = check_embedding_model(model_dir)
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        self.model_dir = root
        self.batch_size = batch_size
        try:
            # Imported here so modules that only need the constants stay light.
            import onnxruntime as ort
            from tokenizers import Tokenizer
        except ImportError as e:
            raise EmbeddingModelLoadError(f"The embedding runtime could not be imported ({e}).") from e
        if hasattr(ort, "disable_telemetry_events"):
            ort.disable_telemetry_events()  # Windows ETW events; the app reports nothing anywhere

        def unusable(file_name: str, error: BaseException) -> EmbeddingModelLoadError:
            return EmbeddingModelLoadError(
                f"The embedding model file {file_name} in {root} could not be loaded ({type(error).__name__}). "
                f"{FETCH_HINT} again, or reinstall the app."
            )

        try:
            tokenizer = Tokenizer.from_file(str(root / "tokenizer.json"))
        except OSError as e:
            raise unusable("tokenizer.json", e) from e
        except Exception as e:
            # tokenizers reports an invalid tokenizer.json with the base Exception class itself
            # (measured: type(e) is Exception), so there is no narrower type to name. Only that exact
            # class is a damaged file; any subclass (TypeError, AttributeError, ...) is a programming
            # error and is re-raised as it is, never remembered as "model unavailable".
            if type(e) is not Exception:
                raise
            raise unusable("tokenizer.json", e) from e
        tokenizer.enable_truncation(max_length=MAX_SEQ_LENGTH)
        pad_id = tokenizer.token_to_id("[PAD]")
        tokenizer.enable_padding(pad_id=0 if pad_id is None else pad_id, pad_token="[PAD]")  # longest in batch
        options = ort.SessionOptions()
        options.log_severity_level = 3
        model_load_errors: tuple[type[Exception], ...] = (OSError, *_onnxruntime_load_errors(ort))
        try:
            session = ort.InferenceSession(
                str(root / "onnx" / "model.onnx"), sess_options=options, providers=["CPUExecutionProvider"]
            )
        except model_load_errors as e:
            raise unusable("onnx/model.onnx", e) from e
        self._tokenizer = tokenizer
        self._session = session
        self._input_names = {i.name for i in self._session.get_inputs()}
        logger.info("embedding_model_loaded", model=MODEL_NAME, revision=MODEL_REVISION[:12], runtime="onnx")

    def encode(self, sentences: str | Sequence[str], **_ignored: object) -> Embeddings:
        """Embed one string or a list; always returns shape ``(n, 384)``."""
        texts = [sentences] if isinstance(sentences, str) else list(sentences)
        if not texts:
            return np.zeros((0, EMBEDDING_DIM), dtype=np.float32)
        chunks = [self._encode_batch(texts[i : i + self.batch_size]) for i in range(0, len(texts), self.batch_size)]
        return np.concatenate(chunks, axis=0)

    def _encode_batch(self, texts: list[str]) -> Embeddings:
        encoded = self._tokenizer.encode_batch(texts)
        input_ids = np.array([e.ids for e in encoded], dtype=np.int64)
        attention = np.array([e.attention_mask for e in encoded], dtype=np.int64)
        feeds = {"input_ids": input_ids, "attention_mask": attention}
        if "token_type_ids" in self._input_names:
            feeds["token_type_ids"] = np.array([e.type_ids for e in encoded], dtype=np.int64)
        hidden = self._session.run(None, feeds)[0]  # (batch, seq, 384) last_hidden_state
        return mean_pool_normalize(np.asarray(hidden, dtype=np.float32), attention)


# One encoder per process and model directory: the model is a fixed, stateless artifact (~90 MB as
# an ONNX session), so every NLQueryEngine shares it. Any load failure (model missing, damaged, or
# the runtime not importable) is remembered as well: it is tried once per process, not per query.
_ENCODERS: dict[Path, OnnxSentenceEncoder | EmbeddingModelMissingError | EmbeddingModelLoadError] = {}
_ENCODERS_LOCK: Final = threading.Lock()

# Embeddings of fixed text lists (golden examples, query templates), shared by every engine.
# Value: the encoder that produced them (compared by identity, so another encoder never reuses
# them) and the read-only array. Engines are built concurrently in API worker threads, so the
# dictionaries are guarded by a lock; the lock is held only to read and write them, never while
# encoding. _TEXT_EMBEDDINGS_PENDING marks a key some thread is encoding right now: others asking
# for the same key wait for that one encode, others asking for a different key are not held up.
_TEXT_EMBEDDINGS: dict[Hashable, tuple[object, Embeddings]] = {}
_TEXT_EMBEDDINGS_PENDING: dict[Hashable, threading.Event] = {}
_TEXT_EMBEDDINGS_LOCK: Final = threading.Lock()
TEXT_EMBEDDINGS_MAX_ENTRIES: Final = 16


def get_encoder() -> OnnxSentenceEncoder:
    """The shared encoder for the current model directory, loaded on first use.

    Raises:
        EmbeddingModelMissingError: the model is not there.
        EmbeddingModelLoadError: the model is there but could not be loaded.

    Either error is raised again on later calls without trying again (``ENCODER_LOAD_ERRORS``);
    a repaired model needs a restart or ``reset_encoder_cache()``.
    """
    root = embedding_model_dir()
    with _ENCODERS_LOCK:
        cached = _ENCODERS.get(root)
        if cached is None:
            try:
                cached = OnnxSentenceEncoder(root)
            except ENCODER_LOAD_ERRORS as e:
                logger.error(
                    "embedding_model_unavailable",
                    error_type=type(e).__name__,
                    error=str(e),
                    effect="semantic matching disabled until the model is repaired and the app restarted",
                )
                cached = e
            _ENCODERS[root] = cached
    if isinstance(cached, ENCODER_LOAD_ERRORS):
        raise cached
    return cached


def encode_cached(encoder: OnnxSentenceEncoder, key: Hashable, texts: Sequence[str]) -> Embeddings:
    """Embeddings of ``texts``, computed once per ``key`` and encoder; the array is read-only.

    ``key`` must change when the texts change (for a file: path, size and mtime; for a list: a
    hash of it). At most ``TEXT_EMBEDDINGS_MAX_ENTRIES`` lists are kept; the oldest is dropped
    first. Encoding runs outside the lock: threads asking for the same key wait for the first one
    to finish and then reuse its result, and threads asking for another key do not wait at all.
    If the encode fails, the error goes to the thread that ran it and nothing is cached; a thread
    that was waiting tries for itself.
    """
    while True:
        with _TEXT_EMBEDDINGS_LOCK:
            cached = _TEXT_EMBEDDINGS.get(key)
            if cached is not None and cached[0] is encoder and cached[1].shape[0] == len(texts):
                return cached[1]
            in_progress = _TEXT_EMBEDDINGS_PENDING.get(key)
            if in_progress is None:
                done = _TEXT_EMBEDDINGS_PENDING[key] = threading.Event()
                break
        in_progress.wait()
    try:
        embeddings = encoder.encode(texts)
        if isinstance(embeddings, np.ndarray):  # a stand-in encoder (tests) may return something else
            embeddings.setflags(write=False)
            with _TEXT_EMBEDDINGS_LOCK:
                _TEXT_EMBEDDINGS.pop(key, None)
                while len(_TEXT_EMBEDDINGS) >= TEXT_EMBEDDINGS_MAX_ENTRIES:
                    _TEXT_EMBEDDINGS.pop(next(iter(_TEXT_EMBEDDINGS)))
                _TEXT_EMBEDDINGS[key] = (encoder, embeddings)
        return embeddings
    finally:
        with _TEXT_EMBEDDINGS_LOCK:
            _TEXT_EMBEDDINGS_PENDING.pop(key, None)
        done.set()


def texts_key(kind: str, texts: Sequence[str]) -> tuple[str, str]:
    """A cache key for a list of texts that has no file behind it: its kind and a hash of the list."""
    digest = hashlib.sha256()
    for text in texts:
        digest.update(text.encode("utf-8"))
        digest.update(b"\x00")
    return (kind, digest.hexdigest())


def reset_encoder_cache() -> None:
    """Forget the shared encoders, remembered failures and cached embeddings (which hold encoders)."""
    with _ENCODERS_LOCK:
        _ENCODERS.clear()
    with _TEXT_EMBEDDINGS_LOCK:
        _TEXT_EMBEDDINGS.clear()


def mean_pool_normalize(hidden: npt.NDArray[np.float32], attention_mask: npt.NDArray[np.int64]) -> Embeddings:
    """sentence-transformers Pooling(mean) + Normalize, with the same epsilons as torch."""
    mask = attention_mask[..., None].astype(np.float32)
    summed = (hidden * mask).sum(axis=1)
    counts = np.clip(mask.sum(axis=1), 1e-9, None)
    pooled = summed / counts
    norms = np.clip(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-12, None)
    result: Embeddings = (pooled / norms).astype(np.float32)
    return result


def rank_by_similarity(scores: npt.ArrayLike, top_k: int = 1, tolerance: float = SCORE_TIE_TOLERANCE) -> list[int]:
    """
    Indices of the ``top_k`` highest scores, the same on every CPU.

    Scores within ``tolerance`` of the best remaining score count as tied with it, and a tie goes to
    the lowest index. A plain ``argmax``/``argsort`` lets float noise pick between duplicate entries
    (two golden examples with the same text), so two machines could retrieve different examples.

    Raises:
        ValueError: scores are not one-dimensional, hold NaN/inf, or ``top_k``/``tolerance`` < 0.
    """
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError(f"scores must be one-dimensional, got shape {values.shape}")
    if top_k < 0 or tolerance < 0:
        raise ValueError(f"top_k and tolerance must be >= 0, got top_k={top_k}, tolerance={tolerance}")
    if not np.isfinite(values).all():
        raise ValueError("scores contain NaN or infinite values")

    remaining = values.copy()
    ranked: list[int] = []
    for _ in range(min(top_k, values.size)):
        # argmax of a boolean array is the first True: the lowest index among the tied best
        pick = int(np.argmax(remaining >= float(np.max(remaining)) - tolerance))
        ranked.append(pick)
        remaining[pick] = -np.inf
    return ranked
