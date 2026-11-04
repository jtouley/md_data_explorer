"""Sentence embeddings with ONNX Runtime and HF ``tokenizers`` (no torch, no network).

Reproduces ``SentenceTransformer("all-MiniLM-L6-v2").encode()`` exactly in structure:
BERT tokenisation (max 256 tokens, longest-first truncation, pad to the longest in the batch),
the official ONNX export of the same weights, mean pooling over the attention mask, then L2
normalisation (the model's Normalize module). tests/core/test_embeddings_equivalence.py checks
cosine >= 0.999 against recorded torch outputs and identical golden-example top-k.

The model is shipped with the app (``clinical_analytics/resources/models``), fetched once at
build time by ``scripts/fetch_embedding_model.py`` (``make fetch-embedding-model``) at a pinned
revision with SHA-256 checks. It is never downloaded at runtime: a missing model raises
``EmbeddingModelMissingError`` naming the fix.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
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

Embeddings = npt.NDArray[np.float32]


class EmbeddingModelMissingError(FileNotFoundError):
    """The bundled embedding model is not where the app expects it."""


def bundled_model_dir() -> Path:
    """Where the build puts the model inside the package."""
    return resources_root() / "models" / MODEL_NAME


def embedding_model_dir() -> Path:
    """Model directory: ``CLINICAL_ANALYTICS_EMBEDDING_MODEL_DIR`` (bundlers, tests) or the bundled copy."""
    override = os.environ.get(EMBEDDING_MODEL_DIR_ENV, "").strip()
    return Path(override).expanduser() if override else bundled_model_dir()


def check_embedding_model(model_dir: Path | None = None) -> Path:
    """Return the model directory, or raise naming the missing files and the fix."""
    root = model_dir if model_dir is not None else embedding_model_dir()
    missing = [name for name in RUNTIME_FILES if not (root / name).is_file()]
    if missing:
        raise EmbeddingModelMissingError(
            f"The embedding model {MODEL_NAME} is missing from {root} ({', '.join(missing)}). "
            f"{FETCH_HINT}; the app never downloads it at runtime."
        )
    return root


class OnnxSentenceEncoder:
    """Drop-in for ``SentenceTransformer.encode`` returning normalised float32 embeddings."""

    def __init__(self, model_dir: Path | None = None, *, batch_size: int = 32) -> None:
        # Imported here so modules that only need the constants stay light.
        import onnxruntime as ort
        from tokenizers import Tokenizer

        root = check_embedding_model(model_dir)
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        self.model_dir = root
        self.batch_size = batch_size
        if hasattr(ort, "disable_telemetry_events"):
            ort.disable_telemetry_events()  # Windows ETW events; the app reports nothing anywhere
        tokenizer = Tokenizer.from_file(str(root / "tokenizer.json"))
        tokenizer.enable_truncation(max_length=MAX_SEQ_LENGTH)
        pad_id = tokenizer.token_to_id("[PAD]")
        tokenizer.enable_padding(pad_id=0 if pad_id is None else pad_id, pad_token="[PAD]")  # longest in batch
        self._tokenizer = tokenizer
        options = ort.SessionOptions()
        options.log_severity_level = 3
        self._session = ort.InferenceSession(
            str(root / "onnx" / "model.onnx"), sess_options=options, providers=["CPUExecutionProvider"]
        )
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


def mean_pool_normalize(hidden: npt.NDArray[np.float32], attention_mask: npt.NDArray[np.int64]) -> Embeddings:
    """sentence-transformers Pooling(mean) + Normalize, with the same epsilons as torch."""
    mask = attention_mask[..., None].astype(np.float32)
    summed = (hidden * mask).sum(axis=1)
    counts = np.clip(mask.sum(axis=1), 1e-9, None)
    pooled = summed / counts
    norms = np.clip(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-12, None)
    result: Embeddings = (pooled / norms).astype(np.float32)
    return result
