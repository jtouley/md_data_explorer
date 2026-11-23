# SPDX-License-Identifier: MIT
"""Worked question -> plan examples, and the few most similar to a question (few-shot).

The pool is ``resources/config/model_query_examples.yaml``: synthetic questions on made-up tables
whose column names no real dataset uses, each plan with every slot filled in. Similarity is the
app's own sentence embedding model (on this machine); without it the first examples are used.
"""

from __future__ import annotations

import functools
import json
from dataclasses import dataclass
from typing import Any, Final

import numpy as np
import structlog
import yaml
from pydantic import BaseModel, ConfigDict, ValidationError

from clinical_analytics.core import embeddings
from clinical_analytics.core.resources import config_file
from clinical_analytics.model_query.plan import Plan

logger = structlog.get_logger()

EXAMPLES_FILE: Final = "model_query_examples.yaml"
DEFAULT_TOP_K: Final = 4
SLOTS: Final = tuple(Plan.model_fields)


class _Table(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str
    columns: list[str]


class _Example(BaseModel):
    model_config = ConfigDict(extra="forbid")

    table: str
    question: str
    plan: dict[str, Any]


class _Pool(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tables: dict[str, _Table]
    examples: list[_Example]


@dataclass(frozen=True)
class Example:
    question: str
    table: str  # the made-up table's description
    columns: tuple[str, ...]
    plan: dict[str, Any]

    def text(self) -> str:
        reply = {slot: self.plan[slot] for slot in SLOTS}
        return f"Table: {self.table}\nQuestion: {self.question}\nReply: {json.dumps(reply)}"


@functools.lru_cache(maxsize=1)
def load_examples() -> tuple[Example, ...]:
    """The shipped pool.

    Raises:
        ValueError: the file is missing or not a valid pool (a packaging error: fail loud).
    """
    path = config_file(EXAMPLES_FILE)
    try:
        pool = _Pool.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
    except (OSError, yaml.YAMLError, ValidationError) as e:
        raise ValueError(f"model query examples at {path} cannot be read: {e}") from e
    unknown = sorted({e.table for e in pool.examples} - set(pool.tables))
    if unknown:
        raise ValueError(f"model query examples name unknown tables: {unknown}")
    incomplete = [e.question for e in pool.examples if set(e.plan) != set(SLOTS)]
    if incomplete:
        raise ValueError(f"model query examples must fill every slot: {incomplete}")
    return tuple(
        Example(
            question=e.question,
            table=" ".join(pool.tables[e.table].text.split()),
            columns=tuple(pool.tables[e.table].columns),
            plan=e.plan,
        )
        for e in pool.examples
    )


def similar_examples(question: str, top_k: int = DEFAULT_TOP_K) -> list[Example]:
    """The ``top_k`` examples whose questions are closest to ``question``, most similar last (the
    one right above the question weighs most for a small model)."""
    pool = load_examples()
    if top_k >= len(pool):
        return list(pool)
    try:
        encoder = embeddings.get_encoder()
        questions = [e.question for e in pool]
        matrix = embeddings.encode_cached(encoder, embeddings.texts_key("model_query_examples", questions), questions)
        query = encoder.encode([question])
        scores = np.asarray(matrix) @ np.asarray(query)[0]
    except embeddings.ENCODER_LOAD_ERRORS as e:
        logger.warning("model_query_examples_unranked", error_type=type(e).__name__)
        return list(pool[:top_k])
    best = np.argsort(-scores, kind="stable")[:top_k]
    return [pool[int(i)] for i in reversed(best)]
