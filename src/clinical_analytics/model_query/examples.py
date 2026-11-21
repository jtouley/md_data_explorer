# SPDX-License-Identifier: MIT
"""Worked question -> reply examples, and the few most similar to a question (few-shot).

The pool is ``resources/config/model_query_examples.yaml``: synthetic questions on made-up tables.
Similarity is the app's own sentence embedding model (on this machine); without it the first
examples of the pool are used, so the path still works.
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
from clinical_analytics.model_query.plan import LISTING, SQL, STAT_TESTS, ReplyFormat

logger = structlog.get_logger()

EXAMPLES_FILE: Final = "model_query_examples.yaml"
DEFAULT_TOP_K: Final = 4
_PLAN_FIELDS: Final = ("thought", "answer", "where", "group_by", "numerator", "number", "outcome")


class _Example(BaseModel):
    model_config = ConfigDict(extra="forbid")

    table: str
    question: str
    plan: dict[str, Any]
    sql: str | None = None


class _Pool(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tables: dict[str, str]
    examples: list[_Example]


@dataclass(frozen=True)
class Example:
    question: str
    table: str  # the made-up table's one-line description
    plan: dict[str, Any]
    sql: str | None

    def reply(self, fmt: ReplyFormat) -> dict[str, Any]:
        """The reply the model should give for this example, in ``fmt`` (keys in schema order)."""
        plan = {name: self.plan.get(name) for name in _PLAN_FIELDS}
        if fmt == "plan":
            return plan
        rest = {name: plan[name] for name in _PLAN_FIELDS[2:]}
        if plan["answer"] in (*STAT_TESTS, LISTING):
            return {"thought": plan["thought"], "answer": plan["answer"], "sql": None, **rest}
        return {"thought": plan["thought"], "answer": SQL, "sql": self.sql, **dict.fromkeys(rest)}

    def text(self, fmt: ReplyFormat) -> str:
        return f"Table: {self.table}\nQuestion: {self.question}\nReply: {json.dumps(self.reply(fmt))}"


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
    return tuple(
        Example(question=e.question, table=" ".join(pool.tables[e.table].split()), plan=e.plan, sql=e.sql)
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


# A table with at most this many columns is short enough to read whole; a wider one gets its
# columns closest to the question repeated right above the question.
SHORTLIST_MIN_COLUMNS: Final = 15
SHORTLIST_SIZE: Final = 6


def related_columns(question: str, described: dict[str, str], top_k: int = SHORTLIST_SIZE) -> list[str]:
    """The names of the columns whose description (``described``: name -> one line saying what it
    holds) is closest to the question, best first; none for a narrow table or without the encoder."""
    if len(described) <= SHORTLIST_MIN_COLUMNS:
        return []
    names = list(described)
    texts = [described[name] for name in names]
    try:
        encoder = embeddings.get_encoder()
        matrix = embeddings.encode_cached(encoder, embeddings.texts_key("model_query_columns", texts), texts)
        scores = np.asarray(matrix) @ np.asarray(encoder.encode([question]))[0]
    except embeddings.ENCODER_LOAD_ERRORS as e:
        logger.warning("model_query_columns_unranked", error_type=type(e).__name__)
        return []
    return [names[int(i)] for i in np.argsort(-scores, kind="stable")[:top_k]]
