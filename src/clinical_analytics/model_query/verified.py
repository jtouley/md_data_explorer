# SPDX-License-Identifier: MIT
"""Verified plans: questions whose answers the user confirmed, kept per dataset as worked
examples for similar questions on the same table.

What is stored is the question masked (its numbers, the plan's values and the plan's column
names replaced by placeholders, so no value the user typed is kept) and the plan. A stored plan is
never reused as an answer: the one or two closest are shown to the model as prior turns, and the
new plan goes through every check. Retrieval needs the same unit, timepoint and negation words as
the question (questions about patients and about encounters embed alike), then similarity.

The idea follows "verified queries" in open-source text-to-SQL tools and question masking from
DAIL-SQL; the code is this app's own. Stored in the data home, on this machine only.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from datetime import UTC, datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Final

import numpy as np
import structlog

from clinical_analytics.core import embeddings
from clinical_analytics.model_query import checks, linking
from clinical_analytics.model_query.context import DatasetContext
from clinical_analytics.model_query.plan import PlanError, parse_plan

logger = structlog.get_logger()

MAX_STORED: Final = 200
TOP_K: Final = 2
MIN_SIMILARITY: Final = 0.8
_DIR: Final = ("model_query", "verified")


def _path(data_dir: Path, dataset_id: str) -> Path:
    if not re.fullmatch(r"[\w-]+", dataset_id):
        raise ValueError("dataset id is not a plain name")
    return data_dir.joinpath(*_DIR, f"{dataset_id}.json")


def mask(question: str, plan: dict[str, Any]) -> str:
    """The question with its numbers, the plan's values and column names as placeholders."""
    text = re.sub(r"(?<![\w.])\d+(?:\.\d+)?(?![\w])", "<number>", question)
    values = {
        str(v)
        for slot in ("filters", "numerator")
        for cond in plan.get(slot) or []
        for v in [*(cond.get("values") or []), cond.get("value")]
        if v is not None and not isinstance(v, int | float)
    }
    for value in sorted(values, key=len, reverse=True):
        text = re.sub(rf"(?<!\w){re.escape(value)}(?!\w)", "<value>", text, flags=re.I)
    columns = {c.get("column") for slot in ("filters", "numerator") for c in plan.get(slot) or []}
    columns |= {plan.get(s) for s in ("measure", "totals_by") if isinstance(plan.get(s), str)}
    columns |= set(plan.get("group_by") or []) if isinstance(plan.get("group_by"), list) else {plan.get("group_by")}
    for name in sorted((c for c in columns if isinstance(c, str)), key=len, reverse=True):
        spoken = re.sub(r"[_.]+", " ", name)
        text = re.sub(rf"(?<!\w)(?:{re.escape(name)}|{re.escape(spoken)})(?!\w)", "<column>", text, flags=re.I)
    return " ".join(text.split())


def _load(data_dir: Path, dataset_id: str) -> list[dict[str, Any]]:
    path = _path(data_dir, dataset_id)
    if not path.is_file():
        return []
    try:
        entries = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        logger.warning("verified_plans_unreadable", dataset_id=dataset_id)
        return []
    return [e for e in entries if isinstance(e, dict)] if isinstance(entries, list) else []


def save(data_dir: Path, dataset_id: str, version: str | None, question: str, plan: dict[str, Any]) -> None:
    """Keep a confirmed question (masked) and its plan for the dataset (the newest first)."""
    try:
        parse_plan(plan)
    except PlanError as e:
        raise ValueError("not a plan") from e
    masked = mask(question, plan)
    entries = [e for e in _load(data_dir, dataset_id) if e.get("masked") != masked]
    entries.insert(0, {"masked": masked, "plan": plan, "version": version, "at": datetime.now(UTC).isoformat()})
    path = _path(data_dir, dataset_id)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as tmp:
        json.dump(entries[:MAX_STORED], tmp)
    os.replace(tmp.name, path)
    logger.info("verified_plan_saved", dataset_id=dataset_id, stored=min(len(entries), MAX_STORED))


def _same_reading(question: str, masked: str, context: DatasetContext) -> bool:
    """The question and a stored one say the same unit, timepoint and negation."""
    a, b = checks.read_cues(question, context), checks.read_cues(masked, context)
    return (a.persons, a.records, a.episodes, a.times, a.negated) == (
        b.persons,
        b.records,
        b.episodes,
        b.times,
        b.negated,
    )


def _similarity(question: str, texts: list[str]) -> list[float]:
    try:
        encoder = embeddings.get_encoder()
        matrix = np.asarray(encoder.encode(texts))
        query = np.asarray(encoder.encode([question]))[0]
        return [float(x) for x in matrix @ query]
    except embeddings.ENCODER_LOAD_ERRORS as e:
        logger.warning("verified_plans_unranked", error_type=type(e).__name__)
        return [SequenceMatcher(None, question.casefold(), t.casefold()).ratio() for t in texts]


def similar(
    data_dir: Path, dataset_id: str, question: str, context: DatasetContext, top_k: int = TOP_K
) -> list[tuple[str, dict[str, Any]]]:
    """The closest confirmed (masked question, plan) pairs for ``question``: plans whose columns
    the table still has, with the same unit, timepoint and negation words, most similar last."""
    names = {c.name for c in context.columns}
    candidates = []
    for entry in _load(data_dir, dataset_id):
        plan, masked = entry.get("plan"), entry.get("masked")
        if not isinstance(plan, dict) or not isinstance(masked, str):
            continue
        try:
            parsed = parse_plan(plan)
        except PlanError:
            continue
        used = {c.column for c in (*parsed.filters, *parsed.numerator)} | {
            n for n in (parsed.measure, parsed.group_by, parsed.totals_by) if n
        }
        if used <= names and _same_reading(question, masked, context):
            candidates.append((masked, plan))
    if not candidates:
        return []
    # The question masked like the stored ones: the values it names (linked to the table) too.
    named = [{"column": x.column, "values": [x.said]} for x in linking.link_values(question, context) if x.exact]
    scores = _similarity(mask(question, {"filters": named}), [m for m, _p in candidates])
    ranked = sorted(zip(scores, range(len(candidates)), strict=True), reverse=True)
    best = [candidates[i] for score, i in ranked[:top_k] if score >= MIN_SIMILARITY]
    logger.info("verified_plans_retrieved", dataset_id=dataset_id, found=len(best))
    return list(reversed(best))
