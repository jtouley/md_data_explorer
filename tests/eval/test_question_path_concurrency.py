# SPDX-License-Identifier: MIT
"""Concurrency: questions asked at once from many threads on one layer give the plan, hold
decision and answer recorded when they were asked one at a time (question_path_snapshot.json).

The DuckDB lock covers DuckDB work only, so parses overlap. A question's per-question parser
state, a level merge between its parse and its execution, or a DuckDB call made without the
lock would each show up here as an answer that differs from the serial one.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from typing import Any

import pytest

from clinical_analytics.eval.catalog import repo_root
from clinical_analytics.eval.headless import service_for
from tests.eval.question_path_snapshot import SNAPSHOT_PATH, record, suite_questions

_THREADS = 8
# Switch threads every few microseconds instead of every 5 ms, so interleavings that would be
# rare in the app happen on every run.
_SWITCH_INTERVAL_S = 1e-5
# Every dataset of the snapshot; test_everySnapshotDataset_isAskedConcurrently keeps this complete.
_DATASETS = (
    "cohort",
    "snap_coh",
    "snap_own",
    "snap_golden",
    "snap_rev",
    "snap_codes",
    "snap_reverse",
    "snap_regions",
    "snap_creat",
)


@lru_cache(maxsize=1)
def _recorded() -> dict[str, dict[str, Any]]:
    loaded: dict[str, dict[str, Any]] = json.loads(SNAPSHOT_PATH.read_text())
    return loaded


@lru_cache(maxsize=1)
def _by_dataset() -> dict[str, tuple[tuple[str, str], ...]]:
    grouped: dict[str, list[tuple[str, str]]] = {}
    for suite, dataset, query in suite_questions():
        grouped.setdefault(dataset, []).append((suite, query))
    return {dataset: tuple(items) for dataset, items in grouped.items()}


@pytest.fixture
def fast_thread_switching() -> Iterator[None]:
    previous = sys.getswitchinterval()
    sys.setswitchinterval(_SWITCH_INTERVAL_S)
    try:
        yield
    finally:
        sys.setswitchinterval(previous)


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.parametrize("dataset", _DATASETS)
def test_questionPath_threadsAskOneLayerAtOnce_everyAnswerEqualsSerial(
    dataset: str, fast_thread_switching: None
) -> None:
    # Arrange: one shared service (one layer, one parser) per dataset, as the API has.
    questions = _by_dataset()[dataset]
    service_for(dataset, repo_root())

    # Act
    with ThreadPoolExecutor(max_workers=_THREADS) as pool:
        answers = list(pool.map(lambda item: (item, record(dataset, item[1])), questions))

    # Assert
    differences = {
        query: (_recorded()[f"{suite}|{dataset}|{query}"], now)
        for (suite, query), now in answers
        if now != _recorded()[f"{suite}|{dataset}|{query}"]
    }
    assert len(answers) == len(questions) > 0
    assert differences == {}


def test_everySnapshotDataset_isAskedConcurrently() -> None:
    # Arrange
    recorded = {key.split("|")[1] for key in _recorded()}

    # Act
    missing = recorded - set(_DATASETS)

    # Assert
    assert missing == set()
