# SPDX-License-Identifier: MIT
"""Concurrency: questions asked at once from many threads on one layer give the plan, hold
decision and answer recorded when they were asked one at a time (question_path_snapshot.json).

The DuckDB lock covers DuckDB work only, so parses overlap. A question's per-question parser
state, a level merge between its parse and its execution, or a DuckDB call made without the
lock would each show up here as an answer that differs from the serial one.

The confirm-while-asking mix adds what the snapshot cannot: one thread confirms the grain (and
the category spellings) while eight others ask, every one of them prepared before the
confirmation was stored. Each must be answered as if asked alone after it.
"""

from __future__ import annotations

import json
import sys
import threading
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from typing import Any

import polars as pl
import pytest

from clinical_analytics.contracts.declarations import MemoryDeclarationStore
from clinical_analytics.core import answer_checks
from clinical_analytics.core.query_service import QueryService
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
    "snap_geo",
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


# --- One thread confirms while eight ask --------------------------------------------------------

_MIX_DATASET = "visits"
_MIX_VERSION = "v1"
_MIX_QUESTIONS = (
    "how many patients over 50",
    "how many male patients",
    "average age by sex",
    "how many patients",
    "average age",
    "how many patients by sex",
    "how many female patients",
    "average age of male patients",
)
_WAIT_S = 30.0


def _visit_rows() -> pl.DataFrame:
    """Synthetic: 80 visit rows of 60 patients; sex is written four ways (three are one category)."""
    ids = [f"P{i:03d}" for i in range(60)] + [f"P{i:03d}" for i in range(20)]
    sexes = ["Male", "male", "M", "Female"]
    return pl.DataFrame(
        {
            "patient_id": ids,
            "age": [30 + (int(pid[1:]) * 7) % 50 for pid in ids],
            "sex": [sexes[int(pid[1:]) % 4] for pid in ids],
        }
    )


def _shown(result: Any) -> tuple[Any, ...]:
    return (result.requires_confirmation, result.failure_reason, result.claim, result.plan_line)


def _ask(service: QueryService, question: str, plan_hash: str | None = None) -> Any:
    return service.ask(
        question,
        dataset_id=_MIX_DATASET,
        dataset_version=_MIX_VERSION,
        confirmed=plan_hash is not None,
        confirm_plan_hash=plan_hash,
    )


def _confirm(service: QueryService, question: str) -> None:
    held = _ask(service, question)
    assert held.requires_confirmation, held.failure_reason
    assert not _ask(service, question, held.plan_hash).requires_confirmation


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.parametrize("confirmations", [("how many patients",), ("how many patients", "how many patients by sex")])
def test_confirmWhileAsking_oneConfirmsEightAsk_everyAnswerEqualsSerialAfterTheConfirmations(
    make_semantic_layer: Callable[..., Any], fast_thread_switching: None, confirmations: tuple[str, ...]
) -> None:
    # Arrange: each asker's first question is prepared, then parks in its parse until every
    # confirmation is stored, so all eight are checked after declarations they did not read.
    store = MemoryDeclarationStore()
    shared = QueryService(make_semantic_layer(dataset_name="clinic", data=_visit_rows()), declarations=store)
    parse = shared.nl_engine.parse_query
    all_parked = threading.Barrier(_THREADS + 1, timeout=_WAIT_S)
    confirmed = threading.Event()
    parked = threading.local()

    def parse_query(*args: Any, **kwargs: Any) -> Any:
        if threading.current_thread().name.startswith("asker") and not getattr(parked, "done", False):
            parked.done = True
            all_parked.wait()
            assert confirmed.wait(_WAIT_S)
        return parse(*args, **kwargs)

    shared.nl_engine.parse_query = parse_query  # type: ignore[method-assign]

    def asker(offset: int) -> list[tuple[str, tuple[Any, ...]]]:
        order = _MIX_QUESTIONS[offset:] + _MIX_QUESTIONS[:offset]
        return [(question, _shown(_ask(shared, question))) for question in order]

    def confirmer() -> None:
        all_parked.wait()
        try:
            for question in confirmations:
                _confirm(shared, question)
        finally:
            confirmed.set()

    # Act
    askers = [
        threading.Thread(target=lambda i=i: results.__setitem__(i, asker(i % len(_MIX_QUESTIONS))), name=f"asker-{i}")
        for i in range(_THREADS)
    ]
    results: dict[int, list[tuple[str, tuple[Any, ...]]]] = {}
    confirming = threading.Thread(target=confirmer, name="confirmer")
    for thread in [*askers, confirming]:
        thread.start()
    for thread in [*askers, confirming]:
        thread.join(_WAIT_S * 2)
    serial_service = QueryService(
        make_semantic_layer(dataset_name="clinic", data=_visit_rows(), workspace_name="serial"),
        declarations=MemoryDeclarationStore(),
    )
    serial_service.declarations.save(_MIX_DATASET, store.load(_MIX_DATASET))
    serial = {question: _shown(_ask(serial_service, question)) for question in _MIX_QUESTIONS}

    # Assert: answered as if asked alone after the confirmations; none held for one again.
    answers = [item for thread_answers in results.values() for item in thread_answers]
    differences = [(question, shown, serial[question]) for question, shown in answers if shown != serial[question]]
    assert len(answers) == _THREADS * len(_MIX_QUESTIONS)
    given = [answer_checks.grain_hold_reason("patient_id")]
    if len(confirmations) > 1:
        given.append("may be one category written differently")
    reheld = [question for question, shown in answers if shown[0] and any(r in (shown[1] or "") for r in given)]
    assert differences == []
    assert reheld == []
