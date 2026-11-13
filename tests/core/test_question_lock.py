# SPDX-License-Identifier: MIT
"""The per-layer DuckDB lock covers DuckDB work only: parsing (LLM calls included) overlaps
across questions on one layer, while every DuckDB call still runs with the lock held."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from typing import Any
from unittest.mock import MagicMock

import polars as pl
import pytest

from clinical_analytics.contracts.declarations import Declarations, GrainDeclaration, MemoryDeclarationStore
from clinical_analytics.core.query_service import MAX_PARSE_ATTEMPTS, STATE_KEPT_CHANGING_REASON, QueryService

_WAIT_S = 5.0
_FAST_S = 1.0


def _held_by_caller(lock: Any) -> bool:
    """Whether the calling thread holds ``lock``: another thread cannot take it at once."""
    taken: list[bool] = []

    def probe() -> None:
        if lock.acquire(blocking=False):
            taken.append(True)
            lock.release()

    other = threading.Thread(target=probe)
    other.start()
    other.join()
    return not taken


def _mock_layer_service() -> tuple[MagicMock, QueryService]:
    layer = MagicMock()
    layer._con_lock = threading.RLock()
    layer.state_generation = 0
    service = QueryService(layer)
    service.conversation_manager.normalize_query = lambda question: question.strip()  # type: ignore[method-assign]
    return layer, service


def test_ask_twoQuestionsParsingAtOnce_parsesOverlap() -> None:
    # Arrange: each parse waits until the other question is parsing too. With the lock held
    # across the parse (an LLM call can take seconds), the second parse never starts.
    _layer, service = _mock_layer_service()
    both_parsing = threading.Barrier(2, timeout=_WAIT_S)
    overlapped: list[bool] = []

    def parse_query(*_args: Any, **_kwargs: Any) -> None:
        try:
            both_parsing.wait()
            overlapped.append(True)
        except threading.BrokenBarrierError:
            overlapped.append(False)
        return None

    service.nl_engine.parse_query = parse_query  # type: ignore[method-assign]
    threads = [threading.Thread(target=service.ask, kwargs={"question": q, "dataset_id": "ds"}) for q in ("a", "b")]

    # Act
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=2 * _WAIT_S)

    # Assert
    assert overlapped == [True, True]


def test_ask_levelsMergedWhileParsing_parsedAgainWithoutTheLock() -> None:
    # Arrange: while this question parses unlocked, another question merges levels (the layer's
    # state_generation moves). Its first parse read the old levels, so it must not be executed,
    # and the second parse must not hold the lock either (an LLM parse can take seconds).
    layer, service = _mock_layer_service()
    parses_locked: list[bool] = []

    def parse_query(*_args: Any, **_kwargs: Any) -> None:
        parses_locked.append(_held_by_caller(layer._con_lock))
        if len(parses_locked) == 1:
            layer.state_generation += 1
        return None

    service.nl_engine.parse_query = parse_query  # type: ignore[method-assign]

    # Act
    result = service.ask("how many patients", dataset_id="ds")

    # Assert
    assert parses_locked == [False, False]
    assert result.plan is None
    assert result.failure_reason is None


def test_ask_stateChangesDuringEveryParse_givesUpWithPlainRetryMessage() -> None:
    # Arrange: another question changes the layer during every one of this question's parses.
    layer, service = _mock_layer_service()
    parses_locked: list[bool] = []

    def parse_query(*_args: Any, **_kwargs: Any) -> None:
        parses_locked.append(_held_by_caller(layer._con_lock))
        layer.state_generation += 1
        return None

    service.nl_engine.parse_query = parse_query  # type: ignore[method-assign]

    # Act
    result = service.ask("how many patients", dataset_id="ds")

    # Assert
    assert parses_locked == [False] * MAX_PARSE_ATTEMPTS
    assert result.requires_confirmation is False
    assert result.failure_reason == STATE_KEPT_CHANGING_REASON
    assert result.run_key is None


@pytest.mark.parametrize(
    ("stored_version", "expected_parses"),
    [
        pytest.param("v1", 2, id="this_version_parsedAgain"),
        pytest.param("v0", 1, id="retired_version_notParsedAgain"),
    ],
)
def test_ask_declarationStoredElsewhereWhileParsing_parsedAgainOnlyForItsVersion(
    stored_version: str, expected_parses: int
) -> None:
    # Arrange: another process (or a retired layer after a re-upload) stores a grain while this
    # v1 question parses; no generation on this layer moves.
    _layer, service = _mock_layer_service()
    store = MemoryDeclarationStore()
    service.declarations = store
    parses = 0

    def parse_query(*_args: Any, **_kwargs: Any) -> None:
        nonlocal parses
        parses += 1
        if parses == 1:
            grain = GrainDeclaration(key="patient_id", rows=2, patients=1, dataset_version=stored_version)
            store.save("ds", Declarations(grains=(grain,)))
        return None

    service.nl_engine.parse_query = parse_query  # type: ignore[method-assign]

    # Act
    service.ask("how many patients", dataset_id="ds", dataset_version="v1")

    # Assert
    assert parses == expected_parses


class _LockCheckedConnection:
    """Stands in for a layer's raw DuckDB connection and records every use made without the
    layer's lock held by the calling thread."""

    def __init__(self, raw: Any, lock: Any) -> None:
        object.__setattr__(self, "_raw", raw)
        object.__setattr__(self, "_lock", lock)
        object.__setattr__(self, "unlocked_uses", [])

    def __getattr__(self, name: str) -> Any:
        if not _held_by_caller(self._lock):
            self.unlocked_uses.append(name)
        return getattr(self._raw, name)


def _clinic_frame_data(rows: int = 60) -> dict[str, list[Any]]:
    regions = ["North", "North East", "South"]
    return {
        "patient_id": list(range(1, rows + 1)),
        "age": [30 + (i * 7) % 50 for i in range(rows)],
        "sex": ["Male" if i % 2 else "Female" for i in range(rows)],
        "region": [regions[i % 3] for i in range(rows)],
        "treatment": ["A" if i % 3 else "B" for i in range(rows)],
        "ldl": [80.0 + (i * 13) % 100 for i in range(rows)],
    }


_CLINIC_QUESTIONS = (
    "how many patients?",
    "how many patients in North East?",
    "average age by treatment",
    "average ldl for patients over 50",
    "how many female patients in the South region?",
    "compare ldl by sex",
    "what is the median age of patients with treatment B?",
    "how many patients by region",
)


def test_ask_realLayer_everyDuckdbCallHoldsTheLayerLock(make_semantic_layer: Any) -> None:
    # Arrange
    layer = make_semantic_layer(dataset_name="clinic", data=_clinic_frame_data())
    checked = _LockCheckedConnection(layer.con.con, layer._con_lock)
    layer.con.con = checked
    service = QueryService(layer)

    # Act
    for question in _CLINIC_QUESTIONS:
        service.ask(question, dataset_id="clinic", dataset_version="v1")

    # Assert
    assert checked.unlocked_uses == []


# --- PR #99 review: cache hits never wait for another question's DuckDB step ----------------


def _cached_reads(layer: Any, service: QueryService) -> dict[str, Callable[[], Any]]:
    engine = service.nl_engine
    columns = list(layer.get_base_view().columns)
    return {
        "distinct_values": lambda: engine._distinct_values("sex"),
        "refresh_value_index": lambda: engine.refresh_value_index(columns),
        "base_value_counts": lambda: layer.base_value_counts("sex"),
        "base_missing_count": lambda: layer.base_missing_count("patient_id"),
        "base_count": lambda: layer._base_count(None),
        "base_count_by_key": lambda: layer._base_count("patient_id"),
    }


@pytest.mark.parametrize(
    "read",
    [
        "distinct_values",
        "refresh_value_index",
        "base_value_counts",
        "base_missing_count",
        "base_count",
        "base_count_by_key",
    ],
)
def test_cachedRead_whileAnotherQuestionHoldsTheLock_returnsWithoutWaiting(make_semantic_layer: Any, read: str) -> None:
    # Arrange: the value is cached; then another question runs a long DuckDB step.
    layer = make_semantic_layer(dataset_name="clinic", data=_clinic_frame_data())
    service = QueryService(layer)
    call = _cached_reads(layer, service)[read]
    call()
    held, release = threading.Event(), threading.Event()

    def long_execution() -> None:
        with layer._con_lock:
            held.set()
            release.wait(_WAIT_S)

    other = threading.Thread(target=long_execution)
    other.start()
    assert held.wait(_WAIT_S)

    # Act
    started = time.monotonic()
    try:
        call()
        elapsed = time.monotonic() - started
    finally:
        release.set()
        other.join()

    # Assert
    assert elapsed < _FAST_S


# --- PR #99 review: a confirmation stored while a question parses ----------------------------


def _visit_rows() -> pl.DataFrame:
    """80 visit rows of 60 patients; sex is written four ways (three of them one category)."""
    ids = [f"P{i:03d}" for i in range(60)] + [f"P{i:03d}" for i in range(20)]
    sexes = ["Male", "male", "M", "Female"]
    return pl.DataFrame(
        {
            "patient_id": ids,
            "age": [30 + (int(pid[1:]) * 7) % 50 for pid in ids],
            "sex": [sexes[int(pid[1:]) % 4] for pid in ids],
        }
    )


def _shown(result: Any) -> tuple[bool, str | None, str | None]:
    return (result.requires_confirmation, result.failure_reason, result.claim)


def test_ask_grainConfirmedWhileParsing_answeredWithTheConfirmationNotHeldAgain(make_semantic_layer: Any) -> None:
    # Arrange: A is prepared before B confirms that rows are visits, and checked after.
    layer = make_semantic_layer(dataset_name="clinic", data=_visit_rows())
    service = QueryService(layer, declarations=MemoryDeclarationStore())
    held = service.ask("how many patients", dataset_id="clinic", dataset_version="v1")
    parse = service.nl_engine.parse_query
    a_parsing, b_confirmed = threading.Event(), threading.Event()
    answers: dict[str, Any] = {}

    def parse_query(*args: Any, **kwargs: Any) -> Any:
        if threading.current_thread().name == "A" and not b_confirmed.is_set():
            a_parsing.set()
            assert b_confirmed.wait(_WAIT_S)
        return parse(*args, **kwargs)

    service.nl_engine.parse_query = parse_query  # type: ignore[method-assign]

    def ask_a() -> None:
        answers["A"] = service.ask("how many patients over 50", dataset_id="clinic", dataset_version="v1")

    asker = threading.Thread(target=ask_a, name="A")

    # Act
    asker.start()
    assert a_parsing.wait(_WAIT_S)
    confirmed = service.ask(
        "how many patients", dataset_id="clinic", dataset_version="v1", confirmed=True, confirm_plan_hash=held.plan_hash
    )
    b_confirmed.set()
    asker.join(_WAIT_S)
    later = service.ask("how many patients", dataset_id="clinic", dataset_version="v1")

    # Assert: A counted patients, as it would have alone after the confirmation; the shared
    # layer still counts patients for the next question.
    assert held.requires_confirmation is True
    assert _shown(confirmed) == (False, None, "Total count: 60 patients.")
    assert _shown(answers["A"]) == (False, None, "Total count: 34 patients.")
    assert _shown(later) == (False, None, "Total count: 60 patients.")
    assert layer.count_entity_key == "patient_id"
