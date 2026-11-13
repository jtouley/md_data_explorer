# SPDX-License-Identifier: MIT
"""The per-layer DuckDB lock covers DuckDB work only: parsing (LLM calls included) overlaps
across questions on one layer, while every DuckDB call still runs with the lock held."""

from __future__ import annotations

import threading
from typing import Any
from unittest.mock import MagicMock

from clinical_analytics.core.connection_lock import ConnectionLock
from clinical_analytics.core.query_service import QueryService

_WAIT_S = 5.0


def _mock_layer_service() -> tuple[MagicMock, QueryService]:
    layer = MagicMock()
    layer._con_lock = ConnectionLock()
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


def test_ask_levelsMergedWhileParsing_parsedAgainUnderTheLock() -> None:
    # Arrange: while this question parses unlocked, another question merges levels (the layer's
    # state_generation moves). Its first parse read the old levels, so it must not be executed.
    layer, service = _mock_layer_service()
    parses_locked: list[bool] = []

    def parse_query(*_args: Any, **_kwargs: Any) -> None:
        parses_locked.append(layer._con_lock.held_by_current_thread())
        if len(parses_locked) == 1:
            layer.state_generation += 1
        return None

    service.nl_engine.parse_query = parse_query  # type: ignore[method-assign]

    # Act
    result = service.ask("how many patients", dataset_id="ds")

    # Assert
    assert parses_locked == [False, True]
    assert result.plan is None


class _LockCheckedConnection:
    """Stands in for a layer's raw DuckDB connection and records every use made without the
    layer's lock held by the calling thread."""

    def __init__(self, raw: Any, lock: Any) -> None:
        object.__setattr__(self, "_raw", raw)
        object.__setattr__(self, "_lock", lock)
        object.__setattr__(self, "unlocked_uses", [])

    def __getattr__(self, name: str) -> Any:
        if not self._lock.held_by_current_thread():
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


def test_connectionLock_heldByCurrentThread_onlyForTheHolder() -> None:
    # Arrange
    lock = ConnectionLock()
    seen_elsewhere: list[bool] = []

    # Act
    with lock, lock:
        held_nested = lock.held_by_current_thread()
        other = threading.Thread(target=lambda: seen_elsewhere.append(lock.held_by_current_thread()))
        other.start()
        other.join()
    held_after = lock.held_by_current_thread()

    # Assert
    assert held_nested is True
    assert seen_elsewhere == [False]
    assert held_after is False


def test_connectionLock_releasedByAnotherThread_raises() -> None:
    # Arrange
    lock = ConnectionLock()
    errors: list[type[BaseException]] = []

    def release_elsewhere() -> None:
        try:
            lock.release()
        except RuntimeError as e:
            errors.append(type(e))

    # Act
    with lock:
        other = threading.Thread(target=release_elsewhere)
        other.start()
        other.join()

    # Assert
    assert errors == [RuntimeError]
