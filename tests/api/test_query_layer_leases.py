# SPDX-License-Identifier: MIT
"""Questions through the API routes hold a lease on their layer, and the bounded layer cache
never loses a dataset's query state.

The scenarios of the PR #97 review: eviction while a question is in flight, a re-upload during
a question, a poll (and a follow-up) after the dataset's layer was evicted, and a cold load
beside a dataset whose question is busy in DuckDB.
"""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import Callable, Iterator
from typing import Any
from unittest.mock import patch

import pytest

from clinical_analytics.api import dependencies
from clinical_analytics.api.layer_cache import LayerCache
from clinical_analytics.api.models.schemas import QueryRequest
from clinical_analytics.api.routes import queries
from clinical_analytics.api.services.query_service import AsyncQueryService, StaleLayerLeaseError
from clinical_analytics.core.connection_lock import ConnectionLock
from clinical_analytics.core.query_plan import QueryPlan
from clinical_analytics.core.query_service import QueryResult as CoreQueryResult

pytestmark = pytest.mark.usefixtures("isolated_data_home")

_FAST_S = 1.0
_BLOCKED_S = 10.0


class FakeLayer:
    def __init__(self, dataset_id: str, number: int) -> None:
        self.dataset_id = dataset_id
        self.number = number
        self._con_lock = ConnectionLock()
        self.closed = False

    def close(self) -> None:
        with self._con_lock:
            self.closed = True


class FakeCore:
    """Stands in for core QueryService: records each question and the layer it ran on."""

    gates: dict[str, threading.Event] = {}  # question text -> released when set
    in_duckdb: dict[str, threading.Event] = {}  # question text -> set once inside the lock
    asked: list[dict[str, Any]] = []

    def __init__(self, layer: FakeLayer) -> None:
        self.semantic_layer = layer

    def ask(self, question: str, **kwargs: Any) -> CoreQueryResult:
        layer = self.semantic_layer
        with layer._con_lock:  # a DuckDB step, as the real execution holds it
            if question in FakeCore.in_duckdb:
                FakeCore.in_duckdb[question].set()
            gate = FakeCore.gates.get(question)
            if gate is not None:
                assert gate.wait(_BLOCKED_S)
            FakeCore.asked.append(
                {
                    "question": question,
                    "layer": layer,
                    "closed_during_ask": layer.closed,
                    "history": kwargs.get("conversation_history"),
                }
            )
        plan = QueryPlan(intent="COUNT", entity_key="patient_id", confidence=0.95)
        return CoreQueryResult(plan=plan, issues=[], result=None, confidence=0.95, run_key="run", context=None)


class Builds:
    def __init__(self) -> None:
        self.count: dict[str, int] = {}
        self.layers: dict[str, list[FakeLayer]] = {}

    def __call__(self, dataset_id: str) -> FakeLayer:
        self.count[dataset_id] = self.count.get(dataset_id, 0) + 1
        layer = FakeLayer(dataset_id, self.count[dataset_id])
        self.layers.setdefault(dataset_id, []).append(layer)
        return layer


@pytest.fixture
def builds() -> Builds:
    return Builds()


@pytest.fixture
def one_layer_cache(builds: Builds) -> Iterator[LayerCache[Any]]:
    """The API's layer cache with room for one dataset, building FakeLayers, and FakeCore."""
    cache: LayerCache[Any] = LayerCache(1, builds)
    FakeCore.gates, FakeCore.in_duckdb, FakeCore.asked = {}, {}, []
    queries._query_services.clear()
    with (
        patch.object(dependencies, "_layer_cache", cache),
        patch.object(AsyncQueryService, "_make_core", lambda _self, layer: FakeCore(layer)),
        patch("clinical_analytics.api.services.query_service.active_dataset_version", return_value=None),
    ):
        yield cache
    queries._query_services.clear()
    cache.shutdown()


async def _ask(dataset_id: str, text: str, session_id: str = "sess") -> str:
    response = await queries.submit_query(QueryRequest(session_id=session_id, dataset_id=dataset_id, query_text=text))
    return response.query_id


async def _until(condition: Callable[[], bool], timeout: float = _BLOCKED_S) -> None:
    deadline = time.monotonic() + timeout
    while not condition():
        assert time.monotonic() < deadline, "condition not met in time"
        await asyncio.sleep(0.01)


async def _completed(query_id: str) -> str:
    async def status() -> str:
        return (await queries.get_query_result(query_id)).status

    deadline = time.monotonic() + _BLOCKED_S
    while (current := await status()) in ("pending", "processing"):
        assert time.monotonic() < deadline
        await asyncio.sleep(0.01)
    return current


async def test_question_layerEvictedWhileInFlight_answeredOnOpenLayerThenClosed(
    one_layer_cache: LayerCache[Any], builds: Builds
) -> None:
    # Arrange: a question on A is running when B is asked about (B pushes A out of the cache).
    FakeCore.gates["slow on A"] = threading.Event()
    FakeCore.in_duckdb["slow on A"] = threading.Event()
    slow = await _ask("A", "slow on A")
    await asyncio.to_thread(FakeCore.in_duckdb["slow on A"].wait, _BLOCKED_S)

    # Act
    other = await _ask("B", "quick on B")
    other_status = await _completed(other)
    FakeCore.gates["slow on A"].set()
    slow_status = await _completed(slow)
    third = await _ask("C", "quick on C")
    await _completed(third)
    one_layer_cache.wait_for_closes()

    # Assert
    layer_a = builds.layers["A"][0]
    assert (other_status, slow_status) == ("completed", "completed")
    assert [item["closed_during_ask"] for item in FakeCore.asked] == [False, False, False]
    assert layer_a.closed is True


async def test_question_reuploadDuringQuestion_neitherBlocksNorClosesItsLayer(
    one_layer_cache: LayerCache[Any], builds: Builds
) -> None:
    # Arrange: a question on A is inside a DuckDB step when A is uploaded again.
    FakeCore.gates["before"] = threading.Event()
    FakeCore.in_duckdb["before"] = threading.Event()
    before = await _ask("A", "before")
    await asyncio.to_thread(FakeCore.in_duckdb["before"].wait, _BLOCKED_S)

    # Act: the upload route calls this on the event loop.
    started = time.monotonic()
    dependencies.invalidate_semantic_layer_cache("A")
    invalidate_s = time.monotonic() - started
    after = await _ask("A", "after")
    after_status = await _completed(after)
    FakeCore.gates["before"].set()
    before_status = await _completed(before)
    one_layer_cache.wait_for_closes()

    # Assert: the old question ran on the old layer, closed only after it; the new one on new data.
    old_layer, new_layer = builds.layers["A"]
    ran_on = {item["question"]: item for item in FakeCore.asked}
    assert invalidate_s < _FAST_S
    assert (before_status, after_status) == ("completed", "completed")
    assert ran_on["before"]["layer"] is old_layer
    assert ran_on["after"]["layer"] is new_layer
    assert ran_on["before"]["closed_during_ask"] is False
    assert old_layer.closed is True
    assert new_layer.closed is False


async def test_poll_afterLayerEvicted_queryFoundAndFollowUpKeepsHistory(
    one_layer_cache: LayerCache[Any], builds: Builds
) -> None:
    # Arrange: a finished question on A, then A's layer is evicted by a question on B.
    first = await _ask("A", "count patients", session_id="doctor")
    await _completed(first)
    service_a = queries._query_services["A"]
    await _completed(await _ask("B", "count patients on B", session_id="other"))
    one_layer_cache.wait_for_closes()
    evicted = builds.layers["A"][0].closed

    # Act
    polled = await queries.get_query_result(first)
    follow_up = await _ask("A", "now by sex", session_id="doctor")
    await _completed(follow_up)

    # Assert: same service and state; its core now runs on a rebuilt layer for A.
    assert evicted is True
    assert polled.query_id == first
    assert polled.status == "completed"
    assert queries._query_services["A"] is service_a
    assert FakeCore.asked[-1]["layer"] is builds.layers["A"][1]
    assert [turn["query"] for turn in FakeCore.asked[-1]["history"]] == ["count patients"]


async def test_stream_afterLayerEvicted_eventsStillServed(one_layer_cache: LayerCache[Any]) -> None:
    # Arrange
    first = await _ask("A", "count patients")
    await _completed(first)
    await _completed(await _ask("B", "count on B"))
    one_layer_cache.wait_for_closes()

    # Act
    found = await queries.find_query_result(first)
    assert found is not None
    events = [event.event async for event in found[0].stream_events(first)]

    # Assert
    assert events[0] == "query_started"
    assert events[-1] == "stream_end"


async def test_question_coldLoadBesideBusyLayer_answeredWithoutWaiting(
    one_layer_cache: LayerCache[Any], builds: Builds
) -> None:
    # Arrange: A's question is stuck inside a DuckDB step; B has never been loaded.
    FakeCore.gates["busy"] = threading.Event()
    FakeCore.in_duckdb["busy"] = threading.Event()
    busy = await _ask("A", "busy")
    await asyncio.to_thread(FakeCore.in_duckdb["busy"].wait, _BLOCKED_S)

    # Act
    started = time.monotonic()
    cold = await _ask("B", "cold")
    cold_status = await _completed(cold)
    cold_s = time.monotonic() - started
    FakeCore.gates["busy"].set()
    busy_status = await _completed(busy)

    # Assert
    assert cold_s < _FAST_S
    assert (cold_status, busy_status) == ("completed", "completed")
    assert builds.count == {"A": 1, "B": 1}


async def test_submit_emptyQuestion_leaseReleased(one_layer_cache: LayerCache[Any], builds: Builds) -> None:
    # Arrange
    with patch.object(AsyncQueryService, "submit_query", side_effect=ValueError("Query cannot be empty")):
        # Act
        with pytest.raises(queries.HTTPException):
            await _ask("A", "x")
    dependencies.invalidate_semantic_layer_cache("A")
    one_layer_cache.wait_for_closes()

    # Assert: no lease was left behind, so the dropped layer closed at once.
    assert builds.layers["A"][0].closed is True


async def test_submit_leaseOlderThanBoundLayer_rejectedAndReleased(
    one_layer_cache: LayerCache[Any], builds: Builds
) -> None:
    # Arrange: a lease taken before a re-upload, submitted after the service moved to new data.
    service, old_lease = await asyncio.to_thread(queries.lease_query_service, "A")
    dependencies.invalidate_semantic_layer_cache("A")
    _service, new_lease = await asyncio.to_thread(queries.lease_query_service, "A")
    new_lease.release()

    # Act / Assert: never run on the old layer behind the service's back.
    with pytest.raises(StaleLayerLeaseError):
        await service.submit_query(query="count", dataset_id="A", lease=old_lease)
    one_layer_cache.wait_for_closes()
    assert old_lease.released is True
    assert builds.layers["A"][0].closed is True
    assert FakeCore.asked == []
