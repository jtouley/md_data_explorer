# SPDX-License-Identifier: MIT
"""Questions through the API routes hold a lease on their layer, and the bounded layer cache
never loses a dataset's query state.

The scenarios of the PR #97 review: eviction while a question is in flight, a re-upload during
a question, a poll (and a follow-up) after the dataset's layer was evicted, and a cold load
beside a dataset whose question is busy in DuckDB.
"""

from __future__ import annotations

import asyncio
import gc
import threading
import time
import weakref
from collections.abc import Callable, Iterator
from typing import Any
from unittest.mock import patch

import pytest

from clinical_analytics.api import dependencies
from clinical_analytics.api.layer_cache import LayerCache
from clinical_analytics.api.models.schemas import QueryRequest
from clinical_analytics.api.routes import queries
from clinical_analytics.api.services.query_service import AsyncQueryService
from clinical_analytics.core.query_plan import QueryPlan
from clinical_analytics.core.query_service import QueryResult as CoreQueryResult
from tests.api.conftest import Builds, FakeLayer

pytestmark = pytest.mark.usefixtures("isolated_data_home")

_FAST_S = 1.0
_BLOCKED_S = 10.0


class FakeCore:
    """Stands in for core QueryService: records each question and the layer it ran on.

    A question first "parses" without the layer's lock (``parse_gates``, ``parsing``), then
    runs a DuckDB step under it (``gates``, ``in_duckdb``), as the real ask does.
    """

    gates: dict[str, threading.Event] = {}  # question text -> released when set
    in_duckdb: dict[str, threading.Event] = {}  # question text -> set once inside the lock
    parse_gates: dict[str, threading.Event] = {}  # question text -> parse released when set
    parsing: dict[str, threading.Event] = {}  # question text -> set once its parse started
    asked: list[dict[str, Any]] = []
    built: list[FakeLayer] = []

    def __init__(self, layer: FakeLayer) -> None:
        self.semantic_layer = layer
        FakeCore.built.append(layer)

    def ask(self, question: str, **kwargs: Any) -> CoreQueryResult:
        layer = self.semantic_layer
        if question in FakeCore.parsing:
            FakeCore.parsing[question].set()
        parse_gate = FakeCore.parse_gates.get(question)
        if parse_gate is not None:
            assert parse_gate.wait(_BLOCKED_S)
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
                    "dataset_version": kwargs.get("dataset_version"),
                }
            )
        plan = QueryPlan(intent="COUNT", entity_key="patient_id", confidence=0.95)
        return CoreQueryResult(plan=plan, issues=[], result=None, confidence=0.95, run_key="run", context=None)


def _cache_with(capacity: int, builds: Builds) -> Iterator[LayerCache[Any]]:
    """The API's layer cache with room for ``capacity`` datasets, building FakeLayers, and FakeCore."""
    cache: LayerCache[Any] = LayerCache(capacity, builds)
    FakeCore.gates, FakeCore.in_duckdb, FakeCore.asked = {}, {}, []
    FakeCore.parse_gates, FakeCore.parsing, FakeCore.built = {}, {}, []
    queries._query_services.clear()
    with (
        patch.object(dependencies, "_layer_cache", cache),
        patch.object(AsyncQueryService, "_make_core", lambda _self, layer: FakeCore(layer)),
    ):
        yield cache
    queries._query_services.clear()
    cache.shutdown()


@pytest.fixture
def one_layer_cache(builds: Builds) -> Iterator[LayerCache[Any]]:
    """The API's layer cache with room for one dataset (see _cache_with)."""
    yield from _cache_with(1, builds)


@pytest.fixture
def two_layer_cache(builds: Builds) -> Iterator[LayerCache[Any]]:
    """The API's layer cache with room for two datasets (see _cache_with)."""
    yield from _cache_with(2, builds)


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
    layer_a = builds.by_dataset["A"][0]
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
    old_layer, new_layer = builds.by_dataset["A"]
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
    evicted = builds.by_dataset["A"][0].closed

    # Act
    polled = await queries.get_query_result(first)
    follow_up = await _ask("A", "now by sex", session_id="doctor")
    await _completed(follow_up)

    # Assert: same service and state; its core now runs on a rebuilt layer for A.
    assert evicted is True
    assert polled.query_id == first
    assert polled.status == "completed"
    assert queries._query_services["A"] is service_a
    assert FakeCore.asked[-1]["layer"] is builds.by_dataset["A"][1]
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
    assert builds.by_dataset["A"][0].closed is True


# ---------------------------------------------------------------------------
# PR #99 review: the lease lasts exactly as long as the worker thread uses the layer
# ---------------------------------------------------------------------------


def _cancel_tasks(service: AsyncQueryService) -> None:
    for task in list(service._tasks):
        task.cancel()


async def test_question_cancelledDuringParseThenInvalidated_layerClosedOnlyAfterItsThread(
    one_layer_cache: LayerCache[Any], builds: Builds
) -> None:
    # Arrange: the question's thread is in its unlocked parse when its task is cancelled.
    FakeCore.parse_gates["q"] = threading.Event()
    FakeCore.parsing["q"] = threading.Event()
    await _ask("A", "q")
    await asyncio.to_thread(FakeCore.parsing["q"].wait, _BLOCKED_S)
    _cancel_tasks(queries._query_services["A"])
    await asyncio.sleep(0.05)

    # Act: the dataset is uploaded again while the cancelled question's thread still runs.
    dependencies.invalidate_semantic_layer_cache("A")
    await asyncio.to_thread(one_layer_cache.wait_for_closes, 0.2)
    closed_mid_question = builds.by_dataset["A"][0].closed
    FakeCore.parse_gates["q"].set()
    await _until(lambda: len(FakeCore.asked) == 1)
    await _until(lambda: builds.by_dataset["A"][0].closed)

    # Assert
    assert closed_mid_question is False
    assert FakeCore.asked[0]["closed_during_ask"] is False


async def test_executeQuery_cancelledDuringParseThenInvalidated_layerClosedOnlyAfterItsThread(
    one_layer_cache: LayerCache[Any], builds: Builds
) -> None:
    # Arrange
    FakeCore.parse_gates["q"] = threading.Event()
    FakeCore.parsing["q"] = threading.Event()
    service, lease = await asyncio.to_thread(queries.lease_query_service, "A")
    running = asyncio.create_task(service.execute_query(query="q", dataset_id="A", lease=lease))
    await asyncio.to_thread(FakeCore.parsing["q"].wait, _BLOCKED_S)
    running.cancel()
    with pytest.raises(asyncio.CancelledError):
        await running

    # Act
    dependencies.invalidate_semantic_layer_cache("A")
    await asyncio.to_thread(one_layer_cache.wait_for_closes, 0.2)
    closed_mid_question = builds.by_dataset["A"][0].closed
    FakeCore.parse_gates["q"].set()
    await _until(lambda: len(FakeCore.asked) == 1)
    await _until(lambda: builds.by_dataset["A"][0].closed)

    # Assert
    assert closed_mid_question is False
    assert FakeCore.asked[0]["closed_during_ask"] is False


async def test_submit_cancelledWhileLeasing_leaseReleasedFromTheThreadsResult(
    one_layer_cache: LayerCache[Any], builds: Builds
) -> None:
    # Arrange: the request is cancelled while its worker thread is still building the layer.
    building, finish = threading.Event(), threading.Event()

    def stall(_dataset_id: str) -> None:
        building.set()
        assert finish.wait(_BLOCKED_S)

    builds.before_return = stall
    request = asyncio.create_task(_ask("A", "q"))
    await asyncio.to_thread(building.wait, _BLOCKED_S)
    request.cancel()
    with pytest.raises(asyncio.CancelledError):
        await request

    # Act: the thread finishes and takes the lease for a request that is gone.
    builds.before_return = None
    finish.set()
    await _until(lambda: "A" in builds.by_dataset)
    dependencies.invalidate_semantic_layer_cache("A")

    # Assert: no lease was leaked, so the dropped layer closes.
    await _until(lambda: builds.by_dataset["A"][0].closed)
    assert FakeCore.asked == []


async def test_submit_taskCancelledBeforeItsFirstStep_leaseReleased(one_layer_cache: LayerCache[Any]) -> None:
    # Arrange
    service, lease = await asyncio.to_thread(queries.lease_query_service, "A")
    await service.submit_query(query="q", dataset_id="A", lease=lease)

    # Act: cancelled before the loop ran the task at all (shutdown, app teardown).
    _cancel_tasks(service)
    await _until(lambda: not service._tasks)

    # Assert
    assert lease.released is True
    assert FakeCore.asked == []


@pytest.mark.parametrize("method", ["submit_query", "execute_query"])
async def test_question_withoutLease_rejected(one_layer_cache: LayerCache[Any], method: str) -> None:
    # Arrange: no unleased path, so no question can run on a layer that was closed.
    service = AsyncQueryService()

    # Act / Assert
    with pytest.raises(TypeError, match="lease"):
        await getattr(service, method)(query="q", dataset_id="A")


# ---------------------------------------------------------------------------
# PR #99 review: no blocking lock on the event loop; closed layers are not kept reachable
# ---------------------------------------------------------------------------


async def test_submit_whileANewLayersCoreBuilds_eventLoopNotBlocked(
    one_layer_cache: LayerCache[Any], builds: Builds
) -> None:
    # Arrange: B holds a lease on A's first layer; A is uploaded again and the next question's
    # core build for the new layer is slow (golden examples, embeddings).
    await _completed(await _ask("A", "first"))
    service, b_lease = await asyncio.to_thread(queries.lease_query_service, "A")
    dependencies.invalidate_semantic_layer_cache("A")
    building, finish = threading.Event(), threading.Event()

    def slow_core(_self: AsyncQueryService, layer: FakeLayer) -> FakeCore:
        if layer.number == 2:
            building.set()
            assert finish.wait(_BLOCKED_S)
        return FakeCore(layer)

    with patch.object(AsyncQueryService, "_make_core", slow_core):
        rebuilt = asyncio.create_task(_ask("A", "on new data"))
        await asyncio.to_thread(building.wait, _BLOCKED_S)

        # Act: B's question is submitted on the event loop meanwhile.
        started = time.monotonic()
        b_query = await service.submit_query(query="on old data", dataset_id="A", lease=b_lease)
        submit_s = time.monotonic() - started
        b_status = await _completed(b_query)
        finish.set()
        rebuilt_status = await _completed(await rebuilt)

    # Assert
    ran_on = {item["question"]: item["layer"] for item in FakeCore.asked}
    assert submit_s < _FAST_S
    assert (b_status, rebuilt_status) == ("completed", "completed")
    assert ran_on["on old data"] is builds.by_dataset["A"][0]
    assert ran_on["on new data"] is builds.by_dataset["A"][1]


async def test_queries_manyDatasetsAskedOnce_atMostCacheSizeLayersReachable(
    two_layer_cache: LayerCache[Any], builds: Builds
) -> None:
    # Arrange
    datasets = [f"D{index}" for index in range(6)]

    # Act
    for dataset in datasets:
        await _completed(await _ask(dataset, "count patients"))
    await asyncio.to_thread(two_layer_cache.wait_for_closes)
    alive = [weakref.ref(layer) for layer in builds.layers]
    builds.layers.clear()
    builds.by_dataset.clear()
    FakeCore.asked.clear()
    FakeCore.built.clear()
    gc.collect()

    # Assert: results stay; only the layers the cache keeps (and their cores) are reachable.
    reachable = [ref for ref in alive if ref() is not None]
    assert set(queries._query_services) == set(datasets)
    assert len(reachable) <= two_layer_cache.capacity


async def test_question_reuploadBetweenLeaseAndFinish_labelledWithItsLayersVersion(
    one_layer_cache: LayerCache[Any], builds: Builds
) -> None:
    # Arrange: the question's lease is on v1's layer; then the dataset is uploaded again (v2).
    builds.versions["A"] = "v1"
    service, lease = await asyncio.to_thread(queries.lease_query_service, "A")
    builds.versions["A"] = "v2"
    dependencies.invalidate_semantic_layer_cache("A")

    # Act
    await _completed(await service.submit_query(query="on v1", dataset_id="A", lease=lease))
    await _completed(await _ask("A", "on v2"))

    # Assert: each answer is labelled, and matched to declarations, with the rows it read.
    versions = {item["question"]: item["dataset_version"] for item in FakeCore.asked}
    assert versions == {"on v1": "v1", "on v2": "v2"}
