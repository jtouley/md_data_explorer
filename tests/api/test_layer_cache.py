# SPDX-License-Identifier: MIT
"""The bounded semantic-layer cache: leases, eviction, invalidation and background close.

Each test is one failure the first bounded cache (PR #97) had: a layer closed while a request
still used it, a re-upload or a cold load waiting on another question's DuckDB lock, and a
layer closed on the request path.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterator
from typing import Any
from unittest.mock import patch

import pytest

from clinical_analytics.api import layer_cache
from clinical_analytics.api.layer_cache import LayerCache, LayerCacheError
from tests.api.conftest import Builds, FakeLayer

_FAST_S = 1.0  # what "returns at once" allows; the blocked work below waits far longer
_BLOCKED_S = 10.0


@pytest.fixture
def make_cache(builds: Builds) -> Iterator[Callable[[int], LayerCache[FakeLayer]]]:
    caches: list[LayerCache[FakeLayer]] = []

    def _make(capacity: int) -> LayerCache[FakeLayer]:
        cache: LayerCache[FakeLayer] = LayerCache(capacity, builds)
        caches.append(cache)
        return cache

    yield _make
    for cache in caches:
        cache.shutdown()


def _hold_lock_in_thread(lock: Any) -> tuple[threading.Event, threading.Thread]:
    """Hold ``lock`` on another thread (a question's DuckDB step) until the event is set."""
    held, release = threading.Event(), threading.Event()

    def hold() -> None:
        with lock:
            held.set()
            release.wait(_BLOCKED_S)

    thread = threading.Thread(target=hold, daemon=True)
    thread.start()
    assert held.wait(_FAST_S)
    return release, thread


# ---------------------------------------------------------------------------
# Eviction while a query is in flight
# ---------------------------------------------------------------------------


def test_lease_layerEvictedWhileLeased_closedOnlyAfterRelease(
    make_cache: Callable[[int], LayerCache[FakeLayer]],
) -> None:
    # Arrange: a question on A is in flight (holds a lease) while other datasets are asked about
    # in a 1-entry cache; each of them would push A out.
    cache = make_cache(1)
    lease_a = cache.lease("A")
    layer_a = lease_a.layer

    # Act
    for other in ("B", "C"):
        with cache.lease(other):
            pass
    cache.wait_for_closes()
    closed_while_leased = layer_a.closed
    lease_a.release()
    with cache.lease("D"):
        pass
    cache.wait_for_closes()

    # Assert
    assert closed_while_leased is False
    assert layer_a.closed is True
    assert cache.cached_ids() == ["D"]


def test_lease_overCapacity_leastRecentlyUsedUnleasedLayerClosed(
    make_cache: Callable[[int], LayerCache[FakeLayer]],
) -> None:
    # Arrange
    cache = make_cache(2)
    with cache.lease("A") as lease_a:
        layer_a = lease_a.layer
    with cache.lease("B") as lease_b:
        layer_b = lease_b.layer
    with cache.lease("A"):  # A is now the most recently used
        pass

    # Act
    with cache.lease("C"):
        pass
    cache.wait_for_closes()

    # Assert
    assert layer_b.closed is True
    assert layer_a.closed is False
    assert cache.cached_ids() == ["A", "C"]


def test_lease_allLayersLeased_cacheGrowsThenTrimsOnRelease(
    make_cache: Callable[[int], LayerCache[FakeLayer]], builds: Builds
) -> None:
    # Arrange: every cached layer is in use, so none can be closed yet.
    cache = make_cache(1)
    lease_a = cache.lease("A")
    lease_b = cache.lease("B")

    # Act
    leased_ids = cache.cached_ids()
    lease_a.release()
    cache.wait_for_closes()

    # Assert: no second copy of A was built while it was over capacity, and A closed once free.
    assert leased_ids == ["A", "B"]
    assert cache.cached_ids() == ["B"]
    assert builds.layers[0].closed is True
    assert builds.count == {"A": 1, "B": 1}
    lease_b.release()


def test_lease_sameDatasetTwice_oneBuildTwoLeases(
    make_cache: Callable[[int], LayerCache[FakeLayer]], builds: Builds
) -> None:
    # Arrange
    cache = make_cache(2)

    # Act
    with cache.lease("A") as first, cache.lease("A") as second:
        same = first.layer is second.layer

    # Assert
    assert same is True
    assert builds.count == {"A": 1}


def test_release_calledTwice_countedOnce(make_cache: Callable[[int], LayerCache[FakeLayer]]) -> None:
    # Arrange: two leases on A, one released twice. A double count would close A under the other.
    cache = make_cache(1)
    first = cache.lease("A")
    second = cache.lease("A")
    layer_a = first.layer

    # Act
    first.release()
    first.release()
    cache.invalidate("A")
    cache.wait_for_closes()
    closed_while_second_held = layer_a.closed
    second.release()
    cache.wait_for_closes()

    # Assert
    assert closed_while_second_held is False
    assert layer_a.closed is True


def test_lease_layerReadAfterRelease_raises(make_cache: Callable[[int], LayerCache[FakeLayer]]) -> None:
    # Arrange
    cache = make_cache(1)
    lease = cache.lease("A")
    lease.release()

    # Act / Assert: a released lease no longer keeps the layer open, so it must not hand it out.
    with pytest.raises(LayerCacheError, match="released"):
        _ = lease.layer


# ---------------------------------------------------------------------------
# Re-upload (invalidation) during a question
# ---------------------------------------------------------------------------


def test_invalidate_layerBusyInQuestion_returnsAtOnceAndClosesAfterRelease(
    make_cache: Callable[[int], LayerCache[FakeLayer]],
) -> None:
    # Arrange: a question on A holds a lease and is inside a DuckDB step (holds A's lock).
    cache = make_cache(2)
    lease_a = cache.lease("A")
    old_layer = lease_a.layer
    release_lock, holder = _hold_lock_in_thread(old_layer._con_lock)

    # Act: the upload route drops A. It is called on the event loop, so it must not wait.
    started = time.monotonic()
    cache.invalidate("A")
    invalidate_s = time.monotonic() - started
    with cache.lease("A") as new_lease:
        new_layer = new_lease.layer
    release_lock.set()
    holder.join(_FAST_S)
    cache.wait_for_closes()
    closed_while_leased = old_layer.closed
    lease_a.release()
    cache.wait_for_closes()

    # Assert
    assert invalidate_s < _FAST_S
    assert new_layer is not old_layer
    assert closed_while_leased is False
    assert old_layer.closed is True
    assert new_layer.closed is False


def test_lease_invalidatedWhileBuilding_neverReturnsTheStaleLayer(
    make_cache: Callable[[int], LayerCache[FakeLayer]], builds: Builds
) -> None:
    # Arrange: the dataset is re-uploaded while its first build is still reading the old data.
    cache = make_cache(2)

    def reupload_during_first_build(dataset_id: str) -> None:
        if builds.count[dataset_id] == 1:
            cache.invalidate(dataset_id)

    builds.before_return = reupload_during_first_build

    # Act
    with cache.lease("A") as lease:
        returned = lease.layer
    cache.wait_for_closes()

    # Assert
    assert returned.number == 2
    assert builds.layers[0].closed is True


def test_lease_invalidatedOnEveryBuild_raisesInsteadOfLooping(
    make_cache: Callable[[int], LayerCache[FakeLayer]], builds: Builds
) -> None:
    # Arrange
    cache = make_cache(2)
    builds.before_return = cache.invalidate

    # Act / Assert
    with pytest.raises(LayerCacheError, match="changed while"):
        cache.lease("A")
    cache.wait_for_closes()
    assert all(layer.closed for layer in builds.layers)


def test_lease_buildFails_nothingCachedAndNextLeaseRebuilds(
    make_cache: Callable[[int], LayerCache[FakeLayer]], builds: Builds
) -> None:
    # Arrange
    cache = make_cache(2)

    def fail_first(dataset_id: str) -> None:
        if builds.count[dataset_id] == 1:
            raise FileNotFoundError(dataset_id)

    builds.before_return = fail_first

    # Act
    with pytest.raises(FileNotFoundError):
        cache.lease("A")
    with cache.lease("A") as lease:
        number = lease.layer.number

    # Assert
    assert number == 2
    assert cache.cached_ids() == ["A"]


# ---------------------------------------------------------------------------
# Cold load beside a busy layer
# ---------------------------------------------------------------------------


def test_lease_coldLoadBesideBusyLayer_doesNotWaitForIt(
    make_cache: Callable[[int], LayerCache[FakeLayer]],
) -> None:
    # Arrange: A is in a long DuckDB step; B is not built yet and the cache holds one layer.
    cache = make_cache(1)
    lease_a = cache.lease("A")
    release_lock, holder = _hold_lock_in_thread(lease_a.layer._con_lock)

    # Act
    started = time.monotonic()
    with cache.lease("B") as lease_b:
        cold_s = time.monotonic() - started
        layer_b = lease_b.layer
    cache.wait_for_closes()
    lease_a_closed_while_leased = lease_a.layer.closed
    release_lock.set()
    holder.join(_FAST_S)
    lease_a.release()
    cache.wait_for_closes()

    # Assert: B was served at once, and A, still leased, was left open for its question.
    assert cold_s < _FAST_S
    assert layer_b.number == 1
    assert lease_a_closed_while_leased is False


def test_lease_coldLoadBesideAnotherCold_buildsInParallel(
    make_cache: Callable[[int], LayerCache[FakeLayer]], builds: Builds
) -> None:
    # Arrange: each build waits until the other dataset's build has started.
    cache = make_cache(2)
    both_building = threading.Barrier(2, timeout=_FAST_S * 5)
    builds.before_return = lambda _dataset_id: both_building.wait()
    results: list[str] = []

    def lease(dataset_id: str) -> None:
        try:
            with cache.lease(dataset_id):
                results.append(dataset_id)
        except threading.BrokenBarrierError:
            results.append("waited")

    threads = [threading.Thread(target=lease, args=(dataset_id,)) for dataset_id in ("A", "B")]

    # Act
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(_BLOCKED_S)

    # Assert
    assert sorted(results) == ["A", "B"]


# ---------------------------------------------------------------------------
# Close runs on a background thread
# ---------------------------------------------------------------------------


def test_eviction_closesOnBackgroundThread_neverOnTheCaller(
    make_cache: Callable[[int], LayerCache[FakeLayer]],
) -> None:
    # Arrange
    cache = make_cache(1)
    with cache.lease("A") as lease_a:
        layer_a = lease_a.layer

    # Act
    with cache.lease("B"):
        pass
    cache.wait_for_closes()

    # Assert
    assert layer_a.closed is True
    assert layer_a.closed_on is not None
    assert layer_a.closed_on != threading.current_thread().name
    assert layer_a.closed_on.startswith("semantic-layer-close")


def test_eviction_closeFails_loggedAndCacheKeepsWorking(make_cache: Callable[[int], LayerCache[FakeLayer]]) -> None:
    # Arrange
    cache = make_cache(1)
    with cache.lease("A") as lease_a:
        broken = lease_a.layer

    def fail_close() -> None:
        raise RuntimeError("disk gone")

    broken.close = fail_close  # type: ignore[method-assign]

    # Act
    with patch.object(layer_cache, "logger") as log:
        with cache.lease("B"):
            pass
        cache.wait_for_closes()
    with cache.lease("A") as again:
        rebuilt = again.layer

    # Assert
    assert [call.args[0] for call in log.error.call_args_list] == ["semantic_layer_close_failed"]
    assert rebuilt is not broken


@pytest.mark.parametrize("capacity", [0, -1])
def test_layerCache_capacityBelowOne_raises(capacity: int, builds: Builds) -> None:
    # Arrange / Act / Assert
    with pytest.raises(ValueError, match="capacity"):
        LayerCache(capacity, builds)
