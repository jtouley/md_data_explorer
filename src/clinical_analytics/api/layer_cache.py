# SPDX-License-Identifier: MIT
"""A bounded cache of semantic layers (one in-memory DuckDB per dataset), safe to evict.

Rules, each one a failure of the first bounded cache (PR #97):

- A request uses a layer only through a lease, and releases it in ``finally`` (``with``). A layer
  leaves the cache when it is evicted or invalidated, but it is closed only once no lease on it
  is left, so nothing ever runs on a closed connection.
- Closing happens on a background thread. ``lease``, ``release`` and ``invalidate`` never wait
  for a layer's DuckDB lock, so neither a re-upload (called on the event loop) nor a cold load
  of another dataset waits for a question that is running.
- Only the dataset being built waits for its build; other datasets build in parallel.
- Eviction takes the least recently used layer that no request holds. When every layer is held
  the cache runs over capacity for a while rather than build a second copy of a dataset, and
  trims back as leases are released.
"""

from __future__ import annotations

import itertools
import threading
from collections import OrderedDict
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import wait as wait_futures
from dataclasses import dataclass
from types import TracebackType
from typing import Generic, Protocol, TypeVar

import duckdb
import structlog

from clinical_analytics.core.log_redaction import error_fields

logger = structlog.get_logger()

# Builds of one dataset that a re-upload may interrupt before a lease gives up and says so.
MAX_BUILD_ATTEMPTS = 3
CLOSE_THREAD_PREFIX = "semantic-layer-close"


class LayerCacheError(RuntimeError):
    """A lease used after release, or a dataset that kept changing while it was being loaded."""


class Closable(Protocol):
    def close(self) -> None: ...


LayerT = TypeVar("LayerT", bound=Closable)


@dataclass(eq=False)
class _Entry(Generic[LayerT]):
    dataset_id: str
    layer: LayerT
    generation: int
    leases: int = 0
    retired: bool = False  # evicted or invalidated: closed once leases reaches 0
    close_scheduled: bool = False


class LayerLease(Generic[LayerT]):
    """One request's hold on a layer. Release it exactly once, in ``finally`` (or use ``with``)."""

    def __init__(self, cache: LayerCache[LayerT], entry: _Entry[LayerT]) -> None:
        self._cache = cache
        self._entry = entry
        self._released = False
        self._release_lock = threading.Lock()

    @property
    def dataset_id(self) -> str:
        return self._entry.dataset_id

    @property
    def generation(self) -> int:
        """Increases with every layer the cache builds: a higher one is newer data or a rebuild."""
        return self._entry.generation

    @property
    def layer(self) -> LayerT:
        if self._released:
            raise LayerCacheError(f"lease on '{self.dataset_id}' already released: lease the layer again")
        return self._entry.layer

    @property
    def released(self) -> bool:
        return self._released

    def release(self) -> None:
        """Give the layer back. A second call does nothing, so it never counts twice."""
        with self._release_lock:
            if self._released:
                return
            self._released = True
        self._cache._release(self._entry)

    def __enter__(self) -> LayerLease[LayerT]:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.release()


class LayerCache(Generic[LayerT]):
    """At most ``capacity`` layers kept open (more only while all of them are leased)."""

    def __init__(self, capacity: int, build: Callable[[str], LayerT]) -> None:
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 1:
            raise ValueError(f"capacity must be 1 or more, got {capacity!r}")
        self._capacity = capacity
        self._build = build
        self._guard = threading.Lock()  # entries, lease counts, epochs: held briefly, never for I/O
        self._entries: OrderedDict[str, _Entry[LayerT]] = OrderedDict()  # least recently used first
        # Bumped by invalidate: a build that started under an older epoch read the old data.
        self._epochs: dict[str, int] = {}
        self._build_locks: dict[str, threading.Lock] = {}
        self._generations = itertools.count(1)
        self._closer = ThreadPoolExecutor(max_workers=1, thread_name_prefix=CLOSE_THREAD_PREFIX)
        self._pending_closes: set[Future[None]] = set()

    @property
    def capacity(self) -> int:
        return self._capacity

    def build_lock(self, dataset_id: str) -> threading.Lock:
        """The lock a build of ``dataset_id`` holds; the delete guard takes it to stop builds."""
        with self._guard:
            return self._build_locks.setdefault(dataset_id, threading.Lock())

    def lease(self, dataset_id: str) -> LayerLease[LayerT]:
        """The dataset's layer, built if needed, held until the lease is released.

        Raises:
            LayerCacheError: the dataset was re-uploaded during every one of MAX_BUILD_ATTEMPTS
                builds. Whatever the build raises propagates, and nothing is cached.
        """
        for _attempt in range(MAX_BUILD_ATTEMPTS):
            leased = self._lease_cached(dataset_id)
            if leased is not None:
                return leased
            with self.build_lock(dataset_id):
                # Another request may have built it while this one waited for the build lock.
                leased = self._lease_cached(dataset_id)
                if leased is not None:
                    return leased
                with self._guard:
                    epoch = self._epochs.get(dataset_id, 0)
                layer = self._build(dataset_id)
                with self._guard:
                    entry: _Entry[LayerT] | None = None
                    if self._epochs.get(dataset_id, 0) != epoch:
                        # Re-uploaded while this build read the old data: close it, build again.
                        to_close = [_Entry(dataset_id, layer, 0, retired=True)]
                    else:
                        entry = _Entry(dataset_id, layer, next(self._generations), leases=1)
                        self._entries[dataset_id] = entry
                        to_close = self._trim_locked()
                self._close_later(to_close)
                if entry is not None:
                    logger.info("semantic_layer_built", dataset_id=dataset_id, open_layers=len(self._entries))
                    return LayerLease(self, entry)
                logger.info("semantic_layer_build_superseded", dataset_id=dataset_id)
        raise LayerCacheError(f"Dataset '{dataset_id}' changed while it was being loaded. Try again.")

    def invalidate(self, dataset_id: str) -> None:
        """Drop the dataset's layer (new data was saved). Never waits: a leased layer is closed
        once its last lease is released, an idle one on the background thread now."""
        with self._guard:
            self._epochs[dataset_id] = self._epochs.get(dataset_id, 0) + 1
            entry = self._entries.pop(dataset_id, None)
            to_close = self._retire_locked(entry) if entry is not None else []
        self._close_later(to_close)

    def cached_ids(self) -> list[str]:
        """Dataset ids with a cached layer, least recently used first."""
        with self._guard:
            return list(self._entries)

    def cached_layer(self, dataset_id: str) -> LayerT | None:
        """The cached layer without a lease, for checks that never touch its connection."""
        with self._guard:
            entry = self._entries.get(dataset_id)
            return entry.layer if entry is not None else None

    def clear(self) -> None:
        """Invalidate every dataset (tests)."""
        for dataset_id in self.cached_ids():
            self.invalidate(dataset_id)

    def wait_for_closes(self, timeout: float = 30.0) -> None:
        """Wait until every close scheduled so far has run (tests, shutdown)."""
        with self._guard:
            pending = set(self._pending_closes)
        wait_futures(pending, timeout=timeout)

    def shutdown(self) -> None:
        """Close every idle layer and stop the close thread (process exit, tests)."""
        self.clear()
        self._closer.shutdown(wait=True)

    def _lease_cached(self, dataset_id: str) -> LayerLease[LayerT] | None:
        with self._guard:
            entry = self._entries.get(dataset_id)
            if entry is None:
                return None
            entry.leases += 1
            self._entries.move_to_end(dataset_id)
        return LayerLease(self, entry)

    def _release(self, entry: _Entry[LayerT]) -> None:
        with self._guard:
            entry.leases -= 1
            if entry.leases < 0:
                raise LayerCacheError(f"lease count of '{entry.dataset_id}' below zero")
            to_close = self._retire_locked(entry) if entry.retired else self._trim_locked()
        self._close_later(to_close)

    def _retire_locked(self, entry: _Entry[LayerT]) -> list[_Entry[LayerT]]:
        entry.retired = True
        if entry.leases == 0 and not entry.close_scheduled:
            entry.close_scheduled = True
            return [entry]
        return []

    def _trim_locked(self) -> list[_Entry[LayerT]]:
        """Evict least recently used, unleased layers down to capacity."""
        to_close: list[_Entry[LayerT]] = []
        while len(self._entries) > self._capacity:
            idle = next((entry for entry in self._entries.values() if entry.leases == 0), None)
            if idle is None:
                logger.warning("semantic_layer_cache_over_capacity", open_layers=len(self._entries))
                break
            del self._entries[idle.dataset_id]
            logger.info("semantic_layer_evicted", dataset_id=idle.dataset_id)
            to_close += self._retire_locked(idle)
        return to_close

    def _close_later(self, entries: list[_Entry[LayerT]]) -> None:
        for entry in entries:
            entry.close_scheduled = True
            future = self._closer.submit(self._close, entry)
            with self._guard:
                self._pending_closes.add(future)
            future.add_done_callback(self._close_done)

    def _close_done(self, future: Future[None]) -> None:
        with self._guard:
            self._pending_closes.discard(future)
        error = future.exception()
        if error is not None:  # _close logs what it expects; anything else is a bug, said loudly
            logger.error("semantic_layer_close_crashed", **error_fields(error))

    @staticmethod
    def _close(entry: _Entry[LayerT]) -> None:
        try:
            entry.layer.close()
        except (duckdb.Error, RuntimeError, OSError) as e:
            # The layer is out of the cache either way; a failed close leaks memory, not answers.
            logger.error("semantic_layer_close_failed", dataset_id=entry.dataset_id, **error_fields(e))
            return
        logger.info("semantic_layer_closed", dataset_id=entry.dataset_id)
