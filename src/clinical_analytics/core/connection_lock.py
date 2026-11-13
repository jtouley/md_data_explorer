# SPDX-License-Identifier: MIT
"""The lock around one semantic layer's DuckDB connection.

A DuckDB connection must not be used by two threads at once, and a cursor would not help: a
cursor is a separate connection that cannot see the layer's registered source or its temporary
cohort views (checked on DuckDB 2.0.0). So every DuckDB call on a layer runs under this lock.

It is re-entrant, because DuckDB-reading helpers (value counts, row counts) are called both on
their own and from inside a locked question step, and it can say whether the calling thread
holds it, so a step that must run locked can check that it does.
"""

from __future__ import annotations

import threading
from types import TracebackType


class ConnectionLock:
    """Re-entrant lock that knows which thread holds it."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._owner: int | None = None
        self._depth = 0

    def acquire(self, blocking: bool = True, timeout: float = -1) -> bool:
        acquired = self._lock.acquire(blocking, timeout)
        if acquired:
            self._owner = threading.get_ident()
            self._depth += 1
        return acquired

    def release(self) -> None:
        if self._owner != threading.get_ident():
            raise RuntimeError("ConnectionLock released by a thread that does not hold it")
        self._depth -= 1
        if self._depth == 0:
            self._owner = None
        self._lock.release()

    def held_by_current_thread(self) -> bool:
        return self._owner == threading.get_ident()

    def __enter__(self) -> ConnectionLock:
        self.acquire()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.release()
