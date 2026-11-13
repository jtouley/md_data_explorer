# SPDX-License-Identifier: MIT
"""Stop the backend when the app that started it is gone or asks it to (no orphaned servers).

Electron starts the API as a sidecar. Two lifelines tie the backend to it:

* ``--stop-on-stdin-close``: Electron holds the write end of the backend's stdin. Closing it is
  the graceful stop request on every platform (on Windows there is no SIGTERM for a windowless
  console child), and it also closes when Electron dies, however it dies.
* ``--exit-with-parent PID``: checks once a second that process PID is still running. PID is the
  app, which need not be our direct parent: a launcher that starts the interpreter without
  ``exec`` (a shell wrapper, ``uv run``, the venv ``python.exe`` on Windows, a bundled-runtime
  launcher) sits in between, so ``os.getppid()`` is not compared. On POSIX the process must exist
  and, where the platform tells us (Linux ``/proc``), still be the one that was there when the
  watch began and not a zombie. On Windows the process is opened once when the watch begins and
  that one handle is waited on, so a PID that Windows gives to another process later is never
  mistaken for the app. Where a reused PID cannot be told apart (macOS), the stdin lifeline is
  what notices the app is gone.

Either one asks uvicorn to shut down gracefully by setting ``server.should_exit`` once the server
has started (stop accepting, let in-flight requests such as an upload or a DuckDB write finish).
No signal is raised, so a lifeline that fires before uvicorn has installed its signal handlers
cannot kill the process outright. Only if the shutdown has not finished after
``HARD_EXIT_AFTER_S`` does the process exit hard.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, BinaryIO, Protocol

import structlog

logger = structlog.get_logger()

CHECK_INTERVAL_S = 1.0
HARD_EXIT_AFTER_S = 120.0
STARTED_POLL_S = 0.05


class StoppableServer(Protocol):
    """The part of ``uvicorn.Server`` the lifelines use."""

    started: bool
    should_exit: bool


_WIN_SYNCHRONIZE = 0x00100000
_WIN_WAIT_TIMEOUT = 0x00000102


class WindowsProcessHandle:
    """One Windows process, held open by handle from the moment the watch begins.

    A handle names the process object itself, not its PID: while it is open the object stays, and
    waiting on it reports that exact process ending. Reopening by PID on every check (what this
    replaced) would find whichever process owns the PID now. ``kernel32`` is injectable so the
    logic is tested without Windows.
    """

    def __init__(self, pid: int, kernel32: Any) -> None:
        self.pid = pid
        self._kernel32 = kernel32
        self._handle = kernel32.OpenProcess(_WIN_SYNCHRONIZE, False, pid)
        self._ended = not self._handle
        if self._ended:
            # Already gone, or not ours to watch: either way there is no app to stay alive for.
            logger.warning("parent_process_not_openable", parent_pid=pid)

    def alive(self) -> bool:
        """Whether the process is still running (False from the first time it is seen ended)."""
        if self._ended:
            return False
        # WAIT_TIMEOUT: not signalled, still running. Anything else (signalled, or the wait itself
        # failed) counts as ended: an app we cannot confirm must not keep the backend alive.
        if self._kernel32.WaitForSingleObject(self._handle, 0) == _WIN_WAIT_TIMEOUT:
            return True
        self.close()
        return False

    def close(self) -> None:
        """Release the handle (safe to call more than once); the process then counts as ended."""
        if self._handle:
            self._kernel32.CloseHandle(self._handle)
            self._handle = None
        self._ended = True


def _windows_kernel32() -> Any:  # pragma: no cover - Windows only
    import ctypes

    kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
    kernel32.OpenProcess.restype = ctypes.c_void_p
    kernel32.OpenProcess.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32]
    kernel32.WaitForSingleObject.restype = ctypes.c_uint32
    kernel32.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
    return kernel32


def process_start_token(pid: int, proc_root: Path = Path("/proc")) -> str | None:
    """What identifies this run of ``pid``: its start time from Linux ``/proc``.

    Returns None when the platform has no ``/proc`` (macOS), and ``""`` when the process is gone
    or a zombie (exited, not yet reaped): neither is a running app.
    """
    if not proc_root.is_dir():
        return None
    try:
        stat = (proc_root / str(pid) / "stat").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    fields = stat.rpartition(")")[2].split()  # after "pid (comm)": state is field 0, starttime field 19
    if len(fields) < 20 or fields[0] == "Z":
        return ""
    return fields[19]


def _posix_pid_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)  # signal 0: existence check only, nothing is delivered
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # it exists; it belongs to someone else
    return True


def parent_alive(parent_pid: int, start_token: str | None = None) -> bool:
    """Whether process ``parent_pid`` (the app) is still running; never signals anything.

    ``start_token`` is ``process_start_token(parent_pid)`` taken when the watch began; with it a
    PID that was reused by another process counts as gone.
    """
    if parent_pid <= 0:
        return False
    if sys.platform == "win32":  # pragma: no cover - os.kill(pid, 0) would send CTRL_C_EVENT there
        # A one-off answer only; the watchdog keeps one handle open instead (parent_liveness_check).
        handle = WindowsProcessHandle(parent_pid, _windows_kernel32())
        try:
            return handle.alive()
        finally:
            handle.close()
    if not _posix_pid_exists(parent_pid):
        return False
    current = process_start_token(parent_pid)
    if current is None:
        return True  # no /proc: existence is all we can tell (the stdin lifeline covers PID reuse)
    if current == "":
        return False
    return start_token is None or current == start_token


def parent_liveness_check(
    parent_pid: int, *, platform: str = sys.platform, kernel32: Any = None
) -> Callable[[int], bool]:
    """The check the watchdog repeats, bound to the process that has ``parent_pid`` right now.

    Windows: one handle, opened here and waited on at every check. POSIX: the PID must exist and,
    on Linux, keep the start time read here.
    """
    if platform == "win32":
        handle = WindowsProcessHandle(parent_pid, kernel32 if kernel32 is not None else _windows_kernel32())
        return lambda _pid: handle.alive()
    start_token = process_start_token(parent_pid)
    return lambda pid: parent_alive(pid, start_token)


def graceful_shutdown_request(
    server: StoppableServer,
    *,
    hard_exit_after_s: float = HARD_EXIT_AFTER_S,
    started_poll_s: float = STARTED_POLL_S,
) -> Callable[[], None]:
    """A callable for the lifelines: ask ``server`` to shut down gracefully, once.

    It waits until the server has started and then sets ``should_exit``, which uvicorn's main loop
    reads on its next tick: the listener closes, in-flight requests finish and the lifespan
    shutdown runs. Setting it earlier would make uvicorn return right after startup and skip that
    shutdown. A hard exit is armed first as the last resort (a server that never starts, or a
    shutdown that never finishes).
    """
    requested = threading.Event()

    def _hard_exit() -> None:
        logger.error("api_shutdown_hard_exit", waited_s=hard_exit_after_s, reason="graceful shutdown did not finish")
        os._exit(1)

    def _request() -> None:
        if requested.is_set():
            return
        requested.set()
        killer = threading.Timer(hard_exit_after_s, _hard_exit)
        killer.daemon = True
        killer.start()
        while not server.started:
            time.sleep(started_poll_s)
        server.should_exit = True
        logger.info("api_shutdown_requested", how="server.should_exit")

    return _request


def start_parent_watchdog(
    parent_pid: int,
    *,
    on_parent_gone: Callable[[], None],
    interval_s: float = CHECK_INTERVAL_S,
    is_alive: Callable[[int], bool] | None = None,
) -> threading.Thread:
    """Watch ``parent_pid`` on a daemon thread; call ``on_parent_gone`` once when it disappears."""
    if parent_pid <= 0:
        raise ValueError(f"parent PID must be positive, got {parent_pid}")
    if is_alive is None:
        is_alive = parent_liveness_check(parent_pid)

    def _run() -> None:
        while is_alive(parent_pid):
            time.sleep(interval_s)
        logger.warning("parent_process_gone", parent_pid=parent_pid, action="shutting down the API")
        on_parent_gone()

    thread = threading.Thread(target=_run, name="parent-watchdog", daemon=True)
    thread.start()
    return thread


def start_stdin_watchdog(
    stream: BinaryIO | None = None,
    *,
    on_close: Callable[[], None],
) -> threading.Thread:
    """Call ``on_close`` once stdin reaches EOF (the app closed it, or the app is gone)."""
    source = stream if stream is not None else sys.stdin.buffer

    def _run() -> None:
        while source.read(4096):
            pass  # the app sends nothing; any bytes are ignored
        logger.info("api_stdin_closed", action="shutting down the API")
        on_close()

    thread = threading.Thread(target=_run, name="stdin-watchdog", daemon=True)
    thread.start()
    return thread
