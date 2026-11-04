"""Stop the backend when the app that started it is gone or asks it to (no orphaned servers).

Electron starts the API as a sidecar. Two lifelines tie the backend to it:

* ``--stop-on-stdin-close``: Electron holds the write end of the backend's stdin. Closing it is
  the graceful stop request on every platform (on Windows there is no SIGTERM for a windowless
  console child), and it also closes when Electron dies, however it dies.
* ``--exit-with-parent PID``: checks once a second that the parent is still there. On POSIX this
  compares ``os.getppid()`` (a reused PID can never look alive); on Windows, where the venv
  ``python.exe`` launcher sits between Electron and the interpreter, it waits on a process handle.

Either one asks uvicorn to shut down gracefully (SIGTERM: stop accepting, let in-flight requests
such as an upload or a DuckDB write finish). Only if that has not finished after
``HARD_EXIT_AFTER_S`` does the process exit hard.
"""

from __future__ import annotations

import os
import signal
import sys
import threading
import time
from collections.abc import Callable
from typing import BinaryIO

import structlog

logger = structlog.get_logger()

CHECK_INTERVAL_S = 1.0
HARD_EXIT_AFTER_S = 120.0
_shutdown_requested = threading.Event()


def _windows_pid_alive(pid: int) -> bool:  # pragma: no cover - exercised on Windows only
    import ctypes

    synchronize, wait_timeout = 0x00100000, 0x00000102
    kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
    handle = kernel32.OpenProcess(synchronize, False, pid)
    if not handle:
        return False
    try:
        return bool(kernel32.WaitForSingleObject(handle, 0) == wait_timeout)
    finally:
        kernel32.CloseHandle(handle)


def parent_alive(parent_pid: int) -> bool:
    """Whether the process that started us is still our parent (never signals anything)."""
    if parent_pid <= 0:
        return False
    if sys.platform == "win32":  # os.kill(pid, 0) would send CTRL_C_EVENT on Windows
        return _windows_pid_alive(parent_pid)
    # Reparenting (to init or a subreaper) is the only way getppid() changes: the parent exited.
    return os.getppid() == parent_pid


def request_graceful_shutdown(*, hard_exit_after_s: float = HARD_EXIT_AFTER_S) -> None:
    """SIGTERM ourselves (uvicorn drains in-flight requests), with a hard exit only as a last resort."""
    if _shutdown_requested.is_set():
        return
    _shutdown_requested.set()

    def _hard_exit() -> None:
        logger.error("api_shutdown_hard_exit", waited_s=hard_exit_after_s, reason="graceful shutdown did not finish")
        os._exit(1)

    killer = threading.Timer(hard_exit_after_s, _hard_exit)
    killer.daemon = True
    killer.start()
    signal.raise_signal(signal.SIGTERM)


def start_parent_watchdog(
    parent_pid: int,
    *,
    interval_s: float = CHECK_INTERVAL_S,
    is_alive: Callable[[int], bool] = parent_alive,
    on_parent_gone: Callable[[], None] = request_graceful_shutdown,
) -> threading.Thread:
    """Watch ``parent_pid`` on a daemon thread; call ``on_parent_gone`` once when it disappears."""
    if parent_pid <= 0:
        raise ValueError(f"parent PID must be positive, got {parent_pid}")

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
    on_close: Callable[[], None] = request_graceful_shutdown,
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
