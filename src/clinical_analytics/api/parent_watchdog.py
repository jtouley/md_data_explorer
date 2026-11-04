"""Stop the backend when the app that started it is gone (no orphaned servers).

Electron starts the API as a sidecar and stops it on quit. If Electron is killed instead (crash,
force quit, ``kill -9``), nothing would stop the backend; ``serve.py --exit-with-parent PID``
starts this watchdog, which checks once a second that PID still exists and, when it does not,
asks uvicorn to shut down (SIGTERM), with a hard exit if that has not finished after a grace
period.
"""

from __future__ import annotations

import os
import signal
import sys
import threading
import time
from collections.abc import Callable

import structlog

logger = structlog.get_logger()

CHECK_INTERVAL_S = 1.0
HARD_EXIT_AFTER_S = 15.0


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


def pid_alive(pid: int) -> bool:
    """Whether a process with this PID exists (never signals it)."""
    if pid <= 0:
        return False
    if sys.platform == "win32":  # os.kill(pid, 0) would send CTRL_C_EVENT on Windows
        return _windows_pid_alive(pid)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists, owned by someone else
    return True


def _request_shutdown() -> None:
    """Graceful uvicorn shutdown, then a hard exit if it hangs."""
    killer = threading.Timer(HARD_EXIT_AFTER_S, lambda: os._exit(1))
    killer.daemon = True
    killer.start()
    signal.raise_signal(signal.SIGTERM)


def start_parent_watchdog(
    parent_pid: int,
    *,
    interval_s: float = CHECK_INTERVAL_S,
    is_alive: Callable[[int], bool] = pid_alive,
    on_parent_gone: Callable[[], None] = _request_shutdown,
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
