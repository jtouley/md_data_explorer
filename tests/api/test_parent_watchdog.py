"""The API stops itself when the app that started it disappears (no orphaned backend)."""

from __future__ import annotations

import os
import subprocess
import sys
import threading
from unittest.mock import patch

import pytest

from clinical_analytics.api import parent_watchdog, serve


def test_pid_alive_ownProcess_true():
    # Act / Assert
    assert parent_watchdog.pid_alive(os.getpid()) is True


@pytest.mark.parametrize("pid", [0, -1], ids=["zero", "negative"])
def test_pid_alive_nonPositivePid_false(pid):
    # Act / Assert
    assert parent_watchdog.pid_alive(pid) is False


def test_pid_alive_exitedProcess_false():
    # Arrange: a process that has exited and been reaped
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()

    # Act / Assert
    assert parent_watchdog.pid_alive(proc.pid) is False


def test_start_parent_watchdog_parentDisappears_callsShutdownOnce():
    # Arrange: parent alive for two checks, then gone
    answers = iter([True, True, False])
    gone = threading.Event()
    calls: list[int] = []

    def on_gone() -> None:
        calls.append(1)
        gone.set()

    # Act
    thread = parent_watchdog.start_parent_watchdog(
        4242, interval_s=0.001, is_alive=lambda _pid: next(answers), on_parent_gone=on_gone
    )

    # Assert
    assert gone.wait(timeout=5)
    thread.join(timeout=5)
    assert calls == [1]
    assert thread.daemon


@pytest.mark.parametrize("pid", [0, -5], ids=["zero", "negative"])
def test_start_parent_watchdog_invalidPid_raisesValueError(pid):
    # Act / Assert
    with pytest.raises(ValueError, match="parent PID must be positive"):
        parent_watchdog.start_parent_watchdog(pid)


@pytest.mark.parametrize(("argv", "expected"), [([], None), (["--exit-with-parent", "1234"], 1234)])
def test_main_exitWithParentFlag_startsWatchdogOnlyWhenGiven(argv, expected):
    # Arrange
    with (
        patch.object(serve, "check_bundled_model", return_value=None),
        patch.object(serve, "check_store", return_value=None),
        patch.object(serve, "start_parent_watchdog") as watchdog,
        patch.object(serve.uvicorn, "run") as run,
    ):
        # Act
        serve.main(argv)

    # Assert
    run.assert_called_once()
    if expected is None:
        watchdog.assert_not_called()
    else:
        watchdog.assert_called_once_with(expected)


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX process groups")
def test_watchdog_realParentKilled_childExits(tmp_path):
    """A child watching its parent exits after the parent is SIGKILLed (the Electron-crash case)."""
    # Arrange: parent spawns a child that watches the parent, prints the child pid, then sleeps
    child_code = (
        "import os, sys, time\n"
        "from clinical_analytics.api.parent_watchdog import start_parent_watchdog\n"
        "start_parent_watchdog(int(sys.argv[1]), interval_s=0.05, on_parent_gone=lambda: os._exit(0))\n"
        "time.sleep(60)\n"
    )
    parent_code = (
        "import os, subprocess, sys, time\n"
        f"c = subprocess.Popen([sys.executable, '-c', {child_code!r}, str(os.getpid())])\n"
        "print(c.pid, flush=True)\n"
        "time.sleep(60)\n"
    )
    parent = subprocess.Popen([sys.executable, "-c", parent_code], stdout=subprocess.PIPE, text=True)
    child_pid = int(parent.stdout.readline())  # type: ignore[union-attr]

    # Act
    parent.kill()
    parent.wait()

    # Assert: the orphaned child notices within a few seconds and exits
    for _ in range(100):
        if not parent_watchdog.pid_alive(child_pid) or _is_zombie(child_pid):
            break
        threading.Event().wait(0.05)
    assert not parent_watchdog.pid_alive(child_pid) or _is_zombie(child_pid)


def _is_zombie(pid: int) -> bool:
    try:
        with open(f"/proc/{pid}/stat") as f:
            return f.read().split(")")[-1].split()[0] == "Z"
    except FileNotFoundError:
        return False
