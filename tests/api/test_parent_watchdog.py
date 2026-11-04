"""The API stops itself when the app that started it disappears or closes its stdin."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
from unittest.mock import patch

import pytest

from clinical_analytics.api import parent_watchdog, serve


@pytest.fixture(autouse=True)
def _fresh_shutdown_flag():
    parent_watchdog._shutdown_requested.clear()
    yield
    parent_watchdog._shutdown_requested.clear()


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX parent check")
def test_parent_alive_ourParent_true():
    # Act / Assert
    assert parent_watchdog.parent_alive(os.getppid()) is True


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX parent check")
@pytest.mark.parametrize("pid_source", ["self", "init"], ids=["alive_but_not_parent", "pid_1_not_parent"])
def test_parent_alive_livePidThatIsNotOurParent_false(pid_source):
    """A live PID that is not our parent (e.g. a reused PID) must not count as the app."""
    # Arrange
    pid = os.getpid() if pid_source == "self" else 1
    if os.getppid() == pid:
        pytest.skip("test process is a direct child of init")

    # Act / Assert
    assert parent_watchdog.parent_alive(pid) is False


@pytest.mark.parametrize("pid", [0, -1], ids=["zero", "negative"])
def test_parent_alive_nonPositivePid_false(pid):
    # Act / Assert
    assert parent_watchdog.parent_alive(pid) is False


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


@pytest.mark.parametrize("payload", [b"", b"ignored bytes\n"], ids=["closed_immediately", "data_then_close"])
def test_start_stdin_watchdog_streamCloses_callsOnClose(payload):
    # Arrange
    read_fd, write_fd = os.pipe()
    closed = threading.Event()
    with os.fdopen(read_fd, "rb") as reader:
        thread = parent_watchdog.start_stdin_watchdog(reader, on_close=closed.set)

        # Act
        os.write(write_fd, payload)
        assert not closed.wait(timeout=0.1), "must not stop while the app keeps stdin open"
        os.close(write_fd)

        # Assert
        assert closed.wait(timeout=5)
        thread.join(timeout=5)


def test_request_graceful_shutdown_firstCall_sigtermThenHardExitOnlyAfterBound():
    # Arrange
    with (
        patch.object(parent_watchdog.signal, "raise_signal") as raise_signal,
        patch.object(parent_watchdog.threading, "Timer") as timer,
    ):
        # Act
        parent_watchdog.request_graceful_shutdown()
        parent_watchdog.request_graceful_shutdown()  # a second lifeline firing changes nothing

    # Assert: graceful first; the hard exit is a long fallback, armed once
    raise_signal.assert_called_once_with(signal.SIGTERM)
    timer.assert_called_once()
    assert timer.call_args.args[0] == parent_watchdog.HARD_EXIT_AFTER_S >= 60


@pytest.mark.parametrize(
    ("argv", "parent", "stdin"),
    [([], None, False), (["--exit-with-parent", "1234"], 1234, False), (["--stop-on-stdin-close"], None, True)],
    ids=["none", "parent_pid", "stdin_close"],
)
def test_main_lifelineFlags_startWatchdogsOnlyWhenGiven(argv, parent, stdin):
    # Arrange
    with (
        patch.object(serve, "check_bundled_model", return_value=None),
        patch.object(serve, "check_store", return_value=None),
        patch.object(serve, "start_parent_watchdog") as parent_watch,
        patch.object(serve, "start_stdin_watchdog") as stdin_watch,
        patch.object(serve.uvicorn, "run") as run,
    ):
        # Act
        serve.main(argv)

    # Assert
    run.assert_called_once()
    assert parent_watch.call_args_list == ([] if parent is None else [((parent,),)])
    assert stdin_watch.called is stdin


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX process semantics")
def test_watchdog_realParentKilled_childExits():
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
    env = {k: v for k, v in os.environ.items() if k != "CLINICAL_ANALYTICS_API_TOKEN"}
    parent = subprocess.Popen([sys.executable, "-c", parent_code], stdout=subprocess.PIPE, text=True, env=env)
    child_pid = int(parent.stdout.readline())  # type: ignore[union-attr]

    # Act
    parent.kill()
    parent.wait()

    # Assert: the orphaned child notices within a few seconds and exits
    for _ in range(100):
        if _gone(child_pid):
            break
        threading.Event().wait(0.05)
    assert _gone(child_pid)


def _gone(pid: int) -> bool:
    try:
        with open(f"/proc/{pid}/stat") as f:
            return f.read().split(")")[-1].split()[0] == "Z"
    except FileNotFoundError:
        return True
