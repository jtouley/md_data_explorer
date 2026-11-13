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


class FakeServer:
    """Stands in for uvicorn.Server: the two attributes the lifelines touch."""

    def __init__(self, started: bool = False) -> None:
        self.started = started
        self.should_exit = False


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX parent check")
def test_parent_alive_ourParent_true():
    # Act / Assert
    assert parent_watchdog.parent_alive(os.getppid()) is True


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX parent check")
@pytest.mark.parametrize("wrapper", ["direct", "shell_that_does_not_exec"])
def test_parent_alive_appBehindNonExecLauncher_stillSeenAsAlive(wrapper):
    """The app's PID is watched, not getppid(): a launcher between app and interpreter is normal."""
    # Arrange: this test process is "the app"; the interpreter is its child or grandchild
    probe = (
        "import sys; from clinical_analytics.api.parent_watchdog import parent_alive, process_start_token as t; "
        "pid = int(sys.argv[1]); print(parent_alive(pid, t(pid)))"
    )
    command = [sys.executable, "-c", probe, str(os.getpid())]
    if wrapper == "shell_that_does_not_exec":
        # The trailing command keeps sh alive as the interpreter's parent (no implicit exec).
        command = ["sh", "-c", '"$@"; status=$?; exit $status', "launcher", *command]

    # Act
    out = subprocess.run(command, capture_output=True, text=True, timeout=60)

    # Assert
    assert out.stdout.strip() == "True", out.stderr[-2000:]


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX process semantics")
@pytest.mark.parametrize("state", ["reaped", "zombie"])
def test_parent_alive_processExited_false(state):
    # Arrange: a process that has exited; "zombie" is not reaped yet (its PID still exists)
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    token = parent_watchdog.process_start_token(proc.pid)
    if state == "reaped":
        proc.wait()
    else:
        if sys.platform != "linux":
            proc.wait()
            pytest.skip("a zombie is only told apart through /proc")
        for _ in range(200):
            if parent_watchdog.process_start_token(proc.pid) == "":
                break
            threading.Event().wait(0.02)

    # Act
    alive = parent_watchdog.parent_alive(proc.pid, token)
    proc.wait()

    # Assert
    assert alive is False


@pytest.mark.skipif(sys.platform != "linux", reason="start time comes from /proc")
@pytest.mark.parametrize(
    ("recorded", "expected"),
    [(None, True), ("same", True), ("another-run", False)],
    ids=["no_token_recorded", "same_process", "pid_reused_by_another_process"],
)
def test_parent_alive_startToken_reusedPidCountsAsGone(recorded, expected):
    # Arrange: a live PID (ours); the token recorded when the watch began may belong to an earlier process
    pid = os.getpid()
    token = parent_watchdog.process_start_token(pid) if recorded == "same" else recorded

    # Act / Assert
    assert parent_watchdog.parent_alive(pid, token) is expected


@pytest.mark.parametrize(
    ("stat", "expected"),
    [
        ("4242 (electron) S 1 4242 4242 0 -1 4194560 " + " ".join(["0"] * 12) + " 987654 0 0", "987654"),
        ("4242 (we ird) name) R 1 4242 4242 0 -1 4194560 " + " ".join(["0"] * 12) + " 111 0 0", "111"),
        ("4242 (electron) Z 1 4242 4242 0 -1 4194560 " + " ".join(["0"] * 12) + " 987654 0 0", ""),
        ("4242 (electron) S 1", ""),
        (None, ""),
    ],
    ids=["running", "name_with_spaces_and_paren", "zombie", "truncated", "no_such_process"],
)
def test_process_start_token_procStat_startTimeOrEmpty(tmp_path, stat, expected):
    # Arrange: a stand-in /proc
    if stat is not None:
        (tmp_path / "4242").mkdir()
        (tmp_path / "4242" / "stat").write_text(stat)

    # Act / Assert
    assert parent_watchdog.process_start_token(4242, proc_root=tmp_path) == expected


def test_process_start_token_noProcFilesystem_none(tmp_path):
    # Act / Assert: macOS has no /proc; the caller falls back to existence only
    assert parent_watchdog.process_start_token(4242, proc_root=tmp_path / "missing") is None


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
        4242, on_parent_gone=on_gone, interval_s=0.001, is_alive=lambda _pid: next(answers)
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
        parent_watchdog.start_parent_watchdog(pid, on_parent_gone=lambda: None)


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


def test_graceful_shutdown_request_beforeServerStarted_waitsThenSetsShouldExitWithoutSignal():
    """A lifeline that fires before uvicorn is serving must not kill the process with a signal."""
    # Arrange
    server = FakeServer(started=False)
    with (
        patch.object(signal, "raise_signal") as raise_signal,
        patch.object(parent_watchdog.threading, "Timer") as timer,
    ):
        stop = parent_watchdog.graceful_shutdown_request(server, started_poll_s=0.001)
        requester = threading.Thread(target=stop, daemon=True)

        # Act: the request arrives early; later the server finishes starting
        requester.start()
        requester.join(timeout=0.1)
        exit_flag_while_starting = server.should_exit
        server.started = True
        requester.join(timeout=5)

    # Assert: nothing until the server can shut down properly, then the graceful flag, never a signal
    assert exit_flag_while_starting is False
    assert server.should_exit is True
    raise_signal.assert_not_called()
    timer.assert_called_once()
    assert timer.call_args.args[0] == parent_watchdog.HARD_EXIT_AFTER_S >= 60


@pytest.mark.parametrize("calls", [1, 3])
def test_graceful_shutdown_request_severalLifelinesFire_hardExitArmedOnce(calls):
    # Arrange
    server = FakeServer(started=True)
    with patch.object(parent_watchdog.threading, "Timer") as timer:
        stop = parent_watchdog.graceful_shutdown_request(server)

        # Act
        for _ in range(calls):
            stop()

    # Assert
    assert server.should_exit is True
    timer.assert_called_once()


@pytest.mark.parametrize(
    ("argv", "parent", "stdin"),
    [(["--exit-with-parent", "1234"], 1234, False), (["--stop-on-stdin-close"], None, True)],
    ids=["parent_pid", "stdin_close"],
)
def test_main_lifelineFlags_startWatchdogsBoundToTheServerTheyStop(argv, parent, stdin):
    # Arrange
    with (
        patch.object(serve, "check_bundled_model", return_value=None),
        patch.object(serve, "check_store", return_value=None),
        patch.object(serve, "start_parent_watchdog") as parent_watch,
        patch.object(serve, "start_stdin_watchdog") as stdin_watch,
        patch.object(serve.uvicorn, "Server") as server_cls,
        patch.object(serve.uvicorn, "run") as run,
        # A real hard-exit timer would os._exit this test process 120 s later.
        patch.object(parent_watchdog.threading, "Timer") as timer,
    ):
        server_cls.return_value.started = True

        # Act
        serve.main(argv)
        watch = parent_watch if parent is not None else stdin_watch
        stop = watch.call_args.kwargs["on_parent_gone" if parent is not None else "on_close"]
        stop()

    # Assert: the server object exists before the lifelines, and their callback stops that server
    run.assert_not_called()
    server_cls.return_value.run.assert_called_once()
    assert server_cls.call_args.args[0].log_config is None
    assert parent_watch.called is (parent is not None)
    assert stdin_watch.called is stdin
    if parent is not None:
        assert parent_watch.call_args.args == (parent,)
    assert server_cls.return_value.should_exit is True
    timer.assert_called_once()


def test_main_noLifelineFlags_plainUvicornRunNoWatchdogs():
    # Arrange
    with (
        patch.object(serve, "check_bundled_model", return_value=None),
        patch.object(serve, "check_store", return_value=None),
        patch.object(serve, "start_parent_watchdog") as parent_watch,
        patch.object(serve, "start_stdin_watchdog") as stdin_watch,
        patch.object(serve.uvicorn, "run") as run,
    ):
        # Act
        serve.main([])

    # Assert
    run.assert_called_once()
    parent_watch.assert_not_called()
    stdin_watch.assert_not_called()


@pytest.mark.parametrize("lifeline", [["--exit-with-parent", "1234"], ["--stop-on-stdin-close"]])
def test_main_reloadWithLifeline_rejectedBeforeAnythingStarts(lifeline, capsys):
    # Arrange
    with patch.object(serve.uvicorn, "run") as run, pytest.raises(SystemExit) as exited:
        # Act
        serve.main(["--reload", *lifeline])

    # Assert
    assert exited.value.code == 2
    assert "--reload cannot be combined" in capsys.readouterr().err
    run.assert_not_called()


def test_main_serverNeverStarted_exitsWithStartupFailure():
    # Arrange
    with (
        patch.object(serve, "check_bundled_model", return_value=None),
        patch.object(serve, "check_store", return_value=None),
        patch.object(serve, "start_stdin_watchdog"),
        patch.object(serve.uvicorn, "Server") as server_cls,
        pytest.raises(SystemExit) as exited,
    ):
        server_cls.return_value.started = False

        # Act
        serve.main(["--stop-on-stdin-close"])

    # Assert
    assert exited.value.code == serve.STARTUP_FAILED_EXIT


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.skipif(sys.platform == "win32", reason="POSIX exit status")
def test_serve_stdinAlreadyClosedAtStart_shutsDownGracefullyNotKilledBySignal(tmp_path):
    """The app quits (stdin closes) while the backend is still importing: a clean exit, not SIGTERM."""
    # Arrange: stdin at EOF from the first moment, long before uvicorn installs signal handlers
    import socket

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    env = {k: v for k, v in os.environ.items() if k != "CLINICAL_ANALYTICS_API_TOKEN"}
    env["CLINICAL_ANALYTICS_DATA_HOME"] = str(tmp_path / "data-home")
    env["CLINICAL_ANALYTICS_API_TOKEN"] = "t" * 43

    # Act
    proc = subprocess.run(
        [sys.executable, "-m", "clinical_analytics.api.serve", "--port", str(port), "--stop-on-stdin-close"],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
    )

    # Assert: uvicorn started, then ran its own shutdown (lifespan included); no signal death
    output = proc.stdout + proc.stderr
    assert proc.returncode == 0, output[-3000:]
    assert "api_stdin_closed" in output
    assert "Application shutdown complete" in output or "Finished server process" in output, output[-3000:]


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


class FakeKernel32:
    """Windows process handles without Windows: PIDs map to process objects, handles to objects.

    ``pid_owner`` says which process holds each PID right now, so a reused PID can be simulated:
    reopening by PID finds the new process, the old handle still names the old one.
    """

    WAIT_OBJECT_0 = 0x00000000
    WAIT_TIMEOUT = 0x00000102
    WAIT_FAILED = 0xFFFFFFFF

    def __init__(self, pid_owner: dict[int, str], running: set[str], *, wait_fails: bool = False) -> None:
        self.pid_owner = pid_owner
        self.running = running
        self.wait_fails = wait_fails
        self.handles: dict[int, str] = {}
        self.opened: list[int] = []
        self.waited: list[int] = []
        self.closed: list[int] = []

    def OpenProcess(self, _access: int, _inherit: bool, pid: int) -> int:  # noqa: N802 - Win32 name
        self.opened.append(pid)
        owner = self.pid_owner.get(pid)
        if owner is None:
            return 0
        handle = 100 + len(self.handles)
        self.handles[handle] = owner
        return handle

    def WaitForSingleObject(self, handle: int, _ms: int) -> int:  # noqa: N802 - Win32 name
        self.waited.append(handle)
        if self.wait_fails:
            return self.WAIT_FAILED
        return self.WAIT_TIMEOUT if self.handles[handle] in self.running else self.WAIT_OBJECT_0

    def CloseHandle(self, handle: int) -> int:  # noqa: N802 - Win32 name
        self.closed.append(handle)
        return 1


def test_parent_liveness_check_windowsPidReusedAfterAppExits_seenAsGone():
    # Arrange: Electron (PID 4242) is watched, then exits and Windows gives 4242 to another process
    kernel32 = FakeKernel32({4242: "electron"}, {"electron"})
    is_alive = parent_watchdog.parent_liveness_check(4242, platform="win32", kernel32=kernel32)
    assert is_alive(4242) is True

    # Act
    kernel32.running = {"unrelated"}
    kernel32.pid_owner[4242] = "unrelated"
    answers = [is_alive(4242) for _ in range(3)]

    # Assert: the one handle opened at the start is what is waited on; the PID is never reopened
    assert answers == [False, False, False]
    assert kernel32.opened == [4242]
    assert set(kernel32.waited) == {100}
    assert kernel32.closed == [100]


@pytest.mark.parametrize(
    ("pid_owner", "wait_fails", "expected_opened_handles"),
    [({}, False, 0), ({4242: "electron"}, True, 1)],
    ids=["not_openable", "wait_failed"],
)
def test_parent_liveness_check_windowsCannotConfirm_countsAsGone(pid_owner, wait_fails, expected_opened_handles):
    # Arrange
    kernel32 = FakeKernel32(pid_owner, {"electron"}, wait_fails=wait_fails)

    # Act
    is_alive = parent_watchdog.parent_liveness_check(4242, platform="win32", kernel32=kernel32)
    answers = [is_alive(4242), is_alive(4242)]

    # Assert: an app we cannot confirm does not keep the backend alive
    assert answers == [False, False]
    assert kernel32.opened == [4242]
    assert len(kernel32.handles) == expected_opened_handles
    assert len(kernel32.waited) == expected_opened_handles  # never waited again once ended


def test_windows_process_handle_oneOffCheckOfRunningApp_closesItsHandle():
    # Arrange
    kernel32 = FakeKernel32({4242: "electron"}, {"electron"})

    # Act
    handle = parent_watchdog.WindowsProcessHandle(4242, kernel32)
    alive = handle.alive()
    handle.close()
    handle.close()  # safe twice

    # Assert
    assert alive is True
    assert kernel32.closed == [100]
    assert handle.alive() is False
