"""Tests for scripts/model_lock.py (one local-model user at a time)."""

import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from scripts import model_lock


@pytest.fixture
def lock_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "locks" / "model.lock"
    monkeypatch.setenv(model_lock.LOCK_ENV, str(path))
    return path


def test_lock_path_envOverride_wins(lock_file: Path) -> None:
    # Act / Assert
    assert model_lock.lock_path() == lock_file


def test_lock_path_noOverride_usesXdgCache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange
    monkeypatch.delenv(model_lock.LOCK_ENV, raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))

    # Act / Assert
    assert model_lock.lock_path() == tmp_path / "mdde" / "model.lock"


def test_model_lock_heldElsewhere_noWaitRaisesBusyNamingHolder(lock_file: Path) -> None:
    # Arrange / Act
    with model_lock.model_lock(lock_file, wait=False):
        with pytest.raises(model_lock.LockBusyError, match="held by: pid"):
            with model_lock.model_lock(lock_file, wait=False):
                pass

    # Assert: released on exit, so it can be taken again
    with model_lock.model_lock(lock_file, wait=False):
        pass


def test_model_lock_heldElsewhere_waitBlocksUntilReleased(lock_file: Path) -> None:
    # Arrange
    order: list[str] = []
    first = model_lock.model_lock(lock_file, wait=False)
    first.__enter__()

    def release() -> None:
        order.append("released")
        first.__exit__(None, None, None)

    timer = threading.Timer(0.3, release)
    timer.start()

    # Act
    with model_lock.model_lock(lock_file, wait=True):
        order.append("acquired")
    timer.join()

    # Assert
    assert order == ["released", "acquired"]


@pytest.mark.parametrize("exit_code", [0, 3], ids=["success", "failure"])
def test_main_freeLock_runsCommandAndReturnsItsExitCode(exit_code: int, lock_file: Path) -> None:
    # Act
    result = model_lock.main(["--", sys.executable, "-c", f"raise SystemExit({exit_code})"])

    # Assert
    assert result == exit_code
    assert lock_file.read_text(encoding="utf-8") == ""  # holder line cleared on release


def test_main_lockBusy_exitsTempfailWithoutRunningCommand(
    lock_file: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # Arrange
    marker = tmp_path / "ran"

    # Act
    with model_lock.model_lock(lock_file, wait=False):
        result = model_lock.main(["--", sys.executable, "-c", f"open({str(marker)!r}, 'w')"])

    # Assert
    assert result == model_lock.EX_TEMPFAIL
    assert not marker.exists()
    assert "Another model user is running" in capsys.readouterr().err


def test_main_commandSeesLockHeld_byItsWrapper(lock_file: Path) -> None:
    # Arrange: the child tries the lock itself; it must find it busy
    probe = (
        "import fcntl, sys\n"
        f"f = open({str(lock_file)!r}, 'a')\n"
        "try:\n    fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)\nexcept BlockingIOError:\n    sys.exit(42)\n"
    )

    # Act
    result = model_lock.main(["--", sys.executable, "-c", probe])

    # Assert
    assert result == 42


@pytest.mark.parametrize("argv", [[], ["--wait"], ["--wait", "--"]], ids=["empty", "wait_only", "separator_only"])
def test_main_noCommand_exitsTwo(argv: list[str], lock_file: Path) -> None:
    # Act / Assert
    with pytest.raises(SystemExit) as exc:
        model_lock.main(argv)
    assert exc.value.args == (2,)


def test_main_missingExecutable_exitsTwo(lock_file: Path, capsys: pytest.CaptureFixture[str]) -> None:
    # Act
    result = model_lock.main(["--", "definitely-not-a-command-mdde"])

    # Assert
    assert result == 2
    assert "definitely-not-a-command-mdde" in capsys.readouterr().err


@pytest.mark.parametrize(
    "command", [[sys.executable, "-c", "pass"], ["definitely-not-a-command-mdde"]], ids=["ran", "failed"]
)
def test_main_anyOutcome_restoresPreviousSignalHandlers(command: list[str], lock_file: Path) -> None:
    # Arrange
    before = {signum: signal.getsignal(signum) for signum in model_lock.FORWARDED_SIGNALS}

    # Act
    model_lock.main(["--", *command])

    # Assert
    assert {signum: signal.getsignal(signum) for signum in model_lock.FORWARDED_SIGNALS} == before


@pytest.mark.parametrize("signum", model_lock.FORWARDED_SIGNALS, ids=lambda s: signal.Signals(s).name)
def test_main_signalToWrapper_isForwardedToCommand(signum: int, lock_file: Path, tmp_path: Path) -> None:
    # Arrange: the child exits 100 + signum when it gets the signal; "ready" says its handler is in
    ready = tmp_path / "ready"
    child = (
        "import signal, sys, time\n"
        f"signal.signal({signum}, lambda s, f: sys.exit(100 + s))\n"
        f"open({str(ready)!r}, 'w').close()\n"
        "time.sleep(30)\nsys.exit(1)\n"
    )

    def send_when_ready() -> None:
        deadline = time.monotonic() + 20
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        os.kill(os.getpid(), signum)

    sender = threading.Thread(target=send_when_ready)
    sender.start()

    # Act
    result = model_lock.main(["--", sys.executable, "-c", child])
    sender.join()

    # Assert
    assert result == 100 + signum
    assert lock_file.read_text(encoding="utf-8") == ""


def test_main_signalBeforeCommandStarts_isDeliveredToCommandNotWrapper(
    lock_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange: SIGTERM lands while the command is being started
    real_popen = subprocess.Popen

    def popen_after_signal(command: list[str]) -> subprocess.Popen[bytes]:
        os.kill(os.getpid(), signal.SIGTERM)
        return real_popen(command)

    monkeypatch.setattr(model_lock.subprocess, "Popen", popen_after_signal)

    # Act: without the early handler this SIGTERM would end the test process
    result = model_lock.main(["--", sys.executable, "-c", "import time; time.sleep(30)"])

    # Assert
    assert result == -signal.SIGTERM
    assert lock_file.read_text(encoding="utf-8") == ""
