"""Tests for scripts/model_lock.py (one local-model user at a time)."""

import sys
import threading
from pathlib import Path

import pytest

from scripts import model_lock


@pytest.fixture
def lock_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "locks" / "model.lock"
    monkeypatch.setenv(model_lock.LOCK_ENV, str(path))
    return path


def test_lock_path_envOverride_wins(lock_file):
    # Act / Assert
    assert model_lock.lock_path() == lock_file


def test_lock_path_noOverride_usesXdgCache(tmp_path, monkeypatch):
    # Arrange
    monkeypatch.delenv(model_lock.LOCK_ENV, raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))

    # Act / Assert
    assert model_lock.lock_path() == tmp_path / "mdde" / "model.lock"


def test_model_lock_heldElsewhere_noWaitRaisesBusyNamingHolder(lock_file):
    # Arrange / Act
    with model_lock.model_lock(lock_file, wait=False):
        with pytest.raises(model_lock.LockBusyError, match="held by: pid"):
            with model_lock.model_lock(lock_file, wait=False):
                pass

    # Assert: released on exit, so it can be taken again
    with model_lock.model_lock(lock_file, wait=False):
        pass


def test_model_lock_heldElsewhere_waitBlocksUntilReleased(lock_file):
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
def test_main_freeLock_runsCommandAndReturnsItsExitCode(exit_code, lock_file):
    # Act
    result = model_lock.main(["--", sys.executable, "-c", f"raise SystemExit({exit_code})"])

    # Assert
    assert result == exit_code
    assert lock_file.read_text(encoding="utf-8") == ""  # holder line cleared on release


def test_main_lockBusy_exitsTempfailWithoutRunningCommand(lock_file, tmp_path, capsys):
    # Arrange
    marker = tmp_path / "ran"

    # Act
    with model_lock.model_lock(lock_file, wait=False):
        result = model_lock.main(["--", sys.executable, "-c", f"open({str(marker)!r}, 'w')"])

    # Assert
    assert result == model_lock.EX_TEMPFAIL
    assert not marker.exists()
    assert "Another model user is running" in capsys.readouterr().err


def test_main_commandSeesLockHeld_byItsWrapper(lock_file):
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
def test_main_noCommand_exitsTwo(argv, lock_file):
    # Act / Assert
    with pytest.raises(SystemExit) as exc:
        model_lock.main(argv)
    assert exc.value.args == (2,)


def test_main_missingExecutable_exitsTwo(lock_file, capsys):
    # Act
    result = model_lock.main(["--", "definitely-not-a-command-mdde"])

    # Assert
    assert result == 2
    assert "definitely-not-a-command-mdde" in capsys.readouterr().err
