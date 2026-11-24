"""Tests for the pre-push gate in .pre-commit-config.yaml (installed by ``make install-pre-commit``).

The gate runs ``make check-fast`` then ``make test-fast`` and blocks the push when either fails.
Its entry is run here against a fake ``make`` on PATH, so the tests check the real command line
without running the suite inside itself.
"""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

from tests.scripts.conftest import MakeGitRepo

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / ".pre-commit-config.yaml"
GATE_ID = "pre-push-gate"
FAKE_MAKE = """#!/usr/bin/env bash
echo "$*" >> "$FAKE_MAKE_LOG"
case "$1" in
  check-fast) exit "$CHECK_FAST_RC" ;;
  test-fast) exit "$TEST_FAST_RC" ;;
esac
exit 99
"""


def _config() -> dict[str, Any]:
    loaded = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


def _hooks(config: dict[str, Any]) -> list[dict[str, Any]]:
    return [hook for repo in config["repos"] for hook in repo["hooks"]]


def _gate() -> dict[str, Any]:
    (gate,) = [hook for hook in _hooks(_config()) if hook["id"] == GATE_ID]
    return gate


def test_config_installHookTypes_includePrePushAndPreCommit() -> None:
    # Act
    types = _config()["default_install_hook_types"]

    # Assert
    assert set(types) == {"pre-commit", "pre-push"}


def test_config_prePushStage_onlyTheGateRuns() -> None:
    # Arrange
    config = _config()
    default = config.get("default_stages")

    # Act
    on_push = [hook["id"] for hook in _hooks(config) if "pre-push" in hook.get("stages", default or ["pre-push"])]

    # Assert: commit hooks (ruff, mypy, ...) are not run a second time on push.
    assert on_push == [GATE_ID]


def test_config_gate_runsOnEveryPushWithoutFilenames() -> None:
    # Act
    gate = _gate()

    # Assert
    assert gate["language"] == "system"
    assert gate["pass_filenames"] is False
    assert gate["always_run"] is True


@pytest.mark.parametrize(
    ("check_rc", "test_rc", "blocked", "calls"),
    [
        (0, 0, False, ["check-fast", "test-fast"]),
        (2, 0, True, ["check-fast"]),
        (0, 2, True, ["check-fast", "test-fast"]),
        (2, 2, True, ["check-fast"]),
    ],
    ids=["all_pass", "checks_fail_skips_tests", "tests_fail", "both_fail"],
)
def test_gate_entry_makeResults_blocksPushOnAnyFailure(
    check_rc: int, test_rc: int, blocked: bool, calls: list[str], tmp_path: Path
) -> None:
    # Arrange
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake = bin_dir / "make"
    fake.write_text(FAKE_MAKE, encoding="utf-8")
    fake.chmod(0o755)
    log = tmp_path / "make.log"
    env = {
        **os.environ,
        "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        "FAKE_MAKE_LOG": str(log),
        "CHECK_FAST_RC": str(check_rc),
        "TEST_FAST_RC": str(test_rc),
    }

    # Act
    result = subprocess.run(shlex.split(_gate()["entry"]), cwd=tmp_path, env=env, check=False)

    # Assert
    assert (result.returncode != 0) is blocked
    made = log.read_text(encoding="utf-8").splitlines()
    assert [line.split()[0] for line in made] == calls


def test_install_preCommitInstall_writesPrePushHook(make_git_repo: MakeGitRepo) -> None:
    # Arrange
    repo = make_git_repo({".pre-commit-config.yaml": CONFIG_PATH.read_bytes()})
    env = {**os.environ, "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1"}

    # Act: what `make install-pre-commit` runs.
    result = subprocess.run(
        [sys.executable, "-m", "pre_commit", "install"], cwd=repo, env=env, capture_output=True, text=True, check=False
    )

    # Assert
    assert result.returncode == 0, result.stderr
    hook = repo / ".git" / "hooks" / "pre-push"
    assert hook.is_file()
    assert os.access(hook, os.X_OK)
    assert "--hook-type=pre-push" in hook.read_text(encoding="utf-8")
    assert (repo / ".git" / "hooks" / "pre-commit").is_file()


def test_makefile_installPreCommit_usesLockedPreCommitWithoutPip() -> None:
    # Arrange
    makefile = (REPO_ROOT / "Makefile").read_text(encoding="utf-8")

    # Act
    body = makefile.split("\ninstall-pre-commit:", 1)[1].split("\n\n", 1)[0]

    # Assert: a uv venv has no pip; pre-commit is a dev dependency.
    assert "pre_commit install" in body
    assert "pip install" not in body
