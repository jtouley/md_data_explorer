"""Tests for scripts/check_push_tree.py (the pre-push gate's guard that the tested tree is the pushed commit)."""

from __future__ import annotations

import os
import subprocess
from collections.abc import Callable
from pathlib import Path

import pytest

from scripts import check_push_tree as cpt
from tests.scripts.conftest import BranchRepo

SRC = "src/clinical_analytics/core/schema.py"


@pytest.fixture
def pushed_repo(make_branch_repo: BranchRepo) -> Path:
    """A clean branch repo whose HEAD is the commit about to be pushed."""
    return make_branch_repo(
        {SRC: "def f():\n    return 1\n", ".gitignore": "*.log\n"}, {SRC: "def f():\n    return 2\n"}
    )


def _run(repo: Path, monkeypatch: pytest.MonkeyPatch, to_ref: str | None) -> int:
    monkeypatch.chdir(repo)
    if to_ref is None:
        monkeypatch.delenv(cpt.TO_REF_ENV, raising=False)
    else:
        monkeypatch.setenv(cpt.TO_REF_ENV, to_ref)
    return cpt.main()


@pytest.mark.parametrize("to_ref", ["HEAD", "work", None], ids=["sha_of_head", "branch_at_head", "run_by_hand"])
def test_main_cleanTreeAtPushedCommit_passes(
    to_ref: str | None, pushed_repo: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # Arrange
    (pushed_repo / "ignored.log").write_text("ignored files do not count\n", encoding="utf-8")

    # Act
    rc = _run(pushed_repo, monkeypatch, to_ref)

    # Assert
    assert rc == 0
    assert "PASS" in capsys.readouterr().out


@pytest.mark.parametrize(
    "dirty",
    [
        lambda repo: (repo / "helper.py").write_text("x = 1\n", encoding="utf-8"),
        lambda repo: (
            (repo / SRC).write_text("def f():\n    return 3\n", encoding="utf-8"),
            subprocess.run(["git", "add", SRC], cwd=repo, check=True),
        ),
        lambda repo: (repo / SRC).write_text("def f():\n    return 4\n", encoding="utf-8"),
    ],
    ids=["untracked_file", "staged_edit", "unstaged_edit"],
)
def test_main_treeNotThePushedCommit_fails(
    dirty: Callable[[Path], object],
    pushed_repo: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Arrange
    dirty(pushed_repo)

    # Act
    rc = _run(pushed_repo, monkeypatch, "HEAD")

    # Assert
    assert rc == 1
    assert "would be tested but not pushed" in capsys.readouterr().err


def test_main_pushOfAnotherBranch_fails(
    pushed_repo: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # Act: `git push origin base` while `work` is checked out.
    rc = _run(pushed_repo, monkeypatch, "base")

    # Assert
    assert rc == 1
    assert "is not the checked-out HEAD" in capsys.readouterr().err


def test_main_unknownPushedRef_exitsTwo(
    pushed_repo: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # Act
    rc = _run(pushed_repo, monkeypatch, "0" * 40)

    # Assert
    assert rc == 2
    assert "cannot run" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("pushed", "dirty", "count"),
    [("a" * 40, [], 0), ("b" * 40, [], 1), (None, ["?? x.py"], 1), ("b" * 40, ["M  y.py"], 2), (None, [], 0)],
    ids=["clean_at_head", "other_commit", "untracked", "both", "by_hand_clean"],
)
def test_problems_headPushedDirty_onePerProblem(pushed: str | None, dirty: list[str], count: int) -> None:
    # Act
    found = cpt.problems("a" * 40, pushed, dirty)

    # Assert
    assert len(found) == count


def test_problems_manyDirtyEntries_listsFirstTenAndCount() -> None:
    # Arrange
    dirty = [f"?? f{i}.py" for i in range(13)]

    # Act
    (problem,) = cpt.problems("a" * 40, None, dirty)

    # Assert
    assert "f9.py" in problem
    assert "f10.py" not in problem
    assert "and 3 more" in problem


@pytest.mark.parametrize("variable", ["GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_COMMON_DIR"])
def test_testSession_gitHookRepoVariable_notInherited(variable: str) -> None:
    # Arrange: git exports GIT_DIR to a hook run from a linked worktree (the pre-push gate);
    # tests/conftest.py drops these before any test runs git.

    # Act
    inherited = os.environ.get(variable)

    # Assert: a test's `git init` in a temp dir can never reach the real repo
    assert inherited is None
