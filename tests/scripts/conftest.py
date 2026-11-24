"""Fixtures shared by the repository-guard script tests (check_spdx_headers, check_no_data_files)."""

from __future__ import annotations

import subprocess
from collections.abc import Callable
from pathlib import Path

import pytest

MakeGitRepo = Callable[[dict[str, bytes]], Path]


@pytest.fixture
def make_git_repo(tmp_path: Path) -> MakeGitRepo:
    """Factory: a throwaway git repository with the given files written and staged.

    Staged is enough: ``git ls-files`` lists the index, so no commit (and no identity) is needed.
    """

    def _make(files: dict[str, bytes]) -> Path:
        root = tmp_path / "repo"
        root.mkdir()
        subprocess.run(["git", "init", "-q", str(root)], check=True)
        for rel, content in files.items():
            path = root / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        subprocess.run(["git", "add", "-A"], cwd=root, check=True)
        return root

    return _make


BranchRepo = Callable[[dict[str, str], dict[str, str | None]], Path]
GIT_IDENTITY = ["-c", "user.name=Test", "-c", "user.email=test@example.invalid", "-c", "commit.gpgsign=false"]


@pytest.fixture
def make_branch_repo(tmp_path: Path) -> BranchRepo:
    """Factory: a git repository with ``base`` committed on branch ``base`` and ``branch`` committed on top.

    For guards that diff against a merge base (check_eval_tokens, check_pr_size, mutation_changed).
    A ``None`` value in ``branch`` deletes that file on the branch.
    """

    def _commit(root: Path, files: dict[str, str | None], message: str) -> None:
        for rel, content in files.items():
            path = root / rel
            if content is None:
                path.unlink()
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
        subprocess.run(["git", "add", "-A"], cwd=root, check=True)
        subprocess.run(["git", *GIT_IDENTITY, "commit", "-q", "--allow-empty", "-m", message], cwd=root, check=True)

    def _make(base: dict[str, str], branch: dict[str, str | None]) -> Path:
        root = tmp_path / "branch_repo"
        root.mkdir()
        subprocess.run(["git", "init", "-q", "-b", "base", str(root)], check=True)
        _commit(root, dict(base), "base")
        subprocess.run(["git", "checkout", "-q", "-b", "work"], cwd=root, check=True)
        _commit(root, branch, "work")
        return root

    return _make
