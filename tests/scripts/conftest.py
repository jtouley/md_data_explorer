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
