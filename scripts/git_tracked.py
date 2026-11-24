"""Tracked files of the enclosing git repository, independent of the current directory.

``git ls-files`` prints paths relative to the cwd, so a guard that matches ``src/*.py`` against
them silently checks nothing when it is run from a subdirectory. These helpers always work from
the repository's top level: paths come back repo-relative (``--full-name``) and callers read
them relative to :func:`repo_root`.
"""

from __future__ import annotations

import subprocess
from pathlib import Path


class GitQueryError(RuntimeError):
    """A git command the guards depend on failed (not a repository, git missing, ...)."""


def _git(args: list[str], cwd: Path) -> bytes:
    try:
        return subprocess.run(["git", *args], cwd=cwd, capture_output=True, check=True).stdout
    except FileNotFoundError as exc:
        raise GitQueryError("git is not installed or not on PATH") from exc
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode(errors="replace").strip()
        raise GitQueryError(f"`git {' '.join(args)}` failed in {cwd}: {stderr}") from exc


def repo_root(start: Path | None = None) -> Path:
    """Top level of the git work tree containing ``start`` (default: the current directory)."""
    out = _git(["rev-parse", "--show-toplevel"], start or Path.cwd())
    return Path(out.decode().strip())


def tracked_files(root: Path) -> list[str]:
    """Every path git tracks under ``root``, relative to ``root`` with "/" separators."""
    out = _git(["ls-files", "-z", "--full-name"], root)
    return [p for p in out.decode().split("\0") if p]


def diff_against_base(root: Path, base: str, args: list[str], paths: tuple[str, ...] = ()) -> str:
    """``git diff --merge-base <base> <args> -- <paths>``: the working tree against the fork point.

    Covers committed and uncommitted changes to tracked files (untracked files are not in a diff
    until they are added). A missing ``base`` ref (shallow clone, no ``git fetch origin main``)
    raises :class:`GitQueryError` naming the ref, never an empty diff.
    """
    try:
        _git(["rev-parse", "--verify", "--quiet", f"{base}^{{commit}}"], root)
    except GitQueryError as exc:
        raise GitQueryError(f"base ref {base!r} not found; run `git fetch origin main` or pass another base") from exc
    cmd = ["-c", "core.quotePath=false", "diff", "--no-color", "--no-ext-diff", "--no-textconv", "--merge-base", base]
    out = _git([*cmd, *args, "--", *paths], root)
    return out.decode("utf-8", errors="replace")
