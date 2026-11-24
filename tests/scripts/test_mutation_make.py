"""Tests for the ``make mutation-changed`` and ``make mutation-module`` recipes (Makefile).

The real recipes run in a throwaway branch repository against a fake ``uv`` on PATH: ``uv run
python`` runs this interpreter, ``uv run mutmut run`` records its arguments and plays back a given
output and exit code, and ``uv run mutmut results`` prints given results. So the tests check the
shell between the script and mutmut (argument building, exit code and log capture) without
running mutmut.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from scripts import mutation_changed as mc
from tests.scripts.conftest import BranchRepo

REPO_ROOT = Path(__file__).resolve().parents[2]
MOD = "clinical_analytics.core.schema"
SCHEMA = "src/clinical_analytics/core/schema.py"
PYPROJECT = '[tool.mutmut]\nsource_paths = ["src/clinical_analytics/"]\ndo_not_mutate = ["*/__init__.py"]\n'
SOURCE = "def keep(x):\n    return x + 1\n"
FAKE_UV = """#!/usr/bin/env bash
if [ "$1 $2" = "run python" ]; then shift 2; exec "$REAL_PYTHON" "$@"; fi
if [ "$1 $2 $3" = "run mutmut run" ]; then
  shift 3; printf '%s\\n' "$@" > "$MUTMUT_ARGS_LOG"; printf '%s\\n' "$FAKE_RUN_OUTPUT"; exit "$FAKE_RUN_RC"
fi
if [ "$1 $2 $3" = "run mutmut results" ]; then printf '%s' "$FAKE_RESULTS"; exit 0; fi
exit 99
"""


def _make(
    repo: Path, target: list[str], tmp_path: Path, run_output: str, run_rc: int, results: str = ""
) -> tuple[int, str, list[str]]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    fake = bin_dir / "uv"
    fake.write_text(FAKE_UV, encoding="utf-8")
    fake.chmod(0o755)
    args_log = tmp_path / "mutmut_args.log"
    env = {
        **os.environ,
        "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        "REAL_PYTHON": sys.executable,
        "MUTMUT_ARGS_LOG": str(args_log),
        "FAKE_RUN_OUTPUT": run_output,
        "FAKE_RUN_RC": str(run_rc),
        "FAKE_RESULTS": results,
    }
    env.pop("MAKEFLAGS", None)
    result = subprocess.run(
        ["make", "-s", "-f", str(REPO_ROOT / "Makefile"), *target],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    args = args_log.read_text(encoding="utf-8").splitlines() if args_log.exists() else []
    return result.returncode, result.stdout + result.stderr, args


@pytest.fixture
def mutation_repo(make_branch_repo: BranchRepo) -> Path:
    """A branch repo that adds ``x_added`` to the schema module, with the scripts the recipes run and
    a stale ``mutants/`` meta file from an earlier run that knows only ``x_keep``."""
    scripts = {
        f"scripts/{name}": (REPO_ROOT / "scripts" / name).read_text(encoding="utf-8")
        for name in ("mutation_changed.py", "git_tracked.py")
    }
    repo = make_branch_repo(
        {"pyproject.toml": PYPROJECT, SCHEMA: SOURCE, ".gitignore": ".venv/\nmutants/\n", **scripts},
        {SCHEMA: SOURCE + "\n\ndef added(x):\n    return x * 2\n"},
    )
    (repo / ".venv").mkdir()
    meta = repo / "mutants" / f"{SCHEMA}.meta"
    meta.parent.mkdir(parents=True)
    meta.write_text(json.dumps({"exit_code_by_key": {f"{MOD}.x_keep__mutmut_1": 1}}), encoding="utf-8")
    return repo


@pytest.mark.parametrize(
    ("run_output", "rc", "head"),
    [
        (f"Traceback (most recent call last):\nAssertionError: {mc.NOTHING_MATCHES}", 0, "mutation-changed: SKIP"),
        ("Traceback (most recent call last):\nRuntimeError: create_mutants failed", 1, "mutation-changed: FAIL"),
    ],
    ids=["nothing_matches_is_skip", "crash_with_stale_meta_is_fail"],
)
@pytest.mark.skipif(shutil.which("make") is None, reason="needs make")
def test_mutationChanged_mutmutExitsNonZero_skipOnlyForNothingMatches(
    run_output: str, rc: int, head: str, mutation_repo: Path, tmp_path: Path
) -> None:
    # Act
    code, output, _args = _make(mutation_repo, ["mutation-changed", "MUTATION_BASE=base"], tmp_path, run_output, 1)

    # Assert
    assert (code == 0) is (rc == 0), output  # make exits 2 when the recipe fails
    assert head in output
    assert run_output.splitlines()[-1] in output  # mutmut's output is still shown


@pytest.mark.skipif(shutil.which("make") is None, reason="needs make")
def test_mutationChanged_changedFunction_mutmutGetsTestSelectionGlob(mutation_repo: Path, tmp_path: Path) -> None:
    # Arrange
    results = f"    {MOD}.x_added__mutmut_1: killed\n"

    # Act
    code, output, args = _make(mutation_repo, ["mutation-changed", "MUTATION_BASE=base"], tmp_path, "done", 0, results)

    # Assert
    assert code == 0, output
    assert "mutation-changed: PASS: 1/1 killed" in output
    assert args[-2:] == [f"{MOD}.x_added__mutmut_*", f"{MOD}.x_adde[d*]"]


@pytest.mark.parametrize(
    ("funcs", "expected"),
    [
        (
            ["FUNCS=x_keep xǁAǁm"],
            [f"{MOD}.x_keep__mutmut_*", f"{MOD}.x_kee[p*]", f"{MOD}.xǁAǁm__mutmut_*", f"{MOD}.xǁAǁ[m*]"],
        ),
        ([], [f"{MOD}.*"]),
    ],
    ids=["funcs", "whole_module"],
)
@pytest.mark.skipif(shutil.which("make") is None, reason="needs make")
def test_mutationModule_funcs_mutmutGetsTestSelectionGlobs(
    funcs: list[str], expected: list[str], mutation_repo: Path, tmp_path: Path
) -> None:
    # Act
    code, output, args = _make(mutation_repo, ["mutation-module", f"MODULE={MOD}", *funcs], tmp_path, "done", 0)

    # Assert
    assert code == 0, output
    assert args[-len(expected) :] == expected
    assert "--max-children" in args
