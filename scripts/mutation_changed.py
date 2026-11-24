#!/usr/bin/env python3
"""Mutation testing scoped to the src modules a branch changed (``make mutation-changed``).

``patterns``: print one mutmut name pattern (``clinical_analytics.core.schema.*``) per mutable
src module added, modified or renamed against the merge base with ``--base``. Only modules under
``[tool.mutmut] source_paths`` and not excluded by ``do_not_mutate`` (pyproject.toml) count. No
output means nothing to mutate.

``summary PATTERN...``: read ``mutmut results --all true`` on stdin, keep the mutants matching the
patterns and print per-module counts plus every mutant not killed. Only ``PASSING_STATUSES``
(killed, timeout, suspicious, caught by type check) pass; every other status fails, including
survived, no tests, not checked, skipped, segfault and "check was interrupted by user" (pytest's
exit 2, also returned for collection errors). Kill such a mutant with a test, or, for an
equivalent mutant, add ``# pragma: no mutate`` to its line and say why there.

A module with no results is told apart by mutmut's ``mutants/<path>.meta`` file, written when
mutants are generated: a meta file with no mutants (constants or dataclasses only, every line
``# pragma: no mutate``) is a SKIP; no meta file means the run never got to the module and fails.

Exit codes: 0 all passed (or nothing to mutate), 1 unkilled mutants or a module without results,
2 the step could not run.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import re
import sys
import tomllib
from collections import Counter
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Final

if __package__:  # imported as scripts.mutation_changed (tests)
    from scripts.git_tracked import GitQueryError, diff_against_base, repo_root
else:  # run as a file: scripts/ is sys.path[0]
    from git_tracked import GitQueryError, diff_against_base, repo_root

DEFAULT_BASE: Final = "origin/main"
SRC_ROOT: Final = "src/"
MUTANTS_DIR: Final = Path("mutants")
# mutmut statuses that count as tested; anything else (survived, no tests, not checked, skipped,
# segfault, check was interrupted by user, a status a later mutmut adds) fails the gate.
PASSING_STATUSES: Final = frozenset({"killed", "timeout", "suspicious", "caught by type check"})
RESULT_RE: Final = re.compile(r"^\s*(\S+): (.+?)\s*$")


def _mutmut_config(pyproject_text: str) -> dict[str, object]:
    config = tomllib.loads(pyproject_text).get("tool", {}).get("mutmut", {})
    return config if isinstance(config, dict) else {}


def do_not_mutate(pyproject_text: str) -> tuple[str, ...]:
    """``[tool.mutmut] do_not_mutate`` globs, so the scope matches ``make mutation``."""
    return tuple(str(g) for g in _as_list(_mutmut_config(pyproject_text).get("do_not_mutate", ())))


def source_paths(pyproject_text: str) -> tuple[str, ...]:
    """``[tool.mutmut] source_paths`` (default ``src/``): mutmut generates mutants only under these."""
    return tuple(str(p) for p in _as_list(_mutmut_config(pyproject_text).get("source_paths", [SRC_ROOT])))


def _as_list(value: object) -> list[object]:
    return [value] if isinstance(value, str) else list(value) if isinstance(value, list | tuple) else []


def _under(path: str, root: str) -> bool:
    root = root.removeprefix("./")
    return path == root or path.startswith(root.rstrip("/") + "/")


def module_patterns(paths: Iterable[str], excluded: Iterable[str], sources: Iterable[str] = (SRC_ROOT,)) -> list[str]:
    """Sorted mutmut patterns for the mutable ``src/**.py`` paths among ``paths`` under ``sources``."""
    excluded, sources = tuple(excluded), tuple(sources)
    modules = {
        path.removeprefix(SRC_ROOT).removesuffix(".py").replace("/", ".")
        for path in paths
        if path.startswith(SRC_ROOT)
        and path.endswith(".py")
        and any(_under(path, root) for root in sources)
        and not any(fnmatch.fnmatch(path, g) for g in excluded)
    }
    return [f"{module}.*" for module in sorted(modules)]


def generated_mutants(mutants_dir: Path, module: str) -> int | None:
    """How many mutants mutmut generated for ``module``; None when it has no meta file yet.

    mutmut writes ``mutants/src/<module path>.py.meta`` for every file it mutates, with an empty
    ``exit_code_by_key`` when the file holds nothing mutable.
    """
    meta = mutants_dir / f"{SRC_ROOT}{module.replace('.', '/')}.py.meta"
    try:
        data = json.loads(meta.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    keys = data.get("exit_code_by_key") if isinstance(data, dict) else None
    if not isinstance(keys, dict):
        raise ValueError(f"{meta}: no 'exit_code_by_key' object; mutmut's meta format changed?")
    return len(keys)


def parse_results(text: str) -> dict[str, str]:
    """mutant name -> status from ``mutmut results`` lines ("    name: status")."""
    return {m.group(1): m.group(2) for m in map(RESULT_RE.match, text.splitlines()) if m}


def summarise(
    results: dict[str, str], patterns: list[str], generated: Callable[[str], int | None] = lambda _module: None
) -> tuple[str, bool]:
    """Report text and whether every matched mutant passed (and every pattern matched one).

    ``generated(module)`` is how many mutants mutmut generated for a module without results: 0 is a
    module with nothing to mutate (SKIP), None one the run never reached (FAIL).
    """
    lines: list[str] = []
    ok = True
    skipped = 0
    totals: Counter[str] = Counter()
    for pattern in patterns:
        matched = {name: status for name, status in results.items() if fnmatch.fnmatchcase(name, pattern)}
        counts = Counter(matched.values())
        totals.update(counts)
        module = pattern.removesuffix(".*")
        if not matched:
            if generated(module) == 0:
                lines.append(f"  {module}: SKIP: nothing to mutate (mutmut generated no mutants)")
                skipped += 1
            else:
                lines.append(f"  {module}: no mutants in the results (did the run finish?)")
                ok = False
            continue
        detail = ", ".join(f"{n} {status}" for status, n in sorted(counts.items()))
        lines.append(f"  {module}: {len(matched)} mutants: {detail}")
        for name, status in sorted(matched.items()):
            if status != "killed":
                lines.append(f"    {status}: {name}  (mutmut show {name})")
    unkilled = sum(n for status, n in totals.items() if status not in PASSING_STATUSES)
    ok = ok and unkilled == 0
    total = sum(totals.values())
    if ok and patterns and skipped == len(patterns):
        head = "mutation-changed: SKIP: no mutants in the changed modules"
    else:
        head = f"mutation-changed: {'PASS' if ok else 'FAIL'}: {totals['killed']}/{total} killed, {unkilled} unkilled"
    return "\n".join([head, *lines]), ok


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    patterns_cmd = sub.add_parser("patterns", help="print a mutmut pattern per changed src module")
    patterns_cmd.add_argument("--base", default=DEFAULT_BASE)
    summary_cmd = sub.add_parser("summary", help="summarise `mutmut results --all true` from stdin")
    summary_cmd.add_argument("patterns", nargs="+")
    summary_cmd.add_argument("--mutants-dir", type=Path, default=MUTANTS_DIR, help="mutmut's mutants/ tree")
    args = parser.parse_args(argv)

    if args.command == "summary":
        try:
            text, ok = summarise(
                parse_results(sys.stdin.read()), args.patterns, lambda module: generated_mutants(args.mutants_dir, module)
            )
        except (OSError, ValueError) as exc:  # json.JSONDecodeError is a ValueError
            print(f"mutation_changed: cannot read mutmut's meta files: {exc}", file=sys.stderr)
            return 2
        print(text)
        return 0 if ok else 1

    try:
        root: Path = repo_root()
        names = diff_against_base(root, args.base, ["--name-only", "-z", "--diff-filter=AMR"], (SRC_ROOT,))
        pyproject = (root / "pyproject.toml").read_text(encoding="utf-8")
        excluded, sources = do_not_mutate(pyproject), source_paths(pyproject)
    except (GitQueryError, OSError, tomllib.TOMLDecodeError) as exc:
        print(f"mutation_changed: cannot run: {exc}", file=sys.stderr)
        return 2
    for pattern in module_patterns(names.split("\0"), excluded, sources):
        print(pattern)
    return 0


if __name__ == "__main__":
    sys.exit(main())
