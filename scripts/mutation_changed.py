#!/usr/bin/env python3
"""Mutation testing scoped to the src modules a branch changed (``make mutation-changed``).

``patterns``: print one mutmut name pattern (``clinical_analytics.core.schema.*``) per mutable
src module added, modified or renamed against the merge base with ``--base``. Modules excluded by
``[tool.mutmut] do_not_mutate`` in pyproject.toml are left out. No output means nothing to mutate.

``summary PATTERN...``: read ``mutmut results --all true`` on stdin, keep the mutants matching the
patterns and print per-module counts plus every mutant not killed. Exits 1 when any mutant
survived, had no covering test or was never checked: kill it with a test, or, for an equivalent
mutant, add ``# pragma: no mutate`` to its line and say why there. Timeouts and suspicious
mutants are listed but do not fail.

Exit codes: 0 all killed, 1 unkilled mutants (or no results for a pattern), 2 the step could not run.
"""

from __future__ import annotations

import argparse
import fnmatch
import re
import sys
import tomllib
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Final

if __package__:  # imported as scripts.mutation_changed (tests)
    from scripts.git_tracked import GitQueryError, diff_against_base, repo_root
else:  # run as a file: scripts/ is sys.path[0]
    from git_tracked import GitQueryError, diff_against_base, repo_root

DEFAULT_BASE: Final = "origin/main"
SRC_ROOT: Final = "src/"
FAILING_STATUSES: Final = frozenset({"survived", "no tests", "not checked"})
RESULT_RE: Final = re.compile(r"^\s*(\S+): (.+?)\s*$")


def do_not_mutate(pyproject_text: str) -> tuple[str, ...]:
    """``[tool.mutmut] do_not_mutate`` globs, so the scope matches ``make mutation``."""
    config = tomllib.loads(pyproject_text).get("tool", {}).get("mutmut", {})
    return tuple(config.get("do_not_mutate", ()))


def module_patterns(paths: Iterable[str], excluded: Iterable[str]) -> list[str]:
    """Sorted mutmut patterns for the mutable ``src/**.py`` paths among ``paths``."""
    excluded = tuple(excluded)
    modules = {
        path.removeprefix(SRC_ROOT).removesuffix(".py").replace("/", ".")
        for path in paths
        if path.startswith(SRC_ROOT) and path.endswith(".py") and not any(fnmatch.fnmatch(path, g) for g in excluded)
    }
    return [f"{module}.*" for module in sorted(modules)]


def parse_results(text: str) -> dict[str, str]:
    """mutant name -> status from ``mutmut results`` lines ("    name: status")."""
    return {m.group(1): m.group(2) for m in map(RESULT_RE.match, text.splitlines()) if m}


def summarise(results: dict[str, str], patterns: list[str]) -> tuple[str, bool]:
    """Report text and whether every matched mutant was killed (and every pattern matched one)."""
    lines: list[str] = []
    ok = True
    totals: Counter[str] = Counter()
    for pattern in patterns:
        matched = {name: status for name, status in results.items() if fnmatch.fnmatchcase(name, pattern)}
        counts = Counter(matched.values())
        totals.update(counts)
        module = pattern.removesuffix(".*")
        if not matched:
            lines.append(f"  {module}: no mutants in the results (did the run finish?)")
            ok = False
            continue
        detail = ", ".join(f"{n} {status}" for status, n in sorted(counts.items()))
        lines.append(f"  {module}: {len(matched)} mutants: {detail}")
        for name, status in sorted(matched.items()):
            if status != "killed":
                lines.append(f"    {status}: {name}  (mutmut show {name})")
    unkilled = sum(n for status, n in totals.items() if status in FAILING_STATUSES)
    ok = ok and unkilled == 0
    total = sum(totals.values())
    head = f"mutation-changed: {'PASS' if ok else 'FAIL'}: {totals['killed']}/{total} killed, {unkilled} unkilled"
    return "\n".join([head, *lines]), ok


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    patterns_cmd = sub.add_parser("patterns", help="print a mutmut pattern per changed src module")
    patterns_cmd.add_argument("--base", default=DEFAULT_BASE)
    summary_cmd = sub.add_parser("summary", help="summarise `mutmut results --all true` from stdin")
    summary_cmd.add_argument("patterns", nargs="+")
    args = parser.parse_args(argv)

    if args.command == "summary":
        text, ok = summarise(parse_results(sys.stdin.read()), args.patterns)
        print(text)
        return 0 if ok else 1

    try:
        root: Path = repo_root()
        names = diff_against_base(root, args.base, ["--name-only", "-z", "--diff-filter=AMR"], (SRC_ROOT,))
        excluded = do_not_mutate((root / "pyproject.toml").read_text(encoding="utf-8"))
    except (GitQueryError, OSError, tomllib.TOMLDecodeError) as exc:
        print(f"mutation_changed: cannot run: {exc}", file=sys.stderr)
        return 2
    for pattern in module_patterns(names.split("\0"), excluded):
        print(pattern)
    return 0


if __name__ == "__main__":
    sys.exit(main())
