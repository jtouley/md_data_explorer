#!/usr/bin/env python3
"""Mutation testing scoped to the src functions a branch changed (``make mutation-changed``).

``patterns``: print one mutmut name pattern per function or method the branch changed against
the merge base with ``--base`` (``clinical_analytics.core.schema.x_load__mutmut_*``,
``clinical_analytics.core.schema.xǁSchemaǁvalidate__mutmut_*``). A function counts as changed when
a diff hunk (``git diff -U0``) adds or edits a line between its first decorator and its last
line, or removes one from it (read from the merge base's copy of the file), so a new module yields
every function in it. Lines outside any function (imports, constants, class attributes) are not
mutated by mutmut at all and yield nothing. Only modules under
``[tool.mutmut] source_paths`` and not excluded by ``do_not_mutate`` (pyproject.toml) count. No
output means nothing to mutate. ``--whole-modules`` prints ``<module>.*`` per changed module
instead (the old scope: every mutant in the file, which can take days).

``summary PATTERN...``: read ``mutmut results --all true`` on stdin, keep the mutants matching the
patterns and print per-pattern counts plus every mutant not killed. Only ``PASSING_STATUSES``
(killed, timeout, suspicious, caught by type check) pass; every other status fails, including
survived, no tests, not checked, skipped, segfault and "check was interrupted by user" (pytest's
exit 2, also returned for collection errors). Kill such a mutant with a test, or, for an
equivalent mutant, add ``# pragma: no mutate`` to its line and say why there.

A pattern with no results is told apart by mutmut's ``mutants/<path>.meta`` file, written when
mutants are generated: a meta file with no mutant matching the pattern (a decorated function
mutmut leaves alone, a body of ``pass``, every line ``# pragma: no mutate``) is a SKIP; no meta
file means the run never got to the module and fails. ``--run-exit-code`` is ``mutmut run``'s
exit code: mutmut exits non-zero when no mutant matches any pattern, which is a SKIP when every
pattern is one and a failure otherwise.

Exit codes: 0 all passed (or nothing to mutate), 1 unkilled mutants or a module without results,
2 the step could not run.
"""

from __future__ import annotations

import argparse
import ast
import fnmatch
import json
import re
import sys
import tomllib
from collections import Counter
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Final, NamedTuple

if __package__:  # imported as scripts.mutation_changed (tests)
    from scripts.git_tracked import GitQueryError, diff_against_base, file_at_merge_base, repo_root
else:  # run as a file: scripts/ is sys.path[0]
    from git_tracked import GitQueryError, diff_against_base, file_at_merge_base, repo_root

DEFAULT_BASE: Final = "origin/main"
SRC_ROOT: Final = "src/"
MUTANTS_DIR: Final = Path("mutants")
# mutmut statuses that count as tested; anything else (survived, no tests, not checked, skipped,
# segfault, check was interrupted by user, a status a later mutmut adds) fails the gate.
PASSING_STATUSES: Final = frozenset({"killed", "timeout", "suspicious", "caught by type check"})
RESULT_RE: Final = re.compile(r"^\s*(\S+): (.+?)\s*$")
HUNK_RE: Final = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
# mutmut's mangled names (mutmut/mutation/trampoline_templates.py): x_<function>, xǁ<Class>ǁ<method>.
CLASS_SEP: Final = "ǁ"


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


def mutable_paths(paths: Iterable[str], excluded: Iterable[str], sources: Iterable[str] = (SRC_ROOT,)) -> list[str]:
    """Sorted unique mutable ``src/**.py`` paths among ``paths``: under ``sources``, not ``excluded``."""
    excluded, sources = tuple(excluded), tuple(sources)
    return sorted(
        {
            path
            for path in paths
            if path.startswith(SRC_ROOT)
            and path.endswith(".py")
            and any(_under(path, root) for root in sources)
            and not any(fnmatch.fnmatch(path, g) for g in excluded)
        }
    )


def module_name(path: str) -> str:
    """``src/clinical_analytics/core/schema.py`` -> ``clinical_analytics.core.schema``."""
    return path.removeprefix(SRC_ROOT).removesuffix(".py").replace("/", ".")


def module_patterns(paths: Iterable[str], excluded: Iterable[str], sources: Iterable[str] = (SRC_ROOT,)) -> list[str]:
    """Sorted whole-module mutmut patterns for the mutable ``src/**.py`` paths among ``paths``."""
    return [f"{module_name(path)}.*" for path in mutable_paths(paths, excluded, sources)]


class FileChange(NamedTuple):
    """One file's ``git diff -U0`` hunks: the lines removed or edited on the old side, the lines
    added or edited on the new side, and the old path (None for a file the branch added)."""

    old_path: str | None
    old_lines: frozenset[int]
    new_lines: frozenset[int]


def _header_path(header: str) -> str | None:
    name = header[4:].removesuffix("\t")
    return None if name == "/dev/null" else name[2:]  # strip the fixed a/ or b/ prefix


def changed_lines(diff: str) -> dict[str, FileChange]:
    """Each file's :class:`FileChange` keyed by its new path, from ``git diff -U0`` output run with
    ``--src-prefix=a/ --dst-prefix=b/``.

    Hunk ``@@ -a,b +c,d @@`` removes old lines a..a+b-1 and adds new lines c..c+d-1 (a count of 0
    is an empty run). A deleted file and a pure rename (no hunk) are left out. File headers are
    read only between ``diff --git`` and the first hunk, so an added line that reads ``++ x``
    (``+++ x`` in the diff) is never taken for one.
    """
    changes: dict[str, FileChange] = {}
    old_path: str | None = None
    new_path: str | None = None
    in_header = False
    for line in diff.splitlines():
        if line.startswith("diff --git "):
            in_header, old_path, new_path = True, None, None
        elif in_header and line.startswith("--- "):
            old_path = _header_path(line)
        elif in_header and line.startswith("+++ "):
            new_path = _header_path(line)
        elif new_path is not None and (hunk := HUNK_RE.match(line)):
            in_header = False
            a, b, c, d = (int(g) if g is not None else 1 for g in hunk.groups())
            prev = changes.get(new_path, FileChange(old_path, frozenset(), frozenset()))
            changes[new_path] = FileChange(
                old_path, prev.old_lines | set(range(a, a + b)), prev.new_lines | set(range(c, c + d))
            )
    return changes


def function_spans(source: str) -> list[tuple[str, int, int]]:
    """(mutmut mangled name, first line, last line) of every function mutmut mutates in ``source``.

    Module-level functions are ``x_<name>``; methods of classes, nested classes included, are
    ``xǁ<Class>ǁ<name>`` with nested class names joined by ".", as mutmut mangles them. A function
    defined inside another one is part of the outer one's span, as mutmut mutates it. The first
    line is the first decorator's. Raises SyntaxError for a file Python cannot parse.
    """
    spans: list[tuple[str, int, int]] = []

    def visit(body: list[ast.stmt], class_name: str) -> None:
        for node in body:
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                prefix = f"x{CLASS_SEP}{class_name}{CLASS_SEP}" if class_name else "x_"
                first = min([node.lineno, *(d.lineno for d in node.decorator_list)])
                spans.append((f"{prefix}{node.name}", first, node.end_lineno or node.lineno))
            elif isinstance(node, ast.ClassDef):
                visit(node.body, f"{class_name}.{node.name}" if class_name else node.name)

    visit(ast.parse(source).body, "")
    return spans


def touched_functions(source: str, lines: Iterable[int]) -> set[str]:
    """Mangled names of the functions in ``source`` whose span holds one of ``lines``."""
    lines = frozenset(lines)
    return {name for name, first, last in function_spans(source) if any(first <= n <= last for n in lines)}


def function_patterns(
    module: str, source: str, lines: Iterable[int], old_source: str | None = None, old_lines: Iterable[int] = ()
) -> list[str]:
    """mutmut patterns for the functions of ``module`` (text ``source``) the change touches.

    A function counts when one of the new-side ``lines`` falls in it, or one of the old-side
    ``old_lines`` fell in a function of ``old_source`` that still exists under the same name (a
    line deleted from a body, the function's last line included). A function deleted outright
    yields nothing: there is nothing left to mutate.
    """
    names = touched_functions(source, lines)
    if old_source is not None:
        current = {name for name, _first, _last in function_spans(source)}
        names |= touched_functions(old_source, old_lines) & current
    return [f"{module}.{name}__mutmut_*" for name in sorted(names)]


def pattern_module(pattern: str) -> str:
    """The module a pattern scopes: ``m.*``, ``m.x_f__mutmut_*``, ``m.xǁA.Bǁf__mutmut_*`` -> ``m``.

    Nested class names hold dots, so a method pattern is cut at its ``.xǁ``; the other two have
    no dot after the module.
    """
    if CLASS_SEP in pattern:
        return pattern.split(f".x{CLASS_SEP}", 1)[0]
    return pattern.rpartition(".")[0]


def generated_mutants(mutants_dir: Path, pattern: str) -> int | None:
    """How many mutants matching ``pattern`` mutmut generated; None when the module has no meta file yet.

    mutmut writes ``mutants/src/<module path>.py.meta`` for every file it mutates, with an empty
    ``exit_code_by_key`` when the file holds nothing mutable.
    """
    module = pattern_module(pattern)
    meta = mutants_dir / f"{SRC_ROOT}{module.replace('.', '/')}.py.meta"
    try:
        data = json.loads(meta.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    keys = data.get("exit_code_by_key") if isinstance(data, dict) else None
    if not isinstance(keys, dict):
        raise ValueError(f"{meta}: no 'exit_code_by_key' object; mutmut's meta format changed?")
    return sum(1 for key in keys if fnmatch.fnmatchcase(key, pattern))


def parse_results(text: str) -> dict[str, str]:
    """mutant name -> status from ``mutmut results`` lines ("    name: status")."""
    return {m.group(1): m.group(2) for m in map(RESULT_RE.match, text.splitlines()) if m}


def summarise(
    results: dict[str, str],
    patterns: list[str],
    generated: Callable[[str], int | None] = lambda _pattern: None,
    run_exit_code: int = 0,
) -> tuple[str, bool]:
    """Report text and whether every matched mutant passed (and every pattern matched one).

    ``generated(pattern)`` is how many mutants mutmut generated for a pattern without results: 0 is
    a pattern with nothing to mutate (SKIP), None one the run never reached (FAIL). A non-zero
    ``run_exit_code`` fails the summary unless every pattern is a SKIP (mutmut exits non-zero when
    no mutant matches any pattern).
    """
    lines: list[str] = []
    ok = True
    skipped = 0
    totals: Counter[str] = Counter()
    for pattern in patterns:
        matched = {name: status for name, status in results.items() if fnmatch.fnmatchcase(name, pattern)}
        counts = Counter(matched.values())
        totals.update(counts)
        scope = pattern.removesuffix(".*").removesuffix("__mutmut_*")
        if not matched:
            if generated(pattern) == 0:
                lines.append(f"  {scope}: SKIP: nothing to mutate (mutmut generated no mutants)")
                skipped += 1
            else:
                lines.append(f"  {scope}: no mutants in the results (did the run finish?)")
                ok = False
            continue
        detail = ", ".join(f"{n} {status}" for status, n in sorted(counts.items()))
        lines.append(f"  {scope}: {len(matched)} mutants: {detail}")
        for name, status in sorted(matched.items()):
            if status != "killed":
                lines.append(f"    {status}: {name}  (mutmut show {name})")
    unkilled = sum(n for status, n in totals.items() if status not in PASSING_STATUSES)
    ok = ok and unkilled == 0
    total = sum(totals.values())
    all_skipped = bool(patterns) and skipped == len(patterns)
    if run_exit_code and not all_skipped:
        lines.insert(0, f"  mutmut run exited {run_exit_code}")
        ok = False
    if ok and all_skipped:
        head = "mutation-changed: SKIP: no mutants in the changed functions"
    else:
        head = f"mutation-changed: {'PASS' if ok else 'FAIL'}: {totals['killed']}/{total} killed, {unkilled} unkilled"
    return "\n".join([head, *lines]), ok


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    patterns_cmd = sub.add_parser("patterns", help="print a mutmut pattern per changed src function")
    patterns_cmd.add_argument("--base", default=DEFAULT_BASE)
    patterns_cmd.add_argument("--whole-modules", action="store_true", help="one <module>.* pattern per changed module")
    summary_cmd = sub.add_parser("summary", help="summarise `mutmut results --all true` from stdin")
    summary_cmd.add_argument("patterns", nargs="+")
    summary_cmd.add_argument("--mutants-dir", type=Path, default=MUTANTS_DIR, help="mutmut's mutants/ tree")
    summary_cmd.add_argument("--run-exit-code", type=int, default=0, help="`mutmut run`'s exit code")
    args = parser.parse_args(argv)

    if args.command == "summary":
        try:
            text, ok = summarise(
                parse_results(sys.stdin.read()),
                args.patterns,
                lambda pattern: generated_mutants(args.mutants_dir, pattern),
                args.run_exit_code,
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
        paths = mutable_paths(names.split("\0"), do_not_mutate(pyproject), source_paths(pyproject))
        if args.whole_modules:
            patterns = [f"{module_name(path)}.*" for path in paths]
        else:
            # Same pathspec as the name list, so renames pair up; fixed prefixes whatever diff.noprefix says.
            hunk_args = ["-U0", "--diff-filter=AMR", "--src-prefix=a/", "--dst-prefix=b/"]
            changes = changed_lines(diff_against_base(root, args.base, hunk_args, (SRC_ROOT,))) if paths else {}
            patterns = []
            for path in paths:
                change = changes.get(path)
                if change is None:  # renamed without an edit
                    continue
                old = file_at_merge_base(root, args.base, change.old_path) if change.old_path else None
                source = (root / path).read_text(encoding="utf-8")
                patterns += function_patterns(module_name(path), source, change.new_lines, old, change.old_lines)
    except (GitQueryError, OSError, tomllib.TOMLDecodeError, SyntaxError, UnicodeDecodeError) as exc:
        print(f"mutation_changed: cannot run: {exc}", file=sys.stderr)
        return 2
    for pattern in patterns:
        print(pattern)
    return 0


if __name__ == "__main__":
    sys.exit(main())
