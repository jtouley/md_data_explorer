#!/usr/bin/env python3
"""Definition of done before review (``make done-check``): run each gate, print a PR-ready summary.

Runs ``make check-fast``, ``make test-fast``, ``make mutation-changed`` and ``make pr-size`` in
order, streaming their output, and then prints a Markdown block to paste into the pull request:
one row per step (PASS, FAIL or SKIP) with its key line, the commit and the base. Every step runs
even after a failure, so the summary is complete. Exit 0 only when no step failed.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Final, Literal

Result = Literal["PASS", "FAIL", "SKIP"]

STEPS: Final[tuple[str, ...]] = ("check-fast", "test-fast", "mutation-changed", "pr-size")
# The line of each step's output that goes into the summary (last match wins).
KEY_LINE: Final[dict[str, re.Pattern[str]]] = {
    "check-fast": re.compile(r"All fast checks passed|check_eval_tokens: .*|\*\*\*.*Error.*"),
    "test-fast": re.compile(r"=+ .*\b(passed|failed|error|errors|no tests ran)\b.* in [\d.]+s.* =+"),
    "mutation-changed": re.compile(r"^mutation-changed: .*"),
    "pr-size": re.compile(r"^pr-size: .*"),
}
SKIP_MARK: Final = "SKIP"
ANSI_RE: Final = re.compile(r"\x1b\[[0-9;]*m")


@dataclass(frozen=True)
class StepOutcome:
    step: str
    result: Result
    detail: str
    notes: tuple[str, ...] = ()  # lines printed under the table (the unkilled mutants)


def key_line(step: str, output: str) -> str:
    """The last output line matching the step's KEY_LINE pattern, without colour codes or "=" rules."""
    pattern = KEY_LINE.get(step)
    if pattern is None:
        return ""
    lines = [ANSI_RE.sub("", line).strip() for line in output.splitlines()]
    matches = [line.strip("= ") for line in lines if pattern.search(line)]
    return matches[-1] if matches else ""


def mutation_notes(output: str) -> tuple[str, ...]:
    """The per-module lines that follow the mutation-changed head line."""
    lines = [ANSI_RE.sub("", line).rstrip() for line in output.splitlines()]
    heads = [i for i, line in enumerate(lines) if line.startswith("mutation-changed: ")]
    if not heads:
        return ()
    return tuple(line for line in lines[heads[-1] + 1 :] if line.startswith("  "))


def classify(step: str, exit_code: int, output: str) -> StepOutcome:
    detail = key_line(step, output)
    notes = mutation_notes(output) if step == "mutation-changed" else ()
    if exit_code != 0:
        return StepOutcome(step, "FAIL", detail or f"exit {exit_code}", notes)
    if step == "mutation-changed" and f": {SKIP_MARK}" in detail:
        return StepOutcome(step, "SKIP", detail, notes)
    return StepOutcome(step, "PASS", detail, notes)


def render(outcomes: Sequence[StepOutcome], head: str, base: str) -> str:
    overall = "FAIL" if any(o.result == "FAIL" for o in outcomes) else "PASS"
    rows = [f"| `make {o.step}` | {o.result} | {o.detail.replace('|', '/')} |" for o in outcomes]
    return "\n".join(
        [
            "### make done-check",
            "",
            f"Commit `{head}` against `{base}`. Overall: **{overall}**",
            "",
            "| Step | Result | Detail |",
            "|---|---|---|",
            *rows,
            *(["", "```", *notes, "```"] if (notes := [n for o in outcomes for n in o.notes]) else []),
        ]
    )


def run_step(make: str, step: str, extra: Sequence[str]) -> tuple[int, str]:
    """Run ``make <step>``, echoing its output live, and return (exit status, output)."""
    proc = subprocess.Popen(
        [make, "--no-print-directory", step, *extra],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
    )
    assert proc.stdout is not None  # stdout=PIPE
    captured: list[str] = []
    for line in proc.stdout:
        sys.stdout.write(line)
        captured.append(line)
    return proc.wait(), "".join(captured)


def _git_head() -> str:
    out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=False)
    dirty = subprocess.run(["git", "status", "--porcelain", "-uno"], capture_output=True, text=True, check=False)
    return out.stdout.strip() + (" + uncommitted changes" if dirty.stdout.strip() else "")


def main(argv: list[str] | None = None, runner: Callable[[str, str, Sequence[str]], tuple[int, str]] = run_step) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--make", default="make")
    parser.add_argument("--base", default="origin/main")
    parser.add_argument("make_args", nargs="*", help="extra VAR=value arguments passed to every step")
    args = parser.parse_args(argv)

    outcomes = []
    for step in STEPS:
        print(f"\n===== done-check: make {step} =====", flush=True)
        status, output = runner(args.make, step, args.make_args)
        outcomes.append(classify(step, status, output))
    print()
    print(render(outcomes, _git_head(), args.base))
    return 1 if any(o.result == "FAIL" for o in outcomes) else 0


if __name__ == "__main__":
    sys.exit(main())
