#!/usr/bin/env python3
"""Ask a golden-suite dataset from the command line (dev tool, not a production surface).

Datasets are the committed synthetic cohorts in tests/fixtures/synthetic. Each question
is classified as answered (matches ground truth), refused (success False), or
wrong-success (success True with a value that does not match). Wrong-success is the
failure that matters: it is a silently wrong answer.

Examples:
    uv run python scripts/headless_ask.py --dataset cohort --query "how many patients?"
    uv run python scripts/headless_ask.py --suite
    uv run python scripts/headless_ask.py --suite-hard
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from clinical_analytics.eval.catalog import DATASET_FILES, questions_for_dataset  # noqa: E402
from clinical_analytics.eval.hard_questions import all_hard_questions, oracle_value  # noqa: E402
from clinical_analytics.eval.headless import _result_frame, ask, classify  # noqa: E402


def _report(label: str, outcomes: list[tuple[dict[str, Any], str, Any, dict[str, Any]]]) -> int:
    counts = {"answered": 0, "held": 0, "refused": 0, "wrong": 0}
    for _question, outcome, _actual, _result in outcomes:
        counts[outcome] += 1
    print(f"{label}: checked {len(outcomes)}, " + ", ".join(f"{key} {value}" for key, value in counts.items()))
    for question, outcome, actual, result in outcomes:
        if outcome == "answered":
            continue
        reasons = [str(item.get("message")) for item in result["issues"] if item.get("severity") == "error"]
        print(
            f"  [{outcome}] {question['id']} query={question['query']!r} expected={question['expected']!r} "
            f"actual={actual!r} intent={result['intent']} metric={result['metric']} "
            f"group_by={result['group_by']} reasons={reasons[:2]}"
        )
    return 1 if counts["wrong"] else 0


def run_suite(workspace: Path) -> int:
    outcomes = []
    for dataset_id in DATASET_FILES:
        for question in questions_for_dataset(dataset_id, workspace):
            result = ask(dataset_id, question["query"], workspace)
            outcome, actual = classify(question, question["expected"], result)
            outcomes.append((question, outcome, actual, result))
    return _report("golden", outcomes)


def run_hard_suite(workspace: Path) -> int:
    outcomes = []
    for question in all_hard_questions():
        expected = oracle_value(question, workspace)
        result = ask(question["dataset_id"], question["query"], workspace)
        outcome, actual = classify(question, expected, result)
        outcomes.append(({**question, "expected": expected}, outcome, actual, result))
    return _report("hard", outcomes)


def main() -> int:
    parser = argparse.ArgumentParser(description="Headless golden-question ask")
    parser.add_argument("--dataset", choices=sorted(DATASET_FILES))
    parser.add_argument("--query")
    parser.add_argument("--suite", action="store_true", help="Run the generated golden questions")
    parser.add_argument("--suite-hard", action="store_true", help="Run the SQL-oracle hard questions")
    parser.add_argument("--workspace", type=Path, default=ROOT)
    args = parser.parse_args()
    if args.suite or args.suite_hard:
        status = 0
        if args.suite:
            status |= run_suite(args.workspace)
        if args.suite_hard:
            status |= run_hard_suite(args.workspace)
        return status
    if not args.dataset or not args.query:
        parser.error("--dataset and --query are required unless --suite or --suite-hard is set")
    result = ask(args.dataset, args.query, args.workspace)
    printable = {key: value for key, value in result.items() if key != "payload"}
    frame = _result_frame(result.get("payload"))
    if frame is not None:
        printable["rows"] = frame.head(50).to_dicts()
    print(json.dumps(printable, default=str, indent=2))
    return 0 if result["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
