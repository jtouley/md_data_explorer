#!/usr/bin/env python3
"""Ask a golden-suite dataset from the command line.

Examples:
    uv run python scripts/headless_ask.py --dataset gdsi --query "how many patients?"
    uv run python scripts/headless_ask.py --suite
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from clinical_analytics.eval.catalog import DATASET_TABLES, questions_for_dataset  # noqa: E402
from clinical_analytics.eval.hard_questions import all_hard_questions, oracle_value  # noqa: E402
from clinical_analytics.eval.headless import actual_value, ask, values_match  # noqa: E402


def run_suite(workspace: Path) -> int:
    failures: list[str] = []
    checked = 0
    for dataset_id in DATASET_TABLES:
        for question in questions_for_dataset(dataset_id, workspace):
            checked += 1
            result = ask(dataset_id, question["query"], workspace)
            actual = actual_value(
                question["kind"],
                question.get("stat"),
                question.get("group_by"),
                result.get("payload"),
            )
            if not result["success"] or not values_match(question["expected"], actual):
                failures.append(
                    f"{question['id']} query={question['query']!r} "
                    f"expected={question['expected']!r} actual={actual!r} "
                    f"intent={result['intent']} metric={result['metric']} "
                    f"group_by={result['group_by']} issues={result['issues']}"
                )
    print(f"checked {checked} questions, failures {len(failures)}")
    for line in failures[:30]:
        print(line)
    return 1 if failures else 0


def run_hard_suite(workspace: Path) -> int:
    failures: list[str] = []
    checked = 0
    for question in all_hard_questions():
        checked += 1
        expected = oracle_value(question, workspace)
        result = ask(question["dataset_id"], question["query"], workspace)
        actual = actual_value(
            question["kind"],
            question.get("stat"),
            question.get("group_by"),
            result.get("payload"),
        )
        if not result["success"] or not values_match(expected, actual):
            failures.append(
                f"{question['id']} query={question['query']!r} "
                f"expected={expected!r} actual={actual!r} "
                f"intent={result['intent']} metric={result['metric']} "
                f"group_by={result['group_by']} issues={result['issues']}"
            )
    print(f"checked {checked} hard questions, failures {len(failures)}")
    for line in failures[:40]:
        print(line)
    return 1 if failures else 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Headless golden-question ask")
    parser.add_argument("--dataset", choices=sorted(DATASET_TABLES))
    parser.add_argument("--query")
    parser.add_argument("--suite", action="store_true", help="Run all 100 questions per dataset")
    parser.add_argument(
        "--suite-hard",
        action="store_true",
        help="Run 50 analytical and 50 provider questions per dataset",
    )
    parser.add_argument("--workspace", type=Path, default=ROOT)
    args = parser.parse_args()
    if args.suite:
        return run_suite(args.workspace)
    if args.suite_hard:
        return run_hard_suite(args.workspace)
    if not args.dataset or not args.query:
        parser.error("--dataset and --query are required unless --suite is set")
    result = ask(args.dataset, args.query, args.workspace)
    printable = {key: value for key, value in result.items() if key != "payload"}
    payload = result.get("payload") or {}
    raw = payload.get("result")
    if raw is not None and hasattr(raw, "to_dict"):
        printable["rows"] = raw.to_dict(orient="records")
    elif isinstance(raw, int | float):
        printable["rows"] = [{"count": raw}]
    print(json.dumps(printable, default=str, indent=2))
    return 0 if result["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
