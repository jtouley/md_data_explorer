# SPDX-License-Identifier: MIT
"""Recorded question-path outputs: the plan, the hold decision and the shown answer of every
question in the golden, hard, reviewer, adversarial, negation, over-holding and reverse suites.

The snapshot was recorded on main before the question-path speedups (value index, identifier
skip, projected execution). test_question_path_equivalence.py replays every question and
requires the same output, so a speedup can never change a plan, a hold reason or an answer.

Regenerate only when a change is meant to alter outputs, and say why in the PR:
    uv run python -m tests.eval.question_path_snapshot
"""

from __future__ import annotations

import json
import math
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import polars as pl
import yaml

from clinical_analytics.eval.catalog import DATASET_FILES, questions_for_dataset, repo_root
from clinical_analytics.eval.hard_questions import all_hard_questions
from clinical_analytics.eval.headless import ask, register_frame

SNAPSHOT_PATH = Path(__file__).with_name("question_path_snapshot.json")
_COHORT = "tests/fixtures/synthetic/cohort.csv"
_REGIONS = ["North", "North East", "South"]
_DIABETES = ["Type 1", "Type 2", "None"]


def golden_frame() -> pl.DataFrame:
    """The frame tests/eval/test_golden_questions.py builds its layer from."""
    return pl.DataFrame(
        {
            "patient_id": list(range(1, 101)),
            "age": [25 + i % 40 for i in range(100)],
            "gender": ["M" if i % 2 == 0 else "F" for i in range(100)],
            "status": ["active" if i % 3 != 0 else "inactive" for i in range(100)],
            "treatment": ["A" if i % 2 == 0 else "B" for i in range(100)],
            "cholesterol": [150 + i % 100 for i in range(100)],
            "LDL": [100 + i % 80 for i in range(100)],
            "BMI": [20 + i % 15 for i in range(100)],
            "statin": [i % 6 for i in range(100)],
        }
    )


def region_frame(size: int = 60) -> pl.DataFrame:
    """The overlapping-values frame of tests/eval/test_reviewer_probes.py."""
    return pl.DataFrame(
        {
            "patient_id": [f"Q{index:03d}" for index in range(size)],
            "region": [_REGIONS[index % 3] for index in range(size)],
            "diabetes status": [_DIABETES[(index // 3) % 3] for index in range(size)],
            "age": [30 + (index * 7) % 60 for index in range(size)],
            "ldl": [80.0 + (index * 13) % 100 for index in range(size)],
        }
    )


_REGION_QUERIES = (
    "how many patients in North East?",
    "how many patients in the North East region?",
    "how many patients with type 2 diabetes?",
    "how many patients with type diabetes?",
)


def suite_questions() -> list[tuple[str, str, str]]:
    """(suite, dataset id, question) for every question, in a fixed order. Registers the frames."""
    from tests.eval.adversarial_probes import own_frame, probes
    from tests.eval.negation_probes import PROBES as NEGATION_PROBES
    from tests.eval.negation_probes import codes_frame, review_frame
    from tests.eval.over_holding_probes import ANSWERABLE, HELD, covid_frame, creatinine_frame, lab_tests_frame
    from tests.eval.reverse_probes import reverse_frame, reverse_probes
    from tests.eval.reviewer_probes import PROBES as REVIEWER_PROBES

    root = repo_root()
    cohort = pl.read_csv(root / _COHORT)
    own = own_frame()
    frames = {
        "snap_golden": golden_frame(),
        "snap_regions": region_frame(),
        "snap_coh": cohort,
        "snap_own": own,
        "snap_codes": codes_frame(),
        "snap_rev": review_frame(),
        "snap_creat": creatinine_frame(),
        "snap_cov": covid_frame(),
        "snap_tests": lab_tests_frame(),
        "snap_reverse": reverse_frame(),
    }
    for name, frame in frames.items():
        register_frame(name, frame)

    golden = yaml.safe_load((root / "tests" / "eval" / "golden_questions.yaml").read_text())["questions"]
    found: list[tuple[str, str, str]] = [("golden", "snap_golden", item["query"]) for item in golden]
    for dataset_id in DATASET_FILES:
        found += [("golden_headless", dataset_id, item["query"]) for item in questions_for_dataset(dataset_id, root)]
    found += [("hard", item["dataset_id"], item["query"]) for item in all_hard_questions()]
    found += [("reviewer", "cohort", probe.query) for probe in REVIEWER_PROBES]
    found += [("reviewer", "snap_regions", query) for query in _REGION_QUERIES]
    found += [("adversarial", f"snap_{dataset}", query) for dataset, query, _truth in probes(cohort, own)]
    found += [("negation", f"snap_{probe.dataset}", probe.query) for probe in NEGATION_PROBES]
    found += [("over_holding", f"snap_{probe.dataset}", probe.query) for probe in (*ANSWERABLE, *HELD)]
    found += [("reverse", "snap_reverse", query) for query, _truth in reverse_probes(frames["snap_reverse"])]
    return found


def _plain(value: Any) -> Any:
    """JSON-stable form: tuples as lists, floats rounded so the platform's last bit cannot differ."""
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_plain(item) for item in value]
    if isinstance(value, float):
        return None if math.isnan(value) else round(value, 9)
    if value is None or isinstance(value, str | int | bool):
        return value
    return str(value)


def record(dataset_id: str, query: str) -> dict[str, Any]:
    """What the question path produced: plan, hold decision and the answer the app shows."""
    result = ask(dataset_id, query, repo_root())
    return _plain(
        {
            "intent": result["intent"],
            "metric": result["metric"],
            "group_by": result["group_by"],
            "limits": result["limits"],
            "proportion": result["proportion"],
            "rate": result["rate"],
            "confidence": result["confidence"],
            "held": result["held"],
            "success": result["success"],
            "failure_reason": result["failure_reason"],
            "plan_line": result["plan_line"],
            "issues": [item.get("message") for item in result["issues"]],
            "presentation": result["presentation"],
        }
    )


def records() -> Iterator[tuple[str, dict[str, Any]]]:
    for suite, dataset_id, query in suite_questions():
        yield f"{suite}|{dataset_id}|{query}", record(dataset_id, query)


def main() -> int:
    snapshot = dict(records())
    SNAPSHOT_PATH.write_text(json.dumps(snapshot, indent=1, sort_keys=True) + "\n")
    print(f"wrote {len(snapshot)} records to {SNAPSHOT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
