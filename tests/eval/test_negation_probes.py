"""Adversarial regression set: each question gets its exact plan and true value, or a hold.

Graded on what the app shows. A probe expecting a hold that runs is wrong; a probe expecting a
plan that runs with other limits, or shows another value, is wrong. 0 wrong is the target.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import polars as pl
import pytest

from clinical_analytics.core.query_plan import CohortStep, FilterSpec, QueryPlan
from clinical_analytics.core.word_accounting import unjustified_elements, unused_words
from clinical_analytics.eval.catalog import repo_root
from clinical_analytics.eval.headless import ask, classify_presented_generic, register_frame
from tests.eval.adversarial_probes import own_frame
from tests.eval.negation_probes import PLAN_PROBES, PROBES, ScopeProbe, codes_frame, review_frame


@lru_cache(maxsize=1)
def _frames() -> dict[str, pl.DataFrame]:
    frames = {
        "coh": pl.read_csv(repo_root() / "tests" / "fixtures" / "synthetic" / "cohort.csv"),
        "own": own_frame(),
        "codes": codes_frame(),
        "rev": review_frame(),
    }
    for name, frame in frames.items():
        register_frame(f"negscope_{name}", frame)
    return frames


def grade(probe: ScopeProbe, result: dict[str, Any], frame: pl.DataFrame) -> tuple[str, Any]:
    """(outcome, expected value) of one probe's result: answered, hidden, held, refused or wrong."""
    if not result["success"]:
        return ("held" if result["held"] else "refused"), None
    if probe.limits is None or probe.truth is None:
        return "wrong", None
    if tuple(result["limits"]) != probe.limits:
        return "wrong", probe.limits
    expected = probe.truth(frame)
    return classify_presented_generic(expected, result), expected


def outcome(probe: ScopeProbe) -> tuple[str, dict[str, Any], Any]:
    """(graded outcome, result, expected) for one probe, through the display path."""
    frame = _frames()[probe.dataset]
    result = ask(f"negscope_{probe.dataset}", probe.query)
    graded, expected = grade(probe, result, frame)
    return graded, result, expected


def test_negationProbes_set_coversEveryRequiredFamily() -> None:
    # Arrange
    keys = [(probe.dataset, probe.query) for probe in PROBES]
    queries = [probe.query.lower() for probe in PROBES]

    # Act
    duplicates = {key for key in keys if keys.count(key) > 1}
    holds = sum(1 for probe in PROBES if probe.limits is None)

    # Assert
    assert len(PROBES) >= 60
    assert duplicates == set()
    assert 0 < holds < len(PROBES)
    assert any(" or " in query and ("share" in query or "proportion" in query) for query in queries)
    assert any("more likely" in query and " or " in query for query in queries)
    assert any(probe.dataset == "codes" for probe in PROBES)
    assert any(probe.dataset == "rev" for probe in PROBES)


def _plan(spec: dict[str, Any]) -> QueryPlan:
    proportion = spec.get("proportion")
    return QueryPlan(
        intent=spec.get("intent", "COUNT"),
        metric=spec.get("metric"),
        group_by=spec.get("group_by"),
        statistic=spec.get("statistic"),
        filters=[FilterSpec(column, operator, value) for operator, column, value in spec.get("filters", [])],
        cohort_steps=[CohortStep(op, column, value) for op, column, value in spec.get("steps", [])],
        proportion_column=proportion[0] if proportion else None,
        proportion_value=proportion[1] if proportion else None,
        confidence=0.9,
    )


@pytest.mark.parametrize(
    ("dataset", "question", "spec"),
    [pytest.param(*case, id=f"{case[0]}:{case[1]}:{index}") for index, case in enumerate(PLAN_PROBES)],
)
def test_gate_reviewedWrongPlan_isHeld(dataset: str, question: str, spec: dict[str, Any]) -> None:
    # Arrange
    frame = _frames()[dataset]
    plan = _plan(spec)

    def values_for(column: str) -> list[Any]:
        return frame.get_column(column).unique().to_list()

    # Act
    unused = unused_words(question, plan, frame.columns, values_for)
    unasked = unjustified_elements(question, plan, frame.columns, values_for)

    # Assert
    assert unused or unasked, f"{question!r} ran with {spec!r}"


@pytest.mark.parametrize(
    ("probe", "result", "expected"),
    [
        pytest.param(
            ScopeProbe("coh", "q", None), {"success": True, "held": False}, "wrong", id="hold_expected_but_ran"
        ),
        pytest.param(ScopeProbe("coh", "q", None), {"success": False, "held": True}, "held", id="held"),
        pytest.param(ScopeProbe("coh", "q", None), {"success": False, "held": False}, "refused", id="refused"),
        pytest.param(
            ScopeProbe("coh", "q", (("gt", "age", 65.0),), lambda frame: 3),
            {"success": True, "held": False, "limits": [("le", "age", 65.0)]},
            "wrong",
            id="inverted_limit",
        ),
        pytest.param(
            ScopeProbe("coh", "q", (("gt", "age", 65.0),), lambda frame: 3),
            {
                "success": True,
                "held": False,
                "limits": [("gt", "age", 65.0)],
                "presentation": {"claim": "Total count: 3.", "table": None},
            },
            "answered",
            id="exact_plan_true_value",
        ),
    ],
)
def test_grade_result_flagsEveryWrongPlan(probe: ScopeProbe, result: dict[str, Any], expected: str) -> None:
    # Arrange
    frame = pl.DataFrame({"age": [70]})

    # Act
    graded, _value = grade(probe, result, frame)

    # Assert
    assert graded == expected


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.parametrize("probe", [pytest.param(probe, id=f"{probe.dataset}:{probe.query}") for probe in PROBES])
def test_negationProbe_question_exactPlanOrHold(probe: ScopeProbe) -> None:
    # Arrange: probe above

    # Act
    graded, result, expected = outcome(probe)

    # Assert
    wanted = {"held", "refused"} if probe.limits is None else {"answered"}
    assert graded in wanted, (
        f"{probe.query!r}: {graded}, expected {expected!r}, limits {result.get('limits')!r}, "
        f"shown {(result.get('presentation') or {}).get('claim')!r}, reason {result.get('failure_reason')!r}"
    )
