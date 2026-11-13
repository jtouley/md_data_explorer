"""Grade what the app shows, not the raw execution frame.

Every golden, hard, reviewer and adversarial question that the app answers goes through the
display path (QueryService.ask -> format_execution_result -> present_analysis) and the claim
and table are read the way a clinician reads them. Safety property: an answered question
shows the true value, or hides it for small cells; it never shows a different value.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from clinical_analytics.contracts.config import LEVEL_VARIANT_MAX_LEVELS
from clinical_analytics.core.nl_query_config import SMALL_CELL_THRESHOLD
from clinical_analytics.core.result_presentation import SECONDARY_SUPPRESSED
from clinical_analytics.eval.catalog import questions_for_dataset
from clinical_analytics.eval.hard_questions import all_hard_questions, oracle_value
from clinical_analytics.eval.headless import (
    HIDDEN,
    ask,
    classify_presented,
    classify_presented_generic,
    frame_for,
    presented_answer,
    register_frame,
    shown_labels,
    small_level_labels,
)
from tests.eval.adversarial_probes import own_frame, probes
from tests.eval.dq_probes import DQ_QUESTIONS, SHOWN_PROBES, dq_frames, fold_frame, visits_died_frame
from tests.eval.reviewer_probes import PROBES


def _held(result: dict[str, Any]) -> str:
    return "wrong" if result["success"] else ("held" if result["held"] else "refused")


class TestPresentedAnswer:
    @pytest.mark.parametrize(
        ("presentation", "expected"),
        [
            pytest.param({"claim": "Total count: 42.", "table": None}, 42, id="count"),
            pytest.param({"claim": "Total count: <11.", "table": None}, HIDDEN, id="small_count"),
            pytest.param(
                {"claim": "x", "table": {"columns": ["proportion", "n"], "rows": [{"proportion": 0.42, "n": 50}]}},
                0.42,
                id="proportion",
            ),
            pytest.param(
                {
                    "claim": "x",
                    "table": {"columns": ["proportion", "n"], "rows": [{"proportion": SECONDARY_SUPPRESSED, "n": 50}]},
                },
                HIDDEN,
                id="proportion_hidden",
            ),
            pytest.param(
                {
                    "claim": "x",
                    "table": {
                        "columns": ["sex", "rate", "n"],
                        "rows": [{"sex": "F", "rate": 0.5, "n": 20}, {"sex": "M", "rate": 0.25, "n": 40}],
                    },
                },
                {"F": 0.5, "M": 0.25},
                id="rates",
            ),
            pytest.param(
                {
                    "claim": "x",
                    "table": {
                        "columns": ["sex", "median age", "n"],
                        "rows": [{"sex": "F", "median age": 50.0, "n": 20}],
                    },
                },
                {"F": 50.0},
                id="grouped_median",
            ),
            pytest.param({"claim": None, "table": None}, None, id="nothing_shown"),
            pytest.param(None, None, id="no_presentation"),
        ],
    )
    def test_presentedAnswer_shownTable_readsValueAsUserDoes(self, presentation: Any, expected: Any) -> None:
        # Arrange / Act
        shown = presented_answer(presentation)

        # Assert
        assert shown == expected

    @pytest.mark.parametrize(
        ("result", "expected_outcome"),
        [
            pytest.param({"success": False, "held": True}, "held", id="held"),
            pytest.param({"success": False, "held": False}, "refused", id="refused"),
            pytest.param(
                {"success": True, "presentation": {"claim": "Total count: 0.", "table": None}},
                "wrong",
                id="proportion_shown_as_count",
            ),
            pytest.param(
                {
                    "success": True,
                    "presentation": {
                        "claim": "x",
                        "table": {"columns": ["proportion", "n"], "rows": [{"proportion": 0.42, "n": 50}]},
                    },
                },
                "answered",
                id="proportion_shown",
            ),
        ],
    )
    def test_classifyPresented_proportion_gradesShownValue(self, result: dict[str, Any], expected_outcome: str) -> None:
        # Arrange
        expected = 0.42

        # Act
        outcome = classify_presented("proportion", None, None, expected, result)

        # Assert
        assert outcome == expected_outcome


# Privacy probes (design review): the answer may be refused or held, never a table whose labels
# name levels held by 1-10 patients. On main the first showed 289 columns, one per exact LDL value.
PRIVACY_QUERIES = (
    "compare treatment between ldl groups",
    "count by ldl",
    "average age by ldl",
    "compare ldl between hdl groups",
)


def _label_leak(result: dict[str, Any], workspace: Path | None) -> str | None:
    """A shown label for a level with 1-10 patients in the dataset, or None."""
    if not result["success"]:
        return None
    frame = frame_for(result["dataset_id"], workspace)
    columns = [c for c in (result.get("group_by"), result.get("metric")) if c]
    leaked = shown_labels(result["presentation"]) & small_level_labels(frame, columns, SMALL_CELL_THRESHOLD)
    return f"{result['query']!r} shows labels of small levels: {sorted(leaked)[:5]}" if leaked else None


def _hold_leak(result: dict[str, Any], frame: pl.DataFrame) -> str | None:
    """A held question whose sentence names a level held by 1-10 rows, or None."""
    reason = str(result.get("failure_reason") or "")
    if not result["held"] or not reason:
        return None
    text_columns = [
        c
        for c, dtype in frame.schema.items()
        if dtype == pl.String and c != "patient_id" and frame.get_column(c).n_unique() <= LEVEL_VARIANT_MAX_LEVELS
    ]
    small = small_level_labels(frame, text_columns, SMALL_CELL_THRESHOLD)
    named = sorted(label for label in small if re.search(rf"(?<!\w){re.escape(label)}(?!\w)", reason))
    return f"hold names small levels {named}" if named else None


def _presented_outcomes(project_root: Path) -> tuple[dict[str, int], list[str]]:
    """Ask every question of the four suites and grade the presented answer.

    A presented label that names a level held by 1-10 patients counts as wrong too.
    """
    tally: dict[str, int] = {}
    wrong: list[str] = []

    def record(suite: str, query: str, outcome: str, expected: Any, result: dict[str, Any]) -> None:
        leak = _label_leak(result, project_root if suite in ("golden", "hard", "reviewer", "privacy") else None)
        if leak:
            outcome = "wrong"
        tally[outcome] = tally.get(outcome, 0) + 1
        if leak:
            wrong.append(f"{suite} {leak}")
        elif outcome == "wrong":
            wrong.append(f"{suite} {query!r} expected={expected!r} shown={presented_answer(result['presentation'])!r}")

    for question in questions_for_dataset("cohort", project_root):
        result = ask("cohort", question["query"], project_root)
        outcome = classify_presented(
            question["kind"], question.get("stat"), question.get("group_by"), question["expected"], result
        )
        record("golden", question["query"], outcome, question["expected"], result)

    for question in all_hard_questions():
        expected = oracle_value(question, project_root)
        result = ask(question["dataset_id"], question["query"], project_root)
        outcome = classify_presented(question["kind"], question.get("stat"), question.get("group_by"), expected, result)
        record("hard", question["query"], outcome, expected, result)

    cohort = pl.read_csv(project_root / "tests" / "fixtures" / "synthetic" / "cohort.csv")
    for probe in PROBES:
        result = ask("cohort", probe.query, project_root)
        if probe.kind == "hold" or probe.truth is None:
            record("reviewer", probe.query, _held(result), "hold", result)
            continue
        expected = probe.truth(cohort)
        record(
            "reviewer",
            probe.query,
            classify_presented(probe.kind, probe.stat, probe.group_by, expected, result),
            expected,
            result,
        )

    for query in PRIVACY_QUERIES:
        result = ask("cohort", query, project_root)
        record("privacy", query, _held(result) if not result["success"] else "answered", "no small labels", result)

    # Data-quality probe files: held or refused, never a number (see tests/eval/dq_probes.py),
    # and a hold sentence never names a level held by 1-10 rows.
    frames = dq_frames()
    for name, frame in frames.items():
        register_frame(f"dq_{name}", frame)
    for name, query in DQ_QUESTIONS:
        result = ask(f"dq_{name}", query)
        leak = _hold_leak(result, frames[name])
        record("dq", f"{name}: {query}", "wrong" if leak else _held(result), leak or "held or refused", result)

    # Shown-answer probes: the exact table or claim the app shows (dq_probes.SHOWN_PROBES).
    register_frame("dq_fold", fold_frame())
    register_frame("dq_visits_died", visits_died_frame())
    for name, query, confirm, expected in SHOWN_PROBES:
        result = ask(f"dq_{name}", query, confirm=confirm)
        presentation = result["presentation"]
        shown = presented_answer(presentation) if isinstance(expected, dict) else presentation["claim"]
        record("dq_shown", f"{name}: {query}", "answered" if shown == expected else "wrong", expected, result)

    own = own_frame()
    register_frame("probe_coh", cohort)
    register_frame("probe_own", own)
    for dataset, query, truth in probes(cohort, own):
        result = ask(f"probe_{dataset}", query)
        outcome = _held(result) if truth is None else classify_presented_generic(truth, result)
        record("adversarial", query, outcome, truth, result)
    return tally, wrong


@pytest.mark.slow
@pytest.mark.integration
def test_presentedAnswers_everySuite_neverShowsAWrongValue(project_root: Path) -> None:
    # Arrange / Act
    tally, wrong = _presented_outcomes(project_root)

    # Assert
    assert wrong == [], f"{len(wrong)} presented answers differ from the truth ({tally})"
    assert tally.get("answered", 0) > 0
