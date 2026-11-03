"""Oracle-backed hard and clinician-shaped questions on the synthetic cohort.

Expected values come from DuckDB SQL over tests/fixtures/synthetic/cohort.csv, never from
the parser. Safety property: each question is answered with the oracle's value or refused;
it must never succeed with a different value.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from clinical_analytics.eval.hard_questions import all_hard_questions, oracle_value, questions_for_dataset
from clinical_analytics.eval.headless import ask, classify

# Hard questions the system currently refuses (mostly 0/1 outcome columns with no
# yes/no labels, which the cohort binder will not guess a polarity for). Pinned so a
# lost answer fails loudly; remove an id when it starts being answered correctly.
HARD_REFUSED_IDS: frozenset[str] = frozenset(
    {
        "cohort-analytical-003",
        "cohort-analytical-005",
        "cohort-analytical-020",
        "cohort-analytical-021",
        "cohort-analytical-022",
        "cohort-analytical-023",
        "cohort-analytical-025",
        "cohort-provider-001",
        "cohort-provider-004",
        "cohort-provider-005",
        "cohort-provider-006",
        "cohort-provider-009",
    }
)
HARD_MIN_ANSWERED_RATE = 0.6


def test_hardQuestions_catalog_uniqueQueriesAndCitedProviderQuestions() -> None:
    # Arrange
    questions = questions_for_dataset("cohort")

    # Act
    families = {question["family"] for question in questions}
    queries = [question["query"] for question in questions]
    uncited = [q["id"] for q in questions if q["family"] == "provider" and not q["citation"]]

    # Assert
    assert families == {"analytical", "provider"}
    assert len(set(queries)) == len(queries)
    assert uncited == []


def test_hardQuestions_pinnedRefusals_areKnownIds() -> None:
    # Arrange
    ids = {question["id"] for question in all_hard_questions()}

    # Act
    unknown = HARD_REFUSED_IDS - ids

    # Assert
    assert unknown == set()


def test_oracleSql_everyHardQuestion_returnsAValue(project_root: Path) -> None:
    # Arrange
    questions = all_hard_questions()

    # Act
    missing = [question["id"] for question in questions if oracle_value(question, project_root) is None]

    # Assert
    assert missing == []


def test_oracleSql_countOfMen_matchesPolarsCount(project_root: Path, load_synthetic_dataset) -> None:
    # Arrange
    import polars as pl

    frame = load_synthetic_dataset("cohort")
    question = next(q for q in all_hard_questions() if q["query"] == "How many male patients?")

    # Act
    expected = oracle_value(question, project_root)

    # Assert
    assert expected == frame.filter(pl.col("sex") == "M").height


@pytest.mark.slow
@pytest.mark.integration
def test_hardSuite_everyQuestion_matchesOracleOrRefuses(project_root: Path) -> None:
    # Arrange
    questions = all_hard_questions()

    # Act
    outcomes: dict[str, list[str]] = {"answered": [], "refused": [], "wrong": []}
    details: list[str] = []
    for question in questions:
        expected = oracle_value(question, project_root)
        result = ask(question["dataset_id"], question["query"], project_root)
        outcome, actual = classify(question, expected, result)
        outcomes[outcome].append(question["id"])
        if outcome == "wrong":
            details.append(f"{question['id']} {question['query']!r} expected={expected!r} actual={actual!r}")

    # Assert
    assert details == [], "silently wrong answers"
    assert set(outcomes["refused"]) == HARD_REFUSED_IDS
    assert len(outcomes["answered"]) / len(questions) >= HARD_MIN_ANSWERED_RATE


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.parametrize(
    "query",
    [
        pytest.param("what predicts the outcome?", id="predictors_have_no_verified_statistic"),
        pytest.param("How many patients died?", id="unbound_death_word"),
        pytest.param("tell me about the weather", id="unparseable_question"),
    ],
)
def test_headlessAsk_unanswerableQuestion_isRefused(project_root: Path, query: str) -> None:
    # Arrange / Act
    result = ask("cohort", query, project_root)

    # Assert
    assert result["success"] is False
