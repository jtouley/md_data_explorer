# SPDX-License-Identifier: MIT
"""Equivalence: the question path gives the plan, hold decision and answer recorded before the
speedups (value index, identifier skip, projected execution), for every suite question.

The snapshot (question_path_snapshot.json) was recorded on main @ ad1f06c, before any of those
changes; see question_path_snapshot.py for how to regenerate it.
"""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

import pytest

from tests.eval.question_path_snapshot import SNAPSHOT_PATH, record, suite_questions

_SUITES = ("golden", "golden_headless", "hard", "reviewer", "adversarial", "negation", "over_holding", "reverse")


@lru_cache(maxsize=1)
def _recorded() -> dict[str, dict[str, Any]]:
    loaded: dict[str, dict[str, Any]] = json.loads(SNAPSHOT_PATH.read_text())
    return loaded


@lru_cache(maxsize=1)
def _questions() -> tuple[tuple[str, str, str], ...]:
    return tuple(suite_questions())


def test_questionPathSnapshot_everySuiteQuestion_isRecorded() -> None:
    # Arrange
    keys = {f"{suite}|{dataset}|{query}" for suite, dataset, query in _questions()}

    # Act
    missing = sorted(keys - set(_recorded()))
    stale = sorted(set(_recorded()) - keys)

    # Assert
    assert missing == []
    assert stale == []
    assert {key.split("|", 1)[0] for key in keys} == set(_SUITES)


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.parametrize("suite", _SUITES)
def test_questionPath_suiteQuestions_matchRecordedPlanHoldAndAnswer(suite: str) -> None:
    # Arrange
    questions = [(dataset, query) for name, dataset, query in _questions() if name == suite]

    # Act
    differences = {
        query: (_recorded()[f"{suite}|{dataset}|{query}"], now)
        for dataset, query in questions
        if (now := record(dataset, query)) != _recorded()[f"{suite}|{dataset}|{query}"]
    }

    # Assert
    assert questions
    assert differences == {}
