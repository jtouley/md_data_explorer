# SPDX-License-Identifier: MIT
"""Value index: per-layer lookups over stored values give exactly the old brute-force results,
and stored values no question can name (identifiers, many-valued measures) are never matched."""

from __future__ import annotations

from typing import Any

import pytest

from clinical_analytics.core.cohort_constraints import _value_hits, peel_cohort, yes_no_values
from clinical_analytics.core.query_plan import CohortStep, QueryPlan
from clinical_analytics.core.value_index import ColumnValues, IndexedWords, StemIndex, cached, matches_stored
from clinical_analytics.core.word_accounting import (
    _names_value,
    _value_key,
    _value_words,
    explain_hold,
    numeric_ranges,
    stem_match,
    unjustified_elements,
)

_WORDS = ["diabetes", "diabetic", "north", "northern", "east", "type", "2", "20", "statins", "statin", "s", "ss", "men"]
_COLUMNS = ["patient_id", "region", "diabetes status", "arm", "Smoker 1: Yes 2: No", "sex", "age", "ldl"]
_VALUES: dict[str, list[Any]] = {
    "patient_id": [f"Q{index:03d}" for index in range(60)],
    "region": ["North", "North East", "South", None],
    "diabetes status": ["None", "Type 1", "Type 2"],
    "arm": ["drug A", "drug B", "placebo"],
    "Smoker 1: Yes 2: No": [1, 2],
    "sex": ["F", "M"],
    "age": list(range(30, 90)),
    "ldl": [80.5 + index for index in range(40)],
}
_QUESTIONS = [
    "how many patients in North East?",
    "how many patients in the north east region?",
    "how many patients with type 2 diabetes?",
    "how many patients with type diabetes?",
    "average ldl of northern patients on drug a",
    "how many women smokers on placebo?",
    "how many diabetics in the south or north?",
    "how many patients q001?",
    "",
]


def _indexed(column: str, *, match_stored: bool = True) -> ColumnValues:
    return ColumnValues(column, _VALUES[column], match_stored=match_stored)


def _plain_for(column: str) -> list[Any]:
    return list(_VALUES[column])


def _indexed_for(column: str) -> list[Any]:
    return _indexed(column)


@pytest.mark.parametrize(
    "word", ["diabetes", "diabetics", "north", "nort", "northe", "2", "20s", "statin", "s", "ss", "man", "", "zzzzz"]
)
def test_stemIndex_anyWord_matchesExactlyTheBruteForceStemMatches(word: str) -> None:
    # Arrange
    index = StemIndex(_WORDS, stem_match)

    # Act
    found = index.matches(word)

    # Assert
    assert found == frozenset(target for target in _WORDS if stem_match(word, target))


@pytest.mark.parametrize(("prefix", "expected"), [("nor", ["north", "northern"]), ("diabet", ["diabetes", "diabetic"])])
def test_stemIndex_startingWith_returnsEveryWordWithThePrefix(prefix: str, expected: list[str]) -> None:
    # Arrange
    index = StemIndex(_WORDS, stem_match)

    # Act
    found = index.starting_with(prefix)

    # Assert
    assert found == expected


@pytest.mark.parametrize("word", ["northern", "type", "east", "x", ""])
def test_indexedWords_namedBy_agreesWithBruteForce(word: str) -> None:
    # Arrange
    words = {"north", "east", "type", "1"}
    indexed = IndexedWords(words, stem_match)

    # Act
    named = _names_value(word, indexed)

    # Assert
    assert named == _names_value(word, set(words))
    assert indexed == words


def test_cached_otherColumnOrPlainList_computesAfresh() -> None:
    # Arrange
    values = _indexed("region")
    calls: list[str] = []

    # Act
    cached(values, "region", "probe", lambda: calls.append("own") or 1)
    cached(values, "region", "probe", lambda: calls.append("again") or 2)
    cached(values, "sex", "probe", lambda: calls.append("other column") or 3)
    cached(list(values), "region", "probe", lambda: calls.append("plain list") or 4)

    # Assert
    assert calls == ["own", "other column", "plain list"]
    assert matches_stored(list(values)) is True


@pytest.mark.parametrize("question", _QUESTIONS)
def test_valueHits_indexedValues_sameHitsAsPlainLists(question: str) -> None:
    # Arrange
    text = question.lower()

    # Act
    indexed = _value_hits(text, _COLUMNS, _indexed_for)

    # Assert
    assert indexed == _value_hits(text, _COLUMNS, _plain_for)


@pytest.mark.parametrize("question", _QUESTIONS[:-1])
def test_peelCohort_indexedValues_samePeelAsPlainLists(question: str) -> None:
    # Arrange / Act
    indexed = peel_cohort(question, _COLUMNS, _indexed_for)

    # Assert
    assert indexed == peel_cohort(question, _COLUMNS, _plain_for)


@pytest.mark.parametrize(
    ("question", "plan"),
    [
        (
            "how many patients in North East?",
            QueryPlan(intent="COUNT", cohort_steps=[CohortStep("eq", "region", "North East")]),
        ),
        ("how many women and men?", QueryPlan(intent="COUNT")),
        ("average ldl by region", QueryPlan(intent="DESCRIBE", metric="ldl", group_by="region")),
        ("average ldl for patients over 65", QueryPlan(intent="DESCRIBE", metric="ldl", statistic="mean")),
        ("how many patients with type 2 diabetes?", QueryPlan(intent="COUNT")),
    ],
)
def test_wordAccounting_indexedValues_sameHoldAsPlainLists(question: str, plan: QueryPlan) -> None:
    # Arrange / Act
    indexed = (
        explain_hold(question, plan, _COLUMNS, _indexed_for),
        unjustified_elements(question, plan, _COLUMNS, _indexed_for),
    )

    # Assert
    assert indexed == (
        explain_hold(question, plan, _COLUMNS, _plain_for),
        unjustified_elements(question, plan, _COLUMNS, _plain_for),
    )


@pytest.mark.parametrize(
    ("word", "expected"), [("women", "F"), ("males", "M"), ("north", "North"), ("nobody", "nobody")]
)
def test_valueKey_indexedValues_firstStoredValueTheWordNames(word: str, expected: str) -> None:
    # Arrange
    column = "sex" if word in {"women", "males"} else "region"

    # Act
    key = _value_key(word, column, _indexed_for)

    # Assert
    assert key == expected
    assert key == _value_key(word, column, _plain_for)


def test_identifierColumn_storedValuesSkipped_codebookStillRead() -> None:
    # Arrange
    dexa = "DEXA result 1: Normal 2: Osteopenia"
    values = {
        "patient_id": _indexed("patient_id", match_stored=False),
        dexa: ColumnValues(dexa, [1, 2], match_stored=False),
    }
    columns = list(values)

    # Act
    hits = _value_hits("how many patients q001 with osteopenia", columns, values.__getitem__)

    # Assert
    assert [(column, value) for column, value, _at in hits] == [(dexa, 2)]
    assert _value_words("patient_id", values["patient_id"]) == frozenset()
    assert set(_value_words(dexa, values[dexa])) == {"normal", "osteopenia"}


@pytest.mark.parametrize(
    ("column", "values", "expected"),
    [("age", [30, 40, None, 50], {"age": (30.0, 50.0)}), ("sex", ["F", "M"], {}), ("ldl", [None, None], {})],
)
def test_numericRanges_indexedValues_sameRangeAsPlainList(
    column: str, values: list[Any], expected: dict[str, Any]
) -> None:
    # Arrange
    indexed = ColumnValues(column, values)

    # Act
    ranges = numeric_ranges([column], lambda _column: indexed)

    # Assert
    assert ranges == expected
    assert ranges == numeric_ranges([column], lambda _column: list(values))


def test_yesNoValues_indexedValues_cachedAndUnchanged() -> None:
    # Arrange
    values = ColumnValues("smoker", ["No", "Yes", None])

    # Act
    first = yes_no_values("smoker", values)
    second = yes_no_values("smoker", values)

    # Assert
    assert first == second == ("Yes", "No")
    assert first == yes_no_values("smoker", ["No", "Yes", None])
