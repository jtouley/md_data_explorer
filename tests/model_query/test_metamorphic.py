# SPDX-License-Identifier: MIT
"""Metamorphic: the model path reads meaning from the contract, not from column names.

The same table and questions with every column renamed to a neutral token (the data dictionary
renamed the same way) must compile to the same SQL once the names are mapped back, raise the same
meaning checks (anchored through the dictionary's words, never the names), and give the same
answers. No model runs: the plans are constructed.
"""

from __future__ import annotations

import re
from typing import Any

import pytest

from clinical_analytics.model_query import checks
from clinical_analytics.model_query.plan import compile_plan, parse_plan
from tests.model_query import tables

_cond = tables.cond
NEUTRAL = {name: f"c{i:02d}" for i, name in enumerate(tables.cohort().columns, start=1)}
BACK = {v: k for k, v in NEUTRAL.items()}


def _neutral_dictionary() -> str:
    text = tables.COHORT_DICTIONARY
    for name, token in NEUTRAL.items():
        text = re.sub(rf"(?m)^{re.escape(name)}(\s)", rf"{token}\1", text)
    return text


def _rename_reply(reply: dict[str, Any], names: dict[str, str]) -> dict[str, Any]:
    out = dict(reply)
    for slot in ("filters", "numerator"):
        out[slot] = [{**c, "column": names[c["column"]]} for c in reply[slot]]
    for slot in ("measure", "group_by", "totals_by"):
        if isinstance(reply.get(slot), str):
            out[slot] = names[reply[slot]]
    return out


def _sql_back(sql: str) -> str:
    return re.sub(r'"(c\d\d)"', lambda m: f'"{BACK.get(m.group(1), m.group(1))}"', sql)


@pytest.fixture
def pair(make_query_table):
    original = make_query_table(tables.cohort(), doc_context=tables.COHORT_DICTIONARY)
    neutral = make_query_table(tables.cohort().rename(NEUTRAL), doc_context=_neutral_dictionary())
    return original, neutral


_DIED = _cond("died_1y", "is", values=["yes"])
CASES = [
    ("How many patients died within a year, by sex?", tables.reply(filters=[_DIED], group_by="sex")),
    (
        "What share of stage III patients died within a year?",
        tables.reply(answer="share", filters=[_cond("stage", "is", values=["III"])], numerator=[_DIED]),
    ),
    (
        "What is the average age of the men?",
        tables.reply(answer="mean", measure="age", filters=[_cond("sex", "is", values=["male"])]),
    ),
    (
        "How many patients over 70 have FEV1 recorded?",
        tables.reply(filters=[_cond("age", ">", 70), _cond("fev_pct", "is recorded")]),
    ),
    # A plan that adds a condition the question does not state: caught both ways.
    (
        "How many patients died within a year?",
        tables.reply(filters=[_DIED, _cond("sex", "is", values=["female"])]),
    ),
]


@pytest.mark.parametrize(("question", "reply"), CASES, ids=[q for q, _r in CASES])
def test_compile_and_checks_neutralColumnNames_sameSqlAndSameMeaningChecks(pair, question, reply):
    # Arrange
    (_c1, original), (_c2, neutral) = pair
    plan = parse_plan(reply)
    renamed = parse_plan(_rename_reply(reply, NEUTRAL))

    # Act
    fixed, _ = checks.apply_stated(plan, question, original)
    fixed_n, _ = checks.apply_stated(renamed, question, neutral)
    compiled, compiled_n = compile_plan(fixed, original), compile_plan(fixed_n, neutral)
    kinds = sorted(p.kind for p in checks.problems(compiled, question, original))
    kinds_n = sorted(p.kind for p in checks.problems(compiled_n, question, neutral))

    # Assert: the same SQL once names map back, and the same meaning checks
    assert _sql_back(compiled_n.sql) == compiled.sql
    assert kinds_n == kinds


@pytest.mark.parametrize(("question", "reply"), CASES[:4], ids=[q for q, _r in CASES[:4]])
def test_answer_question_neutralColumnNames_sameAnswer(pair, make_model_ask, question, reply):
    # Arrange
    original, neutral = pair

    # Act
    answer, _ = make_model_ask(original)(question, reply)
    answer_n, _ = make_model_ask(neutral)(question, _rename_reply(reply, NEUTRAL))

    # Assert: answered alike, with the same numbers and the same card
    assert answer.status == answer_n.status == "answered"
    for key in ("total_count", "proportion", "mean"):
        if answer.analysis.get(key) is not None:
            assert answer_n.analysis.get(key) == pytest.approx(answer.analysis[key])
    # A column the dictionary only lists labels for is named by its own name on the card.
    assert re.sub(r"\bc\d\d\b", lambda m: BACK.get(m.group(0), m.group(0)), answer_n.reading or "") == answer.reading


def test_neutral_dictionary_renamesEveryColumn():
    # Act
    text = _neutral_dictionary()

    # Assert: no original column name is left for the checks to read
    assert not any(re.search(rf"(?m)^{re.escape(name)}\s", text) for name in NEUTRAL)
