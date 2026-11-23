# SPDX-License-Identifier: MIT
"""The reading card: clinical words from templates, and the linter every card must pass."""

from __future__ import annotations

import pytest

from clinical_analytics.model_query import card, present
from clinical_analytics.model_query.plan import compile_plan, parse_plan
from clinical_analytics.sql.sandbox import SandboxConnection
from tests.model_query import tables

_SMOKER = {"column": "smoker", "op": "is", "value": None, "values": ["yes"]}
_HIGH_BP = {"column": "systolic_mmhg", "op": ">=", "value": 140, "values": []}
_ARM = {"column": "arm", "op": "is not", "value": None, "values": ["diuretic"]}
_PLANS = [
    {},
    {"filters": [_SMOKER, _HIGH_BP, _ARM]},
    {"filters": [_SMOKER], "which_visit": "last"},
    {"filters": [_SMOKER], "which_visit": "every"},
    {"group_by": "arm", "which_visit": "first"},
    {"count_of": "clinic visits", "group_by": "visit_on", "period": "year"},
    {"answer": "share", "numerator": [_SMOKER], "filters": [_HIGH_BP]},
    {"answer": "share", "numerator": [_SMOKER], "group_by": "sex", "which_visit": "last"},
    {"answer": "mean", "measure": "systolic_mmhg", "which_visit": "change", "group_by": "arm"},
    {"answer": "median", "measure": "age_at_visit", "filters": [_SMOKER]},
    {"answer": "compare_groups", "measure": "systolic_mmhg", "group_by": "arm"},
    {"group_by": "age_at_visit", "cut_points": [50, 65], "which_visit": "last"},
]


@pytest.mark.parametrize("fields", _PLANS, ids=[str(i) for i in range(len(_PLANS))])
def test_reading_lines_everyPlan_passesTheCardLinter(make_query_table, fields):
    # Arrange
    catalog, context = make_query_table(tables.clinic(), doc_context=tables.CLINIC_DICTIONARY)
    compiled = compile_plan(parse_plan(tables.reply(**fields)), context)
    analysis = None
    total = None
    if compiled.kind != "test":
        with SandboxConnection(catalog) as sandbox:
            frame = sandbox.run_model_sql(compiled.sql)
            total = int(sandbox.run_model_sql(compiled.total_sql).item()) if compiled.total_sql else None
        analysis = present.analysis_of(compiled, frame, context, total)

    # Act
    lines = card.reading_lines(compiled, context, analysis, total)

    # Assert: no technical terms, no raw column names, at most 25 words a line, unit named
    assert card.card_problems(lines, context) == []
    assert any(unit in " ".join(lines) for unit in ("patients", "clinic visit"))


def test_reading_lines_count_saysNofMWithPercent(make_query_table):
    # Arrange
    catalog, context = make_query_table(tables.clinic(), doc_context=tables.CLINIC_DICTIONARY)
    compiled = compile_plan(parse_plan(tables.reply(filters=[_SMOKER], which_visit="last")), context)
    with SandboxConnection(catalog) as sandbox:
        frame = sandbox.run_model_sql(compiled.sql)
        total = int(sandbox.run_model_sql(compiled.total_sql).item())

    # Act
    lines = card.reading_lines(compiled, context, present.analysis_of(compiled, frame, context, total), total)

    # Assert
    n = frame.item(0, "count")
    assert lines[0] == "Counted: patients, each once, read at their latest clinic visit with it recorded."
    assert lines[1] == "Included: current smoker at the visit yes."
    assert lines[-1] == f"Result: {n} of {tables.CLINIC_PATIENTS} patients ({100 * n / tables.CLINIC_PATIENTS:.1f}%)."


@pytest.mark.parametrize(
    ("line", "found"),
    [
        pytest.param("Counted rows where x = 1.", True, id="sql_words"),
        pytest.param("Included: systolic_mmhg over 140.", True, id="raw_column"),
        pytest.param("Values that are null were left out.", True, id="null"),
        pytest.param(" ".join(["word"] * 26), True, id="too_long"),
        pytest.param("Counted: patients, each once, read at their latest clinic visit.", False, id="clean"),
    ],
)
def test_card_problems_bannedContent_isFound(make_query_table, line, found):
    # Arrange
    _catalog, context = make_query_table(tables.clinic(), doc_context=tables.CLINIC_DICTIONARY)

    # Act / Assert
    assert bool(card.card_problems([line], context)) is found
