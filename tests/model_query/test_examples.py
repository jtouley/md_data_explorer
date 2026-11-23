# SPDX-License-Identifier: MIT
"""The shipped few-shot examples: valid plans, every slot shown, only their own tables' columns."""

from __future__ import annotations

import pytest

from clinical_analytics.model_query.examples import SLOTS, load_examples
from clinical_analytics.model_query.plan import parse_plan
from tests.model_query import tables

EXAMPLES = load_examples()
# Column names of the test tables and the app's own: an example must never teach one of them.
_FORBIDDEN = {*tables.clinic().columns, *tables.cohort().columns, *tables.ledger().columns, "patient_id"}


@pytest.mark.parametrize("example", EXAMPLES, ids=[e.question for e in EXAMPLES])
def test_example_plan_isValidAndUsesOnlyItsTable(example):
    # Act
    plan = parse_plan(example.plan)

    # Assert
    used = {c.column for c in (*plan.filters, *plan.numerator)} | {n for n in (plan.measure, plan.group_by) if n}
    assert used <= set(example.columns)
    assert not set(example.columns) & _FORBIDDEN
    assert list(example.plan) == list(SLOTS)
