# SPDX-License-Identifier: MIT
"""The keyword matcher answers in front of the model only for a question it fully reads."""

from __future__ import annotations

from typing import Any

import pytest

from clinical_analytics.core.query_plan import FilterSpec, QueryPlan
from clinical_analytics.core.query_service import QueryResult, QueryService
from tests.model_query import tables


def _result(analysis: dict[str, Any], plan: QueryPlan) -> QueryResult:
    return QueryResult(plan=plan, issues=[], result=None, confidence=1.0, run_key="k", context=None, analysis=analysis)


@pytest.fixture
def cohort(make_query_table):
    return make_query_table(tables.cohort(), doc_context=tables.COHORT_DICTIONARY)[1]


@pytest.mark.parametrize(
    ("question", "count", "filters", "expected"),
    [
        pytest.param("How many patients by sex?", 60, [], True, id="fully_read"),
        pytest.param("How many patients are female?", 0, [FilterSpec("sex", "==", "Female")], False, id="zero"),
        pytest.param("How many patients have stage III?", 60, [], False, id="value_dropped"),
        pytest.param("How many patients are over 70?", 12, [FilterSpec("age", ">", 70)], False, id="number"),
    ],
)
def test_matcher_answers_onlyFullyReadAndNonZero(cohort, question, count, filters, expected):
    # Arrange
    plan = QueryPlan(intent="COUNT", group_by="sex" if "by sex" in question else None, filters=filters)
    fast = _result({"type": "count", "total_count": count}, plan)

    # Act / Assert
    assert QueryService._matcher_answers(question, fast, cohort) is expected
