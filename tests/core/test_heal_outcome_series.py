"""DESCRIBE must use the base view the plan validated, not a renamed cohort."""

from __future__ import annotations

import sys
import types

try:
    import streamlit  # noqa: F401
except ModuleNotFoundError:
    sys.modules.setdefault("streamlit", types.ModuleType("streamlit"))

import ibis
import pandas as pd
import polars as pl
import pytest
from clinical_analytics.core.query_plan import QueryPlan
from clinical_analytics.core.semantic import SemanticLayer
from clinical_analytics.ui.components.question_engine import AnalysisContext, AnalysisIntent


def test_describe_uses_base_view_outcome_not_cohort_alias() -> None:
    n = 150
    base = pd.DataFrame(
        {
            "patient_id": [f"p{i}" for i in range(n)],
            "outcome": [100 + i for i in range(n)],
            "age": [40 + (i % 20) for i in range(n)],
        }
    )
    cohort = base.copy()
    cohort["outcome"] = [i % 2 for i in range(n)]
    layer = object.__new__(SemanticLayer)
    layer._base_view = ibis.memtable(base)
    context = AnalysisContext(
        primary_variable="outcome",
        inferred_intent=AnalysisIntent.DESCRIBE,
        query_plan=QueryPlan(intent="DESCRIBE", metric="outcome", confidence=0.9),
        query_text="describe outcome",
    )
    execution_result = {
        "success": True,
        "result": pd.DataFrame({"mean": [float(base["outcome"].mean())]}),
        "run_key": "k",
        "warnings": [],
        "chart_spec": None,
    }

    formatted = layer.format_execution_result(execution_result, context, cohort=pl.from_pandas(cohort))

    assert formatted["mean"] == pytest.approx(float(base["outcome"].mean()))
    assert formatted["mean"] != pytest.approx(0.5)
