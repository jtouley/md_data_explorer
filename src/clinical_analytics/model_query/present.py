# SPDX-License-Identifier: MIT
"""A query's result rows as the analysis the app already presents (claim and table).

The shapes are those of ``core.result_presentation``: a count, a count per group, a share, a rate
per group, one statistic or a statistic per group. Every number is shown as computed (ADR015);
masking of small counts happens only at the export boundary.
"""

from __future__ import annotations

import math
from typing import Any

import polars as pl
import structlog
from patsy import PatsyError  # type: ignore[import-untyped]

from clinical_analytics.analysis.compute import compute_analysis_by_type
from clinical_analytics.core.analysis_types import AnalysisContext, AnalysisIntent
from clinical_analytics.model_query.context import DatasetContext
from clinical_analytics.model_query.plan import Compiled

TEST_FAILED_REASON = (
    "The {test} could not be computed on these columns. Check that the outcome and the group are the right columns."
)
logger = structlog.get_logger()


def _number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _label(compiled: Compiled, context: DatasetContext, value: Any) -> Any:
    """A group's value in the dictionary's words (a code's label, yes/no for a flag)."""
    column = context.column(compiled.group_column or "")
    if value is None or column is None or column.kind not in ("category", "flag"):
        return value
    return column.label(value)


def analysis_of(
    compiled: Compiled, frame: pl.DataFrame, context: DatasetContext, total: int | None = None
) -> dict[str, Any]:
    """The analysis dict for a count, share or statistic (``total``: the units counted out of)."""
    group = compiled.group_column
    rows = [{**r, "grp": _label(compiled, context, r.get("grp"))} for r in frame.to_dicts()]
    base: dict[str, Any] = {"unit": compiled.unit}
    if compiled.kind == "count":
        if group is None:
            return {"type": "count", "total_count": int(frame.item(0, "count") or 0), **base}
        counts = [int(r["count"] or 0) for r in rows]
        overall = total if total is not None else sum(counts)
        overlapping = sum(counts) > overall
        group_counts: list[dict[str, Any]] = [{group: r["grp"], "count": n} for r, n in zip(rows, counts, strict=True)]
        if overall:
            for row in group_counts:
                row["percent"] = round(100.0 * row["count"] / overall, 1)
        return {
            "type": "count", "total_count": overall, "grouped_by": group, "group_counts": group_counts,
            "overlapping": overlapping, **base,
        }  # fmt: skip
    if compiled.kind == "share":
        event = {"event_text": "the condition counted", "limited": bool(compiled.filters)}
        if group is None:
            row = rows[0] if rows else {}
            return {
                "type": "proportion", "proportion": _number(row.get("rate")), "events": row.get("events"),
                "n_used": row.get("n"), **event, **base,
            }  # fmt: skip
        groups = [
            {"group": r["grp"], "rate": _number(r["rate"]), "n": r.get("n"), "events": r.get("events")} for r in rows
        ]
        return {"type": "rate", "group_col": group, "groups": groups, **event, **base}
    statistic = compiled.plan.answer
    column = context.column(compiled.measure or "")
    metric = column.display if column is not None else "value"
    if group is None:
        row = rows[0] if rows else {}
        return {
            "type": "descriptive", "focused_variable": metric, "is_numeric": True, "statistic": statistic,
            statistic: _number(row.get("value")), "non_null_count": row.get("n"), **base,
        }  # fmt: skip
    groups = [{"group": r["grp"], "value": _number(r["value"]), "n": r.get("n")} for r in rows]
    return {"type": "grouped_statistic", "statistic": statistic, "metric": metric, "group_col": group, "groups": groups}


def has_value(analysis: dict[str, Any]) -> bool:
    """Whether the analysis states a value from at least one row (no group, or a count of zero,
    does not: the model is asked once to check its conditions before a zero is reported)."""
    kind = analysis.get("type")
    if kind == "count":
        return bool(analysis.get("group_counts") if analysis.get("grouped_by") else analysis.get("total_count"))
    if kind == "proportion":
        return analysis.get("proportion") is not None
    if kind in ("rate", "grouped_statistic"):
        return bool(analysis.get("groups"))
    if kind == "descriptive":
        return bool(analysis.get("non_null_count"))
    return not analysis.get("error")


_INTENTS = {
    "compare_groups": AnalysisIntent.COMPARE_GROUPS,
    "odds_ratio": AnalysisIntent.FIND_PREDICTORS,
    "correlation": AnalysisIntent.EXPLORE_RELATIONSHIPS,
    "survival": AnalysisIntent.EXAMINE_SURVIVAL,
}


def run_test(call: Compiled, rows: pl.DataFrame, question: str, entity_key: str | None) -> dict[str, Any]:
    """Route a test call to the app's statistics code (``analysis.compute``) on the fetched rows.
    The result may carry ``error`` (too few rows, a non-binary outcome): shown as the reason."""
    test, outcome, group = call.plan.answer, call.test_outcome, call.test_group
    context = AnalysisContext(research_question=question, query_text=question, inferred_intent=_INTENTS[test])
    if test == "compare_groups":
        context.primary_variable, context.grouping_variable = outcome, group
        context.compare_groups = True
    elif test == "odds_ratio":
        context.primary_variable = outcome
        context.predictor_variables = [group] if group else []
        context.find_predictors = True
    elif test == "correlation":
        context.predictor_variables = [v for v in (outcome, group) if v]
    else:
        context.time_variable, context.event_variable, context.grouping_variable = call.test_time, outcome, group
        context.time_to_event = True
    try:
        return compute_analysis_by_type(rows, context, entity_key)
    except (pl.exceptions.PolarsError, ValueError, KeyError, PatsyError) as e:
        # The statistics code could not read these columns as the test needs them: said plainly,
        # and fed back to the model as a reason to choose other variables.
        logger.info("model_query_test_failed", test=test, error_type=type(e).__name__)
        return {"type": "error", "error": TEST_FAILED_REASON.format(test=test.replace("_", " "))}
