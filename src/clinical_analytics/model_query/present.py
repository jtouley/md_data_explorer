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

from clinical_analytics.analysis.compute import compute_analysis_by_type
from clinical_analytics.core.analysis_types import AnalysisContext, AnalysisIntent
from clinical_analytics.model_query.plan import Compiled, TestCall

TEST_FAILED_REASON = (
    "The {test} could not be computed on these columns. Check that the outcome and the group are the right columns."
)
logger = structlog.get_logger()


def _number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _unit_fields(compiled: Compiled) -> dict[str, Any]:
    fields: dict[str, Any] = {"unit": compiled.unit.word}
    if compiled.unit.noun:
        fields["unit_noun"] = compiled.unit.noun
    return fields


def analysis_of(compiled: Compiled, frame: pl.DataFrame, total: int | None = None) -> dict[str, Any]:
    """The analysis dict for a tabulation's rows (``total``: the ungrouped count of a grouped one)."""
    rows = frame.to_dicts()
    group = compiled.group
    base = _unit_fields(compiled)
    if compiled.kind == "count":
        if group is None:
            return {"type": "count", "total_count": int(frame.item(0, frame.width - 1) or 0), **base}
        value_column = next(c for c in frame.columns if c != group)
        counts = [int(r[value_column] or 0) for r in rows]
        overall = total if total is not None else sum(counts)
        overlapping = sum(counts) > overall
        group_counts: list[dict[str, Any]] = [{group: r[group], "count": n} for r, n in zip(rows, counts, strict=True)]
        if not overlapping and overall:
            # Each group's share of the total (not shown when a unit can be in several groups).
            for row in group_counts:
                row["percent"] = round(100.0 * row["count"] / overall, 1)
        return {
            "type": "count",
            "total_count": overall,
            "grouped_by": group,
            "group_counts": group_counts,
            "overlapping": overlapping,
            **base,
        }
    if compiled.kind == "share":
        value_column = "rate" if "rate" in frame.columns else next(c for c in frame.columns if c != group)
        n_column = "n" if "n" in frame.columns else None
        event: dict[str, Any] = {"event_text": compiled.event_text or "the event", "limited": compiled.limited}
        if compiled.rollup:
            event["rollup"] = compiled.rollup
        if group is None:
            row = rows[0] if rows else {}
            return {
                "type": "proportion",
                "proportion": _number(row.get(value_column)),
                "n_used": row.get(n_column) if n_column else None,
                **event,
                **base,
            }
        groups = [
            {"group": r[group], "rate": _number(r[value_column]), "n": r.get(n_column) if n_column else None}
            for r in rows
        ]
        return {"type": "rate", "group_col": group, "groups": groups, **event, **base}
    if compiled.kind == "statistic":
        statistic = compiled.statistic or "mean"
        metric = compiled.value_text or "value"
        if group is None:
            row = rows[0] if rows else {}
            n = row.get("n")
            return {
                "type": "descriptive",
                "focused_variable": metric,
                "is_numeric": True,
                "statistic": statistic,
                **{name: _number(row.get(name)) for name in ("mean", "median", "std", "min", "max")},
                statistic: _number(row.get(statistic)),
                "non_null_count": n,
                "null_count": (row["rows"] - n) if row.get("rows") is not None and n is not None else None,
                **base,
            }
        value_column = statistic if statistic in frame.columns else next(c for c in frame.columns if c != group)
        groups = [{"group": r[group], "value": _number(r[value_column]), "n": r.get("n")} for r in rows]
        return {
            "type": "grouped_statistic",
            "statistic": statistic,
            "metric": metric,
            "group_col": group,
            "groups": groups,
        }
    return {"type": "table", "claim": compiled.reading, "rows": rows}


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
    if kind == "table":
        return bool(analysis.get("rows"))
    return not analysis.get("error")


_INTENTS = {
    "compare_groups": AnalysisIntent.COMPARE_GROUPS,
    "odds_ratio": AnalysisIntent.FIND_PREDICTORS,
    "correlation": AnalysisIntent.EXPLORE_RELATIONSHIPS,
    "survival": AnalysisIntent.EXAMINE_SURVIVAL,
}


def run_test(call: TestCall, rows: pl.DataFrame, question: str) -> dict[str, Any]:
    """Route a test call to the app's statistics code (``analysis.compute``) on the fetched rows.
    The result may carry ``error`` (too few rows, a non-binary outcome): shown as the reason."""
    context = AnalysisContext(research_question=question, query_text=question, inferred_intent=_INTENTS[call.test])
    if call.test == "compare_groups":
        context.primary_variable, context.grouping_variable = call.outcome, call.group
        context.compare_groups = True
    elif call.test == "odds_ratio":
        context.primary_variable = call.outcome
        context.predictor_variables = [call.group] if call.group else []
        context.find_predictors = True
    elif call.test == "correlation":
        context.predictor_variables = [v for v in (call.outcome, call.group) if v]
    else:
        context.time_variable, context.event_variable, context.grouping_variable = call.outcome, call.time, call.group
        context.time_to_event = True
    try:
        return compute_analysis_by_type(rows, context, call.entity_key)
    except (pl.exceptions.PolarsError, ValueError, KeyError) as e:
        # The statistics code could not read these columns as the test needs them: said plainly,
        # and fed back to the model as a reason to choose other variables.
        logger.info("model_query_test_failed", test=call.test, error_type=type(e).__name__)
        return {"type": "error", "error": TEST_FAILED_REASON.format(test=call.test.replace("_", " "))}
