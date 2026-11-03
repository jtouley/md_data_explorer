"""Claim-first presentation of analysis results.

Turns the dicts returned by ``analysis.compute`` (via ``SemanticLayer.format_execution_result``)
into one plain-language claim plus a small statistics table. Pure functions: every number
comes from the analysis payload and nothing is recomputed, so the API, CLI and any UI show
the same answer. A clause whose inputs are missing is left out rather than guessed.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from datetime import date, datetime
from typing import Any, TypedDict

import numpy as np


class TablePayload(TypedDict):
    columns: list[str]
    rows: list[dict[str, Any]]


class Presentation(TypedDict):
    claim: str | None
    table: TablePayload | None


def to_json_safe(value: Any) -> Any:
    """Recursively convert numpy scalars, NaN/inf and dates into JSON-safe values."""
    if isinstance(value, Mapping):
        return {str(k): to_json_safe(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [to_json_safe(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, datetime | date):
        return value.isoformat()
    if value is None or isinstance(value, str | int | float | bool):
        return value
    return str(value)


def _num(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float | np.number):
        return None
    out = float(value)
    return out if math.isfinite(out) else None


def _fmt(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}"


def _fmt_p(p: float) -> str:
    return "p<0.001" if p < 0.001 else f"p={p:.3f}"


def _significance(p: float | None) -> str:
    if p is None:
        return ""
    return "significant" if p < 0.05 else "not significant"


def _plain(text: str) -> str:
    """Strip the markdown bold markers compute.py puts in headlines."""
    return text.replace("**", "")


def _term_label(term: Any) -> str:
    """Readable name for a regression term: drop the formula's boolean-dummy suffix."""
    return str(term).removesuffix("[T.True]")


def _round_cell(value: Any) -> Any:
    return round(value, 4) if isinstance(value, float) else value


def _table(rows: list[dict[str, Any]]) -> TablePayload | None:
    """Display table; floats are rounded here, the full-precision analysis is sent separately."""
    if not rows:
        return None
    safe_rows = [{k: _round_cell(v) for k, v in row.items()} for row in to_json_safe(rows)]
    return {"columns": list(rows[0].keys()), "rows": safe_rows}


def _present_comparison(analysis: Mapping[str, Any]) -> Presentation:
    outcome = analysis.get("outcome_col") or "outcome"
    group_col = analysis.get("group_col") or "group"
    test_type = analysis.get("test_type")
    p = _num(analysis.get("p_value"))
    group_means = analysis.get("group_means") or {}
    group_sizes = analysis.get("group_sizes") or {}

    rows = [
        {group_col: str(g), f"mean {outcome}": _num(m), "n": group_sizes.get(str(g))} for g, m in group_means.items()
    ]
    if not any(r["n"] is not None for r in rows):
        rows = [{k: v for k, v in r.items() if k != "n"} for r in rows]

    claim: str | None = None
    if test_type == "t_test":
        groups = [str(g) for g in analysis.get("groups") or []]
        diff = _num(analysis.get("mean_diff"))
        lo, hi = _num(analysis.get("ci_lower")), _num(analysis.get("ci_upper"))
        if len(groups) == 2 and diff is not None:
            if abs(diff) < 1e-12:
                claim = f"Mean {outcome} did not differ between {groups[0]} and {groups[1]}"
            else:
                # mean_diff and its CI are groups[0] - groups[1]. When the claim names groups[1]
                # as higher, it states groups[1] - groups[0], so the CI flips sign and order too.
                higher, lower = (groups[0], groups[1]) if diff > 0 else (groups[1], groups[0])
                claim = f"Mean {outcome} was {_fmt(abs(diff))} higher in {higher} than {lower}"
                if diff < 0 and lo is not None and hi is not None:
                    lo, hi = -hi, -lo
            details = []
            if lo is not None and hi is not None:
                details.append(f"95% CI {_fmt(lo)} to {_fmt(hi)}")
            if p is not None:
                details.append(f"t-test {_fmt_p(p)}")
            n_total = sum(n for n in group_sizes.values() if isinstance(n, int))
            if n_total:
                details.append(f"n={n_total}")
            if details:
                claim += f" ({'; '.join(details)})"
            claim += f". Unadjusted comparison by {group_col}."
    elif test_type == "anova" and p is not None:
        n_groups = analysis.get("n_groups") or len(group_means)
        verb = "differed" if p < 0.05 else "did not differ significantly"
        claim = f"Mean {outcome} {verb} across {n_groups} {group_col} groups (ANOVA {_fmt_p(p)}). Unadjusted."
    elif test_type == "chi_square" and p is not None:
        verb = "differed" if p < 0.05 else "did not differ significantly"
        claim = f"The distribution of {outcome} {verb} by {group_col} (chi-square {_fmt_p(p)}). Unadjusted."
        contingency = analysis.get("contingency") or []
        if contingency:
            return {"claim": claim, "table": _table(list(contingency))}

    return {"claim": claim, "table": _table(rows)}


def _present_predictor(analysis: Mapping[str, Any]) -> Presentation:
    outcome = analysis.get("outcome_col") or "outcome"
    summary = analysis.get("summary") or {}
    rows = [
        {
            "variable": _term_label(var),
            "odds ratio": _num(stats.get("Odds Ratio")),
            "95% CI low": _num(stats.get("CI Lower")),
            "95% CI high": _num(stats.get("CI Upper")),
            "p": _num(stats.get("P-Value")),
        }
        for var, stats in summary.items()
        if var != "Intercept" and isinstance(stats, Mapping)
    ]
    significant = sorted(analysis.get("significant_predictors") or [], key=lambda s: _num(s.get("p_value")) or 1.0)
    predictors = ", ".join(str(r["variable"]) for r in rows)
    model = f"Logistic regression on {outcome}" + (f" with {predictors}" if predictors else "")
    if significant:
        top = significant[0]
        odds, lo, hi, p = (_num(top.get(k)) for k in ("odds_ratio", "ci_lower", "ci_upper", "p_value"))
        claim = f"{_term_label(top.get('variable'))} was the strongest significant predictor of {outcome}"
        details = []
        if odds is not None:
            details.append(f"OR {_fmt(odds)}")
        if lo is not None and hi is not None:
            details.append(f"95% CI {_fmt(lo)} to {_fmt(hi)}")
        if p is not None:
            details.append(_fmt_p(p))
        if details:
            claim += f" ({'; '.join(details)})"
        claim += f". {model}."
    elif rows:
        claim = f"No predictor reached p<0.05 for {outcome}. {model}."
    else:
        claim = None
    return {"claim": claim, "table": _table(rows)}


def _present_relationship(analysis: Mapping[str, Any]) -> Presentation:
    pairs = [
        {"variable 1": c.get("var1"), "variable 2": c.get("var2"), "r": _num(c.get("correlation"))}
        for c in analysis.get("correlations") or []
        if c.get("var1") != c.get("var2")
    ]
    pairs.sort(key=lambda r: abs(r["r"]) if r["r"] is not None else -1.0, reverse=True)
    n = analysis.get("n_observations")
    claim = None
    if pairs and pairs[0]["r"] is not None:
        top = pairs[0]
        r = top["r"]
        strength = "strong" if abs(r) >= 0.5 else "moderate" if abs(r) >= 0.3 else "weak"
        claim = f"The strongest correlation was {strength}: {top['variable 1']} and {top['variable 2']} (r={_fmt(r)}"
        claim += f"; n={n})." if isinstance(n, int) else ")."
    return {"claim": claim, "table": _table(pairs)}


def _present_survival(analysis: Mapping[str, Any]) -> Presentation:
    time_col = analysis.get("time_col") or "time"
    median = _num(analysis.get("median_survival"))
    if median is not None:
        claim = f"Median time to {analysis.get('event_col') or 'event'} was {_fmt(median, 1)} ({time_col})."
    else:
        claim = f"Median time to {analysis.get('event_col') or 'event'} was not reached during follow-up ({time_col})."
    summary = analysis.get("summary") or []
    return {"claim": claim, "table": _table(list(summary)[:50])}


def _present_descriptive(analysis: Mapping[str, Any]) -> Presentation:
    col = analysis.get("focused_variable")
    if col:
        if analysis.get("is_numeric"):
            stats = {k: _num(analysis.get(k)) for k in ("mean", "median", "std", "min", "max")}
            n = analysis.get("non_null_count")
            claim = None
            if stats["mean"] is not None:
                claim = f"Mean {col} was {_fmt(stats['mean'])}"
                parts = []
                if stats["std"] is not None:
                    parts.append(f"SD {_fmt(stats['std'])}")
                if stats["median"] is not None:
                    parts.append(f"median {_fmt(stats['median'])}")
                if isinstance(n, int):
                    parts.append(f"n={n}")
                if parts:
                    claim += f" ({'; '.join(parts)})"
                claim += "."
            row: dict[str, Any] = {"variable": col, "n": n, **stats}
            return {"claim": claim, "table": _table([row])}
        headline = analysis.get("headline")
        return {
            "claim": _plain(headline) + "." if isinstance(headline, str) else None,
            "table": _table(list(analysis.get("value_counts") or [])),
        }
    rows = analysis.get("row_count")
    cols = analysis.get("column_count")
    claim = f"The dataset has {rows} rows and {cols} columns." if rows is not None and cols is not None else None
    return {"claim": claim, "table": _table(list(analysis.get("summary_stats") or []))}


def _present_count(analysis: Mapping[str, Any]) -> Presentation:
    group_col = analysis.get("grouped_by")
    group_counts = analysis.get("group_counts")
    total = analysis.get("total_count")
    if group_col and group_counts:
        if isinstance(group_counts, Mapping):
            rows = [{group_col: k, "count": v} for k, v in group_counts.items()]
        else:
            rows = list(group_counts)
        noun = "group" if len(rows) == 1 else "groups"
        claim = f"{total} total across {len(rows)} {group_col} {noun}." if total is not None else None
        return {"claim": claim, "table": _table(rows)}
    claim = f"Total count: {total}." if total is not None else None
    return {"claim": claim, "table": None}


_PRESENTERS = {
    "comparison": _present_comparison,
    "predictor": _present_predictor,
    "relationship": _present_relationship,
    "survival": _present_survival,
    "descriptive": _present_descriptive,
    "count": _present_count,
}


def present_analysis(analysis: Mapping[str, Any] | None) -> Presentation:
    """Claim sentence and statistics table for a formatted analysis result.

    Returns an empty presentation for missing, failed or unknown results so callers
    never show a claim that the analysis did not support.
    """
    if not analysis or analysis.get("error"):
        return {"claim": None, "table": None}
    presenter = _PRESENTERS.get(str(analysis.get("type")))
    if presenter is None:
        return {"claim": None, "table": None}
    return presenter(analysis)
