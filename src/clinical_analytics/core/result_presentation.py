# SPDX-License-Identifier: MIT
"""Claim-first presentation of analysis results.

Turns the dicts returned by ``analysis.compute`` (via ``SemanticLayer.format_execution_result``)
into one plain-language claim plus a small statistics table. Pure functions: every number
comes from the analysis payload and nothing is recomputed, so the API, CLI and any UI show
the same answer. A clause whose inputs are missing is left out rather than guessed.

On screen every number is shown as computed, small counts included (ADR015): the user already
holds the rows on this machine, so hiding a count protects nothing and makes the answer wrong.
Anything that leaves the machine as a file or packet goes through ``core.export_boundary``,
which writes counts of 1-10 as "<11".
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from datetime import date, datetime
from typing import Any, Final, Literal, TypeAlias, TypedDict, TypeGuard

import numpy as np

from clinical_analytics.analysis.grouping import MAX_GROUP_LEVELS, too_many_levels_message
from clinical_analytics.analysis.variable_types import display_level, header_has_code_labels, short_name


class TablePayload(TypedDict):
    columns: list[str]
    rows: list[dict[str, Any]]


class Presentation(TypedDict):
    claim: str | None
    table: TablePayload | None


# A cell a public source suppressed itself (ADR014 §10: "Suppressed", "(D)", "<10"): shown as
# this word, never turned into a number. The app hides nothing of its own on screen.
SOURCE_SUPPRESSED: Final = "suppressed"


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


def _is_count(value: Any) -> TypeGuard[int]:
    return isinstance(value, int | np.integer) and not isinstance(value, bool)


def _fmt(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}"


def _fmt_p(p: float) -> str:
    return "p<0.001" if p < 0.001 else f"p={p:.3f}"


_TEST_LABELS = {
    "welch_t_test": "Welch t-test",
    "t_test": "t-test",
    "anova": "ANOVA",
    "chi_square": "chi-square",
    "fisher_exact": "Fisher's exact test",
}


def _unit(analysis: Mapping[str, Any]) -> str | None:
    """What n counts, when the query service knows the dataset's grain: patients, visits, or rows
    (some rows have no patient ID, so rows are not patients)."""
    unit = analysis.get("unit")
    return unit if unit in ("patients", "visits", "rows") else None


def _units_word(analysis: Mapping[str, Any]) -> str | None:
    """What n counts in words: the encounter noun the question used ("admissions", "tests"),
    else ``_unit``."""
    noun = analysis.get("unit_noun")
    return f"{noun}s" if isinstance(noun, str) and noun else _unit(analysis)


def _one(analysis: Mapping[str, Any]) -> str:
    """One counted unit, in words: the encounter noun the question used ("admission"), else
    patient or visit by the grain."""
    noun = analysis.get("unit_noun")
    if isinstance(noun, str) and noun:
        return noun
    return {"patients": "patient", "visits": "visit"}.get(_unit(analysis) or "", "unit")


def _article(word: str) -> str:
    return "An" if word[:1].lower() in "aeiou" else "A"


def _overlap_sentence(analysis: Mapping[str, Any], group_col: str, total: Any) -> str:
    """Why groups of distinct units add up to more than their total: a unit with rows in two
    groups is counted in each (``SemanticLayer._grouped_count_sql``, ``_unit_share_sql``)."""
    one = _one(analysis)
    return (
        f"{_article(one)} {one} with rows in more than one {group_col} is counted in each, so the {group_col} "
        f"counts add up to more than {total}."
    )


_ROLLUP_SENTENCES: Final = {
    "any": "{a} {one} counts as having {event} when at least one of its rows{scope} has it.",
    "all": "{a} {one} counts as having {event} only when all of its rows{scope} that record it have it.",
}


def _rollup_sentence(analysis: Mapping[str, Any], event: str, group_col: str | None = None) -> str | None:
    """How a share rolled each unit up over its rows ("ever" / "always"), or None. Grouped, a unit
    is rolled up over its rows in each group (``SemanticLayer._unit_share_sql``), and says so."""
    template = _ROLLUP_SENTENCES.get(str(analysis.get("rollup")))
    if not template:
        return None
    scope = f" in that {group_col}" if group_col else ""
    return template.format(a=_article(_one(analysis)), one=_one(analysis), event=event, scope=scope)


def _counted(value: Any, analysis: Mapping[str, Any]) -> str:
    """'n=377', 'n=377 patients' or 'n=377 admissions'."""
    unit = _units_word(analysis)
    return f"n={value} {unit}" if unit else f"n={value}"


def _event_phrase(column: str, level: Any) -> str:
    """'Statin use = Yes', or 'event=1' for a bare code."""
    name = short_name(column)
    label = display_level(column, level)
    return label if label.startswith(f"{name}=") else f"{name} = {label}"


def _n_clause(analysis: Mapping[str, Any], n_key: str = "n_used", excluded_key: str = "n_excluded") -> str | None:
    """'n=377; 23 excluded for missing data' or 'n=377'; None without a count."""
    n_used = analysis.get(n_key)
    if not _is_count(n_used):
        return None
    excluded = analysis.get(excluded_key)
    if _is_count(excluded) and int(excluded) > 0:
        return f"{_counted(n_used, analysis)}; {excluded} excluded for missing data"
    return _counted(n_used, analysis)


def _with_details(claim: str, details: list[str | None]) -> str:
    kept = [d for d in details if d]
    return f"{claim} ({'; '.join(kept)})" if kept else claim


def _plain(text: str) -> str:
    """Strip the markdown bold markers compute.py puts in headlines."""
    return text.replace("**", "")


def term_label(term: Any) -> str:
    """Readable regression term: drop the boolean-dummy suffix; decode coded-category dummies.

    'DEXA result 1: Normal 2: Osteopenia 3: Osteoporosis_2' -> 'DEXA result: Osteopenia'.
    """
    text = str(term).removesuffix("[T.True]")
    base, sep, level = text.rpartition("_")
    if sep and header_has_code_labels(base):
        return f"{short_name(base)}: {display_level(base, level)}"
    return text


def _round_cell(value: Any) -> Any:
    return round(value, 4) if isinstance(value, float) else value


def _table(rows: list[dict[str, Any]]) -> TablePayload | None:
    """Display table: JSON-safe values, floats rounded to 4 decimals."""
    if not rows:
        return None
    safe_rows = [{k: _round_cell(v) for k, v in row.items()} for row in to_json_safe(rows)]
    return {"columns": list(rows[0].keys()), "rows": safe_rows}


def _too_many_levels(column: str, levels: int) -> Presentation:
    """No table for a grouping with more levels than a reader can scan."""
    return {"claim": too_many_levels_message(column, levels), "table": None}


def _present_comparison(analysis: Mapping[str, Any]) -> Presentation:
    outcome = analysis.get("outcome_col") or "outcome"
    group_col = analysis.get("group_col") or "group"
    test_type = str(analysis.get("test_type") or "")
    test = _TEST_LABELS.get(test_type, test_type)
    p = _num(analysis.get("p_value"))
    group_means = analysis.get("group_means") or {}
    group_sizes = analysis.get("group_sizes") or {}
    groups = [str(g) for g in analysis.get("groups") or []]
    if len(groups) > MAX_GROUP_LEVELS or len(analysis.get("contingency") or []) > MAX_GROUP_LEVELS:
        dimension = group_col if len(groups) > MAX_GROUP_LEVELS else outcome
        levels = len(groups) if len(groups) > MAX_GROUP_LEVELS else len(analysis.get("contingency") or [])
        return _too_many_levels(str(dimension), levels)
    label = {g: display_level(group_col, g) for g in groups}
    outcome_name = short_name(outcome)
    p_clause = f"{test} {_fmt_p(p)}" if p is not None else None
    n_clause = _n_clause(analysis)

    if test_type in ("welch_t_test", "t_test", "anova"):
        rows = [
            {
                group_col: label.get(str(g), str(g)),
                f"mean {outcome_name}": _num(m),
                "n": group_sizes.get(str(g)),
            }
            for g, m in group_means.items()
        ]
        claim: str | None = None
        if test_type in ("welch_t_test", "t_test"):
            diff = _num(analysis.get("mean_diff"))
            lo, hi = _num(analysis.get("ci_lower")), _num(analysis.get("ci_upper"))
            if len(groups) == 2 and diff is not None:
                first, second = label[groups[0]], label[groups[1]]
                if round(abs(diff), 2) == 0:
                    claim = f"Mean {outcome_name} was almost the same in {first} and {second} (difference under 0.01)"
                else:
                    # mean_diff and its CI are groups[0] - groups[1]. When the claim names groups[1]
                    # as higher, it states groups[1] - groups[0], so the CI flips sign and order too.
                    higher, lower = (first, second) if diff > 0 else (second, first)
                    claim = f"Mean {outcome_name} was {_fmt(abs(diff))} higher in {higher} than {lower}"
                    if diff < 0 and lo is not None and hi is not None:
                        lo, hi = -hi, -lo
                ci_clause = f"95% CI {_fmt(lo)} to {_fmt(hi)}" if lo is not None and hi is not None else None
                if p is not None and p >= 0.05:
                    claim += ", not statistically significant"
                claim = _with_details(claim, [ci_clause, p_clause, n_clause])
                claim += f". Unadjusted comparison by {group_col}."
        elif p is not None:
            verb = "differed" if p < 0.05 else "did not differ significantly"
            n_groups = analysis.get("n_groups") or len(groups) or len(group_means)
            claim = _with_details(
                f"Mean {outcome_name} {verb} across {n_groups} {group_col} groups", [p_clause, n_clause]
            )
            claim += ". Unadjusted."
        return {"claim": claim, "table": _table(rows)}

    # chi-square / Fisher on a contingency table (rows = outcome levels, columns = groups)
    contingency = [dict(row) for row in analysis.get("contingency") or []]
    proportions = analysis.get("group_proportions") or {}
    event_level = analysis.get("event_level")
    claim = None
    if p is not None:
        significance = "statistically significant" if p < 0.05 else "not statistically significant"
        if proportions and event_level is not None:
            shares = " and ".join(
                f"{_fmt(100 * float(proportions[g]), 0)}% in {label[g]}" for g in groups if g in proportions
            )
            claim = _with_details(
                f"{_event_phrase(outcome, event_level)} was {shares}; the difference was {significance}",
                [p_clause, n_clause],
            )
        else:
            verb = "differed" if p < 0.05 else "did not differ significantly"
            claim = _with_details(f"The distribution of {outcome_name} {verb} by {group_col}", [p_clause, n_clause])
        claim += ". Unadjusted."
        warning = analysis.get("warning")
        if warning:
            claim += f" {warning}"
    if not contingency:
        return {"claim": claim, "table": None}
    rows = []
    for row in contingency:
        shown: dict[str, Any] = {outcome_name: display_level(outcome, row.get(outcome))}
        for g in groups:
            shown[label[g]] = row.get(g)
        rows.append(shown)
    return {"claim": claim, "table": _table(rows)}


def _present_predictor(analysis: Mapping[str, Any]) -> Presentation:
    outcome = analysis.get("outcome_col") or "outcome"
    outcome_name = short_name(outcome)
    event_level = analysis.get("event_level")
    event = _event_phrase(outcome, event_level) if event_level is not None else outcome_name
    summary = analysis.get("summary") or {}
    n_clause = _n_clause(analysis)
    rows = [
        {
            "variable": term_label(var),
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
    model = f"Logistic regression of {event}" + (f" on {predictors}" if predictors else "")
    claim: str | None
    if significant:
        top = significant[0]
        odds, lo, hi, p = (_num(top.get(k)) for k in ("odds_ratio", "ci_lower", "ci_upper", "p_value"))
        claim = _with_details(
            f"{term_label(top.get('variable'))} had the strongest evidence of association with {event}",
            [
                f"OR {_fmt(odds)}" if odds is not None else None,
                f"95% CI {_fmt(lo)} to {_fmt(hi)}" if lo is not None and hi is not None else None,
                _fmt_p(p) if p is not None else None,
            ],
        )
        adjustment = "each odds ratio is adjusted for the other predictors" if len(rows) > 1 else "unadjusted"
        claim += f". {_with_details(model, [n_clause])}; {adjustment}."
    elif rows:
        adjustment = "predictors adjusted for each other" if len(rows) > 1 else "unadjusted"
        claim = f"No predictor reached p<0.05 for {event}. {_with_details(model, [n_clause])}; {adjustment}."
    else:
        claim = None
    return {"claim": claim, "table": _table(rows)}


def _present_relationship(analysis: Mapping[str, Any]) -> Presentation:
    pairs: list[dict[str, Any]] = [
        {"variable 1": c.get("var1"), "variable 2": c.get("var2"), "r": _num(c.get("correlation"))}
        for c in analysis.get("correlations") or []
        if c.get("var1") != c.get("var2")
    ]
    # Strongest first; a pair with no r (a constant column) last.
    pairs.sort(key=lambda r: abs(r["r"]) if isinstance(r["r"], float) else -1.0, reverse=True)
    n_clause = _n_clause(analysis, "n_observations")
    claim = None
    if pairs and isinstance(pairs[0]["r"], float):
        top = pairs[0]
        r = top["r"]
        strength = "strong" if abs(r) >= 0.5 else "moderate" if abs(r) >= 0.3 else "weak"
        claim = _with_details(
            f"The strongest correlation was {strength}: {top['variable 1']} and {top['variable 2']}",
            [f"r={_fmt(r)}", n_clause],
        )
        claim += "."
    return {"claim": claim, "table": _table(pairs)}


def _present_survival(analysis: Mapping[str, Any]) -> Presentation:
    time_col = analysis.get("time_col") or "time"
    event_col = analysis.get("event_col") or "event"
    group_col = analysis.get("group_col")
    event_level = analysis.get("event_level")
    event = _event_phrase(event_col, event_level) if event_level is not None else short_name(event_col)
    group_medians = analysis.get("group_medians") or {}
    n_clause = _n_clause(analysis)

    def _median_text(value: Any) -> str:
        m = _num(value)
        return f"{_fmt(m, 1)}" if m is not None else "not reached"

    if group_col and group_medians:
        group_sizes = analysis.get("group_sizes") or {}
        medians_text = ", ".join(f"{display_level(group_col, g)} {_median_text(m)}" for g, m in group_medians.items())
        logrank = _num(analysis.get("logrank_p_value"))
        claim = _with_details(
            f"Median time to {event} ({time_col}) by {group_col}: {medians_text}",
            [f"log-rank {_fmt_p(logrank)}" if logrank is not None else None, n_clause],
        )
        claim += ". Unadjusted."
        rows = [
            {group_col: display_level(group_col, g), "n": group_sizes.get(g), f"median {time_col}": _num(m)}
            for g, m in group_medians.items()
        ]
        return {"claim": claim, "table": _table(rows)}
    n_used, n_events = analysis.get("n_used"), analysis.get("n_events")
    median = _num(analysis.get("median_survival"))
    if median is not None:
        claim = _with_details(f"Median time to {event} was {_fmt(median, 1)} ({time_col})", [n_clause]) + "."
    else:
        claim = _with_details(f"Median time to {event} was not reached during follow-up ({time_col})", [n_clause]) + "."
    # Summary only: the Kaplan-Meier step table is too long to read as an answer.
    if not (_is_count(n_used) and _is_count(n_events)):
        return {"claim": claim, "table": None}
    row = {
        "n": int(n_used),
        "events": int(n_events),
        "censored": int(n_used) - int(n_events),
        f"median {time_col}": median,
    }
    return {"claim": claim, "table": _table([row])}


_STATISTIC_LABELS = {"mean": "Mean", "median": "Median", "min": "Minimum", "max": "Maximum", "std": "SD of"}


def _present_descriptive(analysis: Mapping[str, Any]) -> Presentation:
    col = analysis.get("focused_variable")
    if col:
        name = short_name(col)
        if analysis.get("is_numeric"):
            stats: dict[str, Any] = {k: _num(analysis.get(k)) for k in ("mean", "median", "std", "min", "max")}
            n = analysis.get("non_null_count")
            n_clause = _n_clause({**analysis, "n_used": n, "n_excluded": analysis.get("null_count")})
            asked = str(analysis.get("statistic") or "mean")
            claim = None
            if asked != "mean" and stats.get(asked) is not None:
                # Lead with the statistic the question asked for; the table has the rest.
                claim = _with_details(
                    f"{_STATISTIC_LABELS.get(asked, asked)} {name} was {_fmt(stats[asked])}", [n_clause]
                )
                claim += "."
            elif stats["mean"] is not None:
                claim = _with_details(
                    f"Mean {name} was {_fmt(stats['mean'])}",
                    [
                        f"SD {_fmt(stats['std'])}" if stats["std"] is not None else None,
                        f"median {_fmt(stats['median'])}" if stats["median"] is not None else None,
                        n_clause,
                    ],
                )
                claim += "."
            return {"claim": claim, "table": _table([{"variable": name, "n": n, **stats}])}
        counts = list(analysis.get("value_counts") or [])
        rows = [{name: display_level(col, r.get(col)), "count": r.get("count")} for r in counts]
        table = _table(rows)
        claim = None
        if table and table["rows"]:
            top = table["rows"][0]
            # A frequency counts patients: distinct patients per level once rows are visits.
            noun = _unit(analysis) or "patients"
            claim = f"Most common {name}: {top[name]} ({top['count']} {noun})."
        return {"claim": claim, "table": table}
    rows_n = analysis.get("row_count")
    cols = analysis.get("column_count")
    claim = f"The dataset has {rows_n} rows and {cols} columns." if rows_n is not None and cols is not None else None
    return {"claim": claim, "table": _table(list(analysis.get("summary_stats") or []))}


def _present_count(analysis: Mapping[str, Any]) -> Presentation:
    group_col = analysis.get("grouped_by")
    group_counts = analysis.get("group_counts")
    total = analysis.get("total_count")
    if group_col and group_counts:
        if isinstance(group_counts, Mapping):
            rows = [{group_col: k, "count": v} for k, v in group_counts.items()]
        else:
            rows = [dict(r) for r in group_counts]
        if len(rows) > MAX_GROUP_LEVELS:
            return _too_many_levels(str(group_col), len(rows))
        for row in rows:
            row[group_col] = display_level(group_col, row.get(group_col))
        noun = "group" if len(rows) == 1 else "groups"
        counted = f"{total} {_units_word(analysis)}" if _units_word(analysis) else f"{total} total"
        claim = f"{counted} across {len(rows)} {group_col} {noun}." if total is not None else None
        if claim is not None and analysis.get("overlapping"):
            claim = f"{claim} {_overlap_sentence(analysis, str(group_col), total)}"
        return {"claim": claim, "table": _table(rows)}
    # What was counted, in the question's own noun for encounters ("admissions"), else the grain.
    unit = f" {_units_word(analysis)}" if _units_word(analysis) else ""
    claim = f"Total count: {total}{unit}." if total is not None else None
    return {"claim": claim, "table": None}


def _percent(value: float) -> str:
    return f"{_fmt(100 * value, 1)}%"


def _group_label(column: str, level: Any) -> str:
    return "missing" if level is None else display_level(column, level)


def _present_proportion(analysis: Mapping[str, Any]) -> Presentation:
    """Share of patients with a value, among those with it recorded."""
    column = str(analysis.get("column") or "value")
    event = _event_phrase(column, analysis.get("value"))
    share = _num(analysis.get("proportion"))
    if share is None:
        return {
            "claim": f"No patient who matched has '{column}' recorded, so there is no share to report.",
            "table": None,
        }
    noun = _units_word(analysis) or "patients"
    population = f"{noun} who matched the limits" if analysis.get("limited") else noun
    claim = _with_details(f"{_percent(share)} of {population} had {event}", [_n_clause(analysis)])
    rollup = _rollup_sentence(analysis, event)
    if rollup:
        claim += f". {rollup.removesuffix('.')}"
    return {"claim": claim + ".", "table": _table([{"proportion": share, "n": analysis.get("n_used")}])}


def _present_rate(analysis: Mapping[str, Any]) -> Presentation:
    """Rate of a value in each group."""
    column = str(analysis.get("column") or "value")
    group_col = str(analysis.get("group_col") or "group")
    event = _event_phrase(column, analysis.get("value"))
    groups = list(analysis.get("groups") or [])
    if len(groups) > MAX_GROUP_LEVELS:
        return _too_many_levels(group_col, len(groups))
    rows, parts = [], []
    for g in groups:
        rate = _num(g.get("rate"))
        label = _group_label(group_col, g.get("group"))
        rows.append({group_col: label, "rate": rate, "n": g.get("n")})
        parts.append(f"{label} {'no values' if rate is None else _percent(rate)}")
    claim = _with_details(f"{event} by {group_col}: {', '.join(parts)}", [_n_clause(analysis)]) + ". Unadjusted."
    rollup = _rollup_sentence(analysis, event, group_col)
    if rollup:
        claim += f" {rollup}"
    if analysis.get("overlapping"):
        n_used = analysis.get("n_used")
        claim += f" {_overlap_sentence(analysis, group_col, n_used if _is_count(n_used) else 'the total')}"
    return {"claim": claim, "table": _table(rows)}


def _present_grouped_statistic(analysis: Mapping[str, Any]) -> Presentation:
    """One statistic per group."""
    statistic = str(analysis.get("statistic") or "mean")
    metric = short_name(str(analysis.get("metric") or "value"))
    group_col = str(analysis.get("group_col") or "group")
    groups = list(analysis.get("groups") or [])
    if len(groups) > MAX_GROUP_LEVELS:
        return _too_many_levels(group_col, len(groups))
    value_column = f"{statistic} {metric}"
    rows, parts = [], []
    for g in groups:
        value = _num(g.get("value"))
        label = _group_label(group_col, g.get("group"))
        rows.append({group_col: label, value_column: value, "n": g.get("n")})
        parts.append(f"{label} {'no values' if value is None else _fmt(value)}")
    label = _STATISTIC_LABELS.get(statistic, statistic)
    claim = (
        _with_details(f"{label} {metric} by {group_col}: {', '.join(parts)}", [_n_clause(analysis)]) + ". Unadjusted."
    )
    return {"claim": claim, "table": _table(rows)}


def _present_estimates(analysis: Mapping[str, Any]) -> Presentation:
    """Estimate answers (core/estimates.py) carry their claim and table, built from the file's
    estimate rows; only cells the source suppressed itself are hidden there."""
    table = analysis.get("table")
    return {"claim": analysis.get("claim"), "table": table if isinstance(table, dict) else None}  # type: ignore[typeddict-item]


_PRESENTERS = {
    "estimates": _present_estimates,
    "comparison": _present_comparison,
    "predictor": _present_predictor,
    "relationship": _present_relationship,
    "survival": _present_survival,
    "descriptive": _present_descriptive,
    "count": _present_count,
    "proportion": _present_proportion,
    "rate": _present_rate,
    "grouped_statistic": _present_grouped_statistic,
}


def present_analysis(analysis: Mapping[str, Any] | None) -> Presentation:
    """Claim sentence and statistics table for a formatted analysis result, with its data notes
    after the claim.

    Returns an empty presentation for missing, failed or unknown results so callers never show a
    claim that the analysis did not support.
    """
    if not analysis or analysis.get("error"):
        return {"claim": None, "table": None}
    presenter = _PRESENTERS.get(str(analysis.get("type")))
    if presenter is None:
        return {"claim": None, "table": None}
    presented = presenter(analysis)
    claim = presented["claim"]
    if claim is None:
        return presented
    notes = [_implausible_note(analysis), *(analysis.get("data_notes") or [])]
    kept = [sentence for sentence in (_note_sentence(note) for note in notes) if sentence]
    if kept:
        claim = " ".join([claim, *kept])
    return {"claim": claim, "table": presented["table"]}


NoteForm: TypeAlias = Literal["number", "percent"]
# Where a counted note's count goes in its text. Replaced with ``str.replace``, never
# ``str.format``: a column name with braces in it must not break or reshape the sentence.
AMOUNT: Final = "{amount}"


class CountedNote(TypedDict):
    """A data note that carries a count: the text with ``AMOUNT`` where the count goes, the count,
    the total it is a share of (None when it has none) and how the count is written (a number, or
    a percent of ``total``)."""

    text: str
    count: int
    total: int | None
    form: NoteForm


class TextNote(TypedDict):
    """A data note with no count in it, shown as written."""

    text: str
    counted: Literal[False]


DataNote: TypeAlias = str | CountedNote | TextNote


def text_note(text: str) -> TextNote:
    """A note that carries no count. Never wrap text that holds a count: use ``counted_note``."""
    if AMOUNT in text:
        raise ValueError("A text note has no amount; a note with a count is a counted note")
    return {"text": text, "counted": False}


def counted_note(text: str, count: int, total: int | None, form: NoteForm = "number") -> CountedNote:
    """A note whose ``count`` is written where ``text`` has ``AMOUNT`` (once, and only there)."""
    if text.count(AMOUNT) != 1:
        raise ValueError("A counted note's text must hold its amount placeholder exactly once")
    if form == "percent" and not (_is_count(total) and int(total) > 0):
        raise ValueError("A percent note needs the positive total it is a percent of")
    return {"text": text, "count": count, "total": total, "form": form}


def note_text(note: DataNote) -> str:
    """The note's text as written by its producer (the count still a placeholder): for matching."""
    return note if isinstance(note, str) else str(note.get("text") or "")


def _note_sentence(note: Any) -> str | None:
    """A data note as it is shown, or None when it is empty or malformed (dropped, never guessed)."""
    if isinstance(note, str):
        return note or None
    if not isinstance(note, Mapping):
        return None
    text = note.get("text")
    if not isinstance(text, str) or not text:
        return None
    if note.get("counted") is False:
        return text if AMOUNT not in text else None
    count, total, form = note.get("count"), note.get("total"), note.get("form", "number")
    if text.count(AMOUNT) != 1 or not _is_count(count) or form not in ("number", "percent"):
        return None
    if form == "percent":
        if not (_is_count(total) and int(total) > 0):
            return None
        return text.replace(AMOUNT, f"{100 * int(count) / int(total):.1f}%")
    return text.replace(AMOUNT, str(int(count)))


def _implausible_note(analysis: Mapping[str, Any]) -> DataNote | None:
    """'Includes 12 values of age outside 0–122 years (likely entry errors).'"""
    implausible = analysis.get("implausible")
    if not isinstance(implausible, Mapping):
        return None
    count = implausible.get("count")
    if not _is_count(count) or int(count) <= 0:
        return None
    column = short_name(str(implausible.get("column") or "this column")).replace(AMOUNT, "")
    bounds = str(implausible.get("bounds") or "its plausible range").replace(AMOUNT, "")
    total = implausible.get("line_total")
    return counted_note(
        f"Includes {AMOUNT} values of {column} outside {bounds} (likely entry errors).",
        int(count),
        int(total) if _is_count(total) else None,
    )
