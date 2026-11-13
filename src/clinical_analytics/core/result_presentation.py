"""Claim-first presentation of analysis results.

Turns the dicts returned by ``analysis.compute`` (via ``SemanticLayer.format_execution_result``)
into one plain-language claim plus a small statistics table. Pure functions: every number
comes from the analysis payload and nothing is recomputed, so the API, CLI and any UI show
the same answer. A clause whose inputs are missing is left out rather than guessed.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from datetime import date, datetime
from typing import Any, TypedDict

import numpy as np

from clinical_analytics.analysis.variable_types import display_level, header_has_code_labels, short_name
from clinical_analytics.core.nl_query_config import SMALL_CELL_THRESHOLD


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


_TEST_LABELS = {
    "welch_t_test": "Welch t-test",
    "t_test": "t-test",
    "anova": "ANOVA",
    "chi_square": "chi-square",
    "fisher_exact": "Fisher's exact test",
}


def suppress_count(value: Any, threshold: int | None = None) -> Any:
    """Show a patient count from 1 to threshold-1 as '<threshold' (0 and larger counts unchanged)."""
    limit = SMALL_CELL_THRESHOLD if threshold is None else threshold
    if isinstance(value, bool) or not isinstance(value, int | np.integer):
        return value
    return f"<{limit}" if 0 < int(value) < limit else int(value)


SECONDARY_SUPPRESSED = "suppressed"


def _limit(threshold: int | None) -> int:
    return SMALL_CELL_THRESHOLD if threshold is None else threshold


def _is_count(value: Any) -> bool:
    return isinstance(value, int | np.integer) and not isinstance(value, bool)


def _complementary(cells: dict[Any, Any], hidden: set[Any], limit: int) -> set[Any]:
    """Extra cells to hide in a line whose total is visible.

    The hidden cells of a line are recoverable as total minus the rest, so a line needs at least
    two hidden cells whose sum is at least the threshold ([100, 1, 1] would reveal 1 and 1).
    Hide the smallest visible non-zero cells until that holds or none are left.
    """
    line_hidden = {k for k in cells if k in hidden}
    if not line_hidden:
        return set()
    extra: set[Any] = set()
    visible = sorted(
        (k for k, v in cells.items() if k not in hidden and _is_count(v) and int(v) > 0), key=lambda k: int(cells[k])
    )
    while visible and (len(line_hidden | extra) < 2 or sum(int(cells[k]) for k in line_hidden | extra) < limit):
        extra.add(visible.pop(0))
    return extra


def _suppress_lines(
    cells: Mapping[Any, Any],
    lines: Sequence[Sequence[Any]],
    threshold: int | None = None,
    hidden: set[Any] | None = None,
) -> set[Any]:
    """Cells to hide when each line's total is known (primary + secondary).

    Primary: counts from 1 to threshold-1 (plus any cells already in ``hidden``). Secondary:
    repeat over the lines until none has a hidden cell that its total and the rest reveal.
    """
    limit = _limit(threshold)
    hidden = set(hidden or ()) | {k for k, v in cells.items() if _is_count(v) and 0 < int(v) < limit}
    changed = bool(hidden)
    while changed:
        changed = False
        for line in lines:
            extra = _complementary({k: cells[k] for k in line}, hidden, limit) - hidden
            if extra:
                hidden |= extra
                changed = True
    return hidden


def suppress_count_matrix(rows: list[list[Any]], threshold: int | None = None) -> set[tuple[int, int]]:
    """Cells to hide in a table whose row and column totals are known (primary + secondary)."""
    cells = {(i, j): v for i, row in enumerate(rows) for j, v in enumerate(row)}
    lines = [[(i, j) for j in range(len(row))] for i, row in enumerate(rows)]
    lines += [[(i, j) for i in range(len(rows))] for j in range(len(rows[0]) if rows else 0)]
    return _suppress_lines(cells, lines, threshold)


def _mask(value: Any, threshold: int | None = None) -> str:
    """One marker for every hidden cell, so small (primary) cells are not told apart."""
    return SECONDARY_SUPPRESSED


def suppress_counts(values: list[Any], threshold: int | None = None) -> list[Any]:
    """Suppress a list of counts whose total is visible (primary + secondary)."""
    hidden = suppress_count_matrix([list(values)], threshold)
    return [_mask(v, threshold) if (0, j) in hidden else v for j, v in enumerate(values)]


def _event_phrase(column: str, level: Any) -> str:
    """'Statin use = Yes', or 'event=1' for a bare code."""
    name = short_name(column)
    label = display_level(column, level)
    return label if label.startswith(f"{name}=") else f"{name} = {label}"


# Cell keys for the n clause; tuples so they never collide with a group name or a table cell.
N_USED = ("n", "used")
N_EXCLUDED = ("n", "excluded")


def suppress_with_exclusions(
    analysis: Mapping[str, Any], parts: Mapping[Any, Any], lines: Sequence[Sequence[Any]] = ()
) -> set[Any]:
    """Hidden cells of one answer: its count cells (``parts``) plus n_used and n_excluded.

    n_used + n_excluded is the dataset row count, which the app shows, and the parts (group
    sizes, table cells) add up to n_used. So [n_used, n_excluded] and [*parts, n_excluded] are
    lines with a visible total: either both n counts are shown or both are hidden, and a hidden
    n_excluded hides a part too. ``lines`` are further lines among the parts (rows, columns).
    The parts are first suppressed against n_used as their total, so an exclusion count that
    could stay visible is not used as their complement.
    """
    hidden = _suppress_lines(parts, [*lines, list(parts)])
    cells = {**parts, N_USED: analysis.get("n_used"), N_EXCLUDED: analysis.get("n_excluded")}
    return _suppress_lines(cells, [*lines, [N_USED, N_EXCLUDED], [*parts, N_EXCLUDED]], hidden=hidden)


def _n_clause(analysis: Mapping[str, Any], hidden: set[Any] | None = None) -> str | None:
    """'n=377; 23 excluded for missing data', or 'n hidden; some excluded for missing data'.

    ``hidden`` comes from ``suppress_with_exclusions`` over the answer's count cells.
    """
    n_used = analysis.get("n_used")
    if not isinstance(n_used, int):
        return None
    if hidden is None:
        hidden = suppress_with_exclusions(analysis, {})
    excluded = analysis.get("n_excluded")
    has_excluded = isinstance(excluded, int) and excluded > 0
    if N_USED in hidden or N_EXCLUDED in hidden:
        return "n hidden; some excluded for missing data" if has_excluded else "n hidden"
    return f"n={n_used}; {excluded} excluded for missing data" if has_excluded else f"n={n_used}"


def _with_details(claim: str, details: list[str | None]) -> str:
    kept = [d for d in details if d]
    return f"{claim} ({'; '.join(kept)})" if kept else claim


def _plain(text: str) -> str:
    """Strip the markdown bold markers compute.py puts in headlines."""
    return text.replace("**", "")


def _term_label(term: Any) -> str:
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


def _table(rows: list[dict[str, Any]], count_columns: tuple[str, ...] = ()) -> TablePayload | None:
    """Display table: floats rounded; each patient-count column suppressed down its length
    (primary + secondary), because its total is shown or derivable."""
    if not rows:
        return None
    safe_rows = [{k: _round_cell(v) for k, v in row.items()} for row in to_json_safe(rows)]
    for column in count_columns:
        masked = suppress_counts([row.get(column) for row in safe_rows])
        for row, value in zip(safe_rows, masked, strict=True):
            if column in row:
                row[column] = value
    return {"columns": list(rows[0].keys()), "rows": safe_rows}


def _present_comparison(analysis: Mapping[str, Any]) -> Presentation:
    outcome = analysis.get("outcome_col") or "outcome"
    group_col = analysis.get("group_col") or "group"
    test_type = str(analysis.get("test_type") or "")
    test = _TEST_LABELS.get(test_type, test_type)
    p = _num(analysis.get("p_value"))
    group_means = analysis.get("group_means") or {}
    group_sizes = analysis.get("group_sizes") or {}
    groups = [str(g) for g in analysis.get("groups") or []]
    label = {g: display_level(group_col, g) for g in groups}
    outcome_name = short_name(outcome)
    p_clause = f"{test} {_fmt_p(p)}" if p is not None else None

    if test_type in ("welch_t_test", "t_test", "anova"):
        sizes = {str(g): group_sizes.get(str(g)) for g in group_means}
        hidden = suppress_with_exclusions(analysis, sizes)
        n_clause = _n_clause(analysis, hidden)
        # A mean of a handful of patients is close to their values (one patient: that value),
        # so a group's mean is hidden with its size, as the Kaplan-Meier median is. Hiding only
        # the small groups' means would mark the complements as the large groups.
        rows = [
            {
                group_col: label.get(str(g), str(g)),
                f"mean {outcome_name}": _mask(m) if str(g) in hidden else _num(m),
                "n": _mask(sizes[str(g)]) if str(g) in hidden else sizes[str(g)],
            }
            for g, m in group_means.items()
        ]

    if test_type in ("welch_t_test", "t_test"):
        diff = _num(analysis.get("mean_diff"))
        lo, hi = _num(analysis.get("ci_lower")), _num(analysis.get("ci_upper"))
        claim: str | None = None
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
        return {"claim": claim, "table": _table(rows)}

    if test_type == "anova":
        claim = None
        if p is not None:
            verb = "differed" if p < 0.05 else "did not differ significantly"
            n_groups = analysis.get("n_groups") or len(groups) or len(group_means)
            claim = _with_details(
                f"Mean {outcome_name} {verb} across {n_groups} {group_col} groups", [p_clause, n_clause]
            )
            claim += ". Unadjusted."
        return {"claim": claim, "table": _table(rows)}

    # chi-square / Fisher on a contingency table (rows = outcome levels, columns = groups)
    contingency = [dict(row) for row in analysis.get("contingency") or []]
    cells = {(i, j): row.get(g) for i, row in enumerate(contingency) for j, g in enumerate(groups)}
    lines = [[(i, j) for j in range(len(groups))] for i in range(len(contingency))]
    lines += [[(i, j) for i in range(len(contingency))] for j in range(len(groups))]
    hidden = suppress_with_exclusions(analysis, cells, lines)
    n_clause = _n_clause(analysis, hidden)
    small_groups = any(_is_count(n) and 0 < int(n) < _limit(None) for n in group_sizes.values())
    withheld = bool(hidden & cells.keys()) or small_groups
    proportions = analysis.get("group_proportions") or {}
    event_level = analysis.get("event_level")
    claim = None
    if p is not None:
        significance = "statistically significant" if p < 0.05 else "not statistically significant"
        if proportions and event_level is not None and not withheld:
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
        if withheld:
            claim += f" Group percentages are withheld because a group or cell has fewer than {_limit(None)} patients."
        warning = analysis.get("warning")
        if warning:
            claim += f" {warning}"
    if not contingency:
        return {"claim": claim, "table": None}
    rows = []
    for i, row in enumerate(contingency):
        shown: dict[str, Any] = {outcome_name: display_level(outcome, row.get(outcome))}
        for j, g in enumerate(groups):
            shown[label[g]] = _mask(row.get(g)) if (i, j) in hidden else row.get(g)
        rows.append(shown)
    return {"claim": claim, "table": _table(rows)}


def _present_predictor(analysis: Mapping[str, Any]) -> Presentation:
    outcome = analysis.get("outcome_col") or "outcome"
    outcome_name = short_name(outcome)
    event_level = analysis.get("event_level")
    event = _event_phrase(outcome, event_level) if event_level is not None else outcome_name
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
    model = f"Logistic regression of {event}" + (f" on {predictors}" if predictors else "")
    n_clause = _n_clause(analysis)
    if significant:
        top = significant[0]
        odds, lo, hi, p = (_num(top.get(k)) for k in ("odds_ratio", "ci_lower", "ci_upper", "p_value"))
        claim = _with_details(
            f"{_term_label(top.get('variable'))} had the strongest evidence of association with {event}",
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
        claim = _with_details(
            f"The strongest correlation was {strength}: {top['variable 1']} and {top['variable 2']}",
            [f"r={_fmt(r)}", f"n={suppress_count(n)}" if isinstance(n, int) else None],
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
    if group_col and group_medians:
        group_sizes = analysis.get("group_sizes") or {}

        def _median_text(value: Any) -> str:
            m = _num(value)
            return f"{_fmt(m, 1)}" if m is not None else "not reached"

        # A median of a handful of patients is one patient's time. Hide the median of every
        # group whose size is hidden: showing only the complements' medians would tell the small
        # groups (fewer than the threshold) apart from the complements (at least the threshold).
        sizes = {g: group_sizes.get(g) for g in group_medians}
        hidden = suppress_with_exclusions(analysis, sizes)
        n_clause = _n_clause(analysis, hidden)
        small = {g for g in group_medians if g in hidden}
        medians_text = ", ".join(
            f"{display_level(group_col, g)} {'not shown' if g in small else _median_text(m)}"
            for g, m in group_medians.items()
        )
        logrank = _num(analysis.get("logrank_p_value"))
        claim = _with_details(
            f"Median time to {event} ({time_col}) by {group_col}: {medians_text}",
            [f"log-rank {_fmt_p(logrank)}" if logrank is not None else None, n_clause],
        )
        claim += ". Unadjusted."
        if small:
            claim += " Medians are not shown for groups whose size is hidden to protect privacy."
        rows = [
            {
                group_col: display_level(group_col, g),
                "n": _mask(sizes[g]) if g in small else sizes[g],
                f"median {time_col}": SECONDARY_SUPPRESSED if g in small else _num(m),
            }
            for g, m in group_medians.items()
        ]
        return {"claim": claim, "table": _table(rows)}
    n_used, n_events = analysis.get("n_used"), analysis.get("n_events")
    counted = isinstance(n_used, int) and isinstance(n_events, int)
    parts: dict[str, Any] = {}
    if isinstance(n_used, int) and isinstance(n_events, int):
        parts = {"events": n_events, "censored": n_used - n_events}
    hidden = suppress_with_exclusions(analysis, parts)
    n_clause = _n_clause(analysis, hidden)
    median = _num(analysis.get("median_survival"))
    # With n hidden the median may be one patient's time (and whether it was reached hints at
    # their events); hide it whenever n is hidden, as for groups, so showing it never tells a
    # large n apart from a small one.
    n_hidden = N_USED in hidden
    if n_hidden:
        claim = _with_details(f"Median time to {event} ({time_col}) is not shown because n is hidden", [n_clause])
        claim += "."
    elif median is not None:
        claim = _with_details(f"Median time to {event} was {_fmt(median, 1)} ({time_col})", [n_clause]) + "."
    else:
        claim = _with_details(f"Median time to {event} was not reached during follow-up ({time_col})", [n_clause]) + "."
    # Summary only: the Kaplan-Meier step table lists individual event times and its steps reveal
    # counts in small risk sets.
    if not counted:
        return {"claim": claim, "table": None}
    row = {
        "n": _mask(n_used) if N_USED in hidden else n_used,
        **{k: _mask(v) if k in hidden else v for k, v in parts.items()},
        f"median {time_col}": _mask(median) if n_hidden else median,
    }
    return {"claim": claim, "table": _table([row])}


def _present_descriptive(analysis: Mapping[str, Any]) -> Presentation:
    col = analysis.get("focused_variable")
    if col:
        name = short_name(col)
        if analysis.get("is_numeric"):
            stats: dict[str, Any] = {k: _num(analysis.get(k)) for k in ("mean", "median", "std", "min", "max")}
            n = analysis.get("non_null_count")
            # The table hides a small n, so the claim says "n hidden" rather than mixing in "<11".
            shown_n = suppress_counts([n])[0]
            claim = None
            if shown_n == SECONDARY_SUPPRESSED:
                # With n from 1 to 10, min/max/median are patients' values and the mean of one
                # patient is that value: show none of the statistics, only that n is hidden.
                stats = {k: _mask(v) for k, v in stats.items()}
                claim = f"Statistics for {name} are not shown because n is hidden (n hidden)."
            elif (
                str(analysis.get("statistic") or "mean") != "mean"
                and stats.get(str(analysis.get("statistic"))) is not None
            ):
                # Lead with the statistic the question asked for; the table has the rest.
                asked = str(analysis.get("statistic"))
                claim = _with_details(
                    f"{_STATISTIC_LABELS.get(asked, asked)} {name} was {_fmt(stats[asked])}",
                    [f"n={n}" if isinstance(n, int) else None],
                )
                claim += "."
            elif stats["mean"] is not None:
                claim = _with_details(
                    f"Mean {name} was {_fmt(stats['mean'])}",
                    [
                        f"SD {_fmt(stats['std'])}" if stats["std"] is not None else None,
                        f"median {_fmt(stats['median'])}" if stats["median"] is not None else None,
                        ("n hidden" if shown_n == SECONDARY_SUPPRESSED else f"n={n}") if isinstance(n, int) else None,
                    ],
                )
                claim += "."
            row: dict[str, Any] = {"variable": name, "n": shown_n, **stats}
            return {"claim": claim, "table": _table([row])}
        counts = list(analysis.get("value_counts") or [])
        rows = [{name: display_level(col, r.get(col)), "count": r.get("count")} for r in counts]
        table = _table(rows, count_columns=("count",))
        claim = None
        if rows and table:
            # Quote the top count as the table shows it: it may be the complement of a small cell.
            shown = table["rows"][0]["count"]
            count_text = "count hidden" if shown == SECONDARY_SUPPRESSED else f"{shown} patients"
            claim = f"Most common {name}: {rows[0][name]} ({count_text})."
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
        for row in rows:
            row[group_col] = display_level(group_col, row.get(group_col))
        if _is_count(total) and isinstance(total, int | np.integer) and 0 < int(total) < _limit(None):
            # Every group is then small, and k groups of at least 1 with a total under the
            # threshold can pin each one (10 groups -> all 1), so the groups are not listed.
            return {"claim": f"Total count: {suppress_count(total)}.", "table": None}
        noun = "group" if len(rows) == 1 else "groups"
        claim = f"{total} total across {len(rows)} {group_col} {noun}." if total is not None else None
        return {"claim": claim, "table": _table(rows, count_columns=("count",))}
    claim = f"Total count: {suppress_count(total)}." if total is not None else None
    return {"claim": claim, "table": None}


_STATISTIC_LABELS = {"mean": "Mean", "median": "Median", "min": "Minimum", "max": "Maximum", "std": "SD of"}


def _group_label(column: str, level: Any) -> str:
    return "missing" if level is None else display_level(column, level)


def _counts_small(events: Any, n: Any) -> bool:
    """A share events/n reveals both counts (with and without the value) once n is known.

    True when either is from 1 to threshold-1, or when the counts are missing (fail closed).
    """
    if not _is_count(events) or not _is_count(n):
        return True
    return any(0 < count < _limit(None) for count in (int(events), int(n) - int(events)))


def _percent(value: float) -> str:
    return f"{_fmt(100 * value, 1)}%"


def _present_proportion(analysis: Mapping[str, Any]) -> Presentation:
    """Share of patients with a value. Hidden when either of its counts (with, without) is small:
    the share and n together give both counts."""
    column = str(analysis.get("column") or "value")
    event = _event_phrase(column, analysis.get("value"))
    share = _num(analysis.get("proportion"))
    hidden_n = suppress_with_exclusions(analysis, {})
    n_clause = _n_clause(analysis, hidden_n)
    if share is None:
        return {
            "claim": f"No patient who matched has '{column}' recorded, so there is no share to report.",
            "table": None,
        }
    n_used = analysis.get("n_used")
    withheld = _counts_small(analysis.get("events"), n_used)
    if withheld:
        claim = _with_details(
            f"The share with {event} is not shown because one of its counts is fewer than {_limit(None)} patients",
            [n_clause],
        )
    else:
        population = "patients who matched the limits" if analysis.get("limited") else "patients"
        claim = _with_details(f"{_percent(share)} of {population} had {event}", [n_clause])
    row = {
        "proportion": SECONDARY_SUPPRESSED if withheld else share,
        "n": SECONDARY_SUPPRESSED if N_USED in hidden_n else n_used,
    }
    return {"claim": claim + ".", "table": _table([row])}


def _present_rate(analysis: Mapping[str, Any]) -> Presentation:
    """Rate of a value in each group. A group's n is suppressed as group sizes are; its rate is
    hidden with its n, or when either of its counts (with, without) is small."""
    column = str(analysis.get("column") or "value")
    group_col = str(analysis.get("group_col") or "group")
    event = _event_phrase(column, analysis.get("value"))
    groups = list(analysis.get("groups") or [])
    sizes = {str(g.get("group")): g.get("n") for g in groups}
    hidden = suppress_with_exclusions(analysis, sizes)
    n_clause = _n_clause(analysis, hidden)
    rows, parts, withheld = [], [], False
    for g in groups:
        key, n, events = str(g.get("group")), g.get("n"), g.get("events")
        rate = _num(g.get("rate"))
        label = _group_label(group_col, g.get("group"))
        rate_hidden = key in hidden or _counts_small(events, n)
        withheld = withheld or rate_hidden
        rows.append(
            {
                group_col: label,
                "rate": SECONDARY_SUPPRESSED if rate_hidden else rate,
                "n": SECONDARY_SUPPRESSED if key in hidden else n,
            }
        )
        parts.append(f"{label} {'not shown' if rate_hidden else ('no values' if rate is None else _percent(rate))}")
    claim = _with_details(f"{event} by {group_col}: {', '.join(parts)}", [n_clause]) + ". Unadjusted."
    if withheld:
        claim += f" A rate is not shown when one of its counts is fewer than {_limit(None)} patients."
    return {"claim": claim, "table": _table(rows)}


def _present_grouped_statistic(analysis: Mapping[str, Any]) -> Presentation:
    """One statistic per group. A group's value is hidden with its size: the median, minimum or
    maximum of a few patients is one patient's value."""
    statistic = str(analysis.get("statistic") or "mean")
    metric = short_name(str(analysis.get("metric") or "value"))
    group_col = str(analysis.get("group_col") or "group")
    groups = list(analysis.get("groups") or [])
    sizes = {str(g.get("group")): g.get("n") for g in groups}
    hidden = suppress_with_exclusions(analysis, sizes)
    n_clause = _n_clause(analysis, hidden)
    value_column = f"{statistic} {metric}"
    rows, parts = [], []
    for g in groups:
        key, value = str(g.get("group")), _num(g.get("value"))
        label = _group_label(group_col, g.get("group"))
        shown = SECONDARY_SUPPRESSED if key in hidden else value
        rows.append({group_col: label, value_column: shown, "n": SECONDARY_SUPPRESSED if key in hidden else g.get("n")})
        parts.append(f"{label} {'not shown' if key in hidden else ('no values' if value is None else _fmt(value))}")
    label = _STATISTIC_LABELS.get(statistic, statistic)
    claim = _with_details(f"{label} {metric} by {group_col}: {', '.join(parts)}", [n_clause]) + ". Unadjusted."
    if any(key in hidden for key in sizes):
        claim += " Values are not shown for groups whose size is hidden to protect privacy."
    return {"claim": claim, "table": _table(rows)}


_PRESENTERS = {
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
