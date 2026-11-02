"""Plain-language claim sentences for Ask Questions results (issue #52)."""

from __future__ import annotations

from typing import Any

from clinical_analytics.core.query_plan import QueryPlan

DEFAULT_SMALL_COUNT_THRESHOLD = 11

_TEST_LABELS = {
    "t_test": "Welch t-test",
    "anova": "ANOVA",
    "chi_square": "chi-square test",
    "logistic": "logistic regression",
    "kaplan_meier": "Kaplan–Meier",
    "log_rank": "log-rank test",
    "pearson": "Pearson correlation",
}


def suppress_small_counts(value: int, threshold: int = DEFAULT_SMALL_COUNT_THRESHOLD) -> int | str:
    """Mask patient counts in (0, threshold) as ``<threshold``; keep 0 and >= threshold."""
    if 0 < value < threshold:
        return f"<{threshold}"
    return value


def _format_n(n: int | None, *, excluded: int | None = None) -> str:
    if n is None:
        return ""
    display = suppress_small_counts(int(n))
    clause = f"n={display}"
    if excluded:
        clause += f" after excluding {excluded}"
    return clause


def _unadjusted_clause(plan: QueryPlan, result: dict[str, Any]) -> str:
    """State covariates that were not adjusted for, when known."""
    predictors = result.get("predictors") or []
    if plan.intent == "FIND_PREDICTORS" and predictors:
        return f"model covariates: {', '.join(str(p) for p in predictors)}"
    # Compare / survival / etc.: default clinical honesty clause
    if plan.intent in {"COMPARE_GROUPS", "SURVIVAL", "CORRELATIONS"}:
        return "not adjusted for other covariates"
    if plan.intent == "FIND_PREDICTORS":
        return "univariate associations unless otherwise noted"
    return ""


def _claim_compare(result: dict[str, Any], plan: QueryPlan) -> str:
    outcome = result.get("outcome_col") or plan.metric or "outcome"
    test = _TEST_LABELS.get(str(result.get("test_type", "")), "group comparison")
    parts: list[str] = []

    if "mean_diff" in result:
        diff = float(result["mean_diff"])
        groups = [str(g) for g in (result.get("groups") or [])]
        if abs(diff) < 1e-12:
            parts.append(f"Mean {outcome} did not differ between groups")
        elif len(groups) >= 2:
            # Positive mean_diff means the first listed group has the higher mean.
            higher, lower = (groups[0], groups[1]) if diff > 0 else (groups[1], groups[0])
            parts.append(f"Mean {outcome} was {abs(diff):.1f} higher in {higher} than {lower}")
        elif result.get("headline_text"):
            parts.append(str(result["headline_text"]).replace("**", ""))
        else:
            # Never invent unlabeled direction (C3).
            parts.append(f"Mean {outcome} differed by {abs(diff):.1f} across groups")
        if "ci_lower" in result and "ci_upper" in result and abs(diff) >= 1e-12:
            parts.append(f"(95% CI {result['ci_lower']:.1f} to {result['ci_upper']:.1f}")
            parts[-1] += f"; {test}"
            n_clause = _format_n(result.get("n"), excluded=result.get("n_excluded"))
            if n_clause:
                parts[-1] += f"; {n_clause}"
            parts[-1] += ")"
        elif abs(diff) >= 1e-12 or not groups:
            parts.append(f"({test})")
            n_clause = _format_n(result.get("n"), excluded=result.get("n_excluded"))
            if n_clause and "(" not in parts[-1]:
                parts[-1] = f"({test}; {n_clause})"
    elif result.get("headline_text"):
        parts.append(str(result["headline_text"]).replace("**", ""))
        n_clause = _format_n(result.get("n"), excluded=result.get("n_excluded"))
        extras = [test]
        if n_clause:
            extras.append(n_clause)
        parts.append(f"({'; '.join(extras)})")
    else:
        p = result.get("p_value")
        base = f"Compared {outcome} across groups using {test}"
        if p is not None:
            base += f" (p={p:.3f}"
            n_clause = _format_n(result.get("n"), excluded=result.get("n_excluded"))
            if n_clause:
                base += f"; {n_clause}"
            base += ")"
        else:
            n_clause = _format_n(result.get("n"), excluded=result.get("n_excluded"))
            if n_clause:
                base += f" ({n_clause})"
        parts.append(base)

    unadj = _unadjusted_clause(plan, result)
    if unadj:
        parts.append(unadj)
    return "; ".join(parts) + "."


def _claim_predictors(result: dict[str, Any], plan: QueryPlan) -> str:
    preds = result.get("significant_predictors") or []
    n_clause = _format_n(result.get("sample_size") or result.get("n"))
    if not preds:
        base = "No significant predictors at p<0.05"
        if n_clause:
            base += f" ({n_clause})"
        unadj = _unadjusted_clause(plan, result)
        return f"{base}; {unadj}." if unadj else f"{base}."

    top = preds[0]
    var = top.get("variable", "predictor")
    or_val = top.get("odds_ratio")
    parts = [f"{var} had the strongest association"]
    if or_val is not None:
        clause = f"OR {or_val:.2f}"
        if "ci_lower" in top and "ci_upper" in top:
            clause += f" (95% CI {top['ci_lower']:.2f} to {top['ci_upper']:.2f}"
            if n_clause:
                clause += f"; {n_clause}"
            clause += ")"
        elif n_clause:
            clause += f" ({n_clause})"
        parts.append(clause)
    unadj = _unadjusted_clause(plan, result)
    if unadj:
        parts.append(unadj)
    return "; ".join(parts) + "."


def _claim_count(result: dict[str, Any], _plan: QueryPlan) -> str:
    n = result.get("total_count")
    if n is None and "headline" in result:
        # Fall back to digits in headline if present
        return str(result["headline"]).replace("**", "")
    if n is None:
        n = result.get("row_count")
    if n is None:
        return "Count completed."
    display = suppress_small_counts(int(n))
    return f"Count is {display}."


def _claim_describe(result: dict[str, Any], plan: QueryPlan) -> str:
    n_clause = _format_n(result.get("row_count") or result.get("n") or result.get("filtered_count"))
    if result.get("headline"):
        text = str(result["headline"]).replace("**", "")
        if n_clause and n_clause not in text:
            return f"{text} ({n_clause})."
        return text if text.endswith(".") else f"{text}."
    metric = result.get("focused_variable") or plan.metric or "variable"
    if "mean" in result:
        text = f"Mean {metric} is {result['mean']:.2f}"
        if n_clause:
            text += f" ({n_clause})"
        return text + "."
    return f"Descriptive analysis of {metric} completed."


def _claim_survival(result: dict[str, Any], plan: QueryPlan) -> str:
    parts: list[str] = []
    if result.get("median_survival"):
        parts.append(f"Median survival {result['median_survival']}")
    elif result.get("headline"):
        parts.append(str(result["headline"]).replace("**", ""))
    else:
        parts.append("Survival analysis completed")
    if "p_value" in result:
        parts.append(f"log-rank p={result['p_value']:.3f}")
    n_clause = _format_n(result.get("n") or result.get("sample_size"))
    if n_clause:
        parts.append(n_clause)
    unadj = _unadjusted_clause(plan, result)
    if unadj:
        parts.append(unadj)
    return "; ".join(parts) + "."


def _claim_correlations(result: dict[str, Any], plan: QueryPlan) -> str:
    parts: list[str] = []
    if "strongest_r" in result and "strongest_pair" in result:
        pair = result["strongest_pair"]
        r = result["strongest_r"]
        clause = f"Strongest correlation {pair} r={r:.2f}"
        if "ci_lower" in result and "ci_upper" in result:
            clause += f" (95% CI {result['ci_lower']:.2f} to {result['ci_upper']:.2f}"
            n_clause = _format_n(result.get("n") or result.get("sample_size"))
            if n_clause:
                clause += f"; {n_clause}"
            clause += ")"
        parts.append(clause)
    elif result.get("headline"):
        parts.append(str(result["headline"]).replace("**", ""))
    else:
        parts.append("Correlation analysis completed")
    unadj = _unadjusted_clause(plan, result)
    if unadj:
        parts.append(unadj)
    return "; ".join(parts) + "."


def build_claim(result: dict[str, Any], plan: QueryPlan) -> str:
    """
    Build one plain-language claim from an analysis result payload.

    Numbers come only from ``result`` — this formatter never recomputes statistics.
    Missing optional fields omit their clause rather than inventing values.
    """
    if result.get("error"):
        return f"Analysis could not be completed: {result['error']}"

    intent = str(plan.intent)
    if intent == "COMPARE_GROUPS" or result.get("type") == "comparison":
        return _claim_compare(result, plan)
    if intent == "FIND_PREDICTORS" or result.get("type") == "predictor":
        return _claim_predictors(result, plan)
    if intent == "COUNT" or result.get("type") == "count":
        return _claim_count(result, plan)
    if intent == "DESCRIBE" or result.get("type") == "descriptive":
        return _claim_describe(result, plan)
    if intent == "SURVIVAL" or result.get("type") == "survival":
        return _claim_survival(result, plan)
    if intent == "CORRELATIONS" or result.get("type") in {"relationship", "correlation"}:
        return _claim_correlations(result, plan)
    if result.get("headline"):
        return str(result["headline"]).replace("**", "")
    return "Analysis completed."
