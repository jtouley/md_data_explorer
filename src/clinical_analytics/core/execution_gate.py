"""Decide whether a QueryPlan may execute. Issue #52."""

import dataclasses
import hashlib
import json
from dataclasses import dataclass
from typing import Any

from clinical_analytics.core.column_parser import parse_column_name
from clinical_analytics.core.query_plan import CohortStep, FilterSpec, QueryPlan

HOLD_FAILURE_REASON = "Confirmation required before execution."
PLAN_CHANGED_REASON = (
    "The question was interpreted differently from the plan you confirmed. Review this plan before it runs."
)

_TEST_LABELS = {
    "COMPARE_GROUPS": "group comparison",
    "FIND_PREDICTORS": "logistic regression",
    "CORRELATIONS": "correlation",
    "COUNT": "count",
    "DESCRIBE": "descriptive statistics",
    "SURVIVAL": "Kaplan–Meier",
}


@dataclass(frozen=True)
class GateDecision:
    """Result of the pre-execution hold."""

    allow: bool
    requires_confirmation: bool
    failure_reason: str


def evaluate_execution_gate(
    plan: QueryPlan,
    *,
    threshold: float,
    confirmed: bool,
    is_complete: bool,
    completeness_error: str,
    validation_valid: bool,
    validation_error: str,
) -> GateDecision:
    """Refuse incomplete or invalid plans. Hold complete valid plans below the threshold."""
    if not is_complete:
        return GateDecision(False, False, completeness_error)
    if not validation_valid:
        return GateDecision(False, False, validation_error)
    if plan.confidence < threshold and not confirmed:
        return GateDecision(False, True, HOLD_FAILURE_REASON)
    return GateDecision(True, False, "")


def classify_planned_query(result: dict) -> str:
    """Label a planned-query result.

    Taxonomy:
    - continue: success
    - hold: low-confidence gate awaiting confirmation (requires_confirmation)
    - refused: incomplete/invalid plan (no run_key, not a confirmation hold)
    - execution_error: failed after a run_key was issued
    """
    if result.get("success") is True:
        return "continue"
    if result.get("requires_confirmation"):
        return "hold"
    if "run_key" not in result or result.get("run_key") is None:
        return "refused"
    return "execution_error"


def apply_planned_query_result(result: dict, session: dict, dataset_version: str) -> str:
    """Return the branch label and clear the rerun flag. Rendering stays in the page."""
    session[f"force_rerun:{dataset_version}"] = False
    return classify_planned_query(result)


def _filter_text(spec: FilterSpec) -> str:
    if isinstance(spec.value, list):
        joined = ", ".join(str(item) for item in spec.value)
        text = f"{spec.column} {spec.operator} [{joined}]"
    else:
        text = f"{spec.column} {spec.operator} {spec.value}"
    return text if spec.exclude_nulls else f"{text} (missing kept)"


# QueryPlan fields that never change what executes (display text, gating score, cache key).
# Every other field, present or future, is part of the confirmation hash.
NON_EXECUTION_PLAN_FIELDS: frozenset[str] = frozenset(
    {
        "confidence",
        "explanation",
        "run_key",
        "follow_ups",
        "follow_up_explanation",
        "interpretation",
        "confidence_explanation",
    }
)
_CONTEXT_FIELDS = ("predictor_variables", "time_variable", "event_variable")


def _canonical(value: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {field.name: _canonical(getattr(value, field.name)) for field in dataclasses.fields(value)}
    if isinstance(value, list | tuple):
        return [_canonical(item) for item in value]
    if value is None or isinstance(value, bool | int | float | str):
        return value
    return str(value)


def executable_plan(plan: QueryPlan, context: Any = None) -> dict[str, Any]:
    """Every field that affects execution, in canonical form (order of cohort steps kept)."""
    fields = {
        field.name: _canonical(getattr(plan, field.name))
        for field in dataclasses.fields(plan)
        if field.name not in NON_EXECUTION_PLAN_FIELDS
    }
    fields["context"] = {name: _canonical(getattr(context, name, None)) for name in _CONTEXT_FIELDS}
    return fields


def plan_fingerprint(plan: QueryPlan, context: Any = None) -> str:
    """Hash a confirmation is bound to: sorted canonical JSON of the executable plan."""
    payload = json.dumps(executable_plan(plan, context), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _value_text(column: str, value: Any) -> str:
    """Value as stored, with its code label when the column header defines one ("1 (Yes)")."""
    text = str(value)
    code = text[:-2] if text.endswith(".0") else text
    label = (parse_column_name(column).value_mapping or {}).get(code)
    return f"{text} ({label})" if label else text


def _number(value: Any) -> str:
    try:
        return f"{float(value):g}"
    except (TypeError, ValueError):
        return str(value)


def _step_text(step: CohortStep) -> str:
    column = step.column
    if step.op == "eq":
        return f"{column} = {_value_text(column, step.value)}"
    if step.op == "gt":
        return f"{column} > {_number(step.value)}"
    if step.op == "lt":
        return f"{column} < {_number(step.value)}"
    if step.op == "ge":
        return f"{column} ≥ {_number(step.value)}"
    if step.op == "le":
        return f"{column} ≤ {_number(step.value)}"
    if step.op == "not_null":
        return f"{column} is recorded"
    if step.op == "gt_median":
        return f"{column} above its median among the patients so far"
    if step.op in {"ge_percent_rank", "ge_quantile"}:
        quantile = 0.9 if step.quantile is None else step.quantile
        return f"{column} at or above its {quantile * 100:g}th percentile among the patients so far"
    return f"{column} {step.op} {step.value}"


def format_plan_line(plan: QueryPlan, context: Any = None) -> str:
    """One hold line naming everything that changes the analysis.

    It is what the user reviews and what a confirmation is bound to, so it must include every
    input the analysis uses: predictors and time/event columns come from the analysis context,
    and filter null-handling and scope from the plan. The model has not run, so the test label
    comes from the intent.
    """
    filters = "; ".join(_filter_text(spec) for spec in plan.filters) or "none"
    label = _TEST_LABELS.get(str(plan.intent), "not selected")
    outcome = plan.metric or "none"
    grouping = plan.group_by or "none"
    parts = [f"{plan.intent}", f"outcome {outcome}", f"grouping {grouping}"]
    predictors = list(getattr(context, "predictor_variables", None) or [])
    if predictors and plan.intent != "CORRELATIONS":
        parts.append(f"predictors {', '.join(str(p) for p in predictors)}")
    elif predictors:
        parts.append(f"variables {', '.join(str(p) for p in predictors)}")
    time_col = getattr(context, "time_variable", None)
    event_col = getattr(context, "event_variable", None)
    if time_col or event_col:
        parts.append(f"time {time_col or 'none'} · event {event_col or 'none'}")
    parts.append(f"filters {filters}")
    if plan.cohort_steps:
        ordered = "; ".join(f"{index}) {_step_text(step)}" for index, step in enumerate(plan.cohort_steps, start=1))
        parts.append(f"limits in order {ordered}")
    if plan.proportion_column:
        population = "patients matching the limits" if plan.cohort_steps or plan.filters else "all patients"
        parts.append(
            f"result proportion with {plan.proportion_column} = "
            f"{_value_text(plan.proportion_column, plan.proportion_value)} among {population}"
        )
    if plan.rate_column:
        parts.append(
            f"result rate of {plan.rate_column} = {_value_text(plan.rate_column, plan.rate_value)}"
            f" within each {plan.group_by or 'patient'} group"
        )
    if plan.entity_key:
        parts.append(f"counting distinct {plan.entity_key}")
    if plan.scope and plan.scope != "all":
        parts.append(f"scope {plan.scope}")
    if plan.requires_filters:
        parts.append("filters required")
    if plan.requires_grouping:
        parts.append("grouping required")
    parts.append(f"test {label}")
    return " · ".join(parts)
