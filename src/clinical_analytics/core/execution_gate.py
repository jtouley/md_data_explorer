"""Decide whether a QueryPlan may execute. Issue #52."""

from dataclasses import dataclass
from typing import Any

from clinical_analytics.core.query_plan import FilterSpec, QueryPlan

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
    if plan.scope and plan.scope != "all":
        parts.append(f"scope {plan.scope}")
    parts.append(f"test {label}")
    return " · ".join(parts)
