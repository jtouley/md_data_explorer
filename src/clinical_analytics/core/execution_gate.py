"""Decide whether a QueryPlan may execute. Issue #52."""

from dataclasses import dataclass

from clinical_analytics.core.query_plan import FilterSpec, QueryPlan

HOLD_FAILURE_REASON = "Confirmation required before execution."

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
    """Label a planned-query result. A successful result continues even without a run key."""
    if result.get("success") is True:
        return "continue"
    if "run_key" not in result or result.get("run_key") is None:
        return "hold"
    return "execution_error"


def apply_planned_query_result(result: dict, session: dict, dataset_version: str) -> str:
    """Return the branch label and clear the rerun flag. Rendering stays in the page."""
    session[f"force_rerun:{dataset_version}"] = False
    return classify_planned_query(result)


def _filter_text(spec: FilterSpec) -> str:
    if isinstance(spec.value, list):
        joined = ", ".join(str(item) for item in spec.value)
        return f"{spec.column} {spec.operator} [{joined}]"
    return f"{spec.column} {spec.operator} {spec.value}"


def format_plan_line(plan: QueryPlan) -> str:
    """One hold line. The model has not run, so the test label comes from intent."""
    filters = "; ".join(_filter_text(spec) for spec in plan.filters) or "none"
    label = _TEST_LABELS.get(str(plan.intent), "not selected")
    outcome = plan.metric or "none"
    grouping = plan.group_by or "none"
    return f"{plan.intent} · outcome {outcome} · grouping {grouping} · filters {filters} · test {label}"
