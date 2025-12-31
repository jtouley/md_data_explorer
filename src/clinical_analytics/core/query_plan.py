"""
QueryPlan Contract - ADR001

Defines the structured query plan schema that serves as the contract between
NLU (query parsing) and execution (semantic layer + analysis functions).

All query parsing must produce a QueryPlan, and all execution must consume a QueryPlan.
This prevents ad-hoc dict structures and ensures type safety.
"""

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class FilterSpec:
    """Single filter condition specification."""

    column: str  # Canonical column name (after alias resolution)
    operator: Literal["==", "!=", ">", ">=", "<", "<=", "IN", "NOT_IN"]
    value: str | int | float | list[str | int | float]  # Filter value(s)
    exclude_nulls: bool = True  # Whether to exclude nulls (default: yes)


@dataclass
class ChartSpec:
    """Chart specification for QueryPlan-driven visualization (Phase 3.3)."""

    type: Literal["bar", "line", "hist"]
    x: str | None = None  # X-axis column name
    y: str | None = None  # Y-axis column name
    group_by: str | None = None  # Grouping column
    title: str = ""  # Chart title


@dataclass
class QueryPlan:
    """Structured query plan produced by NLU and consumed by execution layer."""

    intent: Literal["COUNT", "DESCRIBE", "COMPARE_GROUPS", "FIND_PREDICTORS", "CORRELATIONS"]
    metric: str | None = None  # Column name for aggregation (e.g., "LDL mg/dL" for average/describe)
    group_by: str | None = None  # Column name for grouping (e.g., "Nicotine Use")
    filters: list[FilterSpec] = field(default_factory=list)  # Filter conditions
    confidence: float = 0.0  # Parsing confidence (0.0-1.0)
    explanation: str = ""  # Human-readable explanation (shown in UI + logs)
    run_key: str | None = None  # Deterministic key for idempotent execution (includes dataset_version + plan hash)
    # ADR003 Phase 3: Contract fields for execution validation
    requires_filters: bool = False  # True if query explicitly requires filters
    requires_grouping: bool = False  # True if query pattern implies breakdown
    entity_key: str | None = None  # For COUNT: entity to count (e.g., "patient_id")
    scope: Literal["all", "filtered"] = "all"  # For COUNT: count all rows vs filtered cohort


def generate_chart_spec(plan: QueryPlan) -> ChartSpec | None:
    """
    Generate chart specification from QueryPlan (Phase 3.3).

    Deterministic chart spec tied directly to QueryPlan fields.
    Returns None for non-visualizable queries.

    Args:
        plan: QueryPlan to generate chart spec from

    Returns:
        ChartSpec or None if not visualizable
    """
    # DESCRIBE with grouping -> bar chart (group_by on x, metric on y)
    if plan.intent == "DESCRIBE" and plan.group_by and plan.metric:
        return ChartSpec(
            type="bar",
            x=plan.group_by,
            y=plan.metric,
            group_by=plan.group_by,
            title=f"{plan.metric} by {plan.group_by}",
        )

    # DESCRIBE without grouping -> histogram (metric distribution)
    if plan.intent == "DESCRIBE" and plan.metric and not plan.group_by:
        return ChartSpec(
            type="hist",
            x=plan.metric,
            y=None,
            group_by=None,
            title=f"Distribution of {plan.metric}",
        )

    # COUNT with grouping -> bar chart (group_by on x, count on y)
    if plan.intent == "COUNT" and plan.group_by:
        entity = plan.entity_key or "records"
        return ChartSpec(
            type="bar",
            x=plan.group_by,
            y=entity,
            group_by=plan.group_by,
            title=f"Count of {entity} by {plan.group_by}",
        )

    # COMPARE_GROUPS -> bar chart (group_by on x, metric on y)
    if plan.intent == "COMPARE_GROUPS" and plan.group_by and plan.metric:
        return ChartSpec(
            type="bar",
            x=plan.group_by,
            y=plan.metric,
            group_by=plan.group_by,
            title=f"{plan.metric} by {plan.group_by}",
        )

    # CORRELATIONS -> scatter plot not supported yet, return None
    # FIND_PREDICTORS -> no obvious chart, return None
    # COUNT without grouping -> single number, no chart
    return None
