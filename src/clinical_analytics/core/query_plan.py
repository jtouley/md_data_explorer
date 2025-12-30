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
class QueryPlan:
    """Structured query plan produced by NLU and consumed by execution layer."""

    intent: Literal["COUNT", "DESCRIBE", "COMPARE_GROUPS", "FIND_PREDICTORS", "CORRELATIONS"]
    metric: str | None = None  # Column name for aggregation (e.g., "LDL mg/dL" for average/describe)
    group_by: str | None = None  # Column name for grouping (e.g., "Nicotine Use")
    filters: list[FilterSpec] = field(default_factory=list)  # Filter conditions
    confidence: float = 0.0  # Parsing confidence (0.0-1.0)
    explanation: str = ""  # Human-readable explanation (shown in UI + logs)
    run_key: str | None = None  # Deterministic key for idempotent execution (includes dataset_version + plan hash)
