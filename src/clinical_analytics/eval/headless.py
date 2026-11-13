# SPDX-License-Identifier: MIT
"""Run a natural-language question against a golden dataset CSV without the UI.

Dev and test use only: results carry raw rows with no small-cell suppression, so this
module must never be exposed through the production API.
"""

from __future__ import annotations

import re
import tempfile
from pathlib import Path
from typing import Any, Literal, TypeAlias

import polars as pl

from clinical_analytics.analysis.variable_types import display_level
from clinical_analytics.core.query_service import QueryService
from clinical_analytics.core.result_presentation import SECONDARY_SUPPRESSED, present_analysis
from clinical_analytics.core.semantic import SemanticLayer
from clinical_analytics.eval.catalog import DATASET_FILES, load_dataset_frame

_LAYERS: dict[str, SemanticLayer] = {}
_FRAMES: dict[str, pl.DataFrame] = {}
_SERVICES: dict[str, QueryService] = {}
_CACHE_DIR: Path | None = None


def _cache_dir() -> Path:
    """Per-process scratch dir so parallel test workers never share or race on a CSV copy."""
    global _CACHE_DIR
    if _CACHE_DIR is None:
        _CACHE_DIR = Path(tempfile.mkdtemp(prefix="mdde-golden-"))
    return _CACHE_DIR


def dataset_ids() -> list[str]:
    return list(DATASET_FILES)


def frame_for(dataset_id: str, workspace: Path | None = None) -> pl.DataFrame:
    if dataset_id not in _FRAMES:
        _FRAMES[dataset_id] = load_dataset_frame(dataset_id, workspace)
    return _FRAMES[dataset_id]


def register_frame(dataset_id: str, frame: pl.DataFrame) -> None:
    """Serve an in-memory synthetic frame under ``dataset_id`` (probe suites on other schemas)."""
    _FRAMES[dataset_id] = frame
    _LAYERS.pop(dataset_id, None)
    _SERVICES.pop(dataset_id, None)


def layer_for(dataset_id: str, workspace: Path | None = None) -> SemanticLayer:
    """Semantic layer backed by a per-process CSV copy of the dataset."""
    if dataset_id in _LAYERS:
        return _LAYERS[dataset_id]
    frame = frame_for(dataset_id, workspace)
    csv_path = _cache_dir() / f"{dataset_id}.csv"
    frame.write_csv(csv_path)
    config: dict[str, Any] = {
        "init_params": {"source_path": str(csv_path)},
        "column_mapping": {},
        "time_zero": {"value": "2020-01-01"},
        "outcomes": {},
        "analysis": {"default_outcome": "outcome"},
    }
    layer = SemanticLayer(dataset_id, config=config, workspace_root=csv_path.parent)
    _LAYERS[dataset_id] = layer
    return layer


def _result_frame(payload: dict[str, Any] | None) -> pl.DataFrame | None:
    if not payload:
        return None
    raw = payload.get("result")
    if raw is None:
        return None
    if isinstance(raw, pl.DataFrame):
        return raw
    if isinstance(raw, int | float):
        return pl.DataFrame({"count": [raw]})
    try:
        return pl.from_pandas(raw)
    except (TypeError, ValueError):
        return None


def _group_map(frame: pl.DataFrame, group_column: str, value_column: str) -> dict[str, float | int | None]:
    mapped: dict[str, float | int | None] = {}
    for row in frame.iter_rows(named=True):
        key = "null" if row[group_column] is None else str(row[group_column])
        value = row[value_column]
        if value is None:
            mapped[key] = None
        elif value_column == "count":
            mapped[key] = int(value)
        else:
            mapped[key] = float(value)
    return mapped


def actual_value(kind: str, stat: str | None, group_by: str | None, payload: dict[str, Any] | None) -> Any:
    """Pull the comparable value out of an execution payload."""
    frame = _result_frame(payload)
    if frame is None or frame.height == 0:
        return None
    if kind == "count":
        if frame.width == 1:
            return int(frame.item())
        if "count" in frame.columns and frame.height == 1:
            return int(frame.get_column("count")[0])
        return int(frame.height)
    if kind == "proportion":
        if "proportion" not in frame.columns:
            return None
        value = frame.get_column("proportion")[0]
        return None if value is None else float(value)
    if kind == "corr":
        if "corr" not in frame.columns:
            return None
        value = frame.get_column("corr")[0]
        return None if value is None else float(value)
    if kind == "rates":
        if group_by is None or group_by not in frame.columns or "rate" not in frame.columns:
            return None
        return _group_map(frame, group_by, "rate")
    if kind == "crosstab":
        value_columns = [column for column in frame.columns if column != "n"]
        if len(value_columns) != 2 or "n" not in frame.columns:
            return None
        grouped: dict[str, dict[str, int]] = {}
        for row in frame.iter_rows(named=True):
            group_key = "null" if row[value_columns[0]] is None else str(row[value_columns[0]])
            level_key = "null" if row[value_columns[1]] is None else str(row[value_columns[1]])
            grouped.setdefault(group_key, {})[level_key] = int(row["n"])
        return grouped
    if kind == "count_by":
        if group_by is None or group_by not in frame.columns:
            return None
        return _group_map(frame, group_by, "count")
    if kind in {"mean_by", "compare"}:
        if group_by is None or group_by not in frame.columns or "mean" not in frame.columns:
            return None
        return _group_map(frame, group_by, "mean")
    if kind == "stat":
        column = {
            "average": "mean",
            "mean": "mean",
            "median": "median",
            "minimum": "min",
            "min": "min",
            "maximum": "max",
            "max": "max",
            "std": "std",
        }[stat or "mean"]
        if column not in frame.columns:
            return None
        value = frame.get_column(column)[0]
        return None if value is None else float(value)
    return None


def service_for(dataset_id: str, workspace: Path | None = None) -> QueryService:
    """Reuse one query service so pattern config and embeddings load once per dataset."""
    if dataset_id not in _SERVICES:
        _SERVICES[dataset_id] = QueryService(layer_for(dataset_id, workspace))
    return _SERVICES[dataset_id]


Limit: TypeAlias = tuple[str, str, Any]
_FILTER_OPS = {"==": "eq", "!=": "ne", ">": "gt", ">=": "ge", "<": "lt", "<=": "le", "IN": "in", "NOT_IN": "not_in"}


def _limit_value(value: Any) -> Any:
    if isinstance(value, list | tuple):
        return tuple(sorted((_limit_value(item) for item in value), key=repr))
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    return value


def plan_limits(plan: Any) -> list[Limit]:
    """Every limit the plan applies, filters and cohort steps alike, as (op, column, value) in one
    order: two plans that keep the same patients by the same limits compare equal."""
    if plan is None:
        return []
    limits = [
        (_FILTER_OPS.get(spec.operator, spec.operator), str(spec.column), _limit_value(spec.value))
        for spec in plan.filters
    ]
    limits += [(str(step.op), str(step.column), _limit_value(step.value)) for step in plan.cohort_steps]
    return sorted(limits, key=repr)


def ask(dataset_id: str, query: str, workspace: Path | None = None) -> dict[str, Any]:
    """Parse and execute one question. Returns the service result plus plan fields."""
    service = service_for(dataset_id, workspace)
    result = service.ask(query, dataset_id=dataset_id, dataset_version="golden")
    plan = result.plan
    payload = result.result if isinstance(result.result, dict) else None
    issues = list(result.issues)
    # Same rule as production: a hold or any error is not an answer.
    held = bool(result.requires_confirmation or (payload or {}).get("requires_confirmation"))
    executed = payload is not None and payload.get("success") is True
    success = executed and not held and not any(item.get("severity") == "error" for item in issues)
    return {
        "dataset_id": dataset_id,
        "query": query,
        "intent": plan.intent if plan else None,
        "metric": plan.metric if plan else None,
        "group_by": plan.group_by if plan else None,
        "limits": plan_limits(plan),
        "proportion": (plan.proportion_column, plan.proportion_value) if plan and plan.proportion_column else None,
        "rate": (plan.rate_column, plan.rate_value) if plan and plan.rate_column else None,
        "issues": issues,
        "confidence": result.confidence,
        "payload": payload,
        "success": success,
        "held": held,
        "failure_reason": result.failure_reason,
        "plan_line": result.plan_line,
        # What the app shows: the claim and table built from the analysis, nothing else.
        "presentation": present_analysis(result.analysis) if success else {"claim": None, "table": None},
    }


def values_match(expected: Any, actual: Any) -> bool:
    """Compare counts exactly and numeric aggregates within a small tolerance."""
    if isinstance(expected, dict) and isinstance(actual, dict):
        if set(expected) != set(actual):
            return False
        return all(values_match(expected[key], actual[key]) for key in expected)
    if isinstance(expected, float) or isinstance(actual, float):
        if expected is None or actual is None:
            return expected is actual
        return abs(float(expected) - float(actual)) <= 1e-4 + 1e-5 * abs(float(expected))
    return bool(expected == actual)


Outcome = Literal["answered", "held", "refused", "wrong"]


def classify(question: dict[str, Any], expected: Any, result: dict[str, Any]) -> tuple[Outcome, Any]:
    """Answered matches ground truth, refused is success False, wrong is a success that does not match."""
    actual = actual_value(question["kind"], question.get("stat"), question.get("group_by"), result.get("payload"))
    if not result["success"]:
        return ("held" if result.get("held") else "refused"), actual
    if expected is not None and values_match(expected, actual):
        return "answered", actual
    return "wrong", actual


# ----------------------------------------------------------------------------
# Presented-answer grading: read the answer the way the user does, from the claim
# sentence and the table the app shows, never from the raw execution frame.
# ----------------------------------------------------------------------------

HIDDEN = "hidden"
_PRESENTED_TOLERANCE = 1e-4  # table cells are rounded to 4 decimals
_TOTAL_COUNT = re.compile(r"^Total count: (\S+?)\.$")
_STAT_NAMES = ("mean", "median", "std", "min", "max")


def presented_answer(presentation: dict[str, Any] | None) -> Any:
    """The comparable value a reader gets from a presentation, or None when it states none.

    A count: the number in "Total count: N.". A one-row statistics table: its statistics by
    name. A table with one row per group: {group label: value}. A proportion or correlation:
    that number. HIDDEN when the app withheld the value to protect small cells.
    """
    if not presentation:
        return None
    table = presentation.get("table")
    claim = presentation.get("claim") or ""
    if not table or not table.get("rows"):
        match = _TOTAL_COUNT.match(claim)
        if not match:
            return None
        return int(match.group(1)) if match.group(1).isdigit() else HIDDEN
    columns, rows = list(table["columns"]), list(table["rows"])
    for single in ("proportion", "r"):
        if single in columns and len(rows) == 1:
            return _shown(rows[0][single])
    if "variable" in columns and len(rows) == 1:
        return {name: _shown(rows[0].get(name)) for name in _STAT_NAMES if name in columns}
    value_column = next((c for c in columns[1:] if c == "rate" or c == "count" or c.split(" ")[0] in _STAT_NAMES), None)
    if value_column is None:
        return _crosstab(columns, rows)
    return {str(row[columns[0]]): _shown(row[value_column]) for row in rows}


def _shown(value: Any) -> Any:
    return HIDDEN if value == SECONDARY_SUPPRESSED or (isinstance(value, str) and value.startswith("<")) else value


def _crosstab(columns: list[str], rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """A contingency table (first column the level, one column per group) as {group: {level: n}}."""
    return {group: {str(row[columns[0]]): _shown(row[group]) for row in rows} for group in columns[1:]}


def _is_hidden(value: Any) -> bool:
    if isinstance(value, dict):
        return any(_is_hidden(item) for item in value.values())
    return isinstance(value, str) and value == HIDDEN


def _labelled(expected: Any, column: str | None) -> Any:
    """Expected group keys as the app labels them ("1" on a coded header reads "Yes")."""
    if not isinstance(expected, dict) or column is None:
        return expected
    return {("null" if key == "null" else display_level(column, key)): value for key, value in expected.items()}


def presented_matches(expected: Any, shown: Any) -> bool:
    if expected is None or shown is None:
        return False
    if isinstance(expected, dict):
        if not isinstance(shown, dict) or set(expected) != set(shown):
            return False
        return all(presented_matches(expected[key], shown[key]) for key in expected)
    if isinstance(shown, dict) or isinstance(shown, str):
        return False
    if isinstance(expected, int) and not isinstance(expected, bool) and isinstance(shown, int):
        return expected == shown
    try:
        return abs(float(expected) - float(shown)) <= _PRESENTED_TOLERANCE
    except (TypeError, ValueError):
        return False


PresentedOutcome = Literal["answered", "held", "refused", "hidden", "wrong"]


def classify_presented(
    kind: str, stat: str | None, group_by: str | None, expected: Any, result: dict[str, Any]
) -> PresentedOutcome:
    """Grade what the app shows. "hidden": the value was withheld for small cells, which is
    neither an answer nor an error. "wrong": the app shows no value, or a different one."""
    if not result["success"]:
        return "held" if result.get("held") else "refused"
    shown = presented_answer(result.get("presentation"))
    if kind == "stat" and isinstance(shown, dict):
        column = {"average": "mean", "minimum": "min", "maximum": "max"}.get(stat or "mean", stat or "mean")
        shown = shown.get(column)
    if _is_hidden(shown):
        return "hidden"
    if kind == "crosstab" and isinstance(expected, dict) and group_by:
        expected = _labelled(expected, group_by)
    return "answered" if presented_matches(_labelled(expected, group_by), shown) else "wrong"


def classify_presented_generic(expected: Any, result: dict[str, Any]) -> PresentedOutcome:
    """classify_presented for probes that give only a truth: a number, or {group: number}.

    A number is matched against a count, proportion or correlation, or against any statistic
    of a one-row statistics table (the probe does not say which statistic it computed).
    """
    if not result["success"]:
        return "held" if result.get("held") else "refused"
    shown = presented_answer(result.get("presentation"))
    if _is_hidden(shown):
        return "hidden"
    if isinstance(expected, dict):
        return "answered" if presented_matches(_labelled(expected, result.get("group_by")), shown) else "wrong"
    if isinstance(shown, dict):
        return "answered" if any(presented_matches(expected, value) for value in shown.values()) else "wrong"
    return "answered" if presented_matches(expected, shown) else "wrong"
