"""Run a natural-language question against a golden dataset CSV without the UI.

Dev and test use only: results carry raw rows with no small-cell suppression, so this
module must never be exposed through the production API.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Literal

import polars as pl

from clinical_analytics.core.query_service import QueryService
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
        "issues": issues,
        "confidence": result.confidence,
        "payload": payload,
        "success": success,
        "held": held,
        "failure_reason": result.failure_reason,
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
