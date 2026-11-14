"""Heal witnesses for query failure reporting and the DuckDB connection lock."""

from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock, patch

from clinical_analytics.core.query_plan import QueryPlan
from clinical_analytics.core.query_service import QueryService


def _service(layer: MagicMock) -> QueryService:
    service = QueryService(layer)
    service.conversation_manager.normalize_query = lambda question: question.strip()  # type: ignore[method-assign]
    return service


def test_ask_invalid_plan_error_key_becomes_an_issue() -> None:
    """Validation returns {valid: False, error: ...} with no issues list."""
    layer = MagicMock()
    layer._con_lock = threading.RLock()
    layer._generate_run_key.return_value = "run"
    layer._validate_query_plan.return_value = {
        "valid": False,
        "error": "Column 'missing' not found in dataset",
    }
    service = _service(layer)
    intent = MagicMock(confidence=0.4)
    plan = QueryPlan(intent="DESCRIBE", metric="missing", confidence=0.4)
    service.nl_engine.parse_query = MagicMock(return_value=intent)  # type: ignore[method-assign]
    service.nl_engine._intent_to_plan = MagicMock(return_value=plan)  # type: ignore[method-assign]

    result = service.ask("describe missing", dataset_id="ds")

    assert result.result is None
    assert any(item.get("severity") == "error" and "missing" in item.get("message", "") for item in result.issues)
    layer.execute_query_plan.assert_not_called()


def test_ask_execution_success_false_is_an_error_issue() -> None:
    layer = MagicMock()
    layer._con_lock = threading.RLock()
    layer._generate_run_key.return_value = "run"
    layer._validate_query_plan.return_value = {"valid": True, "error": None}
    layer.execute_query_plan.return_value = {
        "success": False,
        "warnings": ["Execution error: boom"],
    }
    service = _service(layer)
    intent = MagicMock(confidence=0.9)
    plan = QueryPlan(intent="COUNT", entity_key="patient_id", confidence=0.9)
    service.nl_engine.parse_query = MagicMock(return_value=intent)  # type: ignore[method-assign]
    service.nl_engine._intent_to_plan = MagicMock(return_value=plan)  # type: ignore[method-assign]

    result = service.ask("how many patients", dataset_id="ds")

    assert any(item.get("severity") == "error" and "boom" in item.get("message", "") for item in result.issues)


def test_ask_holds_connection_lock_across_overlapping_executions() -> None:
    layer = MagicMock()
    layer._con_lock = threading.RLock()
    layer.state_generation = 0
    layer._validate_query_plan.return_value = {"valid": True, "error": None}
    service = _service(layer)
    service.nl_engine.parse_query = MagicMock(return_value=MagicMock(confidence=0.9))  # type: ignore[method-assign]
    service.nl_engine._intent_to_plan = MagicMock(  # type: ignore[method-assign]
        return_value=QueryPlan(intent="COUNT", entity_key="patient_id", confidence=0.9)
    )
    spans: list[tuple[float, float]] = []

    def execute(*_args: object, **_kwargs: object) -> dict[str, object]:
        started = time.monotonic()
        time.sleep(0.15)
        spans.append((started, time.monotonic()))
        return {"success": False, "warnings": ["stub"]}

    layer.execute_query_plan.side_effect = execute
    with patch.object(QueryService, "_accounting_hold_reason", return_value=None):
        threads = [
            threading.Thread(target=service.ask, kwargs={"question": "one", "dataset_id": "ds"}) for _ in range(2)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

    assert len(spans) == 2
    first, second = sorted(spans)
    assert first[1] <= second[0]
