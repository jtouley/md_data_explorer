"""Ask Questions hold: cache, confirm, and refusal rendering. Issue #52."""

import hashlib
import importlib.util
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from clinical_analytics.core.nl_query_config import AUTO_EXECUTE_CONFIDENCE_THRESHOLD
from clinical_analytics.core.query_plan import QueryPlan

PAGE_PATH = (
    Path(__file__).resolve().parents[4] / "src" / "clinical_analytics" / "ui" / "pages" / "03_💬_Ask_Questions.py"
)
QUERY = "average ldl by sex"
DATASET = "v1"


@pytest.fixture(scope="module")
def ask_page():
    """Load the page module without treating ui/pages as a package."""
    spec = importlib.util.spec_from_file_location("ask_questions_hold_page", PAGE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def plan() -> QueryPlan:
    return QueryPlan(intent="COMPARE_GROUPS", metric="ldl", group_by="sex", confidence=0.3)


def _cache_key() -> str:
    digest = hashlib.sha256(QUERY.encode("utf-8")).hexdigest()[:16]
    return f"exec_result:{DATASET}:{digest}"


def _pending_key() -> str:
    digest = hashlib.sha256(QUERY.encode("utf-8")).hexdigest()[:16]
    return f"pending_confirmation:{DATASET}:{digest}"


def _layer(result: dict) -> MagicMock:
    layer = MagicMock()
    layer.execute_query_plan.return_value = result
    return layer


def test_ask_questions_low_confidence_calls_execute_and_does_not_cache(ask_page, monkeypatch, plan):
    """A refusal calls execute with the real threshold and is not cached."""
    session: dict = {}
    monkeypatch.setattr(ask_page.st, "session_state", session)
    monkeypatch.setattr(ask_page.st, "markdown", lambda text: None)
    monkeypatch.setattr(ask_page.st, "button", lambda *args, **kwargs: False)
    layer = _layer(
        {
            "success": False,
            "requires_confirmation": True,
            "failure_reason": "Confirmation required before execution.",
            "result": None,
        }
    )

    result = ask_page.run_planned_query(layer, plan, DATASET, QUERY)

    layer.execute_query_plan.assert_called_once_with(
        plan,
        confidence_threshold=AUTO_EXECUTE_CONFIDENCE_THRESHOLD,
        query_text=QUERY,
        confirmed=False,
    )
    assert _cache_key() not in session
    assert result["success"] is False


def test_ask_questions_cached_success_skips_execute(ask_page, monkeypatch, plan):
    """A successful cache hit does not call execute again."""
    cached = {"success": True, "run_key": "rk", "result": {"n": 1}}
    session = {_cache_key(): cached}
    monkeypatch.setattr(ask_page.st, "session_state", session)
    layer = _layer({"success": False})

    result = ask_page.run_planned_query(layer, plan, DATASET, QUERY)

    layer.execute_query_plan.assert_not_called()
    assert result == cached


def test_ask_questions_force_rerun_calls_execute_despite_cache(ask_page, monkeypatch, plan):
    """force_rerun bypasses a successful cache entry."""
    session = {_cache_key(): {"success": True, "run_key": "old"}, f"force_rerun:{DATASET}": True}
    monkeypatch.setattr(ask_page.st, "session_state", session)
    monkeypatch.setattr(ask_page.st, "markdown", lambda text: None)
    layer = _layer({"success": True, "run_key": "new", "result": {}})

    ask_page.run_planned_query(layer, plan, DATASET, QUERY)

    layer.execute_query_plan.assert_called_once()


def test_ask_questions_low_confidence_shows_plan_line(ask_page, monkeypatch, plan):
    """Confirm's callback stores the pending flag."""
    session: dict = {}
    clicks: dict = {}
    lines: list[str] = []
    monkeypatch.setattr(ask_page.st, "session_state", session)
    monkeypatch.setattr(ask_page.st, "markdown", lambda text: lines.append(text))
    monkeypatch.setattr(ask_page.st, "rerun", lambda: None)

    def button(label, on_click=None, **kwargs):
        clicks[label] = on_click
        return False

    monkeypatch.setattr(ask_page.st, "button", button)
    layer = _layer(
        {
            "success": False,
            "requires_confirmation": True,
            "failure_reason": "Confirmation required before execution.",
            "result": None,
        }
    )

    ask_page.run_planned_query(layer, plan, DATASET, QUERY)

    assert any("COMPARE_GROUPS" in line and "group comparison" in line for line in lines)
    clicks["Confirm"]()
    assert session[_pending_key()] is True


def test_ask_questions_confirm_calls_execute_with_confirmed_true(ask_page, monkeypatch, plan):
    """A stored pending flag is passed as confirmed=True on the next call."""
    session = {_pending_key(): True}
    monkeypatch.setattr(ask_page.st, "session_state", session)
    layer = _layer({"success": True, "run_key": "rk", "result": {}})

    ask_page.run_planned_query(layer, plan, DATASET, QUERY)

    assert layer.execute_query_plan.call_args.kwargs["confirmed"] is True


def test_ask_questions_reject_clears_pending_and_does_not_confirm(ask_page, monkeypatch, plan):
    """Reject deletes the pending flag."""
    session = {_pending_key(): True}
    clicks: dict = {}
    monkeypatch.setattr(ask_page.st, "session_state", session)
    monkeypatch.setattr(ask_page.st, "markdown", lambda text: None)

    def button(label, on_click=None, **kwargs):
        clicks[label] = on_click
        return False

    monkeypatch.setattr(ask_page.st, "button", button)
    layer = _layer(
        {
            "success": False,
            "requires_confirmation": True,
            "failure_reason": "Confirmation required before execution.",
            "result": None,
        }
    )

    ask_page.run_planned_query(layer, plan, DATASET, QUERY)
    clicks["Reject"]()

    assert _pending_key() not in session
    assert layer.execute_query_plan.call_args.kwargs["confirmed"] is True


def test_ask_questions_high_confidence_calls_execute_without_confirmed(ask_page, monkeypatch, plan):
    """No pending flag means confirmed=False."""
    session: dict = {}
    monkeypatch.setattr(ask_page.st, "session_state", session)
    layer = _layer({"success": True, "run_key": "rk", "result": {}})

    ask_page.run_planned_query(layer, plan, DATASET, QUERY)

    assert layer.execute_query_plan.call_args.kwargs["confirmed"] is False
    assert _pending_key() not in session


def test_ask_questions_incomplete_plan_has_no_confirm_control(ask_page, monkeypatch, plan):
    """Incomplete refusals show the plan line and do not create Confirm."""
    labels: list[str] = []
    monkeypatch.setattr(ask_page.st, "session_state", {})
    monkeypatch.setattr(ask_page.st, "markdown", lambda text: None)
    monkeypatch.setattr(ask_page.st, "button", lambda label, **kwargs: labels.append(label))
    layer = _layer(
        {
            "success": False,
            "requires_confirmation": False,
            "failure_reason": "COMPARE_GROUPS intent requires both metric and group_by",
            "result": None,
        }
    )

    ask_page.run_planned_query(layer, plan, DATASET, QUERY)

    assert "Confirm" not in labels


def test_ask_questions_execution_failure_does_not_render_plan_line(ask_page, monkeypatch, plan):
    """An execution failure keeps its run key and does not show Confirm."""
    lines: list[str] = []
    labels: list[str] = []
    session: dict = {}
    monkeypatch.setattr(ask_page.st, "session_state", session)
    monkeypatch.setattr(ask_page.st, "markdown", lambda text: lines.append(text))
    monkeypatch.setattr(ask_page.st, "button", lambda label, **kwargs: labels.append(label))
    layer = _layer({"success": False, "run_key": "rk", "warnings": ["Execution error: boom"], "result": None})

    ask_page.run_planned_query(layer, plan, DATASET, QUERY)

    assert lines == []
    assert labels == []
    assert _cache_key() not in session
