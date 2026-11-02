"""Ask Questions claim-first rendering (issue #52)."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from clinical_analytics.core.query_plan import QueryPlan

PAGE_PATH = (
    Path(__file__).resolve().parents[4] / "src" / "clinical_analytics" / "ui" / "pages" / "03_💬_Ask_Questions.py"
)


@pytest.fixture(scope="module")
def ask_page():
    spec = importlib.util.spec_from_file_location("ask_questions_claim_page", PAGE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_render_result_claim_appears_before_analysis_table(ask_page, monkeypatch):
    """Successful compare result leads with Claim before the analysis renderer."""
    # Arrange
    order: list[str] = []
    monkeypatch.setattr(
        ask_page.st,
        "success",
        lambda text, **kwargs: order.append(f"claim:{text}"),
    )
    monkeypatch.setattr(
        ask_page,
        "render_analysis_by_type",
        lambda *args, **kwargs: order.append("table"),
    )
    monkeypatch.setattr(ask_page, "_render_result_interpretation", lambda *args, **kwargs: None)
    monkeypatch.setattr(ask_page, "_render_interpretation_inline_compact", lambda *args, **kwargs: None)
    monkeypatch.setattr(ask_page, "_render_query_interpretation", lambda *args, **kwargs: None)
    monkeypatch.setattr(ask_page, "_render_llm_follow_ups", lambda *args, **kwargs: None)
    monkeypatch.setattr(ask_page, "_render_proactive_questions", lambda *args, **kwargs: None)

    result = {
        "type": "comparison",
        "test_type": "t_test",
        "outcome_col": "ldl",
        "group_col": "sex",
        "mean_diff": -5.0,
        "ci_lower": -8.0,
        "ci_upper": -2.0,
        "n": 80,
    }
    plan = QueryPlan(intent="COMPARE_GROUPS", metric="ldl", group_by="sex", confidence=0.9)
    context = SimpleNamespace(inferred_intent=MagicMock(value="COMPARE_GROUPS"))

    # Act
    ask_page.render_result(
        result,
        context,
        run_key="rk",
        query_text="compare ldl by sex",
        query_plan=plan,
    )

    # Assert
    assert order[0].startswith("claim:")
    assert "Claim:" in order[0]
    assert "5.0" in order[0] or "5" in order[0]
    assert "table" in order
    assert order.index("table") > 0
    assert result.get("claim")
