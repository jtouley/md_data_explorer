"""
LLM result caching in the Ask Questions page (error translation + result interpretation).

Replaces earlier tests that re-implemented the caching logic inline and asserted on the copy.
These call the production functions with the LLM calls patched and assert on call counts
and on what lands in the ResultCache.

Test name follows: test_unit_scenario_expectedBehavior
"""

from unittest.mock import Mock, patch

import pytest

from clinical_analytics.core.result_cache import ResultCache
from clinical_analytics.ui.components.question_engine import AnalysisIntent

DATASET_VERSION = "llm_caching_v1"


def _run_execute(ask_questions_page, mock_session_state, sample_cohort, sample_context, formatted_result, run_key):
    """Execute analysis once with a semantic layer that returns formatted_result; return the cached result."""
    mock_session_state.clear()
    cache = ResultCache(max_size=10)
    mock_session_state["result_cache"] = cache
    sample_context.inferred_intent = AnalysisIntent.COUNT
    semantic_layer = Mock(spec=["format_execution_result"])
    semantic_layer.format_execution_result.return_value = dict(formatted_result)
    spinner = Mock(spec=["__enter__", "__exit__"])
    spinner.__enter__ = Mock(return_value=spinner)
    spinner.__exit__ = Mock(return_value=None)
    with (
        patch.object(ask_questions_page.st, "session_state", mock_session_state),
        patch.object(ask_questions_page.st, "spinner", Mock(return_value=spinner)),
    ):
        ask_questions_page.execute_analysis_with_idempotency(
            sample_cohort,
            sample_context,
            run_key,
            DATASET_VERSION,
            "query",
            execution_result={"success": True, "result": Mock(), "run_key": run_key},
            semantic_layer=semantic_layer,
        )
    cached = cache.get(run_key, DATASET_VERSION)
    assert cached is not None
    return cached.result


@pytest.mark.parametrize(
    ("existing_friendly", "expected_calls"),
    [(None, 1), ("", 1), ("already translated", 0)],
)
def test_execute_analysis_error_result_translatesOnlyWhenNotCached(
    ask_questions_page, mock_session_state, sample_cohort, sample_context, existing_friendly, expected_calls
):
    # Arrange
    formatted = {"type": "count", "error": "ColumnNotFoundError: ldl"}
    if existing_friendly is not None:
        formatted["friendly_error_message"] = existing_friendly

    # Act
    with patch.object(ask_questions_page, "translate_error_with_llm", return_value="friendly") as translate:
        result = _run_execute(
            ask_questions_page, mock_session_state, sample_cohort, sample_context, formatted, f"err-{existing_friendly}"
        )

    # Assert
    assert translate.call_count == expected_calls
    assert result["friendly_error_message"] == (existing_friendly or "friendly")


@pytest.mark.parametrize(
    ("formatted", "feature_enabled", "expected_calls"),
    [
        ({"type": "count", "total_count": 5}, True, 1),
        ({"type": "count", "total_count": 5, "llm_interpretation": "cached"}, True, 0),
        ({"type": "count", "error": "boom", "friendly_error_message": "x"}, True, 0),
        ({"type": "count", "total_count": 5}, False, 0),
    ],
    ids=["no_interpretation", "already_cached", "error_result", "feature_disabled"],
)
def test_execute_analysis_interpretation_callsLlmOnlyWhenNeeded(
    ask_questions_page, mock_session_state, sample_cohort, sample_context, formatted, feature_enabled, expected_calls
):
    # Arrange
    run_key = f"interp-{expected_calls}-{feature_enabled}-{'error' in formatted}-{'llm_interpretation' in formatted}"

    # Act
    with (
        patch.object(ask_questions_page, "ENABLE_RESULT_INTERPRETATION", feature_enabled),
        patch.object(ask_questions_page, "interpret_result_with_llm", return_value="insight") as interpret,
    ):
        result = _run_execute(ask_questions_page, mock_session_state, sample_cohort, sample_context, formatted, run_key)

    # Assert
    assert interpret.call_count == expected_calls
    if expected_calls:
        assert result["llm_interpretation"] == "insight"


@pytest.mark.parametrize(("cached_translation", "expected_calls"), [("cached", 0), (None, 1), ("", 1)])
def test_render_error_with_translation_cachedTranslation_skipsLlm(
    ask_questions_page, cached_translation, expected_calls
):
    # Arrange / Act
    with (
        patch.object(ask_questions_page, "translate_error_with_llm", return_value="fresh") as translate,
        patch.object(ask_questions_page.st, "error"),
        patch.object(ask_questions_page.st, "info") as info,
    ):
        ask_questions_page._render_error_with_translation("tech error", cached_translation=cached_translation)

    # Assert
    assert translate.call_count == expected_calls
    info.assert_called_once()
    assert (cached_translation or "fresh") in info.call_args.args[0]
