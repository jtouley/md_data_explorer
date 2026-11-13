"""
Tests for AsyncQueryService - async wrapper around core QueryService.

Following TDD: Red phase - tests written before implementation.
"""

import json
import re
from collections.abc import Iterator
from typing import Any
from unittest.mock import MagicMock, patch

import polars as pl
import pytest

from clinical_analytics.analysis.grouping import MAX_GROUP_LEVELS, is_small_level
from clinical_analytics.api.services.query_service import AsyncQueryService
from clinical_analytics.core.result_presentation import to_json_safe

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_semantic_layer():
    """Mock SemanticLayer for unit tests."""
    mock = MagicMock()
    mock.get_column_alias_index.return_value = {"age": "age", "outcome": "outcome"}
    return mock


@pytest.fixture
def mock_core_query_service():
    """Mock core QueryService for unit tests."""
    with patch("clinical_analytics.api.services.query_service.CoreQueryService") as mock_cls:
        mock_instance = MagicMock()
        mock_cls.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def async_query_service(mock_semantic_layer, mock_core_query_service):
    """Create AsyncQueryService with mocked dependencies."""
    return AsyncQueryService(mock_semantic_layer)


# ============================================================================
# Test Classes
# ============================================================================


class TestAsyncQueryServiceDelegation:
    """Tests for AsyncQueryService delegating to core QueryService."""

    @pytest.mark.asyncio
    async def test_async_query_service_delegates_to_core(self, async_query_service, mock_core_query_service):
        """Verify async service delegates query execution to core service."""
        # Arrange
        from clinical_analytics.core.query_service import QueryResult as CoreQueryResult

        mock_core_query_service.ask.return_value = CoreQueryResult(
            plan=MagicMock(),
            issues=[],
            result={"test": "data"},
            confidence=0.95,
            run_key="test_run_key",
            context=MagicMock(),
        )

        # Act
        result = await async_query_service.execute_query(
            query="What is the average age?",
            dataset_id="test_dataset",
        )

        # Assert
        mock_core_query_service.ask.assert_called_once()
        call_args = mock_core_query_service.ask.call_args
        assert call_args.kwargs["question"] == "What is the average age?"
        assert result is not None


class TestAsyncQueryServiceQueryId:
    """Tests for query ID generation."""

    @pytest.mark.asyncio
    async def test_async_query_service_returns_query_id(self, async_query_service, mock_core_query_service):
        """Verify submit_query returns a unique query ID."""
        # Arrange
        from clinical_analytics.core.query_service import QueryResult as CoreQueryResult

        mock_core_query_service.ask.return_value = CoreQueryResult(
            plan=MagicMock(),
            issues=[],
            result={"test": "data"},
            confidence=0.9,
            run_key="run_key",
            context=MagicMock(),
        )

        # Act
        query_id = await async_query_service.submit_query(
            query="What is the average age?",
            dataset_id="test_dataset",
        )

        # Assert
        assert query_id is not None
        assert isinstance(query_id, str)
        assert query_id.startswith("qry_")


class TestAsyncQueryServiceStreaming:
    """Tests for SSE streaming progress events."""

    @pytest.mark.asyncio
    async def test_async_query_service_streams_progress_events(self, async_query_service, mock_core_query_service):
        """Verify stream_events emits SSE events for query progress."""
        # Arrange
        from clinical_analytics.core.query_service import QueryResult as CoreQueryResult

        mock_core_query_service.ask.return_value = CoreQueryResult(
            plan=MagicMock(),
            issues=[],
            result={"test": "data"},
            confidence=0.9,
            run_key="run_key",
            context=MagicMock(),
        )

        query_id = await async_query_service.submit_query(
            query="What is the average age?",
            dataset_id="test_dataset",
        )

        # Act
        events = []
        async for event in async_query_service.stream_events(query_id):
            events.append(event)
            if event.event == "query_completed":
                break

        # Assert
        assert len(events) >= 2  # At least started and completed
        event_types = [e.event for e in events]
        assert "query_started" in event_types
        assert "query_completed" in event_types


class TestAsyncQueryServiceValidation:
    """Tests for input validation."""

    @pytest.mark.asyncio
    async def test_async_query_service_invalid_dataset_raises_error(self, async_query_service, mock_core_query_service):
        """Verify invalid dataset ID raises appropriate error."""
        # Arrange
        mock_core_query_service.ask.side_effect = ValueError("Dataset not found")

        # Act & Assert
        with pytest.raises(ValueError, match="Dataset not found"):
            await async_query_service.execute_query(
                query="What is the average age?",
                dataset_id="nonexistent_dataset",
            )

    @pytest.mark.asyncio
    async def test_async_query_service_empty_query_raises_error(self, async_query_service, mock_core_query_service):
        """Verify empty query raises validation error."""
        # Act & Assert
        with pytest.raises(ValueError, match="Query cannot be empty"):
            await async_query_service.execute_query(
                query="",
                dataset_id="test_dataset",
            )

    @pytest.mark.asyncio
    async def test_async_query_service_whitespace_query_raises_error(
        self, async_query_service, mock_core_query_service
    ):
        """Verify whitespace-only query raises validation error."""
        # Act & Assert
        with pytest.raises(ValueError, match="Query cannot be empty"):
            await async_query_service.execute_query(
                query="   ",
                dataset_id="test_dataset",
            )


class TestAsyncQueryServiceSerialization:
    """Tests for result serialization (DataFrame to JSON-serializable dict)."""

    def test_serialize_result_none_returns_none(self, async_query_service):
        """Verify None result returns None."""
        result = async_query_service._serialize_result(None)
        assert result is None

    def test_serialize_result_dict_returns_dict(self, async_query_service):
        """Verify simple dict is returned as-is."""
        input_dict = {"mean": 45.5, "count": 100}
        result = async_query_service._serialize_result(input_dict)
        assert result == {"mean": 45.5, "count": 100}

    def test_serialize_result_polars_dataframe_converts_to_dict(self, async_query_service):
        """Verify Polars DataFrame is converted to dict with table structure."""
        import polars as pl

        df = pl.DataFrame({"name": ["Alice", "Bob"], "age": [30, 25]})
        result = async_query_service._serialize_result(df)

        assert "table" in result
        assert "row_count" in result
        assert result["row_count"] == 2
        assert result["table"]["columns"] == ["name", "age"]
        assert len(result["table"]["rows"]) == 2
        assert result["table"]["rows"][0] == {"name": "Alice", "age": 30}

    def test_serialize_result_dict_with_dataframe_value(self, async_query_service):
        """Verify dict containing DataFrame value is serialized correctly."""
        import polars as pl

        df = pl.DataFrame({"value": [1, 2, 3]})
        input_dict = {"summary": "test", "data": df}
        result = async_query_service._serialize_result(input_dict)

        assert result["summary"] == "test"
        assert "table" in result["data"]
        assert result["data"]["row_count"] == 3

    def test_get_result_preview_polars_dataframe_serializes(self, async_query_service):
        """Verify result preview serializes DataFrame via _serialize_result."""
        import polars as pl

        df = pl.DataFrame({"id": list(range(100))})
        result = async_query_service._get_result_preview(df)

        # _serialize_result wraps in table/row_count structure
        assert result["row_count"] == 100
        assert "table" in result
        assert len(result["table"]["rows"]) == 5  # Preview limits to 5 rows

    def test_get_result_preview_none_returns_none(self, async_query_service):
        """Verify None input returns None."""
        result = async_query_service._get_result_preview(None)
        assert result is None

    def test_get_result_preview_dict_returns_first_3_keys(self, async_query_service):
        """Verify dict preview returns first 3 keys."""
        input_dict = {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5}
        result = async_query_service._get_result_preview(input_dict)

        assert len(result) == 3
        assert "a" in result
        assert "b" in result

    def test_get_result_preview_nested_dict_with_dataframe_serializes(self, async_query_service):
        """Verify nested dict containing DataFrame is fully serialized."""
        import json

        import polars as pl

        # This mirrors the real result structure from CoreQueryService
        df = pl.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})
        nested_result = {"analysis": {"data": df, "title": "Test"}}

        result = async_query_service._get_result_preview(nested_result)

        # Should be fully JSON serializable - no DataFrame objects
        json_str = json.dumps(result)  # Should not raise
        assert "col1" in json_str
        assert "analysis" in result

    def test_serialize_result_nested_dict_with_dataframe_recursive(self, async_query_service):
        """Verify _serialize_result handles deeply nested DataFrames."""
        import json

        import polars as pl

        df = pl.DataFrame({"x": [1, 2]})
        deep_nested = {"level1": {"level2": {"level3": df}}}

        result = async_query_service._serialize_result(deep_nested)

        # Must be JSON serializable
        json_str = json.dumps(result)
        assert "table" in json_str
        assert "columns" in json_str
        assert "level1" in result


class TestAsyncQueryServiceAnalysisPayload:
    """Completed queries return the claim and statistics table, not the cohort rows."""

    @pytest.mark.asyncio
    async def test_submit_query_withAnalysis_returnsClaimFirstPayload(
        self, async_query_service, mock_core_query_service
    ):
        # Arrange
        from clinical_analytics.core.query_service import QueryResult as CoreQueryResult

        analysis = {
            "type": "count",
            "total_count": 50,
            "grouped_by": "sex",
            "group_counts": [{"sex": "F", "count": 30}, {"sex": "M", "count": 20}],
            "chart_spec": {"type": "bar", "x": "sex"},
        }
        mock_core_query_service.ask.return_value = CoreQueryResult(
            plan=MagicMock(),
            issues=[],
            result={"success": True, "result": "raw cohort rows"},
            confidence=0.9,
            run_key="run_key",
            context=None,
            analysis=analysis,
        )

        # Act
        query_id = await async_query_service.submit_query(query="how many by sex?", dataset_id="ds")
        events = [event async for event in async_query_service.stream_events(query_id)]
        stored = await async_query_service.get_result(query_id)

        # Assert
        assert stored.result["claim"] == "50 total across 2 sex groups."
        assert stored.result["analysis_type"] == "count"
        assert stored.result["table"]["rows"] == [{"sex": "F", "count": 30}, {"sex": "M", "count": 20}]
        assert "analysis" not in stored.result
        assert "raw cohort rows" not in str(stored.result)
        completed = next(e for e in events if e.event == "query_completed")
        preview = completed.data["result_preview"]
        assert preview["claim"] == stored.result["claim"]
        assert "analysis" not in preview

    @pytest.mark.asyncio
    async def test_execute_query_withoutAnalysis_returnsNoRawRows(self, async_query_service, mock_core_query_service):
        # Arrange
        from clinical_analytics.core.query_service import QueryResult as CoreQueryResult

        mock_core_query_service.ask.return_value = CoreQueryResult(
            plan=MagicMock(),
            issues=[],
            result={"test": "data"},
            confidence=0.9,
            run_key="run_key",
            context=None,
        )

        # Act
        result = await async_query_service.execute_query(query="anything", dataset_id="ds")

        # Assert: execution rows are never returned in place of a presented answer
        assert result.result is None


class TestAsyncQueryServiceConversation:
    """session_id selects conversation history for the parser; it is never the upload id."""

    @staticmethod
    def _completed(plan):
        from clinical_analytics.core.query_service import QueryResult as CoreQueryResult

        return CoreQueryResult(
            plan=plan, issues=[], result=None, confidence=0.9, run_key="rk", context=None, analysis=None
        )

    @pytest.mark.asyncio
    async def test_execute_query_secondTurn_receivesFirstTurnAsHistory(
        self, async_query_service, mock_core_query_service
    ):
        # Arrange
        from clinical_analytics.core.query_plan import FilterSpec, QueryPlan

        first_plan = QueryPlan(
            intent="COUNT",
            group_by="sex",
            confidence=0.9,
            filters=[FilterSpec(column="age", operator=">", value=50)],
        )
        mock_core_query_service.ask.return_value = self._completed(first_plan)

        # Act
        await async_query_service.execute_query(query="count by sex", dataset_id="ds", session_id="sess_1")
        await async_query_service.execute_query(query="now by site", dataset_id="ds", session_id="sess_1")
        await async_query_service.execute_query(query="how many?", dataset_id="ds", session_id="sess_other")

        # Assert
        first, second, other = mock_core_query_service.ask.call_args_list
        assert first.kwargs["conversation_history"] == []
        assert second.kwargs["conversation_history"][0] == {
            "query": "count by sex",
            "intent": "COUNT",
            "metric": None,
            "group_by": "sex",
            "filters_applied": [{"column": "age", "operator": ">", "value": 50, "exclude_nulls": True}],
        }
        assert other.kwargs["conversation_history"] == []
        assert all("upload_id" not in call.kwargs for call in mock_core_query_service.ask.call_args_list)

    @pytest.mark.asyncio
    async def test_submit_query_contextWithoutHistory_usedAsPreviousTurn(
        self, async_query_service, mock_core_query_service
    ):
        # Arrange
        mock_core_query_service.ask.return_value = self._completed(None)

        # Act
        query_id = await async_query_service.submit_query(
            query="now by site",
            dataset_id="ds",
            session_id="sess_new",
            context={"previous_intent": "COUNT", "previous_group_by": "sex"},
        )
        _ = [event async for event in async_query_service.stream_events(query_id)]

        # Assert
        history = mock_core_query_service.ask.call_args.kwargs["conversation_history"]
        assert [(turn["intent"], turn["group_by"]) for turn in history] == [("COUNT", "sex")]

    def test_record_turn_manyTurns_keepsMostRecentTwenty(self, async_query_service):
        # Arrange
        from clinical_analytics.api.services.query_service import MAX_CONVERSATION_TURNS
        from clinical_analytics.core.query_plan import QueryPlan

        result = self._completed(QueryPlan(intent="DESCRIBE", metric="age", confidence=0.9))

        # Act
        for i in range(MAX_CONVERSATION_TURNS + 5):
            async_query_service._record_turn("sess", f"q{i}", result)

        # Assert
        history = async_query_service.conversation_history("sess")
        assert len(history) == MAX_CONVERSATION_TURNS
        assert history[0]["query"] == "q5"


class TestContextToTurn:
    @pytest.mark.parametrize(
        ("context", "expected"),
        [
            (None, None),
            ({}, None),
            ({"unrelated": 1}, None),
            (
                {"intent": "COUNT", "group_by": "sex", "query": "count by sex"},
                {"query": "count by sex", "intent": "COUNT", "metric": None, "group_by": "sex", "filters_applied": []},
            ),
            (
                {"previous_intent": "COMPARE_GROUPS", "previous_variables": ["ldl", "treatment"]},
                {
                    "query": None,
                    "intent": "COMPARE_GROUPS",
                    "metric": "ldl",
                    "group_by": "treatment",
                    "filters_applied": [],
                },
            ),
        ],
        ids=["none", "empty", "unrelated_keys", "history_shaped", "previous_variables"],
    )
    def test_context_to_turn_shapes_mappedToHistoryEntry(self, context, expected):
        # Arrange
        from clinical_analytics.api.services.query_service import context_to_turn

        # Act / Assert
        assert context_to_turn(context) == expected


class TestAsyncQueryServiceHeldTurns:
    @pytest.mark.asyncio
    async def test_execute_query_heldPlan_notRecordedAsTurn(self, async_query_service, mock_core_query_service):
        """A held plan has not run, so the next question must not refine it."""
        # Arrange
        from clinical_analytics.core.query_plan import QueryPlan
        from clinical_analytics.core.query_service import QueryResult as CoreQueryResult

        mock_core_query_service.ask.return_value = CoreQueryResult(
            plan=QueryPlan(intent="DESCRIBE", metric="age", confidence=0.5),
            issues=[],
            result=None,
            confidence=0.5,
            run_key=None,
            context=None,
            requires_confirmation=True,
            failure_reason="Confirmation required before execution.",
        )

        # Act
        await async_query_service.execute_query(query="average age", dataset_id="ds", session_id="sess")

        # Assert
        assert async_query_service.conversation_history("sess") == []


def _numbers_in(value: Any) -> Iterator[float]:
    """Every number in a JSON body: leaves, dict keys and numbers written inside strings.

    The random query id is skipped; its hex digits are not data.
    """
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "query_id":
                continue
            yield from _numbers_in(key)
            yield from _numbers_in(item)
    elif isinstance(value, list):
        for item in value:
            yield from _numbers_in(item)
    elif isinstance(value, bool) or value is None:
        return
    elif isinstance(value, int | float):
        yield float(value)
    elif isinstance(value, str):
        yield from (float(n) for n in re.findall(r"\d+(?:\.\d+)?", value))


class TestQueryResultSmallCells:
    """GET /api/queries/{id} returns the presented answer only, never the raw counts."""

    @pytest.mark.asyncio
    async def test_get_query_result_sparseFisher_bodyHasNoRecoverableSmallCount(
        self, async_query_service, mock_core_query_service
    ):
        # Arrange: sparse 2x2 (A: 15 alive / 1 dead, B: 1 alive / 3 dead); every cell is hidden
        from fastapi.testclient import TestClient

        from clinical_analytics.api.main import app
        from clinical_analytics.core.query_service import QueryResult as CoreQueryResult

        contingency = [{"died": "alive", "A": 15, "B": 1}, {"died": "dead", "A": 1, "B": 3}]
        sizes = {"A": 16, "B": 4}
        proportions = {"A": 1 / 16, "B": 3 / 4}
        analysis = {
            "type": "comparison",
            "test_type": "fisher_exact",
            "outcome_col": "died",
            "group_col": "arm",
            "groups": ["A", "B"],
            "group_sizes": sizes,
            "contingency": contingency,
            "event_level": "dead",
            "group_proportions": proportions,
            "odds_ratio": 45.0,
            "p_value": 0.0198,
            "n_used": 20,
            "n_excluded": 0,
            "summary": [{"timeline": 3.0, "at_risk": 4, "observed": 1}],
        }
        raw_rows = pl.DataFrame({"arm": ["A", "B"], "died": [1, 3]})
        mock_core_query_service.ask.return_value = CoreQueryResult(
            plan=MagicMock(),
            issues=[],
            result={"success": True, "result": raw_rows},
            confidence=0.9,
            run_key="run_key",
            context=None,
            analysis=analysis,
        )
        stored = await async_query_service.execute_query(query="death by arm?", dataset_id="ds")
        # cell counts, group sizes, proportions (and their complements) and the odds ratio
        forbidden = {15.0, 1.0, 3.0, 16.0, 4.0, 45.0}
        forbidden |= {round(v, 4) for p in proportions.values() for v in (p, 1 - p, 100 * p, 100 * (1 - p))}

        async def found(_query_id: str) -> tuple[Any, Any]:
            return async_query_service, stored

        # Act
        with patch("clinical_analytics.api.routes.queries.find_query_result", side_effect=found):
            body = TestClient(app).get(f"/api/queries/{stored.query_id}").json()

        # Assert: walk the whole body (confidence 0.9 and the p-value are allowed)
        assert body["result_data"]["claim"].startswith("The distribution of died differed by arm")
        leaked = sorted({n for n in _numbers_in(body) if n in forbidden or round(n, 4) in forbidden})
        assert leaked == [], f"GET body {body} shows {leaked}"

    @pytest.mark.parametrize(
        ("groups", "small"),
        [
            pytest.param({f"{80 + i / 10:.1f}": 1 for i in range(289)}, None, id="exact_ldl_values_289_columns"),
            pytest.param({"North": 60, "South": 50, "Rare": 3}, "Rare", id="site_with_3_patients"),
        ],
    )
    @pytest.mark.asyncio
    async def test_get_query_result_crosstabOfSmallLevels_payloadCarriesNoSmallLabel(
        self, async_query_service, mock_core_query_service, groups, small
    ):
        # Arrange: on main "compare treatment between ldl groups" stored one column per exact LDL
        # value (drug hidden, placebo 0), naming single patients' values and arm
        from fastapi.testclient import TestClient

        from clinical_analytics.api.main import app
        from clinical_analytics.core.query_service import QueryResult as CoreQueryResult

        analysis = {
            "type": "comparison",
            "test_type": "chi_square",
            "outcome_col": "treatment",
            "group_col": "ldl" if small is None else "site",
            "groups": list(groups),
            "group_sizes": groups,
            "contingency": [
                {"treatment": "drug", **{g: n - n // 2 for g, n in groups.items()}},
                {"treatment": "placebo", **{g: n // 2 for g, n in groups.items()}},
            ],
            "p_value": 0.15,
            "n_used": sum(groups.values()),
            "n_excluded": 0,
        }
        mock_core_query_service.ask.return_value = CoreQueryResult(
            plan=MagicMock(),
            issues=[],
            result={"success": True, "result": pl.DataFrame({"x": [1]})},
            confidence=0.9,
            run_key="run_key",
            context=None,
            analysis=analysis,
        )
        stored = await async_query_service.execute_query(query="compare treatment by group", dataset_id="ds")

        async def found(_query_id: str) -> tuple[Any, Any]:
            return async_query_service, stored

        # Act
        with patch("clinical_analytics.api.routes.queries.find_query_result", side_effect=found):
            body = TestClient(app).get(f"/api/queries/{stored.query_id}").json()

        # Assert: the payload itself, not only the claim; the stored result is the same object
        events = [to_json_safe(getattr(e, "data", e)) for e in async_query_service._events.get(stored.query_id, [])]
        text = json.dumps(body) + json.dumps(to_json_safe(stored.result)) + json.dumps(events, default=str)
        forbidden = [g for g in groups if is_small_level(groups[g])] if small else ["ldl=80.0", "ldl=80.1"]
        assert not any(f'"{label}"' in text for label in forbidden), text[:500]
        table = (body.get("result_data") or {}).get("table")
        assert table is None or len(table["columns"]) <= MAX_GROUP_LEVELS + 1

    @pytest.mark.asyncio
    async def test_get_query_result_heldQuery_bodyHasNoRawResult(self, async_query_service, mock_core_query_service):
        # Arrange: a held query whose execution result carries raw rows and analysis keys
        from fastapi.testclient import TestClient

        from clinical_analytics.api.main import app
        from clinical_analytics.core.query_service import QueryResult as CoreQueryResult

        raw = {
            "success": False,
            "requires_confirmation": True,
            "result": pl.DataFrame({"arm": ["A", "B"], "died": [7, 3]}),
            "contingency": [{"died": "dead", "A": 7, "B": 3}],
            "group_proportions": {"A": 0.7, "B": 0.3},
        }
        mock_core_query_service.ask.return_value = CoreQueryResult(
            plan=MagicMock(),
            issues=[],
            result=raw,
            confidence=0.6,
            run_key=None,
            context=None,
            requires_confirmation=True,
            plan_line="Compare died by arm",
        )
        query_id = await async_query_service.submit_query(query="death by arm?", dataset_id="ds")
        _ = [event async for event in async_query_service.stream_events(query_id)]
        stored = await async_query_service.get_result(query_id)

        async def found(_query_id: str) -> tuple[Any, Any]:
            return async_query_service, stored

        # Act
        with patch("clinical_analytics.api.routes.queries.find_query_result", side_effect=found):
            body = TestClient(app).get(f"/api/queries/{query_id}").json()

        # Assert: the plan line is there to confirm; no raw keys or values anywhere
        assert body["status"] == "awaiting_confirmation"
        assert body["plan_line"] == "Compare died by arm"
        text = json.dumps(body)
        assert not any(key in text for key in ('"result"', '"contingency"', '"group_proportions"', '"died"'))
        assert not {7.0, 3.0, 0.7, 0.3} & set(_numbers_in(body)), body
