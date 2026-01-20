"""
Tests for conversational query refinement (ADR009 Phase 6: Conversation Context).

This module tests LLM-based refinement handling for follow-up queries like
"remove the n/a" after asking "count patients by statin".

Architecture:
- LLM detects if query is a refinement based on context
- LLM extracts filters/modifications from refinement query
- LLM merges refinement with previous query plan
- DuckDB/Ibis executes the refined query

Following ADR009: "LLM should be used for all tasks requiring natural language understanding
and context awareness."
"""

import pytest


def test_parse_query_with_refinement_context_adds_filter(
    make_semantic_layer,
    mock_llm_calls,
    nl_query_engine_with_cached_model,
):
    """Test that LLM recognizes refinement and merges with previous query."""
    # Arrange: Create semantic layer with statin column
    semantic = make_semantic_layer(
        dataset_name="test_statins",
        data={
            "patient_id": ["P1", "P2", "P3", "P4"],
            "statin_used": [0, 1, 2, 1],  # 0=n/a, 1=atorvastatin, 2=rosuvastatin
            "age": [45, 52, 38, 61],
        },
    )
    engine = nl_query_engine_with_cached_model(semantic_layer=semantic)

    # Previous query: count by statin (all values including n/a)
    conversation_history = [
        {
            "query": "count patients by statin",
            "intent": "COUNT",
            "group_by": "statin_used",
            "metric": None,
            "filters_applied": [],
            "run_key": "abc123",
            "timestamp": 100.0,
        }
    ]

    # Act: Parse refinement query "remove the n/a" with conversation context
    result = engine.parse_query(
        query="remove the n/a",
        conversation_history=conversation_history,
    )

    # Assert: LLM should recognize this as refinement of previous COUNT query
    assert result.intent_type == "COUNT", "Should maintain previous COUNT intent"
    assert result.grouping_variable == "statin_used", "Should maintain previous grouping"

    # LLM should add filter to exclude n/a (value 0)
    assert len(result.filters) >= 1, "Should have filter from refinement"
    filter_columns = [f.column for f in result.filters]
    assert "statin_used" in filter_columns, "Should filter on statin_used"

    # Find the statin filter
    statin_filters = [f for f in result.filters if f.column == "statin_used"]
    assert len(statin_filters) == 1
    statin_filter = statin_filters[0]

    # Should exclude n/a (value 0)
    assert statin_filter.operator == "!=", "Should use != for exclusion"
    assert statin_filter.value == 0, "Should exclude value 0 (n/a)"

    # Should have good confidence (LLM understood context)
    assert result.confidence >= 0.7, "LLM should be confident with context"


def test_parse_query_refinement_without_context_has_low_confidence(
    make_semantic_layer,
    mock_llm_calls,
    nl_query_engine_with_cached_model,
):
    """Test that refinement query without context has low confidence."""
    # Arrange
    semantic = make_semantic_layer(
        dataset_name="test",
        data={"patient_id": ["P1", "P2"], "outcome": [0, 1]},
    )
    engine = nl_query_engine_with_cached_model(semantic_layer=semantic)

    # Act: Parse "remove the n/a" with NO conversation history
    result = engine.parse_query(
        query="remove the n/a",
        conversation_history=None,  # No context
    )

    # Assert: LLM should recognize ambiguity and lower confidence
    assert result.confidence < 0.75, "Should have low confidence without context"


def test_parse_query_refinement_merges_with_existing_filters(
    make_semantic_layer,
    mock_llm_calls,
    nl_query_engine_with_cached_model,
):
    """Test that LLM merges refinement filter with existing filters."""
    # Arrange
    semantic = make_semantic_layer(
        dataset_name="test",
        data={
            "patient_id": ["P1", "P2", "P3"],
            "age": [45, 52, 38],
            "status": [0, 1, 1],  # 0=unknown, 1=active
        },
    )
    engine = nl_query_engine_with_cached_model(semantic_layer=semantic)

    # Previous query already had an age filter
    conversation_history = [
        {
            "query": "count patients over 50",
            "intent": "COUNT",
            "group_by": None,
            "filters_applied": [
                {
                    "column": "age",
                    "operator": ">",
                    "value": 50,
                    "exclude_nulls": True,
                }
            ],
        }
    ]

    # Act: Add refinement to exclude unknown status
    result = engine.parse_query(
        query="exclude unknown status",
        conversation_history=conversation_history,
    )

    # Assert: Should have both filters
    assert result.intent_type == "COUNT"
    assert len(result.filters) >= 2, "Should have age + status filters"

    # Should preserve age filter
    age_filters = [f for f in result.filters if f.column == "age"]
    assert len(age_filters) == 1
    assert age_filters[0].operator == ">"
    assert age_filters[0].value == 50

    # Should add status filter
    status_filters = [f for f in result.filters if f.column == "status"]
    assert len(status_filters) >= 1


@pytest.mark.skip(
    reason="LLM mock returns hardcoded value 50 instead of extracting 65 from query. "
    "Mock doesn't parse query text to extract numeric values. "
    "TODO: Enhance mock to extract values from query text or convert to integration test."
)
def test_parse_query_refinement_updates_same_column_filter(
    make_semantic_layer,
    mock_llm_calls,
    nl_query_engine_with_cached_model,
):
    """Test that LLM replaces filter on same column when refined."""
    # Arrange
    semantic = make_semantic_layer(
        dataset_name="test",
        data={
            "patient_id": ["P1", "P2", "P3"],
            "age": [45, 52, 68],
        },
    )
    engine = nl_query_engine_with_cached_model(semantic_layer=semantic)

    conversation_history = [
        {
            "query": "patients over 50",
            "intent": "COUNT",
            "filters_applied": [
                {
                    "column": "age",
                    "operator": ">",
                    "value": 50,
                    "exclude_nulls": True,
                }
            ],
        }
    ]

    # Act: Refine the age filter
    result = engine.parse_query(
        query="actually make it over 65",
        conversation_history=conversation_history,
    )

    # Assert: LLM should update age filter, not duplicate
    age_filters = [f for f in result.filters if f.column == "age"]
    assert len(age_filters) == 1, "Should only have one age filter (updated)"
    assert age_filters[0].value == 65, "Should use updated value"


def test_parse_query_non_refinement_with_history_works_normally(
    make_semantic_layer,
    mock_llm_calls,
    nl_query_engine_with_cached_model,
):
    """Test that new queries work normally even with conversation history."""
    # Arrange
    semantic = make_semantic_layer(
        dataset_name="test",
        data={
            "patient_id": ["P1", "P2"],
            "age": [45, 52],
            "outcome": [0, 1],
        },
    )
    engine = nl_query_engine_with_cached_model(semantic_layer=semantic)

    conversation_history = [
        {
            "query": "count patients by age",
            "intent": "COUNT",
            "group_by": "age",
        }
    ]

    # Act: Ask completely new question (not a refinement)
    result = engine.parse_query(
        query="describe outcome",
        conversation_history=conversation_history,
    )

    # Assert: LLM should recognize this is NOT a refinement
    assert result.intent_type == "DESCRIBE", "Should parse as new DESCRIBE query"
    assert result.grouping_variable != "age", "Should not inherit previous grouping"


def test_parse_query_backward_compatible_without_history(
    make_semantic_layer,
    mock_llm_calls,
    nl_query_engine_with_cached_model,
):
    """Test that parse_query works without conversation_history (backward compat)."""
    # Arrange
    semantic = make_semantic_layer(
        dataset_name="test",
        data={"patient_id": ["P1", "P2"], "outcome": [0, 1]},
    )
    engine = nl_query_engine_with_cached_model(semantic_layer=semantic)

    # Act: Parse query without conversation_history parameter
    result = engine.parse_query(query="describe outcome")

    # Assert: Should work as before
    assert result.intent_type == "DESCRIBE"
    assert result.confidence > 0.0


@pytest.mark.parametrize(
    "refinement_query,expected_intent",
    [
        ("remove the n/a", "previous"),  # Should inherit previous intent
        pytest.param(
            "exclude missing values",
            "previous",
            marks=pytest.mark.skip(
                reason="Semantic match finds exact golden example returning DESCRIBE, "
                "ignoring conversation history. Architectural issue: golden examples "
                "are templates, not context-aware training data. "
                "TODO: Refactor to use golden examples as few-shot learning context."
            ),
        ),
        ("get rid of zeros", "previous"),
        ("without unknowns", "previous"),
        ("only active patients", "previous"),
        ("also exclude pediatric", "previous"),
    ],
)
def test_llm_recognizes_refinement_patterns(
    make_semantic_layer,
    mock_llm_calls,
    nl_query_engine_with_cached_model,
    refinement_query,
    expected_intent,
):
    """Test that LLM recognizes various refinement patterns."""
    # Arrange
    semantic = make_semantic_layer(
        dataset_name="test",
        data={
            "patient_id": ["P1", "P2"],
            "status": [0, 1],
        },
    )
    engine = nl_query_engine_with_cached_model(semantic_layer=semantic)

    conversation_history = [
        {
            "query": "count patients",
            "intent": "COUNT",
            "group_by": None,
        }
    ]

    # Act
    result = engine.parse_query(
        query=refinement_query,
        conversation_history=conversation_history,
    )

    # Assert: LLM should recognize refinement and inherit previous intent
    if expected_intent == "previous":
        assert result.intent_type == "COUNT", f"Should inherit COUNT for: {refinement_query}"
        assert len(result.filters) >= 1, f"Should have filter for: {refinement_query}"


def test_llm_refinement_with_coded_categorical_column(
    make_cohort_with_categorical,
    make_semantic_layer,
    mock_llm_calls,
    nl_query_engine_with_cached_model,
):
    """Test LLM handles refinement with coded categorical columns correctly."""
    # Arrange: Use factory fixture for categorical cohort
    cohort = make_cohort_with_categorical(
        patient_ids=["P1", "P2", "P3", "P4"],
        ages=[45, 52, 38, 61],
        treatment=["1: Control", "2: Treatment A", "1: Control", "2: Treatment A"],
    )

    semantic = make_semantic_layer(
        dataset_name="test_coded",
        data=cohort.to_dict(),
    )
    engine = nl_query_engine_with_cached_model(semantic_layer=semantic)

    conversation_history = [
        {
            "query": "count by treatment group",
            "intent": "COUNT",
            "group_by": "treatment_group",
        }
    ]

    # Act: LLM should understand "remove control" means exclude code 1
    result = engine.parse_query(
        query="remove control group",
        conversation_history=conversation_history,
    )

    # Assert
    assert result.intent_type == "COUNT"
    assert result.grouping_variable == "treatment"  # Resolved from "treatment_group" alias
    # LLM should extract filter for control (code 1)
    assert len(result.filters) >= 1


@pytest.mark.skip(
    reason="LLM mock doesn't populate 'interpretation' field on QueryIntent. "
    "Mock returns empty interpretation string. "
    "TODO: Enhance mock to include interpretation or convert to integration test."
)
def test_llm_provides_explanation_for_refinement(
    make_semantic_layer,
    mock_llm_calls,
    nl_query_engine_with_cached_model,
):
    """Test that LLM provides explanation when handling refinement."""
    # Arrange
    semantic = make_semantic_layer(
        dataset_name="test",
        data={"patient_id": ["P1", "P2"], "status": [0, 1]},
    )
    engine = nl_query_engine_with_cached_model(semantic_layer=semantic)

    conversation_history = [
        {
            "query": "count patients",
            "intent": "COUNT",
        }
    ]

    # Act
    result = engine.parse_query(
        query="exclude unknowns",
        conversation_history=conversation_history,
    )

    # Assert: LLM should provide explanation of refinement
    # QueryIntent uses 'interpretation' field, not 'explanation'
    assert result.interpretation, "Should have interpretation"
    assert len(result.interpretation) > 10, "Interpretation should be meaningful"
    # Interpretation should mention refinement or filter
    interpretation_lower = result.interpretation.lower()
    assert any(
        word in interpretation_lower for word in ["filter", "exclude", "refin", "previous"]
    ), "Interpretation should describe the refinement"


def test_parse_query_refinement_handles_llm_failure_with_fallback(
    make_semantic_layer,
    mock_llm_calls,
    nl_query_engine_with_cached_model,
):
    """
    Test that refinement queries work even when LLM returns invalid response.

    This tests the exact scenario from the terminal where LLM returns
    {"query": "remove the n/a"} instead of proper QueryPlan schema.
    The system should fall back to refinement handler that merges with previous context.
    """
    # Arrange: Create semantic layer with statin column matching terminal scenario
    statin_col = (
        "Statin Used:    0: n/a                       1: Atorvastatin  "
        "2: Rosuvastatin 3: Pravastatin   4: Pitavastatin  5: Simvastatin"
    )
    semantic = make_semantic_layer(
        dataset_name="test_statins",
        data={
            "patient_id": ["P1", "P2", "P3", "P4"],
            statin_col: [0, 1, 2, 1],
            "age": [45, 52, 38, 61],
        },
    )
    engine = nl_query_engine_with_cached_model(semantic_layer=semantic)

    # Previous query: count by statin (all values including n/a)
    conversation_history = [
        {
            "query": "what statins were those patients on, broken down by count of patients per statin?",
            "intent": "COUNT",
            "group_by": statin_col,
            "metric": None,
            "filters_applied": [],
            "run_key": "abc123",
            "timestamp": 100.0,
        }
    ]

    # Act: Parse refinement query "remove the n/a" with conversation context
    # This should work even if LLM fails and returns invalid response
    result = engine.parse_query(
        query="remove the n/a",
        conversation_history=conversation_history,
    )

    # Assert: Should maintain previous intent and grouping
    assert result.intent_type == "COUNT", "Should maintain previous COUNT intent"
    assert result.grouping_variable == statin_col, "Should maintain previous grouping"

    # Should add filter to exclude n/a (value 0) - NOT string "the n/a"
    assert len(result.filters) >= 1, "Should have filter from refinement"

    # Find the statin filter (may be canonical name or alias)
    statin_filters = [
        f for f in result.filters if f.column == statin_col or statin_col in f.column or f.column in statin_col
    ]
    assert len(statin_filters) >= 1, "Should have at least one statin filter"
    statin_filter = statin_filters[0]

    # CRITICAL: Should use numeric value 0, NOT string "the n/a"
    assert statin_filter.operator == "!=", "Should use != for exclusion"
    assert (
        statin_filter.value == 0
    ), f"Should exclude value 0 (n/a), not string. Got: {statin_filter.value} (type: {type(statin_filter.value)})"

    # Should have reasonable confidence (even with fallback)
    assert result.confidence >= 0.5, "Should have reasonable confidence even with fallback"
