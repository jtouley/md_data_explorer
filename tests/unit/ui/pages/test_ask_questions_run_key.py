"""
Test stable run_key generation for idempotency.

Test name follows: test_unit_scenario_expectedBehavior
"""

import hashlib
import json

import pytest

from clinical_analytics.ui.components.question_engine import AnalysisContext, AnalysisIntent


@pytest.fixture
def sample_context():
    """Create sample AnalysisContext for testing."""
    context = AnalysisContext(
        inferred_intent=AnalysisIntent.DESCRIBE,
        primary_variable="outcome",
        grouping_variable="treatment",
        predictor_variables=["age", "sex"],
    )
    # Set confidence as attribute (not a dataclass field, but used via getattr)
    context.confidence = 0.9
    return context


def test_run_key_generation_same_query_produces_same_key(sample_context, ask_questions_page):
    """
    Test that same query + variables generates same run_key.

    Test name: test_unit_scenario_expectedBehavior
    """
    # Arrange: Same query, same variables
    dataset_version = "test_dataset_v1"
    query_text = "compare outcome by treatment"

    # Act: Generate run_key twice
    key1 = ask_questions_page.generate_run_key(dataset_version, query_text, sample_context)
    key2 = ask_questions_page.generate_run_key(dataset_version, query_text, sample_context)

    # Assert: Keys are identical
    assert key1 == key2
    assert len(key1) == 64  # SHA256 hex digest length


def test_run_key_generation_whitespace_normalization_produces_same_key(sample_context, ask_questions_page):
    """
    Test that whitespace normalization produces same key.

    Test name: test_unit_scenario_expectedBehavior
    """
    # Arrange: Query with different whitespace
    dataset_version = "test_dataset_v1"
    query1 = "compare outcome by treatment"
    query2 = "compare  outcome   by  treatment"  # Extra spaces
    query3 = " compare outcome by treatment "  # Leading/trailing spaces

    # Act: Generate run_keys
    key1 = ask_questions_page.generate_run_key(dataset_version, query1, sample_context)
    key2 = ask_questions_page.generate_run_key(dataset_version, query2, sample_context)
    key3 = ask_questions_page.generate_run_key(dataset_version, query3, sample_context)

    # Assert: Keys are identical (whitespace normalized)
    assert key1 == key2 == key3


def test_run_key_generation_different_queries_produce_different_keys(sample_context, ask_questions_page):
    """
    Test that different queries produce different keys.

    Test name: test_unit_scenario_expectedBehavior
    """
    # Arrange: Different queries
    dataset_version = "test_dataset_v1"
    query1 = "compare outcome by treatment"
    query2 = "compare outcome by age"

    # Act: Generate run_keys
    key1 = ask_questions_page.generate_run_key(dataset_version, query1, sample_context)
    key2 = ask_questions_page.generate_run_key(dataset_version, query2, sample_context)

    # Assert: Keys are different
    assert key1 != key2


def test_run_key_generation_predictors_ordering_normalized(sample_context, ask_questions_page):
    """
    Test that predictor ordering doesn't affect run_key.

    Test name: test_unit_scenario_expectedBehavior
    """
    # Arrange: Same predictors in different order
    dataset_version = "test_dataset_v1"
    query_text = "find predictors"

    context1 = AnalysisContext(
        inferred_intent=AnalysisIntent.FIND_PREDICTORS,
        primary_variable="outcome",
        predictor_variables=["age", "sex"],
    )
    context1.confidence = 0.9

    context2 = AnalysisContext(
        inferred_intent=AnalysisIntent.FIND_PREDICTORS,
        primary_variable="outcome",
        predictor_variables=["sex", "age"],  # Different order
    )
    context2.confidence = 0.9

    # Act: Generate run_keys
    key1 = ask_questions_page.generate_run_key(dataset_version, query_text, context1)
    key2 = ask_questions_page.generate_run_key(dataset_version, query_text, context2)

    # Assert: Keys are identical (predictors sorted before hashing)
    assert key1 == key2


def test_run_key_generation_empty_values_normalized(ask_questions_page):
    """
    Test that empty/None values are normalized consistently.

    Test name: test_unit_scenario_expectedBehavior
    """
    # Arrange: Contexts with None vs empty string
    dataset_version = "test_dataset_v1"
    query_text = "describe"

    context1 = AnalysisContext(
        inferred_intent=AnalysisIntent.DESCRIBE,
        primary_variable=None,
        grouping_variable=None,
        predictor_variables=[],
    )
    context1.confidence = 0.9

    context2 = AnalysisContext(
        inferred_intent=AnalysisIntent.DESCRIBE,
        primary_variable="",  # Empty string instead of None
        grouping_variable="",
        predictor_variables=[],
    )
    context2.confidence = 0.9

    # Act: Generate run_keys
    key1 = ask_questions_page.generate_run_key(dataset_version, query_text, context1)
    key2 = ask_questions_page.generate_run_key(dataset_version, query_text, context2)

    # Assert: Keys are identical (empty values normalized)
    assert key1 == key2


def test_run_key_generation_canonicalized_json_structure(ask_questions_page):
    """
    Test that run_key uses canonicalized JSON (sort_keys=True).

    Test name: test_unit_scenario_expectedBehavior
    """
    # Arrange: Create payload manually to verify structure
    dataset_version = "test_dataset_v1"
    query_text = "compare outcome by treatment"
    context = AnalysisContext(
        inferred_intent=AnalysisIntent.COMPARE_GROUPS,
        primary_variable="outcome",
        grouping_variable="treatment",
    )
    context.confidence = 0.9

    # Act: Generate run_key
    run_key = ask_questions_page.generate_run_key(dataset_version, query_text, context)

    # Assert: Verify it's a valid SHA256 hash
    assert len(run_key) == 64
    assert all(c in "0123456789abcdef" for c in run_key)

    # Verify payload structure by reconstructing
    payload = {
        "dataset_version": dataset_version,
        "query": " ".join(query_text.strip().split()),  # Normalized
        "intent": context.inferred_intent.value,
        "vars": {
            "primary": context.primary_variable or "",
            "grouping": context.grouping_variable or "",
            "predictors": sorted(context.predictor_variables or []),
            "time": context.time_variable or "",
            "event": context.event_variable or "",
        },
    }
    expected_key = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    # Assert: Keys match
    assert run_key == expected_key
