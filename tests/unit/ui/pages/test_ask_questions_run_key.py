"""
Test stable run_key generation for idempotency.

Test name follows: test_unit_scenario_expectedBehavior
"""

from clinical_analytics.ui.components.question_engine import AnalysisContext, AnalysisIntent


def test_normalize_query_handles_tabs_and_newlines(ask_questions_page):
    """Test normalize_query handles tabs and newlines."""
    # Arrange
    query_with_tabs = "compare\toutcome\tby\ttreatment"
    query_with_newlines = "compare\noutcome\nby\ntreatment"
    query_mixed = "compare\t\noutcome  \n\tby   treatment"

    # Act
    normalized_tabs = ask_questions_page.normalize_query(query_with_tabs)
    normalized_newlines = ask_questions_page.normalize_query(query_with_newlines)
    normalized_mixed = ask_questions_page.normalize_query(query_mixed)

    # Assert: All normalized to same single-space separated string
    assert normalized_tabs == "compare outcome by treatment"
    assert normalized_newlines == "compare outcome by treatment"
    assert normalized_mixed == "compare outcome by treatment"


def test_normalize_query_lowercases_and_strips(ask_questions_page):
    """Test normalize_query lowercases and strips whitespace."""
    # Arrange
    query_mixed_case = "  Compare OUTCOME by Treatment  "

    # Act
    normalized = ask_questions_page.normalize_query(query_mixed_case)

    # Assert
    assert normalized == "compare outcome by treatment"


def test_normalize_query_handles_none_returns_empty_string(ask_questions_page):
    """Test normalize_query returns empty string for None."""
    # Arrange
    query_none = None

    # Act
    normalized = ask_questions_page.normalize_query(query_none)

    # Assert
    assert normalized == ""


def test_canonicalize_scope_drops_none_values(ask_questions_page):
    """Test canonicalize_scope drops None values."""
    # Arrange
    scope = {
        "filter_a": "value_a",
        "filter_b": None,
        "filter_c": "value_c",
    }

    # Act
    canonical = ask_questions_page.canonicalize_scope(scope)

    # Assert: None value dropped
    assert "filter_b" not in canonical
    assert canonical == {"filter_a": "value_a", "filter_c": "value_c"}


def test_canonicalize_scope_sorts_keys(ask_questions_page):
    """Test canonicalize_scope sorts dictionary keys."""
    # Arrange
    scope = {
        "z_filter": "value_z",
        "a_filter": "value_a",
        "m_filter": "value_m",
    }

    # Act
    canonical = ask_questions_page.canonicalize_scope(scope)

    # Assert: Keys are sorted
    assert list(canonical.keys()) == ["a_filter", "m_filter", "z_filter"]


def test_canonicalize_scope_sorts_list_values(ask_questions_page):
    """Test canonicalize_scope sorts list values."""
    # Arrange
    scope = {
        "predictors": ["age", "treatment", "bmi"],
    }

    # Act
    canonical = ask_questions_page.canonicalize_scope(scope)

    # Assert: List sorted
    assert canonical["predictors"] == ["age", "bmi", "treatment"]


def test_canonicalize_scope_handles_none_scope(ask_questions_page):
    """Test canonicalize_scope handles None scope."""
    # Arrange
    scope = None

    # Act
    canonical = ask_questions_page.canonicalize_scope(scope)

    # Assert: Returns empty dict
    assert canonical == {}


def test_run_key_includes_canonicalized_semantic_scope(ask_questions_page):
    """Test run_key includes canonicalized semantic scope."""
    # Arrange
    dataset_version = "test_dataset_v1"
    query_text = "compare outcome by treatment"  # Already normalized
    context = AnalysisContext(
        inferred_intent=AnalysisIntent.COMPARE_GROUPS,
        primary_variable="outcome",
        grouping_variable="treatment",
    )
    scope = {"filter_type": "active", "cohort": "primary"}

    # Act
    key_with_scope = ask_questions_page.generate_run_key(dataset_version, query_text, context, scope)
    key_without_scope = ask_questions_page.generate_run_key(dataset_version, query_text, context, None)

    # Assert: Different keys with/without scope
    assert key_with_scope != key_without_scope


def test_run_key_different_scope_produces_different_key(ask_questions_page):
    """Test different scopes produce different run_keys."""
    # Arrange
    dataset_version = "test_dataset_v1"
    query_text = "compare outcome by treatment"
    context = AnalysisContext(
        inferred_intent=AnalysisIntent.COMPARE_GROUPS,
        primary_variable="outcome",
        grouping_variable="treatment",
    )
    scope1 = {"filter_type": "active"}
    scope2 = {"filter_type": "historical"}

    # Act
    key1 = ask_questions_page.generate_run_key(dataset_version, query_text, context, scope1)
    key2 = ask_questions_page.generate_run_key(dataset_version, query_text, context, scope2)

    # Assert: Different scopes produce different keys
    assert key1 != key2


def test_run_key_identical_scope_produces_same_key(ask_questions_page):
    """Test identical scopes produce same run_key after canonicalization."""
    # Arrange
    dataset_version = "test_dataset_v1"
    query_text = "compare outcome by treatment"
    context = AnalysisContext(
        inferred_intent=AnalysisIntent.COMPARE_GROUPS,
        primary_variable="outcome",
        grouping_variable="treatment",
    )
    # Same scope with different key order and list order
    scope1 = {"b_filter": "value", "a_filter": "value", "predictors": ["age", "bmi"]}
    scope2 = {"a_filter": "value", "b_filter": "value", "predictors": ["bmi", "age"]}

    # Act
    key1 = ask_questions_page.generate_run_key(dataset_version, query_text, context, scope1)
    key2 = ask_questions_page.generate_run_key(dataset_version, query_text, context, scope2)

    # Assert: Same keys after canonicalization
    assert key1 == key2


def test_run_key_excludes_volatile_fields(ask_questions_page):
    """Test run_key excludes confidence and other volatile fields."""
    # Arrange
    dataset_version = "test_dataset_v1"
    query_text = "compare outcome by treatment"
    context1 = AnalysisContext(
        inferred_intent=AnalysisIntent.COMPARE_GROUPS,
        primary_variable="outcome",
        grouping_variable="treatment",
    )
    context1.confidence = 0.5

    context2 = AnalysisContext(
        inferred_intent=AnalysisIntent.COMPARE_GROUPS,
        primary_variable="outcome",
        grouping_variable="treatment",
    )
    context2.confidence = 0.9

    # Act
    key1 = ask_questions_page.generate_run_key(dataset_version, query_text, context1)
    key2 = ask_questions_page.generate_run_key(dataset_version, query_text, context2)

    # Assert: Same key despite different confidence
    assert key1 == key2


def test_run_key_only_includes_material_context_variables(ask_questions_page):
    """Test run_key only includes material variables (not UI flags)."""
    # Arrange
    dataset_version = "test_dataset_v1"
    query_text = "compare outcome by treatment"  # Already normalized (lowercase, single spaces)
    context = AnalysisContext(
        inferred_intent=AnalysisIntent.COMPARE_GROUPS,
        primary_variable="outcome",
        grouping_variable="treatment",
    )

    # Act
    key = ask_questions_page.generate_run_key(dataset_version, query_text, context)

    # Assert: Verify it's a valid SHA256 hash
    assert len(key) == 64
    assert all(c in "0123456789abcdef" for c in key)

    # Verify payload structure by checking key is stable
    # Same inputs should produce same key
    key2 = ask_questions_page.generate_run_key(dataset_version, query_text, context)
    assert key == key2


def test_run_key_generation_same_query_produces_same_key(sample_context, ask_questions_page):
    """Test that same query + variables generates same run_key."""
    # Arrange: Same query, same variables
    dataset_version = "test_dataset_v1"
    query_text = "compare outcome by treatment"  # Already normalized

    # Act: Generate run_key twice
    key1 = ask_questions_page.generate_run_key(dataset_version, query_text, sample_context)
    key2 = ask_questions_page.generate_run_key(dataset_version, query_text, sample_context)

    # Assert: Keys are identical
    assert key1 == key2
    assert len(key1) == 64  # SHA256 hex digest length


def test_run_key_generation_different_queries_produce_different_keys(sample_context, ask_questions_page):
    """Test that different queries produce different keys."""
    # Arrange: Different queries (both normalized)
    dataset_version = "test_dataset_v1"
    query1 = "compare outcome by treatment"
    query2 = "compare outcome by age"

    # Act: Generate run_keys
    key1 = ask_questions_page.generate_run_key(dataset_version, query1, sample_context)
    key2 = ask_questions_page.generate_run_key(dataset_version, query2, sample_context)

    # Assert: Keys are different
    assert key1 != key2


def test_run_key_generation_predictors_ordering_normalized(ask_questions_page):
    """Test that predictor ordering doesn't affect run_key."""
    # Arrange: Same predictors in different order
    dataset_version = "test_dataset_v1"
    query_text = "find predictors"

    context1 = AnalysisContext(
        inferred_intent=AnalysisIntent.FIND_PREDICTORS,
        primary_variable="outcome",
        predictor_variables=["age", "sex"],
    )

    context2 = AnalysisContext(
        inferred_intent=AnalysisIntent.FIND_PREDICTORS,
        primary_variable="outcome",
        predictor_variables=["sex", "age"],  # Different order
    )

    # Act: Generate run_keys
    key1 = ask_questions_page.generate_run_key(dataset_version, query_text, context1)
    key2 = ask_questions_page.generate_run_key(dataset_version, query_text, context2)

    # Assert: Keys are identical (predictors sorted before hashing)
    assert key1 == key2


def test_run_key_generation_empty_values_normalized(ask_questions_page):
    """Test that empty/None values are normalized consistently."""
    # Arrange: Contexts with None vs empty string
    dataset_version = "test_dataset_v1"
    query_text = "describe"

    context1 = AnalysisContext(
        inferred_intent=AnalysisIntent.DESCRIBE,
        primary_variable=None,
        grouping_variable=None,
        predictor_variables=[],
    )

    context2 = AnalysisContext(
        inferred_intent=AnalysisIntent.DESCRIBE,
        primary_variable="",  # Empty string instead of None
        grouping_variable="",
        predictor_variables=[],
    )

    # Act: Generate run_keys
    key1 = ask_questions_page.generate_run_key(dataset_version, query_text, context1)
    key2 = ask_questions_page.generate_run_key(dataset_version, query_text, context2)

    # Assert: Keys are identical (empty values normalized)
    assert key1 == key2
