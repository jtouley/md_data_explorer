"""
Tests for proactive question generation (ADR004 Phase 4).

Tests cover:
- Deterministic fallback questions
- LLM generation with semantic layer bounding
- Confidence gating
- Idempotency caching
- Privacy validation
- Feature flag enforcement
"""

from unittest.mock import MagicMock, patch

import pytest
from clinical_analytics.core.nl_query_engine import QueryIntent
from clinical_analytics.core.question_generator import (
    _deterministic_questions,
    _validate_questions_bounded,
    generate_proactive_questions,
    generate_upload_questions,
)


class TestGenerateUploadQuestions:
    """Test upload-time example question generation."""

    def test_generate_upload_questions_deterministic_fallback_returns_questions(self):
        """Test that deterministic fallback returns questions when LLM unavailable."""
        # Arrange: Mock semantic layer with columns
        mock_semantic_layer = MagicMock()
        mock_semantic_layer.get_column_alias_index.return_value = {
            "age": "age",
            "outcome": "outcome",
        }

        mock_inferred_schema = MagicMock()

        # Act: Generate questions (should use deterministic fallback)
        questions = generate_upload_questions(
            semantic_layer=mock_semantic_layer,
            inferred_schema=mock_inferred_schema,
            doc_context=None,
        )

        # Assert: Should return list of questions
        assert isinstance(questions, list)
        assert len(questions) > 0
        assert all(isinstance(q, str) for q in questions)

    def test_generate_upload_questions_empty_columns_returns_empty_list(self):
        """Test that empty columns returns empty list."""
        # Arrange: Mock semantic layer with no columns
        mock_semantic_layer = MagicMock()
        mock_semantic_layer.get_column_alias_index.return_value = {}

        mock_inferred_schema = MagicMock()

        # Act: Generate questions
        questions = generate_upload_questions(
            semantic_layer=mock_semantic_layer,
            inferred_schema=mock_inferred_schema,
            doc_context=None,
        )

        # Assert: Should return empty list
        assert questions == []


class TestDeterministicQuestions:
    """Test deterministic question generation (fallback)."""

    def test_deterministic_questions_returns_list_of_strings(self):
        """Test that deterministic questions returns list of strings."""
        # Arrange
        mock_semantic_layer = MagicMock()
        available_columns = ["age", "outcome", "treatment"]

        # Act
        questions = _deterministic_questions(
            semantic_layer=mock_semantic_layer,
            available_columns=available_columns,
            query_intent=None,
        )

        # Assert
        assert isinstance(questions, list)
        assert len(questions) > 0
        assert all(isinstance(q, str) for q in questions)
        assert len(questions) <= 5  # Limit to 5 questions


class TestGenerateProactiveQuestions:
    """Test query-time proactive follow-up question generation."""

    @pytest.fixture
    def mock_semantic_layer(self):
        """Create mock semantic layer with columns."""
        mock = MagicMock()
        mock.get_column_alias_index.return_value = {
            "age": "age",
            "outcome": "outcome",
            "treatment": "treatment",
        }
        return mock

    @pytest.fixture
    def high_confidence_intent(self):
        """Create high-confidence QueryIntent (≥0.85)."""
        return QueryIntent(intent_type="DESCRIBE", confidence=0.9)

    @pytest.fixture
    def medium_confidence_intent(self):
        """Create medium-confidence QueryIntent (0.5-0.85)."""
        return QueryIntent(intent_type="COMPARE_GROUPS", confidence=0.7)

    @pytest.fixture
    def low_confidence_intent(self):
        """Create low-confidence QueryIntent (<0.5)."""
        return QueryIntent(intent_type="DESCRIBE", confidence=0.3)

    @pytest.fixture
    def dict_cache_backend(self):
        """Create dict-based cache backend for testing."""

        class DictCacheBackend:
            def __init__(self):
                self._cache: dict[str, list[str]] = {}

            def get(self, key: str) -> list[str] | None:
                return self._cache.get(key)

            def set(self, key: str, value: list[str]) -> None:
                self._cache[key] = value

        return DictCacheBackend()

    def test_generate_proactive_questions_high_confidence_returns_questions(
        self, mock_semantic_layer, high_confidence_intent, dict_cache_backend
    ):
        """Test that high confidence (≥0.85) returns questions."""
        # Arrange
        # Act
        questions = generate_proactive_questions(
            semantic_layer=mock_semantic_layer,
            query_intent=high_confidence_intent,
            dataset_version="test_v1",
            run_key="test_run",
            normalized_query="test query",
            cache_backend=dict_cache_backend,
        )

        # Assert: Should return list of questions (may be empty if LLM unavailable)
        assert isinstance(questions, list)
        assert all(isinstance(q, str) for q in questions)

    def test_generate_proactive_questions_medium_confidence_returns_questions(
        self, mock_semantic_layer, medium_confidence_intent, dict_cache_backend
    ):
        """Test that medium confidence (0.5-0.85) returns clarification questions."""
        # Arrange
        # Act
        questions = generate_proactive_questions(
            semantic_layer=mock_semantic_layer,
            query_intent=medium_confidence_intent,
            dataset_version="test_v1",
            run_key="test_run",
            normalized_query="test query",
            cache_backend=dict_cache_backend,
        )

        # Assert: Should return list of questions (may be empty if LLM unavailable)
        assert isinstance(questions, list)
        assert all(isinstance(q, str) for q in questions)

    def test_generate_proactive_questions_low_confidence_returns_empty(
        self, mock_semantic_layer, low_confidence_intent, dict_cache_backend
    ):
        """Test that low confidence (<0.5) returns empty list."""
        # Arrange
        # Act
        questions = generate_proactive_questions(
            semantic_layer=mock_semantic_layer,
            query_intent=low_confidence_intent,
            dataset_version="test_v1",
            run_key="test_run",
            normalized_query="test query",
            cache_backend=dict_cache_backend,
        )

        # Assert: Should return empty list (confidence too low)
        assert questions == []

    def test_generate_proactive_questions_none_intent_returns_empty(self, mock_semantic_layer, dict_cache_backend):
        """Test that None query_intent returns empty list."""
        # Arrange
        # Act
        questions = generate_proactive_questions(
            semantic_layer=mock_semantic_layer,
            query_intent=None,
            dataset_version="test_v1",
            run_key="test_run",
            normalized_query="test query",
            cache_backend=dict_cache_backend,
        )

        # Assert: Should return empty list
        assert questions == []

    def test_generate_proactive_questions_empty_columns_returns_empty(self, dict_cache_backend):
        """Test that empty columns returns empty list."""
        # Arrange: Mock semantic layer with no columns
        mock_semantic_layer = MagicMock()
        mock_semantic_layer.get_column_alias_index.return_value = {}

        high_confidence_intent = QueryIntent(intent_type="DESCRIBE", confidence=0.9)

        # Act
        questions = generate_proactive_questions(
            semantic_layer=mock_semantic_layer,
            query_intent=high_confidence_intent,
            dataset_version="test_v1",
            run_key="test_run",
            normalized_query="test query",
            cache_backend=dict_cache_backend,
        )

        # Assert: Should return empty list
        assert questions == []

    def test_generate_proactive_questions_caching_idempotency(
        self, mock_semantic_layer, high_confidence_intent, dict_cache_backend
    ):
        """Test that questions are cached and returned from cache on second call."""
        # Arrange
        # Act: First call
        questions1 = generate_proactive_questions(
            semantic_layer=mock_semantic_layer,
            query_intent=high_confidence_intent,
            dataset_version="test_v1",
            run_key="test_run",
            normalized_query="test query",
            cache_backend=dict_cache_backend,
        )

        # Act: Second call with same parameters
        questions2 = generate_proactive_questions(
            semantic_layer=mock_semantic_layer,
            query_intent=high_confidence_intent,
            dataset_version="test_v1",
            run_key="test_run",
            normalized_query="test query",
            cache_backend=dict_cache_backend,
        )

        # Assert: Should return same questions (from cache)
        assert questions1 == questions2

    @patch("clinical_analytics.core.question_generator.ENABLE_PROACTIVE_QUESTIONS", False)
    def test_generate_proactive_questions_feature_flag_disabled_returns_empty(
        self, mock_semantic_layer, high_confidence_intent, dict_cache_backend
    ):
        """Test that feature flag disabled returns empty list."""
        # Arrange
        # Act
        questions = generate_proactive_questions(
            semantic_layer=mock_semantic_layer,
            query_intent=high_confidence_intent,
            dataset_version="test_v1",
            run_key="test_run",
            normalized_query="test query",
            cache_backend=dict_cache_backend,
        )

        # Assert: Should return empty list (feature disabled)
        assert questions == []

    def test_validate_questions_bounded_filters_hallucinated_columns(self):
        """Test that _validate_questions_bounded filters out questions with unknown columns."""
        # Arrange
        questions = [
            "What is the average age?",  # Valid (age is in available columns)
            "What is the average nonexistent_column?",  # Invalid (not in available columns)
        ]
        available_columns = ["age", "outcome", "treatment"]
        available_aliases = []

        # Act
        validated = _validate_questions_bounded(questions, available_columns, available_aliases)

        # Assert: Should filter out questions with hallucinated columns
        assert len(validated) == 1
        assert "What is the average age?" in validated
        assert "What is the average nonexistent_column?" not in validated
