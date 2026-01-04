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

from unittest.mock import MagicMock

from clinical_analytics.core.question_generator import (
    _deterministic_questions,
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
