"""
Tests for automated golden question generation (ADR009 Phase 6).

Tests cover:
- generate_golden_questions_from_logs() function
- analyze_golden_question_coverage() function
- maintain_golden_questions_automatically() function
- Validation for generated golden questions
- Graceful degradation when LLM unavailable
"""

from unittest.mock import MagicMock, patch

from clinical_analytics.core.golden_question_generator import (
    analyze_golden_question_coverage,
    generate_golden_questions_from_logs,
    maintain_golden_questions_automatically,
    validate_golden_question,
)
from clinical_analytics.core.llm_feature import LLMCallResult


class TestGoldenQuestionGeneration:
    """Tests for automated golden question generation."""

    def test_generate_golden_questions_from_logs_function_exists(self):
        """Verify generate_golden_questions_from_logs() function exists."""
        from clinical_analytics.core.golden_question_generator import (
            generate_golden_questions_from_logs,
        )

        assert callable(generate_golden_questions_from_logs)

    def test_analyze_golden_question_coverage_function_exists(self):
        """Verify analyze_golden_question_coverage() function exists."""
        from clinical_analytics.core.golden_question_generator import (
            analyze_golden_question_coverage,
        )

        assert callable(analyze_golden_question_coverage)

    def test_maintain_golden_questions_automatically_function_exists(self):
        """Verify maintain_golden_questions_automatically() function exists."""
        from clinical_analytics.core.golden_question_generator import (
            maintain_golden_questions_automatically,
        )

        assert callable(maintain_golden_questions_automatically)

    @patch("clinical_analytics.core.golden_question_generator.call_llm")
    def test_generate_golden_questions_success(self, mock_call_llm):
        """Test successful golden question generation from logs."""
        # Arrange
        query_logs = [
            {
                "query": "what is average LDL?",
                "intent": "DESCRIBE",
                "confidence": 0.85,
                "success": True,
            },
            {
                "query": "compare BMI by treatment",
                "intent": "COMPARE_GROUPS",
                "confidence": 0.90,
                "success": True,
            },
        ]
        semantic_layer = MagicMock()

        mock_call_llm.return_value = LLMCallResult(
            raw_text='{"golden_questions": ['
            '{"question": "What is the average LDL cholesterol?", "intent": "DESCRIBE"}, '
            '{"question": "How does BMI compare between treatment groups?", "intent": "COMPARE_GROUPS"}'
            "]}",
            payload={
                "golden_questions": [
                    {"question": "What is the average LDL cholesterol?", "intent": "DESCRIBE"},
                    {
                        "question": "How does BMI compare between treatment groups?",
                        "intent": "COMPARE_GROUPS",
                    },
                ]
            },
            latency_ms=2000.0,
            timed_out=False,
            error=None,
        )

        # Act
        result = generate_golden_questions_from_logs(query_logs, semantic_layer)

        # Assert
        assert isinstance(result, list)
        assert len(result) >= 1
        assert all(isinstance(q, dict) for q in result)
        assert all("question" in q for q in result)

    @patch("clinical_analytics.core.golden_question_generator.call_llm")
    def test_generate_golden_questions_graceful_degradation(self, mock_call_llm):
        """Test graceful degradation when LLM unavailable."""
        # Arrange
        query_logs = [{"query": "what is average LDL?", "intent": "DESCRIBE", "confidence": 0.85}]
        semantic_layer = MagicMock()

        mock_call_llm.return_value = LLMCallResult(
            raw_text=None,
            payload=None,
            latency_ms=100.0,
            timed_out=False,
            error="ollama_unavailable",
        )

        # Act
        result = generate_golden_questions_from_logs(query_logs, semantic_layer)

        # Assert: Should return empty list (graceful degradation)
        assert result == []

    @patch("clinical_analytics.core.golden_question_generator.call_llm")
    def test_analyze_golden_question_coverage(self, mock_call_llm):
        """Test coverage gap analysis."""
        # Arrange
        golden_questions = [
            {"question": "What is average LDL?", "intent": "DESCRIBE"},
            {"question": "Compare by treatment", "intent": "COMPARE_GROUPS"},
        ]
        query_logs = [
            {"query": "average LDL", "intent": "DESCRIBE"},
            {"query": "compare by treatment", "intent": "COMPARE_GROUPS"},
            {"query": "find predictors of mortality", "intent": "FIND_PREDICTORS"},
        ]
        semantic_layer = MagicMock()

        mock_call_llm.return_value = LLMCallResult(
            raw_text='{"gaps": [{"intent": "FIND_PREDICTORS", "description": "No predictor questions"}]}',
            payload={"gaps": [{"intent": "FIND_PREDICTORS", "description": "No predictor questions"}]},
            latency_ms=1500.0,
            timed_out=False,
            error=None,
        )

        # Act
        result = analyze_golden_question_coverage(golden_questions, query_logs, semantic_layer)

        # Assert
        assert isinstance(result, dict)
        assert "gaps" in result or "coverage" in result

    def test_validate_golden_question_valid(self):
        """Test validation of valid golden question."""
        # Arrange
        question = {"question": "What is average LDL?", "intent": "DESCRIBE"}

        # Act
        is_valid, error = validate_golden_question(question)

        # Assert
        assert is_valid is True
        assert error is None

    def test_validate_golden_question_missing_question_field(self):
        """Test validation fails for missing question field."""
        # Arrange
        question = {"intent": "DESCRIBE"}

        # Act
        is_valid, error = validate_golden_question(question)

        # Assert
        assert is_valid is False
        assert error is not None
        assert "question" in error.lower()

    def test_validate_golden_question_missing_intent_field(self):
        """Test validation fails for missing intent field."""
        # Arrange
        question = {"question": "What is average LDL?"}

        # Act
        is_valid, error = validate_golden_question(question)

        # Assert
        assert is_valid is False
        assert error is not None
        assert "intent" in error.lower()

    def test_validate_golden_question_invalid_intent(self):
        """Test validation fails for invalid intent."""
        # Arrange
        question = {"question": "What is average LDL?", "intent": "INVALID_INTENT"}

        # Act
        is_valid, error = validate_golden_question(question)

        # Assert
        assert is_valid is False
        assert error is not None
        assert "intent" in error.lower()

    @patch("clinical_analytics.core.golden_question_generator.call_llm")
    def test_maintain_golden_questions_automatically_dry_run(self, mock_call_llm):
        """Test dry-run mode (no modifications)."""
        # Arrange
        query_logs = [{"query": "what is average LDL?", "intent": "DESCRIBE", "confidence": 0.85}]
        existing_golden_questions = [{"question": "What is average LDL?", "intent": "DESCRIBE"}]
        semantic_layer = MagicMock()

        mock_call_llm.return_value = LLMCallResult(
            raw_text='{"recommendation": "keep existing"}',
            payload={"recommendation": "keep existing"},
            latency_ms=1000.0,
            timed_out=False,
            error=None,
        )

        # Act
        result = maintain_golden_questions_automatically(
            query_logs, existing_golden_questions, semantic_layer, dry_run=True
        )

        # Assert
        assert isinstance(result, dict)
        assert "recommendations" in result or "action" in result

    @patch("clinical_analytics.core.golden_question_generator.call_llm")
    def test_maintain_golden_questions_automatically_with_updates(self, mock_call_llm):
        """Test automatic maintenance with updates."""
        # Arrange
        query_logs = [
            {"query": "what is average LDL?", "intent": "DESCRIBE", "confidence": 0.85},
            {"query": "find predictors of mortality", "intent": "FIND_PREDICTORS", "confidence": 0.80},
        ]
        existing_golden_questions = [{"question": "What is average LDL?", "intent": "DESCRIBE"}]
        semantic_layer = MagicMock()

        mock_call_llm.return_value = LLMCallResult(
            raw_text='{"new_questions": [{"question": "What predicts mortality?", "intent": "FIND_PREDICTORS"}]}',
            payload={"new_questions": [{"question": "What predicts mortality?", "intent": "FIND_PREDICTORS"}]},
            latency_ms=2000.0,
            timed_out=False,
            error=None,
        )

        # Act
        result = maintain_golden_questions_automatically(
            query_logs, existing_golden_questions, semantic_layer, dry_run=False
        )

        # Assert
        assert isinstance(result, dict)
