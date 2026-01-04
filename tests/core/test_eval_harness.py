"""
Tests for eval harness functionality (Phase 7.2).

Ensures:
- Golden questions YAML can be loaded
- Eval harness can run questions against SemanticLayer
- Results include intent match, confidence, and correctness
"""

import pytest
from clinical_analytics.core.eval_harness import EvalHarness, load_golden_questions


class TestGoldenQuestionsYAML:
    """Test loading golden questions from YAML."""

    def test_load_golden_questions_from_yaml(self, tmp_path):
        """load_golden_questions should parse YAML file."""
        # Arrange: Create golden questions YAML
        yaml_path = tmp_path / "golden_questions.yaml"
        yaml_path.write_text(
            """
questions:
  - id: count_all_patients
    query: "how many patients?"
    expected_intent: COUNT
    expected_metric: null
    expected_group_by: null
    dataset: "test_dataset"

  - id: describe_age
    query: "what is the average age?"
    expected_intent: DESCRIBE
    expected_metric: "age"
    expected_group_by: null
    dataset: "test_dataset"
"""
        )

        # Act: Load questions
        questions = load_golden_questions(yaml_path)

        # Assert: Questions should be loaded
        assert len(questions) == 2
        assert questions[0]["id"] == "count_all_patients"
        assert questions[0]["query"] == "how many patients?"
        assert questions[0]["expected_intent"] == "COUNT"
        assert questions[1]["id"] == "describe_age"
        assert questions[1]["expected_intent"] == "DESCRIBE"
        assert questions[1]["expected_metric"] == "age"

    def test_golden_questions_yaml_supports_filters(self, tmp_path):
        """Golden questions should support expected filters."""
        # Arrange: YAML with filter expectations
        yaml_path = tmp_path / "golden_questions.yaml"
        yaml_path.write_text(
            """
questions:
  - id: filtered_count
    query: "how many active patients?"
    expected_intent: COUNT
    expected_filters:
      - column: "status"
        operator: "=="
        value: "active"
    dataset: "test_dataset"
"""
        )

        # Act: Load questions
        questions = load_golden_questions(yaml_path)

        # Assert: Filters should be loaded
        assert len(questions) == 1
        assert "expected_filters" in questions[0]
        assert len(questions[0]["expected_filters"]) == 1
        assert questions[0]["expected_filters"][0]["column"] == "status"
        assert questions[0]["expected_filters"][0]["operator"] == "=="


class TestEvalHarness:
    """Test eval harness runner."""

    @pytest.fixture
    def eval_harness(self, make_semantic_layer):
        """Create eval harness with test semantic layer."""
        semantic_layer = make_semantic_layer(
            data={
                "patient_id": [1, 2, 3],
                "age": [25, 35, 45],
                "status": ["active", "inactive", "active"],
            }
        )
        return EvalHarness(semantic_layer)

    def test_eval_harness_runs_golden_question(self, eval_harness):
        """EvalHarness should run a golden question and check results."""
        # Arrange: Golden question
        question = {
            "id": "count_all",
            "query": "how many patients?",
            "expected_intent": "COUNT",
            "expected_metric": None,
            "expected_group_by": None,
        }

        # Act: Run evaluation
        result = eval_harness.evaluate_question(question)

        # Assert: Result should include correctness check
        assert "id" in result
        assert result["id"] == "count_all"
        assert "intent_match" in result
        assert "confidence" in result
        assert "correct" in result

    def test_eval_harness_detects_intent_mismatch(self, eval_harness):
        """EvalHarness should detect when intent doesn't match expectation."""
        # Arrange: Question with wrong expected intent
        question = {
            "id": "test_mismatch",
            "query": "how many patients?",  # Should be COUNT
            "expected_intent": "DESCRIBE",  # Wrong!
            "expected_metric": None,
            "expected_group_by": None,
        }

        # Act: Run evaluation
        result = eval_harness.evaluate_question(question)

        # Assert: Should detect mismatch
        assert result["intent_match"] is False
        assert result["correct"] is False

    def test_eval_harness_batch_evaluation(self, eval_harness):
        """EvalHarness should support batch evaluation of multiple questions."""
        # Arrange: Multiple questions
        questions = [
            {
                "id": "q1",
                "query": "how many patients?",
                "expected_intent": "COUNT",
            },
            {
                "id": "q2",
                "query": "average age?",
                "expected_intent": "DESCRIBE",
                "expected_metric": "age",
            },
        ]

        # Act: Run batch evaluation
        results = eval_harness.evaluate_batch(questions)

        # Assert: Should return results for all questions
        assert len(results) == 2
        assert results[0]["id"] == "q1"
        assert results[1]["id"] == "q2"

    def test_eval_harness_summary_statistics(self, eval_harness):
        """EvalHarness should provide summary statistics (accuracy, avg confidence)."""
        # Arrange: Mixed correct/incorrect questions
        questions = [
            {
                "id": "q1_correct",
                "query": "how many patients?",
                "expected_intent": "COUNT",
            },
            {
                "id": "q2_incorrect",
                "query": "how many patients?",
                "expected_intent": "DESCRIBE",  # Wrong
            },
        ]

        # Act: Run batch and get summary
        results = eval_harness.evaluate_batch(questions)
        summary = eval_harness.get_summary(results)

        # Assert: Summary should include accuracy
        assert "accuracy" in summary
        assert "total_questions" in summary
        assert "correct_count" in summary
        assert "average_confidence" in summary
        assert summary["total_questions"] == 2


class TestEvalHarnessIntegration:
    """Test eval harness with real QueryPlan parsing."""

    @pytest.fixture
    def eval_harness_with_engine(self, make_semantic_layer):
        """Create eval harness with NLQueryEngine."""
        semantic_layer = make_semantic_layer(
            data={
                "patient_id": [1, 2, 3, 4],
                "age": [25, 35, 45, 55],
                "status": ["active", "inactive", "active", "active"],
            }
        )
        return EvalHarness(semantic_layer)

    def test_eval_harness_with_real_parsing(self, eval_harness_with_engine):
        """EvalHarness should work with real NL parsing."""
        # Arrange: Real-world question
        question = {
            "id": "real_count",
            "query": "how many patients?",
            "expected_intent": "COUNT",
        }

        # Act: Evaluate
        result = eval_harness_with_engine.evaluate_question(question)

        # Assert: Should parse and match correctly
        assert result["correct"] is True
        assert result["intent_match"] is True
        assert result["confidence"] > 0
