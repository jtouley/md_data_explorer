"""
Eval Harness - Golden Questions Testing (Phase 7.2).

Enables data-driven improvement by:
- Loading golden questions from YAML
- Evaluating parsing accuracy against expected results
- Tracking regression/improvement over time
"""

import logging
from pathlib import Path
from typing import Any

import yaml

from clinical_analytics.core.nl_query_engine import NLQueryEngine
from clinical_analytics.core.semantic import SemanticLayer

logger = logging.getLogger(__name__)


def load_golden_questions(yaml_path: Path | str) -> list[dict[str, Any]]:
    """
    Load golden questions from YAML file.

    Args:
        yaml_path: Path to YAML file containing golden questions

    Returns:
        List of question dicts with expected outputs

    Example YAML format:
        questions:
          - id: count_all_patients
            query: "how many patients?"
            expected_intent: COUNT
            expected_metric: null
            expected_group_by: null
            dataset: "test_dataset"

          - id: filtered_count
            query: "how many active patients?"
            expected_intent: COUNT
            expected_filters:
              - column: "status"
                operator: "=="
                value: "active"
    """
    yaml_path = Path(yaml_path)

    if not yaml_path.exists():
        raise FileNotFoundError(f"Golden questions YAML not found: {yaml_path}")

    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    if not data or "questions" not in data:
        raise ValueError(f"YAML must have 'questions' key: {yaml_path}")

    questions = data["questions"]
    logger.info(f"Loaded {len(questions)} golden questions from {yaml_path}")

    return questions


class EvalHarness:
    """
    Evaluation harness for testing NL query parsing accuracy.

    Uses golden questions (query + expected output) to measure:
    - Intent matching accuracy
    - Variable extraction correctness
    - Filter parsing accuracy
    - Overall confidence scores
    """

    def __init__(self, semantic_layer: SemanticLayer):
        """
        Initialize eval harness with semantic layer.

        Args:
            semantic_layer: SemanticLayer instance for query execution
        """
        self.semantic_layer = semantic_layer
        self.query_engine = NLQueryEngine(semantic_layer)
        logger.info("EvalHarness initialized")

    def evaluate_question(self, question: dict[str, Any]) -> dict[str, Any]:
        """
        Evaluate a single golden question.

        Args:
            question: Golden question dict with expected outputs

        Returns:
            Evaluation result dict with correctness checks

        Example:
            >>> result = harness.evaluate_question({
            ...     "id": "count_all",
            ...     "query": "how many patients?",
            ...     "expected_intent": "COUNT",
            ... })
            >>> print(result["correct"], result["confidence"])
            True 0.9
        """
        query_text = question["query"]
        expected_intent = question.get("expected_intent")
        expected_metric = question.get("expected_metric")
        expected_group_by = question.get("expected_group_by")
        expected_filters = question.get("expected_filters", [])
        conversation_history = question.get("conversation_history", [])

        # Parse query with conversation history (ADR009 Phase 6)
        query_intent = self.query_engine.parse_query(
            query_text,
            conversation_history=conversation_history if conversation_history else None,
        )

        if query_intent is None:
            # Parsing failed
            return {
                "id": question.get("id", "unknown"),
                "query": query_text,
                "intent_match": False,
                "confidence": 0.0,
                "correct": False,
                "error": "Failed to parse query",
            }

        # Check intent match
        intent_match = query_intent.intent_type == expected_intent if expected_intent else True

        # Check metric match (if specified)
        metric_match = query_intent.primary_variable == expected_metric if expected_metric is not None else True

        # Check group_by match (if specified)
        group_by_match = query_intent.grouping_variable == expected_group_by if expected_group_by is not None else True

        # Check filters match (if specified)
        # For now, just check filter count (can be enhanced to check actual filter specs)
        filters_match = len(query_intent.filters) == len(expected_filters) if expected_filters else True

        # Overall correctness
        correct = intent_match and metric_match and group_by_match and filters_match

        result = {
            "id": question.get("id", "unknown"),
            "query": query_text,
            "intent_match": intent_match,
            "metric_match": metric_match,
            "group_by_match": group_by_match,
            "filters_match": filters_match,
            "confidence": query_intent.confidence,
            "correct": correct,
            "actual_intent": query_intent.intent_type,
            "expected_intent": expected_intent,
        }

        return result

    def evaluate_batch(self, questions: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Evaluate multiple golden questions in batch.

        Args:
            questions: List of golden question dicts

        Returns:
            List of evaluation result dicts

        Example:
            >>> results = harness.evaluate_batch(golden_questions)
            >>> summary = harness.get_summary(results)
            >>> print(f"Accuracy: {summary['accuracy']:.1%}")
            Accuracy: 85.0%
        """
        results = []
        for question in questions:
            try:
                result = self.evaluate_question(question)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to evaluate question {question.get('id')}: {e}")
                results.append(
                    {
                        "id": question.get("id", "unknown"),
                        "query": question.get("query", ""),
                        "correct": False,
                        "error": str(e),
                    }
                )

        logger.info(f"Evaluated {len(results)} questions")
        return results

    def get_summary(self, results: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Compute summary statistics for evaluation results.

        Args:
            results: List of evaluation result dicts from evaluate_batch()

        Returns:
            Summary dict with accuracy, avg confidence, etc.

        Example:
            >>> summary = harness.get_summary(results)
            >>> print(summary)
            {
                'accuracy': 0.85,
                'total_questions': 20,
                'correct_count': 17,
                'incorrect_count': 3,
                'average_confidence': 0.82,
                ...
            }
        """
        total_questions = len(results)
        correct_count = sum(1 for r in results if r.get("correct", False))
        incorrect_count = total_questions - correct_count

        # Average confidence (only for successful parses)
        confidences = [r["confidence"] for r in results if "confidence" in r]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

        # Intent accuracy (only for questions with expected_intent)
        intent_matches = [r for r in results if "intent_match" in r]
        intent_accuracy = (
            sum(1 for r in intent_matches if r["intent_match"]) / len(intent_matches) if intent_matches else 0.0
        )

        summary = {
            "accuracy": correct_count / total_questions if total_questions > 0 else 0.0,
            "total_questions": total_questions,
            "correct_count": correct_count,
            "incorrect_count": incorrect_count,
            "average_confidence": avg_confidence,
            "intent_accuracy": intent_accuracy,
        }

        return summary
