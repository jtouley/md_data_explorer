"""
Golden Questions Evaluation Test.

Run this to evaluate query parsing accuracy against known-good queries:
    pytest tests/eval/test_golden_questions.py -v

This will:
- Load golden questions from golden_questions.yaml
- Parse each query using the NLQueryEngine
- Compare results to expected outputs
- Report accuracy metrics

Regression test: If accuracy drops below 80%, the test fails.
"""

import pytest
from clinical_analytics.core.eval_harness import EvalHarness, load_golden_questions


@pytest.mark.integration
@pytest.mark.slow
def test_golden_questions_evaluation(make_semantic_layer):
    """
    Evaluate all golden questions and ensure accuracy is above threshold.

    This is a regression test - if parsing accuracy drops below 80%,
    the test will fail and alert us to degradation.

    Integration test: Uses real NLQueryEngine and SemanticLayer to test
    end-to-end query parsing pipeline.
    """
    # Arrange: Create semantic layer with test data
    semantic_layer = make_semantic_layer(
        data={
            "patient_id": list(range(1, 101)),
            "age": [25 + i % 40 for i in range(100)],
            "gender": ["M" if i % 2 == 0 else "F" for i in range(100)],
            "status": ["active" if i % 3 != 0 else "inactive" for i in range(100)],
            "treatment": ["A" if i % 2 == 0 else "B" for i in range(100)],
            "cholesterol": [150 + i % 100 for i in range(100)],
            "LDL": [100 + i % 80 for i in range(100)],
            "BMI": [20 + i % 15 for i in range(100)],
            "statin": [i % 6 for i in range(100)],  # 0=n/a, 1-5=different statins
        }
    )

    # Load golden questions
    yaml_path = "tests/eval/golden_questions.yaml"
    questions = load_golden_questions(yaml_path)

    # Act: Run evaluation
    harness = EvalHarness(semantic_layer)
    results = harness.evaluate_batch(questions)
    summary = harness.get_summary(results)

    # Print detailed results for failures
    failures = [r for r in results if not r.get("correct", False)]
    if failures:
        print("\n" + "=" * 80)
        print("FAILURES (detailed)")
        print("=" * 80)
        for result in failures:
            print(f"\n✗ {result['id']}")
            print(f"  Query: {result['query']}")
            exp = result.get("expected_intent")
            act = result.get("actual_intent")
            match = result.get("intent_match")
            print(f"  Intent:   expected={exp}, actual={act}, match={match}")
            print(f"  Metric:   match={result.get('metric_match')}")
            print(f"  GroupBy:  match={result.get('group_by_match')}")
            print(f"  Filters:  match={result.get('filters_match')}")
            print(f"  Confidence: {result.get('confidence', 0.0):.2f}")

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total Questions: {summary['total_questions']}")
    print(f"Correct: {summary['correct_count']}")
    print(f"Incorrect: {summary['incorrect_count']}")
    print(f"Accuracy: {summary['accuracy']:.1%}")
    print(f"Intent Accuracy: {summary['intent_accuracy']:.1%}")
    print(f"Average Confidence: {summary['average_confidence']:.2f}")

    # Assert: Accuracy must be above 80%
    assert summary["accuracy"] >= 0.80, (
        f"Golden questions accuracy ({summary['accuracy']:.1%}) "
        f"is below 80% threshold. This indicates regression in query parsing."
    )

    # Assert: Intent accuracy must be above 85%
    assert (
        summary["intent_accuracy"] >= 0.85
    ), f"Intent accuracy ({summary['intent_accuracy']:.1%}) is below 85% threshold."


def test_load_golden_questions_yaml():
    """Test that golden questions YAML can be loaded."""
    # Act: Load questions
    yaml_path = "tests/eval/golden_questions.yaml"
    questions = load_golden_questions(yaml_path)

    # Assert: Questions loaded successfully
    assert len(questions) > 0, "Should load at least one question"
    assert all("id" in q for q in questions), "All questions should have IDs"
    assert all("query" in q for q in questions), "All questions should have queries"
    assert all("expected_intent" in q for q in questions), "All questions should have expected intent"
