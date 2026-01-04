#!/usr/bin/env python3
"""
Eval Harness Runner - Run golden questions evaluation.

Usage:
    python tests/eval/run_eval.py [yaml_path] [--dataset DATASET_ID]

Examples:
    # Run with default golden questions and test dataset
    python tests/eval/run_eval.py

    # Run with custom questions file
    python tests/eval/run_eval.py tests/eval/custom_questions.yaml

    # Run against uploaded dataset
    python tests/eval/run_eval.py --dataset user_upload_123
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import polars as pl
from clinical_analytics.core.eval_harness import EvalHarness, load_golden_questions
from clinical_analytics.core.semantic import SemanticLayer


def create_test_semantic_layer() -> SemanticLayer:
    """
    Create a test semantic layer with sample data.

    Returns:
        SemanticLayer instance for testing
    """
    # Sample data matching common query patterns
    data = {
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

    df = pl.DataFrame(data)

    # Create semantic layer config
    config = {
        "schema_mapping": {
            "identifier": "patient_id",
        },
        "column_aliases": {
            "age": ["Patient Age", "age"],
            "gender": ["Patient Gender", "sex"],
            "status": ["Patient Status", "active status"],
            "treatment": ["Treatment Group", "treatment arm"],
            "cholesterol": ["Total Cholesterol", "chol"],
            "LDL": ["LDL Cholesterol", "ldl"],
            "BMI": ["Body Mass Index", "bmi"],
            "statin": ["Statin Type", "statin used"],
        },
    }

    # Create semantic layer
    semantic_layer = SemanticLayer(
        dataset_name="test_eval_dataset",
        dataset_version="eval_v1",
        config=config,
    )

    # Register data
    semantic_layer.register_source_table(df, "test_data")

    return semantic_layer


def main():
    """Run eval harness on golden questions."""
    parser = argparse.ArgumentParser(description="Run golden questions evaluation")
    parser.add_argument(
        "yaml_path",
        nargs="?",
        default="tests/eval/golden_questions.yaml",
        help="Path to golden questions YAML (default: tests/eval/golden_questions.yaml)",
    )
    parser.add_argument(
        "--dataset",
        help="Dataset ID to evaluate against (default: test dataset)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print detailed results for each question",
    )

    args = parser.parse_args()

    # Load golden questions
    yaml_path = Path(args.yaml_path)
    if not yaml_path.exists():
        print(f"Error: Golden questions file not found: {yaml_path}")
        sys.exit(1)

    print(f"Loading golden questions from: {yaml_path}")
    questions = load_golden_questions(yaml_path)
    print(f"Loaded {len(questions)} questions\n")

    # Create semantic layer
    if args.dataset:
        print(f"Using dataset: {args.dataset}")
        # TODO: Load actual dataset by ID
        # For now, fall back to test dataset
        print("Warning: Custom dataset loading not implemented yet, using test dataset")
        semantic_layer = create_test_semantic_layer()
    else:
        print("Using test dataset")
        semantic_layer = create_test_semantic_layer()

    # Create eval harness
    harness = EvalHarness(semantic_layer)

    # Run evaluation
    print("Running evaluation...\n")
    results = harness.evaluate_batch(questions)

    # Print results
    if args.verbose:
        print("=" * 80)
        print("DETAILED RESULTS")
        print("=" * 80)
        for result in results:
            status = "✓" if result.get("correct") else "✗"
            print(f"\n{status} {result['id']}")
            print(f"  Query: {result['query']}")
            print(f"  Expected: {result.get('expected_intent', 'N/A')}")
            print(f"  Actual: {result.get('actual_intent', 'N/A')}")
            print(f"  Confidence: {result.get('confidence', 0.0):.2f}")
            if not result.get("correct"):
                print("  ❌ Mismatch:")
                if not result.get("intent_match"):
                    print(f"     Intent: expected {result.get('expected_intent')}, got {result.get('actual_intent')}")
                if not result.get("metric_match", True):
                    print("     Metric mismatch")
                if not result.get("group_by_match", True):
                    print("     Group-by mismatch")
                if not result.get("filters_match", True):
                    print("     Filters mismatch")

    # Print summary
    summary = harness.get_summary(results)
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total Questions: {summary['total_questions']}")
    print(f"Correct: {summary['correct_count']}")
    print(f"Incorrect: {summary['incorrect_count']}")
    print(f"Accuracy: {summary['accuracy']:.1%}")
    print(f"Intent Accuracy: {summary['intent_accuracy']:.1%}")
    print(f"Average Confidence: {summary['average_confidence']:.2f}")

    # Exit with error code if accuracy is below threshold
    if summary["accuracy"] < 0.80:
        print("\n⚠️  Warning: Accuracy below 80% threshold!")
        sys.exit(1)

    print(f"\n✓ Evaluation passed (accuracy: {summary['accuracy']:.1%})")
    sys.exit(0)


if __name__ == "__main__":
    main()
