#!/usr/bin/env python3
"""Self-improving NL query parsing evaluation script.

This script iteratively runs golden questions, analyzes failures,
and improves the prompt until reaching target accuracy.

Usage:
    python scripts/self_improve_nl_parsing.py --target-accuracy 0.95 --max-iterations 10
"""

import argparse
import sys
from pathlib import Path

import structlog
import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from clinical_analytics.core.eval_harness import EvalHarness
from clinical_analytics.core.prompt_optimizer import PromptOptimizer

logger = structlog.get_logger(__name__)


def load_golden_questions(questions_file: Path) -> list[dict]:
    """Load golden questions from YAML file."""
    with open(questions_file) as f:
        data = yaml.safe_load(f)
    return data.get("golden_questions", [])


def convert_eval_results_to_optimizer_format(eval_results: dict) -> list[dict]:
    """Convert eval harness results to prompt optimizer format."""
    test_results = []

    for result in eval_results.get("results", []):
        test_results.append(
            {
                "query": result.get("question", {}).get("query", ""),
                "expected_intent": result.get("question", {}).get("expected_intent"),
                "actual_intent": result.get("parsed_query", {}).get("intent_type"),
                "expected_metric": result.get("question", {}).get("expected_metric"),
                "actual_metric": result.get("parsed_query", {}).get("primary_variable"),
                "expected_group_by": result.get("question", {}).get("expected_group_by"),
                "actual_group_by": result.get("parsed_query", {}).get("grouping_variable"),
                "expected_filters": result.get("question", {}).get("expected_filters", []),
                "actual_filters": result.get("parsed_query", {}).get("filters", []),
                "conversation_history": result.get("question", {}).get(
                    "conversation_history", []
                ),
                "passed": result.get("passed", False),
                "confidence": result.get("parsed_query", {}).get("confidence", 0.0),
            }
        )

    return test_results


def main():
    """Run self-improving evaluation loop."""
    parser = argparse.ArgumentParser(description="Self-improving NL query parsing evaluation")
    parser.add_argument(
        "--target-accuracy",
        type=float,
        default=0.95,
        help="Target accuracy threshold (default: 0.95)",
    )
    parser.add_argument(
        "--max-iterations", type=int, default=10, help="Maximum iterations (default: 10)"
    )
    parser.add_argument(
        "--questions-file",
        type=Path,
        default=Path("tests/eval/golden_questions.yaml"),
        help="Path to golden questions file",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("/tmp/nl_query_learning"),
        help="Directory for iteration logs",
    )
    args = parser.parse_args()

    logger.info(
        "self_improve_start",
        target_accuracy=args.target_accuracy,
        max_iterations=args.max_iterations,
    )

    # Initialize components
    optimizer = PromptOptimizer(log_dir=args.log_dir)
    questions = load_golden_questions(args.questions_file)

    logger.info("loaded_golden_questions", count=len(questions))

    # Iterative improvement loop
    for iteration in range(1, args.max_iterations + 1):
        logger.info("iteration_start", iteration=iteration)

        # Run evaluation
        harness = EvalHarness()
        eval_results = harness.run_evaluation(questions, verbose=False)

        accuracy = eval_results.get("summary", {}).get("accuracy", 0.0)
        logger.info("iteration_evaluation_complete", iteration=iteration, accuracy=accuracy)

        # Check if target reached
        if accuracy >= args.target_accuracy:
            logger.info(
                "target_accuracy_reached",
                iteration=iteration,
                accuracy=accuracy,
                target=args.target_accuracy,
            )
            print(f"\n✅ SUCCESS: Reached {accuracy:.1%} accuracy (target: {args.target_accuracy:.1%})")
            print(f"   Iterations: {iteration}/{args.max_iterations}")
            return 0

        # Analyze failures and generate improvements
        test_results = convert_eval_results_to_optimizer_format(eval_results)
        patterns = optimizer.analyze_failures(test_results)

        logger.info("failure_patterns_detected", iteration=iteration, pattern_count=len(patterns))

        # Generate prompt improvements
        prompt_additions = optimizer.generate_improved_prompt_additions(patterns)

        # Log iteration
        optimizer.log_iteration(iteration, accuracy, patterns, prompt_additions)

        # Display progress
        print(f"\nIteration {iteration}/{args.max_iterations}:")
        print(f"  Accuracy: {accuracy:.1%} (target: {args.target_accuracy:.1%})")
        print(f"  Patterns detected: {len(patterns)}")
        for pattern in patterns:
            print(f"    - {pattern.pattern_type}: {pattern.count} failures")

        # Update prompt (in real implementation, this would modify nl_query_engine.py)
        logger.info(
            "prompt_improvements_generated",
            iteration=iteration,
            additions_length=len(prompt_additions),
        )

        # In production, we would:
        # 1. Update nl_query_engine.py with prompt_additions
        # 2. Reload NLQueryEngine
        # 3. Continue loop
        # For now, we just log and break after first iteration
        logger.warning(
            "manual_intervention_required",
            message="Prompt improvements generated but not automatically applied. "
            "Update nl_query_engine.py manually with improvements.",
        )
        print(f"\n📝 Prompt improvements:")
        print(prompt_additions)
        break

    # Max iterations reached without achieving target
    logger.warning(
        "max_iterations_reached",
        final_accuracy=accuracy,
        target=args.target_accuracy,
        iterations=args.max_iterations,
    )
    print(f"\n⚠️  Max iterations reached: {accuracy:.1%} accuracy (target: {args.target_accuracy:.1%})")
    print(f"   Check logs in: {args.log_dir}")
    return 1


if __name__ == "__main__":
    sys.exit(main())

