#!/usr/bin/env python3
"""Self-improving NL query parsing evaluation script.

This script iteratively runs golden questions, analyzes failures,
and improves the prompt until reaching target accuracy.

Usage:
    python scripts/self_improve_nl_parsing.py --target-accuracy 0.95 --max-iterations 10
"""

import argparse
import os
import sys
from pathlib import Path

import structlog
import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from clinical_analytics.core.eval_harness import EvalHarness
from clinical_analytics.core.prompt_optimizer import PromptOptimizer

logger = structlog.get_logger(__name__)


def write_prompt_overlay(prompt_additions: str, overlay_path: Path) -> None:
    """
    Write prompt additions atomically to overlay file.

    Uses temp + replace to prevent race conditions if engine reads
    while script is writing.

    Args:
        prompt_additions: Generated fixes from failure patterns
        overlay_path: Target overlay file path
    """
    # Ensure parent directory exists
    overlay_path.parent.mkdir(parents=True, exist_ok=True)

    # Write to temp file
    tmp = overlay_path.with_suffix(".tmp")
    tmp.write_text(prompt_additions, encoding="utf-8")

    # Atomic replace (POSIX guarantee)
    os.replace(tmp, overlay_path)

    logger.info("prompt_overlay_written", path=str(overlay_path), length=len(prompt_additions))


def load_golden_questions(questions_file: Path) -> list[dict]:
    """Load golden questions from YAML file."""
    with open(questions_file) as f:
        data = yaml.safe_load(f)
    return data.get("questions", [])


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

        # Run evaluation with FRESH engine (Fix #4 - picks up overlay changes)
        # Note: EvalHarness needs semantic_layer - using mock for self-improvement
        from clinical_analytics.core.semantic import SemanticLayer
        from unittest.mock import MagicMock
        
        # Create mock semantic layer for golden question testing
        mock_layer = MagicMock(spec=SemanticLayer)
        mock_layer.get_column_alias_index.return_value = {}
        mock_layer.available_columns = []
        
        harness = EvalHarness(mock_layer)
        results = harness.evaluate_batch(questions)
        summary = harness.get_summary(results)
        eval_results = {"results": results, "summary": summary}

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

        # Cap patterns to top 5 by priority (Fix #7)
        patterns.sort(key=lambda p: (p.priority, -p.count))
        patterns = patterns[:5]

        # Generate prompt improvements
        prompt_additions = optimizer.generate_improved_prompt_additions(patterns)

        # Hard cap overlay length (prevent bloat) (Fix #7)
        MAX_OVERLAY_LENGTH = 8000
        if len(prompt_additions) > MAX_OVERLAY_LENGTH:
            prompt_additions = prompt_additions[:MAX_OVERLAY_LENGTH]
            logger.warning(
                "prompt_overlay_truncated",
                original_length=len(prompt_additions),
                capped_length=MAX_OVERLAY_LENGTH,
            )

        # Write overlay atomically (Fix #3)
        overlay_path = Path(os.getenv("NL_PROMPT_OVERLAY_PATH", "/tmp/nl_query_learning/prompt_overlay.txt"))
        write_prompt_overlay(prompt_additions, overlay_path)

        # Log iteration
        optimizer.log_iteration(iteration, accuracy, patterns, prompt_additions)

        logger.info(
            "prompt_improvements_applied",
            iteration=iteration,
            additions_length=len(prompt_additions),
        )

        # Display progress
        print(f"\nIteration {iteration}/{args.max_iterations}:")
        print(f"  Accuracy: {accuracy:.1%} (target: {args.target_accuracy:.1%})")
        print(f"  Patterns detected: {len(patterns)}")
        for pattern in patterns:
            print(f"    - {pattern.pattern_type}: {pattern.count} failures")
        print(f"\n📝 Prompt improvements applied:")
        print(f"   {len(prompt_additions)} characters written to {overlay_path.name}")
        print(f"   Top {len(patterns)} patterns addressed")
        print(f"   Re-running evaluation with updated prompt...")

        # NO BREAK - let loop continue to next iteration

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

