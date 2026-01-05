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


def generate_golden_questions_from_logs(
    log_file: Path,
    min_confidence: float = 0.8,
    lookback_days: int = 7,
) -> list[dict]:
    """
    Mine successful queries from logs to generate golden question candidates.

    Args:
        log_file: Path to structured log file
        min_confidence: Minimum confidence threshold for candidates
        lookback_days: How many days back to look in logs

    Returns:
        List of candidate golden questions with query, expected_intent, etc.
    """
    # Placeholder implementation - Phase 6 full integration
    # In production, this would:
    # 1. Parse structlog JSON lines from log_file
    # 2. Filter by confidence >= min_confidence
    # 3. Group by query pattern
    # 4. Return top N unique queries as candidates

    logger.info(
        "golden_question_generation_placeholder",
        log_file=str(log_file),
        min_confidence=min_confidence,
        lookback_days=lookback_days,
    )

    return []  # Placeholder - no candidates for now


def refresh_golden_questions_from_logs(
    log_file: Path,
    golden_questions_file: Path,
    min_confidence: float = 0.8,
    max_new_questions: int = 10,
) -> int:
    """
    Mine successful queries from logs and add to golden questions.

    Integrates ADR009 Phase 6 golden question generation into
    self-improvement loop.

    Args:
        log_file: Path to query logs
        golden_questions_file: Path to golden_questions.yaml
        min_confidence: Minimum confidence for candidates
        max_new_questions: Maximum new questions to add per refresh

    Returns:
        Number of new questions added
    """
    # Load existing corpus
    if not golden_questions_file.exists():
        existing_data = {"questions": []}
    else:
        with open(golden_questions_file) as f:
            existing_data = yaml.safe_load(f) or {"questions": []}

    existing_queries = {q["query"].lower() for q in existing_data.get("questions", [])}

    # Generate candidates from logs
    candidates = generate_golden_questions_from_logs(
        log_file=log_file,
        min_confidence=min_confidence,
        lookback_days=7,  # Last week's queries
    )

    # Deduplicate and limit
    new_questions = [c for c in candidates if c["query"].lower() not in existing_queries][:max_new_questions]

    if new_questions:
        existing_data["questions"].extend(new_questions)
        with open(golden_questions_file, "w") as f:
            yaml.dump(existing_data, f)
        logger.info("golden_questions_refreshed", count=len(new_questions))

    return len(new_questions)


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
                "conversation_history": result.get("question", {}).get("conversation_history", []),
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
    parser.add_argument("--max-iterations", type=int, default=10, help="Maximum iterations (default: 10)")
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

    # Step 0: Refresh golden questions from production logs (Phase 6 integration)
    log_file = Path("/tmp/nl_query.log")
    if log_file.exists():
        new_count = refresh_golden_questions_from_logs(
            log_file=log_file,
            golden_questions_file=args.questions_file,
            min_confidence=0.8,
            max_new_questions=5,  # Add up to 5 new questions per run
        )
        if new_count > 0:
            print(f"📚 Added {new_count} new golden questions from logs")

    # Initialize components
    optimizer = PromptOptimizer(log_dir=args.log_dir)
    questions = load_golden_questions(args.questions_file)

    logger.info("loaded_golden_questions", count=len(questions))

    # Iterative improvement loop
    for iteration in range(1, args.max_iterations + 1):
        logger.info("iteration_start", iteration=iteration)

        # Run evaluation with FRESH engine (Fix #4 - picks up overlay changes)
        # Note: EvalHarness needs semantic_layer - using mock for self-improvement
        from unittest.mock import MagicMock

        from clinical_analytics.core.semantic import SemanticLayer

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
        max_overlay_length = 8000
        if len(prompt_additions) > max_overlay_length:
            prompt_additions = prompt_additions[:max_overlay_length]
            logger.warning(
                "prompt_overlay_truncated",
                original_length=len(prompt_additions),
                capped_length=max_overlay_length,
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
        print("\n📝 Prompt improvements applied:")
        print(f"   {len(prompt_additions)} characters written to {overlay_path.name}")
        print(f"   Top {len(patterns)} patterns addressed")
        print("   Re-running evaluation with updated prompt...")

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
