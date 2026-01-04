#!/usr/bin/env python
"""
CLI tool for automated golden question maintenance (ADR009 Phase 6).

This tool generates, analyzes, and maintains golden questions from query logs.

Usage:
    python tests/eval/maintain_golden_questions.py \\
        --log-dir /path/to/query/logs \\
        --output /path/to/golden_questions.json \\
        [--dry-run]

Examples:
    # Analyze golden questions (dry-run, no modifications)
    python tests/eval/maintain_golden_questions.py \\
        --log-dir data/query_logs \\
        --output golden_questions.json \\
        --dry-run

    # Update golden questions (apply recommendations)
    python tests/eval/maintain_golden_questions.py \\
        --log-dir data/query_logs \\
        --output golden_questions.json
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import structlog
from clinical_analytics.core.golden_question_generator import (
    analyze_golden_question_coverage,
    generate_golden_questions_from_logs,
    maintain_golden_questions_automatically,
    validate_golden_question,
)

logger = structlog.get_logger()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


def load_query_logs(log_dir: Path) -> list[dict[str, Any]]:
    """
    Load query logs from directory.

    Supports:
    - *.json files (one query per file)
    - *.jsonl files (one query per line)

    Args:
        log_dir: Directory containing query log files

    Returns:
        List of query log dictionaries
    """
    logs = []

    if not log_dir.exists():
        logger.warning("log_dir_not_found", path=log_dir)
        return logs

    # Load JSON files
    for json_file in log_dir.glob("*.json"):
        try:
            with open(json_file) as f:
                data = json.load(f)
                if isinstance(data, list):
                    logs.extend(data)
                elif isinstance(data, dict):
                    logs.append(data)
        except Exception as e:
            logger.warning("failed_to_load_json", file=json_file, error=str(e))

    # Load JSONL files
    for jsonl_file in log_dir.glob("*.jsonl"):
        try:
            with open(jsonl_file) as f:
                for line in f:
                    if line.strip():
                        logs.append(json.loads(line))
        except Exception as e:
            logger.warning("failed_to_load_jsonl", file=jsonl_file, error=str(e))

    logger.info("query_logs_loaded", count=len(logs), source_dir=str(log_dir))
    return logs


def load_existing_golden_questions(output_path: Path) -> list[dict[str, Any]]:
    """
    Load existing golden questions from file.

    Args:
        output_path: Path to golden_questions.json

    Returns:
        List of existing golden questions (empty if file doesn't exist)
    """
    if not output_path.exists():
        logger.info("no_existing_golden_questions", path=output_path)
        return []

    try:
        with open(output_path) as f:
            data = json.load(f)
            questions = data.get("golden_questions", []) if isinstance(data, dict) else data
            logger.info("existing_golden_questions_loaded", count=len(questions), path=str(output_path))
            return questions
    except Exception as e:
        logger.warning("failed_to_load_existing_questions", path=output_path, error=str(e))
        return []


def save_golden_questions(questions: list[dict[str, Any]], output_path: Path, dry_run: bool = False) -> None:
    """
    Save golden questions to file.

    Args:
        questions: List of golden questions
        output_path: Path to save golden_questions.json
        dry_run: If True, don't save (just log)
    """
    if dry_run:
        logger.info("dry_run_mode_skipping_save", path=str(output_path), count=len(questions))
        return

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "golden_questions": questions,
            "count": len(questions),
            "intents": list(set(q.get("intent") for q in questions if q.get("intent"))),
        }

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(
            "golden_questions_saved",
            path=str(output_path),
            count=len(questions),
        )
    except Exception as e:
        logger.error("failed_to_save_golden_questions", path=str(output_path), error=str(e))
        raise


def main() -> int:
    """
    Main CLI entry point.

    Returns:
        Exit code (0 on success, 1 on error)
    """
    parser = argparse.ArgumentParser(description="Automated golden question maintenance from query logs")
    parser.add_argument(
        "--log-dir",
        type=Path,
        required=True,
        help="Directory containing query log files (*.json, *.jsonl)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output file for golden questions (golden_questions.json)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run without modifications (show recommendations only)",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.75,
        help="Minimum confidence threshold for included queries (default: 0.75)",
    )

    args = parser.parse_args()

    logger.info(
        "maintenance_started",
        log_dir=str(args.log_dir),
        output=str(args.output),
        dry_run=args.dry_run,
    )

    try:
        # Load query logs
        query_logs = load_query_logs(args.log_dir)
        if not query_logs:
            logger.error("no_query_logs_found")
            return 1

        # Load existing golden questions
        existing_questions = load_existing_golden_questions(args.output)

        # Initialize semantic layer (placeholder - in production, would load from dataset config)
        # For now, create a minimal mock to avoid dependency issues
        semantic_layer = None

        # Generate new candidates
        candidates = generate_golden_questions_from_logs(query_logs, semantic_layer, min_confidence=args.min_confidence)

        logger.info("candidates_generated", count=len(candidates))

        # Analyze coverage
        all_questions = existing_questions + candidates
        coverage_result = (
            analyze_golden_question_coverage(all_questions, query_logs, semantic_layer)
            if semantic_layer and all_questions
            else {}
        )

        logger.info("coverage_analyzed", gaps=len(coverage_result.get("gaps", [])))

        # Perform maintenance
        maintenance_result = maintain_golden_questions_automatically(
            query_logs, existing_questions, semantic_layer, dry_run=args.dry_run
        )

        # Merge results
        final_questions = existing_questions + maintenance_result.get("new_questions", [])

        # Validate all questions
        valid_questions = []
        for q in final_questions:
            is_valid, error = validate_golden_question(q)
            if is_valid:
                valid_questions.append(q)
            else:
                logger.warning("invalid_question", question=q, error=error)

        # Save results
        save_golden_questions(valid_questions, args.output, dry_run=args.dry_run)

        # Print summary
        print("\n=== Golden Question Maintenance Summary ===\n")
        print(f"Existing questions: {len(existing_questions)}")
        print(f"New candidates:     {len(candidates)}")
        print(f"Valid questions:    {len(valid_questions)}")
        print(f"Coverage gaps:      {len(coverage_result.get('gaps', []))}")
        print(f"Dry run mode:       {args.dry_run}")
        print(f"\nOutput: {args.output}")

        logger.info(
            "maintenance_completed",
            existing_count=len(existing_questions),
            new_count=len(candidates),
            valid_count=len(valid_questions),
            gaps=len(coverage_result.get("gaps", [])),
            dry_run=args.dry_run,
        )

        return 0

    except Exception as e:
        logger.error("maintenance_failed", error=str(e), exc_info=True)
        print(f"\n❌ Error: {str(e)}", file=__import__("sys").stderr)
        return 1


if __name__ == "__main__":
    exit(main())
