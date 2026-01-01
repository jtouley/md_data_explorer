"""Storage utilities for performance tracking data."""

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Schema definitions
PERFORMANCE_DATA_SCHEMA = {
    "run_id": str,
    "tests": list,
    "summary": dict,
}

BASELINE_SCHEMA = {
    "baseline_date": str,
    "tests": dict,
    "suite_metrics": dict,
}


def _empty_performance_data() -> dict[str, Any]:
    """Return empty performance data structure."""
    return {
        "run_id": "",
        "tests": [],
        "summary": {
            "total_tests": 0,
            "slow_tests": 0,
            "total_duration": 0.0,
            "average_duration": 0.0,
        },
    }


def _empty_baseline() -> dict[str, Any]:
    """Return empty baseline structure."""
    return {
        "baseline_date": "",
        "tests": {},
        "suite_metrics": {},
    }


def _validate_performance_data(data: dict[str, Any]) -> bool:
    """Validate performance data schema."""
    if not isinstance(data, dict):
        return False
    if "run_id" not in data or "tests" not in data or "summary" not in data:
        return False
    if not isinstance(data["tests"], list):
        return False
    if not isinstance(data["summary"], dict):
        return False
    return True


def _validate_baseline(data: dict[str, Any]) -> bool:
    """Validate baseline schema."""
    if not isinstance(data, dict):
        return False
    if "baseline_date" not in data or "tests" not in data or "suite_metrics" not in data:
        return False
    if not isinstance(data["tests"], dict):
        return False
    if not isinstance(data["suite_metrics"], dict):
        return False
    return True


def load_performance_data(data_file: Path) -> dict[str, Any]:
    """
    Load performance data from JSON file.

    Args:
        data_file: Path to performance data JSON file

    Returns:
        Performance data dictionary, or empty structure if file missing/invalid
    """
    if not data_file.exists():
        logger.warning(f"Performance data file not found: {data_file}")
        return _empty_performance_data()

    try:
        with open(data_file, encoding="utf-8") as f:
            data = json.load(f)

        if not _validate_performance_data(data):
            logger.error(f"Invalid performance data schema in {data_file}")
            return _empty_performance_data()

        return data
    except json.JSONDecodeError as e:
        logger.error(f"Corrupted JSON in performance data file {data_file}: {e}")
        return _empty_performance_data()
    except Exception as e:
        logger.error(f"Error loading performance data from {data_file}: {e}")
        return _empty_performance_data()


def save_performance_data(data: dict[str, Any], data_file: Path) -> None:
    """
    Save performance data to JSON file.

    Args:
        data: Performance data dictionary
        data_file: Path to save performance data JSON file
    """
    data_file.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(data_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        logger.error(f"Error saving performance data to {data_file}: {e}")
        raise


def load_baseline(baseline_file: Path) -> dict[str, Any]:
    """
    Load baseline data from JSON file.

    Args:
        baseline_file: Path to baseline JSON file

    Returns:
        Baseline data dictionary, or empty structure if file missing/invalid
    """
    if not baseline_file.exists():
        logger.warning(f"Baseline file not found: {baseline_file}")
        return _empty_baseline()

    try:
        with open(baseline_file, encoding="utf-8") as f:
            data = json.load(f)

        if not _validate_baseline(data):
            logger.error(f"Invalid baseline schema in {baseline_file}")
            return _empty_baseline()

        return data
    except json.JSONDecodeError as e:
        logger.error(f"Corrupted JSON in baseline file {baseline_file}: {e}")
        return _empty_baseline()
    except Exception as e:
        logger.error(f"Error loading baseline from {baseline_file}: {e}")
        return _empty_baseline()


def save_baseline(baseline: dict[str, Any], baseline_file: Path) -> None:
    """
    Save baseline data to JSON file.

    Args:
        baseline: Baseline data dictionary
        baseline_file: Path to save baseline JSON file
    """
    baseline_file.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(baseline_file, "w", encoding="utf-8") as f:
            json.dump(baseline, f, indent=2)
    except Exception as e:
        logger.error(f"Error saving baseline to {baseline_file}: {e}")
        raise
