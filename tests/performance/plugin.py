"""Pytest plugin for performance tracking."""

import logging
import time
from datetime import datetime
from pathlib import Path

from performance.storage import load_performance_data, save_performance_data

logger = logging.getLogger(__name__)

# Global state for tracking
_tracking_enabled = False
_config = None
_performance_data_file = None
_worker_id = None
_worker_file = None
_test_results = []


def pytest_addoption(parser):
    """Add --track-performance command-line option."""
    parser.addoption(
        "--track-performance",
        action="store_true",
        default=False,
        help="Enable performance tracking for test execution",
    )


def pytest_configure(config):
    """Configure plugin when pytest starts."""
    global _tracking_enabled, _config, _performance_data_file, _worker_id, _worker_file

    _config = config
    _tracking_enabled = _is_tracking_enabled(config)

    if not _tracking_enabled:
        return

    # Get worker ID for parallel execution
    _worker_id = _get_worker_id(config)

    # Set up file paths
    rootdir = Path(config.rootdir)
    tests_dir = rootdir / "tests"
    _performance_data_file = tests_dir / ".performance_data.json"

    # For workers, use worker-specific file
    if _is_worker_process(config):
        _worker_file = tests_dir / f".performance_data_worker_{_worker_id}.json"
        _test_results = []
    else:
        # Master process - will aggregate worker files in sessionfinish
        _test_results = []

    logger.info(
        "performance_tracking_enabled",
        worker_id=_worker_id,
        is_worker=_is_worker_process(config),
        data_file=str(_performance_data_file),
    )


def pytest_runtest_setup(item):
    """Track test start time."""
    if not _tracking_enabled:
        return
    if _should_exclude_test(item):
        return

    item._performance_start = time.perf_counter()


def pytest_runtest_teardown(item, nextitem):
    """Track test end time and record result."""
    if not _tracking_enabled:
        return
    if _should_exclude_test(item):
        return

    if not hasattr(item, "_performance_start"):
        return

    duration = time.perf_counter() - item._performance_start

    # Get test status from stored report or default to unknown
    status = "unknown"
    if hasattr(item, "rep_call") and hasattr(item.rep_call, "outcome"):
        status = item.rep_call.outcome

    # Extract markers
    markers = [marker.name for marker in item.iter_markers()]

    # Extract module name
    module = item.nodeid.split("::")[0].replace("tests/", "").split("/")[0]

    test_result = {
        "nodeid": item.nodeid,
        "duration": round(duration, 3),
        "markers": markers,
        "module": module,
        "status": status,
        "timestamp": datetime.now().isoformat(),
    }

    _test_results.append(test_result)

    # For workers, write to worker file immediately
    if _is_worker_process(_config):
        _write_worker_file()


def pytest_runtest_logreport(report):
    """Store test report for status extraction."""
    if not _tracking_enabled:
        return

    # Store report on item for use in teardown
    if hasattr(report, "nodeid") and report.when == "call":
        # Find the item and attach report
        if hasattr(_config, "session") and hasattr(_config.session, "items"):
            for item in _config.session.items:
                if item.nodeid == report.nodeid:
                    item.rep_call = report
                    break


def pytest_sessionfinish(session, exitstatus):
    """Aggregate worker files and write final performance data."""
    if not _tracking_enabled:
        return

    # Only master process aggregates
    if _is_worker_process(_config):
        # Final write for worker
        _write_worker_file()
        return

    # Master process: aggregate all worker files
    rootdir = Path(_config.rootdir)
    tests_dir = rootdir / "tests"

    # Collect all test results
    all_test_results = list(_test_results)  # From master process

    # Read worker files
    worker_files = list(tests_dir.glob(".performance_data_worker_*.json"))
    for worker_file in worker_files:
        try:
            worker_data = load_performance_data(worker_file)
            all_test_results.extend(worker_data.get("tests", []))
        except Exception as e:
            logger.warning(f"Error reading worker file {worker_file}: {e}")

    # Calculate summary statistics
    total_tests = len(all_test_results)
    slow_tests = sum(1 for t in all_test_results if t.get("duration", 0) > 30.0)
    total_duration = sum(t.get("duration", 0) for t in all_test_results)

    # Extract all durations for statistics
    durations = [t.get("duration", 0) for t in all_test_results]
    average_duration = total_duration / total_tests if total_tests > 0 else 0.0
    min_duration = min(durations) if durations else 0.0
    max_duration = max(durations) if durations else 0.0

    # Create final performance data
    performance_data = {
        "run_id": datetime.now().isoformat(),
        "tests": all_test_results,
        "summary": {
            "total_tests": total_tests,
            "slow_tests": slow_tests,
            "total_duration": round(total_duration, 3),
            "average_duration": round(average_duration, 3),
            "min_duration": round(min_duration, 3),
            "max_duration": round(max_duration, 3),
        },
    }

    # Write final file
    save_performance_data(performance_data, _performance_data_file)

    # Clean up worker files
    for worker_file in worker_files:
        try:
            worker_file.unlink()
        except Exception as e:
            logger.warning(f"Error cleaning up worker file {worker_file}: {e}")

    logger.info(
        "performance_tracking_complete",
        total_tests=total_tests,
        slow_tests=slow_tests,
        total_duration=total_duration,
        average_duration=average_duration,
        min_duration=min_duration,
        max_duration=max_duration,
    )


def _is_tracking_enabled(config) -> bool:
    """Check if performance tracking is enabled."""
    return config.getoption("--track-performance", default=False)


def _should_exclude_test(item) -> bool:
    """Check if test should be excluded from tracking."""
    # Exclude performance system tests to avoid recursive tracking
    if "tests/performance/test_" in item.nodeid:
        return True
    return False


def _get_worker_id(config) -> str:
    """Get worker ID for parallel execution."""
    if hasattr(config, "workerinput") and config.workerinput:
        return config.workerinput.get("workerid", "master")
    return "master"


def _is_worker_process(config) -> bool:
    """Check if running in worker process."""
    return hasattr(config, "workerinput") and config.workerinput is not None


def _get_performance_data_file(config) -> Path:
    """Get path to performance data file."""
    rootdir = Path(config.rootdir)
    return rootdir / "tests" / ".performance_data.json"


def _get_worker_file(config, worker_id: str) -> Path:
    """Get path to worker-specific performance data file."""
    rootdir = Path(config.rootdir)
    return rootdir / "tests" / f".performance_data_worker_{worker_id}.json"


def _write_worker_file():
    """Write test results to worker-specific file."""
    if not _worker_file or not _test_results:
        return

    # Calculate statistics for worker
    durations = [t.get("duration", 0) for t in _test_results]
    total_duration = sum(durations)
    average_duration = total_duration / len(_test_results) if _test_results else 0.0
    min_duration = min(durations) if durations else 0.0
    max_duration = max(durations) if durations else 0.0

    # Create performance data structure for worker
    worker_data = {
        "run_id": datetime.now().isoformat(),
        "tests": _test_results,
        "summary": {
            "total_tests": len(_test_results),
            "slow_tests": sum(1 for t in _test_results if t.get("duration", 0) > 30.0),
            "total_duration": round(total_duration, 3),
            "average_duration": round(average_duration, 3),
            "min_duration": round(min_duration, 3),
            "max_duration": round(max_duration, 3),
        },
    }

    save_performance_data(worker_data, _worker_file)
