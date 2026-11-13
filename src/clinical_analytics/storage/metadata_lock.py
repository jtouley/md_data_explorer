# SPDX-License-Identifier: MIT
"""Locked read-modify-write of a dataset's metadata JSON (the commit point of every save).

Every metadata writer (upload commit, legacy conversion, content-hash backfill, documentation
add, alias add in the semantic layer) goes through ``update_metadata``: under the sidecar lock
``.{id}.json.lock`` the file is re-read, changed and atomically replaced (``storage.atomic``),
so no writer replaces a change it never saw and a crash never leaves the file truncated.
"""

from __future__ import annotations

import json
import platform
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Final

import structlog

from clinical_analytics.storage import atomic

logger = structlog.get_logger()

METADATA_LOCK_TIMEOUT_S: Final = 10.0


class FileLockTimeoutError(Exception):
    """File lock acquisition timeout error."""

    pass


@contextmanager
def file_lock(file_path: Path, timeout: float = 10.0) -> Iterator[Any]:
    """
    Cross-platform file locking context manager.

    Provides exclusive file locking for metadata writes to prevent corruption
    from concurrent Streamlit reruns.

    Platform-specific behavior:
    - Unix/Linux/macOS: Uses fcntl.flock() for full-file advisory lock
    - Windows: Uses msvcrt.locking() to lock 1 byte at file start
      (common hack for file-level locking on Windows, not a full mandatory lock)

    Args:
        file_path: Path to file to lock
        timeout: Maximum time to wait for lock acquisition (seconds)

    Yields:
        File handle (opened for read-write, will not truncate existing content)

    Raises:
        FileLockTimeoutError: If lock cannot be acquired within timeout

    Metadata is never written through the yielded handle: callers lock the sidecar
    ``metadata_lock_path(...)`` and replace the metadata atomically (``_update_metadata``),
    so a crash can never leave it truncated.

    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    # Open file for read-write, create if doesn't exist
    # Use 'r+' mode if file exists, 'w+' if it doesn't (to create it)
    if file_path.exists():
        f = open(file_path, "r+")
    else:
        f = open(file_path, "w+")

    try:
        # Platform-specific locking
        is_windows = platform.system() == "Windows"

        if is_windows:
            # Windows: Use msvcrt
            import msvcrt

            start_time = time.time()
            while True:
                try:
                    msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)  # type: ignore[attr-defined]
                    break  # Lock acquired
                except OSError:
                    if time.time() - start_time > timeout:
                        raise FileLockTimeoutError(f"Could not acquire lock on {file_path} within {timeout}s")
                    time.sleep(0.1)  # Wait and retry
        else:
            # Unix/Linux/macOS: Use fcntl
            import fcntl

            start_time = time.time()
            while True:
                try:
                    fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break  # Lock acquired
                except BlockingIOError:
                    if time.time() - start_time > timeout:
                        raise FileLockTimeoutError(f"Could not acquire lock on {file_path} within {timeout}s")
                    time.sleep(0.1)  # Wait and retry

        # Lock acquired, yield file handle
        yield f

    finally:
        # Release lock (automatic on file close, but explicit is better)
        try:
            if is_windows:
                import msvcrt

                msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)  # type: ignore[attr-defined]
            else:
                import fcntl

                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        except OSError:
            logger.debug("file_lock_unlock_failed", lock=file_path.name)  # closing releases it anyway

        f.close()


def metadata_lock_path(metadata_path: Path) -> Path:
    """Sidecar lock file for a dataset's metadata (the metadata file itself is replaced, not locked)."""
    return metadata_path.with_name(f".{metadata_path.name}.lock")


def update_metadata(
    metadata_path: Path, mutate: Callable[[dict[str, Any]], bool], *, missing_ok: bool = False
) -> dict[str, Any]:
    """Read, change and atomically replace metadata under its lock (``mutate`` False: no write).

    The one metadata write path: every writer re-reads the file under the sidecar lock, so no
    writer replaces a change it never saw. ``missing_ok``: a dataset with no metadata yet starts
    from ``{}`` (a first upload's commit).

    Returns:
        The metadata as it now stands.

    Raises:
        RuntimeError: the lock was not acquired in time.
        FileNotFoundError: no metadata for this dataset (and not ``missing_ok``).
        OSError: the write failed; the previous metadata is unchanged.
        Whatever ``mutate`` raises: nothing is written.
    """
    try:
        with file_lock(metadata_lock_path(metadata_path), timeout=METADATA_LOCK_TIMEOUT_S):
            if missing_ok and not metadata_path.exists():
                current: dict[str, Any] = {}
            else:
                current = json.loads(metadata_path.read_text(encoding="utf-8"))
            if mutate(current):
                atomic.atomic_write_json(metadata_path, current)
            return dict(current)
    except FileLockTimeoutError as e:
        raise RuntimeError(f"Failed to acquire lock on {metadata_path.name} (timeout after 10s)") from e
