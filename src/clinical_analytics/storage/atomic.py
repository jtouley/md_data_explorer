# SPDX-License-Identifier: MIT
"""Crash-safe file writes: the one write path for stored data and metadata.

Every write goes to a hidden ``.<name>.<random>.part`` file in the target's directory, is
fsynced, then moved into place with ``os.replace`` (atomic on POSIX and Windows), and the
directory is fsynced so the rename itself survives a power cut. A crash at any point leaves
either the old file or the new one, never a truncated one; at worst a ``.part`` file is left
behind, which nothing reads.

Two modes:

* ``atomic_write(path, writer)`` replaces ``path`` (metadata, the commit point).
* ``atomic_write(path, writer, exclusive=True)`` never replaces an existing ``path``: data files
  are named by content version, so an existing file already holds these bytes. It returns False
  and leaves the file alone.

``contracts/store.py`` uses the same temp + fsync + rename pattern for contracts and reports.
"""

from __future__ import annotations

import json
import os
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger()

PART_SUFFIX = ".part"


def _part_path(path: Path) -> Path:
    return path.with_name(f".{path.name}.{uuid.uuid4().hex}{PART_SUFFIX}")


def _fsync_file(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _fsync_dir(directory: Path) -> None:
    """Persist a rename. Windows cannot open a directory for fsync; NTFS journals the rename."""
    if os.name == "nt":
        return
    fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_write(path: Path, writer: Callable[[Path], object], *, exclusive: bool = False) -> bool:
    """Write ``path`` crash-safely via ``writer(part_path)``.

    Args:
        path: Final file path.
        writer: Writes the complete content to the path it is given (a temp file); its return is ignored.
        exclusive: Never replace an existing ``path`` (content-versioned data files).

    Returns:
        True when this call published the file, False when ``exclusive`` and it already existed.

    Raises:
        OSError: The write, fsync or rename failed; ``path`` is unchanged.
    """
    path = Path(path)
    log = logger.bind(file=path.name, exclusive=exclusive)
    if exclusive and path.exists():
        log.info("atomic_write_kept_existing")
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    part = _part_path(path)
    try:
        writer(part)
        _fsync_file(part)
        if exclusive:
            # A hard link publishes only if the name is free; no window where a reader sees a
            # half-written file and no overwrite of a file that appeared since the check above.
            try:
                os.link(part, path)
            except FileExistsError:
                log.info("atomic_write_kept_existing")
                return False
            except OSError:
                # File systems without hard links (some network and FAT volumes).
                if path.exists():
                    log.info("atomic_write_kept_existing")
                    return False
                os.replace(part, path)
        else:
            os.replace(part, path)
        _fsync_dir(path.parent)
    finally:
        part.unlink(missing_ok=True)
    log.debug("atomic_write_done")
    return True


def atomic_write_bytes(path: Path, data: bytes, *, exclusive: bool = False) -> bool:
    """``atomic_write`` for a bytes payload."""
    return atomic_write(path, lambda p: p.write_bytes(data), exclusive=exclusive)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Replace a JSON document crash-safely (metadata: always replaced, never exclusive)."""
    text = json.dumps(payload, indent=2)
    atomic_write(path, lambda p: p.write_text(text, encoding="utf-8"))
