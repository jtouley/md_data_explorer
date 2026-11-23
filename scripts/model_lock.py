#!/usr/bin/env python3
"""Run a command while holding the local-model lock, so one model user runs at a time.

Several jobs sharing one Ollama server slow each other down and time out, which turns a
measurement into noise. Every command that calls the local model goes through this wrapper. It
takes an exclusive ``flock`` on ``$MDDE_MODEL_LOCK`` (default ``~/.cache/mdde/model.lock``,
shared by every checkout and worktree of the user), runs the command, and releases the lock when
the command ends. The lock fd is not inherited, so a daemon the command starts (``ollama serve``)
does not keep holding it.

Priority: measurement commands (``make accept*``, eval runs) pass ``--wait`` and queue on the lock.
Everything else (ad-hoc questions, experiments) runs without ``--wait`` and gives up at once with
exit 75 when the lock is held, so it never delays a measurement.

    python scripts/model_lock.py --wait -- make accept DATASET=covid_ms MODEL=llama3.2:3b
    python scripts/model_lock.py -- uv run python scripts/headless_ask.py ...

Exit codes: the command's own; 75 lock busy (without --wait); 2 usage or lock-file error.
"""

from __future__ import annotations

import argparse
import fcntl
import os
import signal
import subprocess
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import FrameType
from typing import Final, TextIO

LOCK_ENV: Final = "MDDE_MODEL_LOCK"
EX_TEMPFAIL: Final = 75


class LockBusyError(RuntimeError):
    """The lock is held by another model user and the caller asked not to wait."""


def lock_path() -> Path:
    override = os.environ.get(LOCK_ENV)
    if override:
        return Path(override)
    cache = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
    return cache / "mdde" / "model.lock"


def _holder(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8").strip() or "unknown"
    except OSError:
        return "unknown"


@contextmanager
def model_lock(path: Path, *, wait: bool) -> Iterator[TextIO]:
    """Hold an exclusive flock on ``path``; raise LockBusyError at once if busy and not ``wait``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            if not wait:
                raise LockBusyError(f"model lock {path} is held by: {_holder(path)}") from None
            print(f"model_lock: waiting for {path} (held by: {_holder(path)})", file=sys.stderr, flush=True)
            fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            handle.seek(0)
            handle.truncate()
            handle.write(f"pid {os.getpid()} since {time.strftime('%Y-%m-%dT%H:%M:%S')}\n")
            handle.flush()
            yield handle
        finally:
            handle.seek(0)
            handle.truncate()
            fcntl.flock(handle, fcntl.LOCK_UN)


def run_locked(command: list[str], path: Path, *, wait: bool) -> int:
    """Run ``command`` under the lock, forwarding SIGTERM/SIGHUP to it; return its exit status."""
    with model_lock(path, wait=wait):
        child = subprocess.Popen(command)

        def forward(signum: int, _frame: FrameType | None) -> None:
            child.send_signal(signum)

        for signum in (signal.SIGTERM, signal.SIGHUP):
            signal.signal(signum, forward)
        while True:
            try:
                return child.wait()
            except KeyboardInterrupt:  # the child got the same SIGINT from the terminal; wait for it
                continue


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--wait", action="store_true", help="queue for the lock (measurement commands)")
    parser.add_argument("command", nargs=argparse.REMAINDER, help="-- command [args...]")
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("no command given; usage: model_lock.py [--wait] -- command [args...]")

    try:
        return run_locked(command, lock_path(), wait=args.wait)
    except LockBusyError as exc:
        print(f"model_lock: {exc}. Another model user is running; retry later, or pass --wait.", file=sys.stderr)
        return EX_TEMPFAIL
    except OSError as exc:
        print(f"model_lock: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
