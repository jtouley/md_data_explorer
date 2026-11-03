"""Pure helpers for repo-context MCP (testable, no FastMCP import required)."""

from __future__ import annotations

import os
import re
from pathlib import Path

_INITIATIVE_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,199}$")


def repo_root() -> Path:
    """Resolve repository root: REPO_CONTEXT_ROOT env, else cwd."""
    env = os.environ.get("REPO_CONTEXT_ROOT", "").strip()
    if env:
        return Path(env).resolve()
    return Path.cwd().resolve()


def parse_initiative_slug(raw: str | None) -> str | None:
    """Return normalized slug or None for empty. Raises ValueError if invalid."""
    if raw is None or not str(raw).strip():
        return None
    s = str(raw).strip()
    if not _INITIATIVE_RE.fullmatch(s):
        msg = "initiative must match [a-zA-Z0-9][a-zA-Z0-9_-]{0,199} (no path segments or '..')"
        raise ValueError(msg)
    return s


def diagnostic_file(repo: Path, initiative: str | None) -> Path:
    """Path to rolling or initiative-specific diagnostic markdown."""
    diag = repo / ".context" / "diagnostics"
    if initiative:
        return diag / f"{initiative}_context.md"
    return diag / "repo_context.md"


ALLOWED_SEARCH_ROOTS = ("docs", ".cursor/plans", ".context")
MAX_SEARCH_FILE_BYTES = 256_000
MAX_SEARCH_MATCHES = 40
MAX_SEARCH_LINES_PER_FILE = 5
