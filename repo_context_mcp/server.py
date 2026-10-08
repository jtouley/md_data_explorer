"""stdio MCP server: repo manifest, rolling diagnostics, bounded plan/doc search.

Run from repository root (Cursor default cwd) or set REPO_CONTEXT_ROOT.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from repo_context_mcp._lib import (
    ALLOWED_SEARCH_ROOTS,
    MAX_SEARCH_FILE_BYTES,
    MAX_SEARCH_LINES_PER_FILE,
    MAX_SEARCH_MATCHES,
    diagnostic_file,
    parse_initiative_slug,
    repo_root,
)

mcp = FastMCP("clinical_analytics_repo_context")


def _read_text(path: Path, limit: int) -> str | None:
    try:
        data = path.read_bytes()[:limit]
    except OSError:
        return None
    try:
        return data.decode("utf-8", errors="replace")
    except OSError:
        return None


@mcp.tool(
    name="get_repo_context_manifest",
    annotations={
        "title": "Repo context manifest",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def get_repo_context_manifest() -> str:
    """Return `.context/mcp/repo_context_manifest.json` (freshness / git_head)."""
    root = repo_root()
    manifest = root / ".context" / "mcp" / "repo_context_manifest.json"
    if not manifest.is_file():
        head = ""
        try:
            r = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=root,
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            head = (r.stdout or "").strip()
        except (OSError, subprocess.TimeoutExpired):
            head = ""
        payload = {
            "error": "manifest not found — run write_repo_context_manifest.py from repo root",
            "git_head": head or "unknown",
            "repo_context_root": str(root),
        }
        return json.dumps(payload, indent=2)
    text = _read_text(manifest, 2_000_000)
    if text is None:
        return json.dumps({"error": "could not read manifest", "path": str(manifest)})
    return text


@mcp.tool(
    name="get_repo_context_diagnostic",
    annotations={
        "title": "Repo context diagnostic markdown",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def get_repo_context_diagnostic(initiative: str = "") -> str:
    """Read rolling `.context/diagnostics/repo_context.md`, or `<initiative>_context.md` if set."""
    root = repo_root()
    try:
        slug = parse_initiative_slug(initiative or None)
    except ValueError as e:
        return f"Error: {e}"
    path = diagnostic_file(root, slug)
    if not path.is_file():
        hint = f"Run the repo-context skill workflow (step 5) or create this file. Expected: {path}"
        return f"Error: diagnostic not found. {hint}"
    text = _read_text(path, 2_000_000)
    if text is None:
        return f"Error: could not read {path}"
    return text


@mcp.tool(
    name="search_plans_and_docs",
    annotations={
        "title": "Search docs and plans (bounded)",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def search_plans_and_docs(query: str, max_results: int = 30) -> str:
    """Case-insensitive substring search under docs/, .cursor/plans/, .context/ only."""
    root = repo_root()
    q = (query or "").strip().lower()
    if len(q) < 2:
        return "Error: query must be at least 2 characters."
    cap = max(1, min(max_results, MAX_SEARCH_MATCHES))
    hit_files: list[Path] = []
    for rel in ALLOWED_SEARCH_ROOTS:
        base = root / rel
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*")):
            if len(hit_files) >= cap:
                break
            if not path.is_file():
                continue
            if path.suffix.lower() not in {
                ".md",
                ".mdc",
                ".yml",
                ".yaml",
                ".toml",
                ".json",
                ".txt",
            }:
                continue
            try:
                st = path.stat()
            except OSError:
                continue
            if st.st_size > MAX_SEARCH_FILE_BYTES:
                continue
            content = _read_text(path, MAX_SEARCH_FILE_BYTES)
            if content is None:
                continue
            if q not in content.lower():
                continue
            hit_files.append(path)
        if len(hit_files) >= cap:
            break
    if not hit_files:
        return f"No matches for {query!r} under {ALLOWED_SEARCH_ROOTS} (capped {cap} files)."
    matches: list[str] = []
    for path in hit_files:
        content = _read_text(path, MAX_SEARCH_FILE_BYTES)
        if content is None:
            continue
        rel_path = path.relative_to(root)
        line_hits = 0
        for i, line in enumerate(content.splitlines(), start=1):
            if q in line.lower():
                matches.append(f"{rel_path}:{i}:{line.strip()[:200]}")
                line_hits += 1
                if line_hits >= MAX_SEARCH_LINES_PER_FILE:
                    break
    return "\n".join(matches)


@mcp.tool(
    name="list_recent_plans",
    annotations={
        "title": "Recent Cursor plan files",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def list_recent_plans(limit: int = 20) -> str:
    """List `.cursor/plans/**/*.plan.md` by mtime, newest first."""
    root = repo_root()
    plans_dir = root / ".cursor" / "plans"
    if not plans_dir.is_dir():
        return f"No .cursor/plans directory at {plans_dir}"
    n = max(1, min(limit, 50))
    entries: list[tuple[float, Path]] = []
    for path in plans_dir.rglob("*.plan.md"):
        if path.is_file():
            try:
                m = path.stat().st_mtime
            except OSError:
                continue
            entries.append((m, path))
    entries.sort(key=lambda x: x[0], reverse=True)
    lines = [f"{p.relative_to(root)}\t{m}" for m, p in entries[:n]]
    return "\n".join(lines) if lines else "No *.plan.md files found."


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
