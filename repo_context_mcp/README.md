# Repo-context MCP (local stdio)

Read-only tools for **manifest**, **rolling diagnostic** (`.context/diagnostics/repo_context.md`), **bounded search** over `docs/`, `.cursor/plans/`, `.context/`, and **recent plan files**.

## One-time setup

1. Install dependencies including the MCP group (from repository root):

   ```bash
   uv sync --all-groups
   ```

   Or add only this group alongside your usual dev install:

   ```bash
   uv sync --group dev --group repo-context-mcp
   ```

   Avoid `uv sync --group repo-context-mcp` alone — it can drop other optional groups.

2. Refresh the manifest (after merges or before heavy context sessions):

   ```bash
   uv run python ~/.cursor/skills/repo-context/scripts/write_repo_context_manifest.py
   ```

   Or keep a copy of that script in-repo and call it the same way.

3. Register the server in **Cursor** → Settings → MCP → Add server:

   - **Name:** `repo-context` (or any label)
   - **Command:** `uv`
   - **Args:**

     ```text
     run
     --group
     repo-context-mcp
     python
     -m
     repo_context_mcp.server
     ```

   - **Cwd:** your clone root (same folder as `pyproject.toml`).

   Optional env var: **`REPO_CONTEXT_ROOT`** — absolute path to the repo if the client cannot set cwd correctly.

## Tools

| Tool | Purpose |
|------|---------|
| `get_repo_context_manifest` | JSON from `.context/mcp/repo_context_manifest.json` (or inline `git_head` if missing) |
| `get_repo_context_diagnostic` | `repo_context.md` by default; pass `initiative` (slug) for `<initiative>_context.md` |
| `search_plans_and_docs` | Substring search, allowlisted roots only |
| `list_recent_plans` | `*.plan.md` under `.cursor/plans/`, newest first |

Handlers re-read the filesystem each call (no stale in-process cache for file content).

## Rolling context file

Maintain **`.context/diagnostics/repo_context.md`** via the **repo-context** Cursor skill. Initiative-specific files use the suffix `_context.md` and a safe slug (letters, digits, `_`, `-`).
