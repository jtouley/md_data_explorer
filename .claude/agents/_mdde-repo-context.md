# Shared context for `mdde-*` agents

Use this as the canonical stack summary when any `mdde-*` agent is invoked.

## Project

**md_data_explorer** — clinical analytics platform (Polars-first pipelines, DuckDB analytics, Ibis SQL generation, Streamlit UI; Electron + TypeScript UI migration per project plans).

## Authoritative docs

- `.claude/CLAUDE.md` — Python, Polars, testing, and review expectations
- `tests/AGENTS.md` — fixtures, markers, `assert_frame_equal`, Makefile test commands
- `Makefile` — `make test-fast`, `make test-core`, `make check-fast`, etc.

## Non-negotiables

- **No new pandas** in application or test code unless an existing justified exception pattern applies.
- **No bare `except Exception`**; catch specific types; fail loud at boundaries.
- **Tests:** `make test-*` or `make test-fast` — not ad-hoc `pytest` / `uv run pytest` as the default workflow.
- **Polars:** lazy `scan_*`, single `collect()` where possible; avoid `map_elements`; prefer expressions and `polars.selectors`.
- **DataFrame asserts:** `polars.testing.assert_frame_equal` for comparisons.

## Optional deep dives

- `taskmaster.yaml` — conceptual agent/workflow map (not runtime wiring)
- `docs/` — architecture and specs

## MCP workbench (when Cursor exposes MCP)

- Load `~/.cursor/skills/references/mcp-workbench.md` whenever `playwright`, `duckdb` (MotherDuck `mcp-server-motherduck`), or `serena` (`serena-agent` / `serena-mcp-server`) servers are available (source of truth in-repo: `.claude/skills-references/mcp-workbench.md`; refresh via `make sync-cursor-skills`).
- Prefer **serena** for symbol-level navigation, **duckdb** for SQL and catalog inspection, **playwright** for live browser verification—without replacing Makefile-driven tests or committed E2E suites.
