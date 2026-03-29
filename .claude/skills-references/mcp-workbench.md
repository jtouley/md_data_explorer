# MCP workbench (md_data_explorer)

Instructions for agents when the host registers official MCP servers alongside normal IDE tools. **Installed copy:** `~/.cursor/skills/references/mcp-workbench.md` (sync from this file via `make sync-cursor-skills` or `uv run python scripts/sync_volt_cursor_skills.py`). Paths below assume the **md_data_explorer** repository root when working in that project.

## Preconditions

- Confirm `playwright`, `duckdb`, and `serena` appear in the client’s MCP server list (see user `mcp.json` or project overrides).
- Prefer MCP tools when they reduce full-file reads, give structured browser state, or run SQL without bespoke shell glue.

## playwright (Microsoft `@playwright/mcp`)

- Use for interactive browser checks: navigation, accessibility snapshots, clicks, forms, console and network logs.
- Treat as **exploration and debugging**; keep **committed** automation on project conventions (`make test-e2e`, Python Playwright tests, `webapp-testing` skill patterns).
- Prefer snapshots and structured locators over screenshot-only reasoning when the server exposes them.

## duckdb (MotherDuck `mcp-server-motherduck`)

- Use for ad hoc catalog inspection and SQL (`list_tables`, `list_columns`, `execute_query`, connection switching when configured).
- Use for analytics and explain plans on representative databases the user has attached or allowed via `--allow-switch-databases`.
- For **transcript search**, still follow the **read-memories** skill SQL shapes; MCP complements that when query targets are ordinary tables or shared `.duckdb` files.

## serena (oraios `serena-agent`, entrypoint `serena-mcp-server`)

- Use for **symbol-scoped** work: find definitions, references, and edits across Python and TypeScript with LSP-backed tools before bulk file reads.
- Use early in large refactors, ship pipelines, and cross-module debugging to narrow scope.
- Fall back to `rg` and normal file reads when Serena is disabled or the workspace is not indexed.

## Suggested order

- **serena** (or precise `rg`) → locate code and call graph impact.
- **duckdb** → validate data, SQL, and plans.
- **playwright** → confirm UI behavior in a real browser when unit tests are insufficient.

## Related global Cursor skills (under `~/.cursor/skills/`)

These skills include MCP callouts that point back here when the workspace is **md_data_explorer**:

- **frontend-design** — Playwright MCP for quick visual checks; Serena for locating UI code.
- **webapp-testing** — Playwright MCP vs committed Python Playwright / Makefile targets.
- **web-artifacts-builder** — Playwright MCP to smoke bundled artifacts; Serena while editing generated React.
- **test-quality-loop** — Playwright MCP for reconnaissance; Serena for symbol-scoped test work.
- **doc-coauthoring** — Serena when quoting code paths accurately.
- **plan-to-pr** — Serena / duckdb / playwright across the plan-to-PR pipeline.
