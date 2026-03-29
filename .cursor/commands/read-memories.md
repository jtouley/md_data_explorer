# Read memories (Cursor + Claude Code)

Search past **Cursor** agent transcripts and/or **Claude Code** session logs using DuckDB, following the global skill.

## Trigger

`read-memories <keyword> [--here | --cursor-all]`

- **`--here`** (default): only transcripts for the current workspace (`~/.cursor/projects/<slug>/agent-transcripts/`).
- **`--cursor-all`**: every project under `~/.cursor/projects/`.

## Rules / skill

Global skill path: `~/.cursor/skills/read-memories/SKILL.md` (open that file or rely on Cursor skill discovery by name **read-memories**).

## Behavior

1. Read and follow **`~/.cursor/skills/read-memories/SKILL.md`** (Mode A: Cursor transcripts; Mode B: Claude Code JSONL).
2. Prefer **`uv run python`** + `duckdb` from the repo root so the same DuckDB as `pyproject.toml` is used.
3. **Internalize** matches—summarize decisions and open threads; do not paste large raw JSONL unless the user asks.
4. If chat search is weak, optionally **`rg`** `.cursor/plans/`, `docs/`, `.context/` for the same keyword.

## Related

- Upstream inspiration: [duckdb-skills — read-memories](https://github.com/duckdb/duckdb-skills)
