# Cursor skills (project folder)

## Sync to `~/.cursor/skills/`

```bash
make sync-cursor-skills
```

This runs `scripts/sync_volt_cursor_skills.py`, which:

1. **Generates** **volt-*** skills (and **volt-mdde-context**) from `.claude/agents/volt-*.md` into `~/.cursor/skills/`.
2. **Copies** every **packaged** skill under **this directory** whose folder contains **`SKILL.md`** (e.g. **ship-feature-spec-pr/**, including `references/`).
3. Copies **mcp-workbench.md** into `~/.cursor/skills/references/`.

Generated volt bodies are **not** committed under `.cursor/skills/`; only repo-authored packaged skills live here.

## Global-only skills (not in this repo)

These stay under **`~/.cursor/skills/`** from other installs; **`make sync-cursor-skills` does not remove them**. Attach in Cursor when needed:

| Skill | Typical path |
|-------|----------------|
| **staff-repo-context-mcp** | `~/.cursor/skills/staff-repo-context-mcp/` |
| **staff-data-driven-test-engineer** | `~/.cursor/skills/staff-data-driven-test-engineer/` |
| **read-memories** | `~/.cursor/skills/read-memories/` |
| **`/staff-consult`** | `.cursor/commands/staff-consult.md` |

## Diagnostics (gitignored)

Handoffs for shipping may live under **`.context/diagnostics/`** (default `.gitignore` includes `.context/`). Example: `electron_ui_migration_context.md` for the Electron track.
