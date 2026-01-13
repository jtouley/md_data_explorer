# Claude Guardrails

Quick reference for the Claude Code governance hooks in this repo.

## Active Hooks (`.claude/settings.json`)

| Hook | Trigger | What it does |
|------|---------|--------------|
| `block-main-edits.sh` | Edit/Write on main/master | Blocks edits, prompts branch creation |
| `auto-format-python.sh` | Edit/Write *.py | Runs `ruff format` + `ruff check --fix` |
| `enforce-tests-updated.sh` | Edit/Write src/clinical_analytics/*.py | Blocks if tests not updated |

## Terminal Commands

**Format a touched file:**
```bash
uv run ruff format path/to/file.py
uv run ruff check --fix path/to/file.py
```

**Run fast tests:**
```bash
make test-fast
```

**Type check (optional):**
```bash
make type-check
```

**Full quality gate before commit:**
```bash
make check-fast
```

## Reminder

When you change production code (`src/`), update or add corresponding tests (`tests/`). The `enforce-tests-updated.sh` hook will block otherwise.

## Files

- `.claude/settings.json` — Hook configuration
- `.claude/hooks/` — Hook scripts
- `.claude/agents/code-reviewer.md` — Review checklist
- `.claude/commands/` — Claude commands
