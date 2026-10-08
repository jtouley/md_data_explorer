# Claude Code Guardrails

This directory contains Claude Code configuration for enforcing quality standards in the md_data_explorer project.

## 📁 Structure

```
.claude/
├── settings.json              # Hook configuration
├── agents/
│   ├── code-reviewer.md       # Code review checklist agent (project-native)
│   ├── _volt-repo-context.md  # Shared stack summary for volt-* agents
│   └── volt-*.md              # Scrubbed subagents (VoltAgent-derived, repo-tuned)
├── commands/
│   ├── code-quality.md        # Quality check command
│   └── pr-review.md           # PR review command
├── hooks/
│   ├── block-main-edits.sh    # Prevent edits on main branch
│   ├── auto-format-python.sh  # Auto-format with ruff
│   └── enforce-tests-updated.sh  # Ensure tests updated with src changes
└── CLAUDE.md                  # Project-specific coding rules

```

## 🔒 Hooks

### PreToolUse Hooks
- **block-main-edits.sh**: Prevents accidental edits on main/master branch

### PostToolUse Hooks
- **auto-format-python.sh**: Automatically formats Python files with ruff after edits
- **enforce-tests-updated.sh**: Blocks commits when src/clinical_analytics changes without test updates

## 🤖 Agents

### code-reviewer.md (project-native)

Comprehensive code review checklist covering:
- Error handling and contract validation
- Idempotency and determinism
- Test coverage and fixture usage
- Polars best practices
- Style compliance

### volt-* agents (scrubbed from VoltAgent / awesome-claude-code-subagents)

These are **short, md_data_explorer–specific** prompts derived from the VoltAgent subagent collection: boilerplate JSON “context manager” protocols, generic web-framework advice, and pandas-first data guidance were removed or replaced with Polars, DuckDB/Ibis, `uv`, and `make test-*` conventions.

| Agent | Use when |
|-------|-----------|
| `volt-python-pro` | Core Python, Polars, typing, project tooling |
| `volt-typescript-pro` | Electron/renderer TypeScript, strict IPC typing |
| `volt-electron-pro` | Electron security, packaging, desktop shell |
| `volt-data-engineer` | Pipelines, ingestion, semantic layer, data quality |
| `volt-sql-duckdb` | DuckDB SQL, explain plans, Ibis SQL review |
| `volt-llm-architect` | NL query / LLM features, safety, evaluation |
| `volt-qa-expert` | Test strategy, risk-based quality, markers/slow tests |
| `volt-test-automator` | pytest, fixtures, `assert_frame_equal` |
| `volt-performance-engineer` | Profiling Polars/DuckDB/UI hot paths |
| `volt-frontend-developer` | Streamlit + future Electron UI |
| `volt-fullstack-developer` | End-to-end slices across UI and analytics core |
| `volt-debugger` | Systematic root-cause and repro |
| `volt-documentation-engineer` | Docs and onboarding aligned with Makefile |

All `volt-*` agents point at **`agents/_volt-repo-context.md`** for shared stack rules.

Upstream ideas: [VoltAgent/awesome-claude-code-subagents](https://github.com/VoltAgent/awesome-claude-code-subagents) (Apache-2.0). This repo’s files are adapted prompts, not a copy of the full upstream set.

### Cursor (Skills)

Cursor loads **Agent Skills** from `~/.cursor/skills/<name>/SKILL.md` (global).

- **`volt-*` + `volt-mdde-context`:** generated from `.claude/agents/volt-*.md` and `_volt-repo-context.md` into `~/.cursor/skills/`.
- **Packaged skills:** any subdirectory of `.cursor/skills/` that contains `SKILL.md` (e.g. **ship-feature-spec-pr**) is copied to `~/.cursor/skills/` on sync.
- **`mcp-workbench.md`:** versioned at `.claude/skills-references/mcp-workbench.md` and copied to `~/.cursor/skills/references/` on sync.

Regenerate global skills after editing agents, packaged skills under `.cursor/skills/`, or the MCP reference:

```bash
make sync-cursor-skills
```

## 📋 Commands

### code-quality.md
Runs comprehensive quality checks:
1. Pre-commit hooks (ruff, mypy, etc.)
2. Fast test suite
3. Coverage regression check

### pr-review.md
Reviews branch changes against main:
1. Shows diff summary
2. Applies code-reviewer checklist
3. Flags missing test coverage
4. Suggests improvements

## 🚀 Usage

### For Claude
Hooks run automatically on Edit/Write operations. No manual action needed.

### For Developers
```bash
# Run quality checks manually
make check-fast

# Review before creating PR
# (code-quality and pr-review commands available through Claude)

# Disable hooks temporarily (not recommended)
# Edit .claude/settings.json and remove/comment out hooks
```

## 🛠️ How It Works

1. **settings.json**: Defines which hooks run on which tool operations
2. **Hooks**: Bash scripts that return JSON with `{"block": true/false, "message": "..."}`
3. **Agents/Commands**: Markdown templates that Claude can invoke for specialized tasks

## 📝 Customization

### Adding New Hooks
1. Create script in `.claude/hooks/`
2. Make it executable: `chmod +x .claude/hooks/your-hook.sh`
3. Add to `.claude/settings.json` under `preToolUse` or `postToolUse`
4. Test with edge cases (empty files, non-existent paths, etc.)

### Modifying Existing Hooks
- **block-main-edits.sh**: Add more protected branches
- **auto-format-python.sh**: Add additional formatters (black, isort, etc.)
- **enforce-tests-updated.sh**: Adjust allowlist patterns or test requirements

### Creating New Commands/Agents
1. Add markdown file to `.claude/commands/` or `.claude/agents/`
2. Follow existing format (clear description, usage, implementation)
3. Reference in other docs as needed

## ⚠️ Troubleshooting

### Hook Failures
```bash
# Test hook manually
CLAUDE_TOOL_INPUT_FILE_PATH="path/to/file.py" bash .claude/hooks/auto-format-python.sh

# Check hook output format (must be valid JSON)
bash .claude/hooks/block-main-edits.sh | jq .
```

### Line Ending Issues
```bash
# Convert to Unix line endings if needed
sed -i 's/\r$//' .claude/hooks/*.sh
```

### Disable All Hooks
Temporarily rename `settings.json`:
```bash
mv .claude/settings.json .claude/settings.json.disabled
```

## 🎯 Goals

These guardrails aim to:
1. **Prevent mistakes**: Block edits on protected branches
2. **Maintain quality**: Auto-format code, enforce test coverage
3. **Provide guidance**: Code review checklists, quality commands
4. **Stay composable**: Small, focused scripts that work together

## 📚 Related Documentation
- `.claude/CLAUDE.md` - Project coding standards
- `tests/AGENTS.md` - Test fixture enforcement
- `.pre-commit-config.yaml` - Git pre-commit hooks
- `Makefile` - Standard development commands
