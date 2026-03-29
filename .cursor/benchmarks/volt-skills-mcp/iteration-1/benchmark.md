# Skill benchmark: volt skills + MCP wiring

**Date:** 2026-03-28 (updated)
**Type:** Structural — verifies versioned MCP reference, repo context, and global `~/.cursor/skills/` volt + manual skills.

## How to run

```bash
uv run python scripts/benchmark_volt_skills_mcp.py
```

## Design notes

- **Sync target:** `make sync-cursor-skills` installs volt-* skills, copies packaged dirs under `.cursor/skills/` that contain `SKILL.md`, and copies `mcp-workbench.md` to `~/.cursor/skills/references/`.
- **Versioned MCP doc:** `.claude/skills-references/mcp-workbench.md` is the git source; sync copies it to `~/.cursor/skills/references/mcp-workbench.md`.

## Limitations

- Does not measure model quality or MCP invocation rate. For skill-creator-style **per-skill YAML validation** of every global skill, run `make benchmark-cursor-skills`.
