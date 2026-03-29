# Cursor skills (project folder)

**Packaged** subfolders (`SKILL.md` under `.cursor/skills/<name>/`) are an optional git mirror. **`~/.cursor/skills/<name>/` is canonical** — `make sync-cursor-skills` does not copy them.

```bash
make sync-cursor-skills          # mdde-* + mdde-context + mcp-workbench → ~/.cursor/skills/
make cursor-packaged-skills      # diff repo packaged vs global (+ validate_cursor_skill)
```

Other packaged actions (same script; set `CURSOR_PACKAGED_ARGS`):

```bash
make cursor-packaged-skills CURSOR_PACKAGED_ARGS=--pull-packaged-from-global
make cursor-packaged-skills CURSOR_PACKAGED_ARGS=--push-packaged-to-global
make cursor-packaged-skills CURSOR_PACKAGED_ARGS=--promote-packaged-to-global
```

`--force` on the script still applies to generated skills, mcp ref, and push. See `uv run python scripts/sync_volt_cursor_skills.py --help`.

**Elsewhere in `~/.cursor/skills/`** (repo-context, test-quality-loop, read-memories, plan-to-pr, …): attach in Cursor as needed; sync does not delete them.

**Diagnostics:** `.context/diagnostics/` (often gitignored).
