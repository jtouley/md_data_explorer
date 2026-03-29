# Paths and commands (any workspace)

Conventions match Cursor projects that use **`.cursor/plans/`**, **`.context/reviews/`**, and slash commands **`/plan-review`**, **`/plan-update`**, **`/spec-driven`**, **`/pr-review`**. Adjust if this repo uses different paths.

## Repository anchors

| Artifact | Path pattern |
|----------|----------------|
| Plans | `.cursor/plans/<name>.plan.md` |
| Plan review output | `.context/reviews/plan_<plan-basename>.md` |
| PR diff (for `/pr-review`) | `.context/diffs/pr<number>.diff` |
| PR review output | `.context/reviews/<number>.md` |
| Checkpoint (optional) | `.context/checkpoints/<task_id>.md` |
| Portable / tracked plans (when above are gitignored) | `docs/implementation/`, `docs/specs/` — confirm per repository |

**md_data_explorer:** `.cursor/plans/` and `.context/` are **gitignored**; see parent `SKILL.md` § *Gitignored `.cursor/plans/` or `.context/`*.

## Cursor slash commands (run in order)

1. **Capture spec** — Write or point to a plan under `.cursor/plans/` (or a tracked doc if plans are gitignored).
2. **`/plan-review <plan-identifier>`** — Chat summary + `.context/reviews/plan_<name>.md`.
3. **`/plan-update <plan-identifier>`** — Apply review feedback to the plan; re-`/plan-review` until execution-ready.
4. **`/spec-driven <task or plan path>`** — TDD implementation per that repo’s rules (tests, lint, commit, push, PR). Before assuming a new PR number, run `gh pr list --head "$(git branch --show-current)"` and reuse an open PR when present.
5. **Prepare PR diff** — If `.context/diffs/prN.diff` is missing:
   ```bash
   mkdir -p .context/diffs
   gh pr diff <N> > .context/diffs/pr<N>.diff
   ```
6. **`/pr-review PR<N>`** — Diff review + `.context/reviews/<N>.md`.

## Plan identifier

- Full path: `.cursor/plans/foo.plan.md`
- Basename `foo` → review artifact `plan_foo.md` (verify against this repo’s `/plan-review` command).

## Quality gates

- **Red phase:** Project may allow focused `pytest` / `uv run pytest` for one test; confirm in repo rules.
- **Green / PR:** Prefer repo’s standard interface (**Makefile**, **npm**, **cargo test**, etc.)—do not assume `make test-fast` unless the project has it.

## Multi-phase plans (same skill, many slices)

- **“All phases”** means **repeat the workflow** per unchecked phase (or per omnibus checkpoint), not “finish everything in one assistant message.”
- After each slice: run the repo’s **verify** commands; optionally **`staff-data-driven-test-engineer`** on the same target bar; fix; re-run.
- If the model runs out of context: emit a **handoff** (next slice, commands, PR `#`, blockers) so the user or the next turn continues **without re-deriving the plan**.

## md_data_explorer

When the workspace is **md_data_explorer**, use **Makefile** (`make test-core`, `make test-fast`, `make check`), **volt-mdde-context**, and `.cursor/rules/` as referenced in that repo’s CLAUDE / AGENTS docs. Ensure dev dependencies are installed before `git push` if pre-push runs `pre_commit` (`make install-dev`).
