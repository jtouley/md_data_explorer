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

## Workflow steps (auto-proceed)

The agent executes these steps in order, **auto-proceeding** through each gate. See parent `SKILL.md` for the authoritative rules.

### 1. Locate or create plan
- Read `.cursor/plans/*.plan.md` (or tracked `MASTER_PLAN.md` / omnibus).
- Identify next `pending` phase whose dependencies are all `done`.

### 2. Plan review (new plans only)
- **Skip for mature plans** (any phase already `done`).
- New plans: `/plan-review` → artifact at `.context/reviews/plan_<name>.md`.
- Auto-proceed on READY TO EXECUTE. Loop `/plan-update` → `/plan-review` on READY WITH CHANGES. Stop only on unresolvable NOT READY.

### 3. Implement next slice
- **Red → Green → Refactor** per repo rules.
- Run verify commands. Fix. Re-run until green.
- Commit and push. Update plan YAML status.

### 4. PR review (internal)
- Resolve PR `N`: `gh pr list --head "$(git branch --show-current)"`. Reuse existing; create only when none exists.
- Save diff:
  ```bash
  mkdir -p .context/diffs
  gh pr diff <N> > .context/diffs/pr<N>.diff
  ```
- Write review artifact to `.context/reviews/<N>.md`.
- **MERGE:** Proceed to next slice. **NO MERGE (fixable):** Fix, re-push, re-review. **NO MERGE (blocker):** DECISIONS NEEDED.

### 5. Repeat or hand off
- More pending phases + context available → loop to step 3.
- Context exhausted → write handoff block (next slice, files, commands, PR `#`, blockers).
- All phases `done` → report completion.

## Plan identifier

- Full path: `.cursor/plans/foo.plan.md`
- Basename `foo` → review artifact `plan_foo.md` (verify against this repo's `/plan-review` command).

## Quality gates

- **Red phase:** Project may allow focused `pytest` / `uv run pytest` for one test; confirm in repo rules.
- **Green / PR:** Prefer repo's standard interface (**Makefile**, **npm**, **cargo test**, etc.) — do not assume `make test-fast` unless the project has it.

## Multi-phase plans (same skill, many slices)

- **"All phases"** means **repeat the workflow** per unchecked phase, not "finish everything in one assistant message."
- After each slice: run verify commands; fix; re-run.
- If the model runs out of context: emit a **handoff** (next slice, commands, PR `#`, blockers) so the next turn continues **without re-deriving the plan**.

## md_data_explorer

When the workspace is **md_data_explorer**, use **Makefile** (`make test-core`, `make test-fast`, `make check`), **mdde-context**, and `.cursor/rules/` as referenced in that repo's CLAUDE / AGENTS docs. Ensure dev dependencies are installed before `git push` if pre-push runs `pre_commit` (`make install-dev`).
