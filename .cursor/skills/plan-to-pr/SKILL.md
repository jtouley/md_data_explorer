---
name: plan-to-pr
description: "Ships features end-to-end: locates or creates a plan, auto-proceeds through plan review (new plans only), spec-driven implementation, and PR review—looping per slice until the plan is complete, context is exhausted, or a genuine blocker surfaces. Use when the user asks to ship a feature, run the full pipeline, or chain plan review with PR review."
---

# Plan to pull request: plan → implement → ship

**Canonical location:** `~/.cursor/skills/plan-to-pr/` only. Packaged skills are not versioned under this repository’s `.cursor/skills/`; use `make sync-cursor-skills` for **mdde-*** / **mcp-workbench** generation only.

## Purpose

Ship code from **plan to merged PR** without manual gate-keeping. Every review gate is an internal checkpoint the agent resolves autonomously; the only valid stops are genuine blockers.

## When to use this skill

- User wants to **ship a feature** from the current or recent thread.
- User asks to **chain** `/plan-review`, `/spec-driven`, and `/pr-review`.
- User says **"run the full pipeline"**, **"global setup"**, **"from this chat to PR"**, **"all phases"**, or **"ship the whole plan"**.

## Auto-proceed rule (non-negotiable)

This skill **ships code**. Every gate (plan review, PR review, CI, verification) is an **internal checkpoint**, not a hand-back point. The agent must **auto-proceed** through each gate when the result is unblocked.

**Valid stops (surface DECISIONS NEEDED):**
- Tests fail and the fix is non-obvious.
- `/plan-review` returns **NOT READY** with a blocker the agent cannot resolve.
- Merge conflict the agent cannot resolve.
- A human-judgment safety question (destructive migration, data loss risk).

**Not valid stops:**
- "Confirm CI is green" — CI is asynchronous; push and note it, keep working.
- "Read the review file" — the agent wrote it; internalize and proceed.
- "Pick the next slice" — the plan YAML defines the order; pick the next `pending` phase whose dependencies are `done`.
- Producing review artifacts without implementing anything.

If context window runs out before the plan is complete, emit a **handoff block** (next slice, commands, PR `#`, blockers) and stop. That is the only non-blocker reason to end a turn.

## Mature-plan bypass

A plan with **any phase marked `done`** has already been reviewed. Skip the `/plan-review` → `/plan-update` ceremony and proceed directly to implementing the **next pending slice**.

**When to run `/plan-review`:**
- Plan was **just created** in this conversation (no phases `done`).
- User explicitly asks for a review.
- Scope shifted significantly since the last review.

**When to skip it:**
- Plan exists with executed phases — it was reviewed when created; re-reviewing wastes a turn.

## Multi-phase plans (entire migrations, YAML phases, omnibus tracks)

- This skill applies to **large plans** (many phases, e.g. Electron 6–10e) as well as single features. **Do not** refuse those requests by saying the work cannot be done **in one model turn** and stopping with only a scope essay.
- **Turn limits ≠ process limits.** Decompose the authoritative plan, then **execute ordered vertical slices**: implement → verify → commit/push → PR review → next slice.
- **Verify–fix loop:** After substantive changes, run the repo's test targets (`make test-fast`, `make test-electron-e2e`, etc.); fix failures; repeat until green or surface **DECISIONS NEEDED** with a concrete blocker.
- **Handoff (mandatory when context ends before the plan ends):** Write a short **continuation block**: next slice(s), owning files, exact verify commands, open PR `#`, and blockers. The **next conversation** continues the **same** pipeline—no restart from zero.
- **Parallelism:** Use **Task/subagents** for readonly review or exploration while implementing when it speeds verification; reconcile findings before merge.

## Prerequisites

- Git repo with the project's branch and PR conventions.
- **`gh` CLI** when creating PRs or running `gh pr diff` for `/pr-review`.
- Plans and reviews live under **`.cursor/plans/`** and **`.context/`** (or equivalent defined in that repo's commands).
- **Pre-push hooks** that run `python -m pre_commit` require dev dependencies in the project venv (for example `make install-dev` / `uv sync` with the dev group). If `git push` fails with `No module named pre_commit`, install dev deps and retry.

### Gitignored `.cursor/plans/` or `.context/` (e.g. md_data_explorer)

Some repositories **gitignore** `.cursor/plans/` and/or `.context/`. In those clones:

- Treat paths under `.context/` as **local agent scratch** unless the team uses `git add -f` or stores artifacts under a **tracked** root (for example `docs/implementation/`).
- **Do not** claim a plan or review file was **committed** unless `git status` shows it staged or committed.
- **Portable plan truth** may live in tracked docs (for example `docs/implementation/MASTER_PLAN.md`, `docs/specs/*.md`). Prefer updating those in the same PR when the team relies on them instead of only writing ignored paths.

## MCP (optional)

When **serena**, **duckdb**, or **playwright** MCP servers are enabled, follow `~/.cursor/skills/references/mcp-workbench.md` for **md_data_explorer** MCP conventions (versioned in-repo as `.claude/skills-references/mcp-workbench.md`; run `make sync-cursor-skills` after edits). Use **serena** to map symbols and blast radius before large edits; **duckdb** for data or SQL checks tied to the feature; **playwright** for browser verification of UI acceptance criteria.

## Workflow (execute in order)

### 1. Locate or create the plan

- If a plan exists (`.cursor/plans/*.plan.md` or tracked `MASTER_PLAN.md` / omnibus), read its YAML todos. Identify the **next pending phase** whose dependencies are all `done`.
- If **`repo-context`** produced a handoff (e.g. `.context/diagnostics/<initiative>_context.md`), merge its **goal**, **non-goals**, and **risk** into context.
- If no plan exists, create one with phases, todos, success criteria, and TDD notes aligned to the repo's plan-execution rules.

### 2. Plan review (new plans only)

- **Mature plans (any phase `done`):** Skip to step 3.
- **New plans:** Run `/plan-review`, write the artifact to `.context/reviews/plan_<name>.md`, read the verdict.
  - **READY TO EXECUTE:** Proceed to step 3 immediately.
  - **READY WITH CHANGES:** Run `/plan-update`, then `/plan-review` again. **Auto-proceed** as soon as the verdict is READY TO EXECUTE.
  - **NOT READY** with a blocker the agent cannot resolve: Surface **DECISIONS NEEDED** and stop.

### 3. Implement the next slice (spec-driven)

- Implement the next pending phase: **Red → Green → Refactor** using the repo's test and lint entrypoints.
- Do not commit without tests where the repo requires them; do not weaken hooks.
- After implementation, run the repo's verification commands (`make test-fast`, `make test-electron-e2e`, etc.). Fix failures. Re-run until green.
- Commit and push. Update the plan YAML todo status to `done` (or `in_progress` if partially complete).

### 4. PR review (internal gate, auto-proceed)

- **Resolve PR number `N`:** `gh pr list --head "$(git branch --show-current)"`. Reuse an existing PR; create one only when none exists.
- Save the diff: `mkdir -p .context/diffs && gh pr diff <N> > .context/diffs/pr<N>.diff`
- Write the review artifact to `.context/reviews/<N>.md`.
- If **MERGE:** Note it and proceed to the next slice (step 3).
- If **NO MERGE** with fixable issues: Fix them, re-push, re-review. **Auto-proceed.**
- If **NO MERGE** with an unfixable blocker: Surface **DECISIONS NEEDED** and stop.

### 5. Repeat or hand off

- If more pending phases remain and context allows: **loop back to step 3**.
- If context is running out: Write a **handoff block** (next slice, owning files, verify commands, PR `#`, blockers) and stop. The next conversation continues from that block.
- If all phases are `done`: Report completion.

## Chaining summary

```text
Plan (existing or new)
  → [new plan only: /plan-review → auto-proceed or DECISIONS NEEDED]
  → implement next slice → verify → fix → commit/push
  → /pr-review (internal) → auto-proceed or DECISIONS NEEDED
  → [next pending slice: repeat implement → verify → pr-review]
  → … until plan complete, context exhausted (handoff), or genuinely blocked
```

## Repository-specific rules

- Read **AGENTS.md**, **CLAUDE.md**, **CONTRIBUTING**, or **Makefile** help for the active workspace; do not assume Python or Make.
- For **md_data_explorer**, load **mdde-context** (generated under `~/.cursor/skills/` via sync) and follow Makefile + Polars-first rules there.

## Detailed paths

- Load **`references/workflow.md`** in this skill directory for the path table and `gh pr diff` snippet.

## What this skill does not do

- Replace human judgment on scope or HITL safety.
- Bypass failing tests or pre-commit; if blocked, surface **DECISIONS NEEDED** in the repo's required output format.
- Stop with "ACTIONS REQUIRED" lists for the human — the agent resolves each gate and keeps moving.
