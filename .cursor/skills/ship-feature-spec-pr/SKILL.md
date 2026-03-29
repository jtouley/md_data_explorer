---
name: ship-feature-spec-pr
description: "Turns a chat-defined feature or multi-phase plan into a reviewed PR by chaining plan authoring, /plan-review, /plan-update, /spec-driven implementation, and /pr-review—repeating per plan slice until done or blocked. Use when the user asks to ship from a conversation, run the full pipeline, ship all phases, or chain plan review with PR review after implementation."
---

# Ship feature: conversation → plan → spec-driven → PR review

**Canonical copy:** `~/.cursor/skills/ship-feature-spec-pr/` (global). Project mirrors (e.g. md_data_explorer `.cursor/skills/`) should stay aligned; update global first when changing this file.

## Purpose

Provide a single procedural workflow so another agent (or human) can move from **conversation** to **reviewed PR** without skipping gates: written plan, plan review, plan updates if needed, spec-driven implementation, and PR diff review.

## When to use this skill

- User wants to **ship a feature** from the current or recent thread.
- User asks to **chain** `/plan-review`, `/spec-driven`, and `/pr-review`.
- User says **"run the full pipeline"**, **"global setup"**, **"from this chat to PR"**, **"all phases"**, or **"ship the whole plan"**.

## Multi-phase plans (entire migrations, YAML phases, omnibus tracks)

- This skill applies to **large plans** (many phases, e.g. Electron 6–10e) as well as single features. **Do not** refuse those requests by saying the work cannot be done **in one model turn** and stopping with only a scope essay.
- **Turn limits ≠ process limits.** The correct response is to **decompose** the authoritative plan (todos in `.cursor/plans/*.plan.md`, or **tracked** `MASTER_PLAN.md` / omnibus when plans are gitignored), then **execute ordered vertical slices**: implement → verify with the repo’s **Makefile / CI** commands → commit/push when appropriate → **PR review** as needed.
- **Verify–fix loop:** After substantive changes, run **`staff-data-driven-test-engineer`** (or the repo’s agreed test targets) on the **same bar** (`make test-fast`, `make test-electron-e2e`, etc.); fix failures; repeat until green or surface **DECISIONS NEEDED** with a concrete blocker.
- **Handoff (mandatory when context ends before the plan ends):** Write a short **continuation block**: next slice(s), owning files, exact verify commands, open PR `#`, and blockers. The **next conversation** continues the **same** pipeline—no restart from zero.
- **Parallelism:** Use **Task/subagents** for readonly review or exploration while implementing when it speeds verification; reconcile findings before merge.

## Prerequisites

- Git repo with the project’s branch and PR conventions.
- **`gh` CLI** when creating PRs or running `gh pr diff` for `/pr-review`.
- Plans and reviews live under **`.cursor/plans/`** and **`.context/`** (or equivalent defined in that repo’s commands).
- **Pre-push hooks** that run `python -m pre_commit` require dev dependencies in the project venv (for example `make install-dev` / `uv sync` with the dev group). If `git push` fails with `No module named pre_commit`, install dev deps and retry.

### Gitignored `.cursor/plans/` or `.context/` (e.g. md_data_explorer)

Some repositories **gitignore** `.cursor/plans/` and/or `.context/`. In those clones:

- Treat paths under `.context/` as **local agent scratch** unless the team uses `git add -f` or stores artifacts under a **tracked** root (for example `docs/implementation/`).
- **Do not** claim a plan or review file was **committed** unless `git status` shows it staged or committed.
- **Portable plan truth** may live in tracked docs (for example `docs/implementation/MASTER_PLAN.md`, `docs/specs/*.md`). Prefer updating those in the same PR when the team relies on them instead of only writing ignored paths.

## MCP (optional)

When **serena**, **duckdb**, or **playwright** MCP servers are enabled, follow `~/.cursor/skills/references/mcp-workbench.md` for **md_data_explorer** MCP conventions (versioned in-repo as `.claude/skills-references/mcp-workbench.md`; run `make sync-cursor-skills` after edits). Use **serena** to map symbols and blast radius before large edits; **duckdb** for data or SQL checks tied to the feature; **playwright** for browser verification of UI acceptance criteria.

## Workflow (execute in order)

### 1. Extract feature spec from conversation

- If **`staff-repo-context-mcp`** produced a handoff (e.g. `.context/diagnostics/<initiative>_context.md`), treat it as the primary **“why”** and **risk** input alongside the chat; merge into **goal**, **non-goals**, and **acceptance criteria** before drafting or updating the plan.
- Capture **goal**, **non-goals**, **acceptance criteria**, likely **files/modules**, and **test commands** for this repository.
- If no plan exists, create **`.cursor/plans/<name>.plan.md`** with phases, todos, success criteria, and TDD notes aligned to **this repo’s** plan-execution rules (if any).
- If a plan exists, record its identifier for `/plan-review`.

### 2. Plan review gate

- Run **`/plan-review <plan-identifier>`**.
- Read **`.context/reviews/plan_<name>.md`** and the chat verdict.
- If **NOT READY** or **READY WITH CHANGES**: run **`/plan-update <plan-identifier>`**, then **`/plan-review`** again until the team’s execution threshold is met (e.g. **READY TO EXECUTE**).

### 3. Implementation (spec-driven)

- Run **`/spec-driven`** with the plan path or task title.
- Follow **Red → Green → Refactor** using **this project’s** test and lint entrypoints (Makefile, package scripts, etc.).
- Do not commit without tests where the repo requires them; do not weaken hooks.

### 4. PR review gate

- **Resolve PR number `N`:** Run `gh pr list --head "$(git branch --show-current)" --json number,url` (or inspect the remote). If a PR already exists for the current branch, **use that `N`**; run `gh pr create` only when none exists.
- Ensure **`.context/diffs/pr<N>.diff`** exists as **optional local input** for `/pr-review`; if not:
  - `mkdir -p .context/diffs && gh pr diff <N> > .context/diffs/pr<N>.diff`
- If `.context/` is gitignored, the diff file remains **local**; that is acceptable for review tooling.
- Run **`/pr-review PR<N>`**.
- Resolve **NO MERGE** items and re-review as needed.

### 5. Optional checkpoint

- For long tasks, write **`.context/checkpoints/<task_id>.md`** if the repo’s spec-driven protocol defines it.

## Chaining summary

```text
Conversation → Plan file → /plan-review → [ /plan-update → /plan-review ]*
  → /spec-driven → (push + existing or new PR #N) → gh pr diff → /pr-review PRN
  → verify (Makefile / CI; optional staff-data-driven-test-engineer) → fix → re-verify
  → [ next plan slice: repeat from /spec-driven or from plan review if scope shifted ]
  → … until plan complete, user stops, or DECISIONS NEEDED
```

## Repository-specific rules

- Read **AGENTS.md**, **CLAUDE.md**, **CONTRIBUTING**, or **Makefile** help for the active workspace; do not assume Python or Make.
- For **md_data_explorer**, load **volt-mdde-context** (project `.cursor/skills/`) and follow Makefile + Polars-first rules there.

## Detailed paths

- Load **`references/workflow.md`** in this skill directory for the path table and `gh pr diff` snippet.

## What this skill does not do

- Replace human judgment on scope or HITL safety.
- Bypass failing tests or pre-commit; if blocked, surface **DECISIONS NEEDED** in the repo’s required output format.
