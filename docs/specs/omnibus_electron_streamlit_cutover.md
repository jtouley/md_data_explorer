---
doc_type: specification
domain: desktop_ui
status: active
title: Omnibus spec — Electron + FastAPI cutover (remove Streamlit)
related_plan: .cursor/plans/electron_ui_migration_b421909d.plan.md
tags:
  - electron
  - fastapi
  - streamlit-removal
  - parity-matrix
last_updated: 2026-03-23
---

# Omnibus specification: Electron desktop UI and full Streamlit removal

This document is the **single entry point** for the migration from **Streamlit** to **Electron + FastAPI** as the only supported interactive UI path. It consolidates intent, parity requirements, cutover order, and verification commands. **Executable phase detail** remains in the Cursor plan file cited below; this spec does not duplicate every task line-by-line.

## Authoritative references (read these first)

| Document | Role |
|----------|------|
| [`.cursor/plans/electron_ui_migration_b421909d.plan.md`](../../.cursor/plans/electron_ui_migration_b421909d.plan.md) | Phases, YAML todos, TDD gates, API sketches, Phase 10 deletion order |
| [`docs/architecture/SEMANTIC_LAYER_FASTAPI_ADAPTER.md`](../architecture/SEMANTIC_LAYER_FASTAPI_ADAPTER.md) | Semantic layer ↔ API boundary |
| [`docs/architecture/LIGHTWEIGHT_UI_ARCHITECTURE.md`](../architecture/LIGHTWEIGHT_UI_ARCHITECTURE.md) | Earlier “lightweight UI” design (Next.js); **desktop target in this repo is Electron** — use for component naming ideas, not as stack-of-record |
| [`docs/implementation/ADR/ADR010.md`](../implementation/ADR/ADR010.md) | Streamlit → lightweight UI migration ADR |
| [`.claude/CLAUDE.md`](../../.claude/CLAUDE.md) | Python workflow, Polars-first, quality expectations |
| [`tests/AGENTS.md`](../../tests/AGENTS.md) | Fixtures, `make test-*`, `assert_frame_equal` |

## Goals and non-goals

### Goals

1. **Primary UX**: Users run a **desktop app** (Electron) that talks to a **local FastAPI** process; no Streamlit dependency in production paths.
2. **Remove Streamlit completely** from the repository: dependency, entrypoints, pages, Streamlit-specific tests, and docs that instruct `streamlit run` / `make run` as the main app.
3. **Preserve clinical workflows** that exist today in Streamlit **or** explicitly deprecate them with product sign-off (recorded in the parity matrix below).

### Non-goals (unless added by product)

- Hosting a public multi-tenant SaaS UI (auth, billing) — out of scope for this cutover spec.
- Rewriting the **semantic layer** or **Polars/DuckDB** analytics core — reuse via API services.

## Current vs target architecture

### Today (dual stack)

- **Streamlit**: `src/clinical_analytics/ui/` (`app.py`, `pages/`, `components/`), `make run` / `run-app*`, `streamlit` in `pyproject.toml`.
- **Electron**: `electron/` — `main.js` loads Vite renderer; `preload.js` uses `fetch` + SSE to **FastAPI** (e.g. `http://localhost:8000`). **Does not embed Streamlit.**

### Target (single stack)

- **Electron** = only shipped interactive UI shell.
- **FastAPI** = HTTP + SSE for datasets, queries, sessions, enrichments (existing routes under `src/clinical_analytics/api/routes/`).
- **Python package** = analytics engine + API; **no** `streamlit` in dependencies.

```text
Electron (main → preload → renderer)
        │  HTTP / SSE
        ▼
FastAPI (routes: datasets, queries, sessions, enrichments)
        │
        ▼
core/ (SemanticLayer, NLQueryEngine, QueryService, …)
```

## Parity matrix: Streamlit → API → Electron

Use this table to **block Streamlit deletion** until each row needed for release is **Implemented** in API + Electron (or **Won’t do** with owner sign-off).

Legend: **S** = Streamlit today | **A** = FastAPI | **E** = Electron renderer

| ID | Streamlit surface (path or feature) | A: endpoint / contract | E: UI | Status |
|----|-------------------------------------|-------------------------|-------|--------|
| P01 | `pages/01_📤_Add_Your_Data.py` — upload, preview, mapping | Extend `datasets` (or dedicated upload routes) per plan Phase 8 | Upload wizard (plan Phase 8) | **Gap** |
| P02 | `pages/02_📊_Your_Dataset.py` — cohort / dataset view | `GET /api/datasets`, preview, detail | Dataset summary + preview | **Partial** |
| P03 | `pages/03_💬_Ask_Questions.py` — NL chat, trust, clarifying | `POST /api/queries`, SSE stream, sessions | Chat + dataset selector (plan Phases 5–7) | **Partial** |
| P20 | `pages/20_📊_Descriptive_Stats.py` | Same query pipeline; typed result payload | Result renderer — descriptive | **Gap** |
| P21 | `pages/21_📈_Compare_Groups.py` | Same | Result renderer — comparison | **Gap** |
| P22 | `pages/22_🎯_Risk_Factors.py` | Same | Result renderer — risk | **Gap** |
| P23 | `pages/23_⏱️_Survival_Analysis.py` | Same | Result renderer — survival | **Gap** |
| P24 | `pages/24_🔗_Correlations.py` | Same | Result renderer — correlations | **Gap** |
| ENR | Enrichment panel / patch history (ADR011 components) | `routes/enrichments.py` | Electron panels (plan Phases 6b–6c) | **Partial** |
| OLL | Ollama / LLM availability feedback | `GET /health` includes `ollama_*` fields (fast probe) | Banner in Electron renderer | **Implemented** |

**Rule:** Do not execute **Phase 10** (delete Streamlit) until every row marked **Required for v1** is **Implemented** or **Won’t do** (explicit).

## Phased delivery (aligned with Cursor plan)

The numeric phases below match **[`electron_ui_migration_b421909d.plan.md`](../../.cursor/plans/electron_ui_migration_b421909d.plan.md)** YAML todos. Status reflects that plan’s `done` / `pending` as of the last plan edit in-repo.

| Phase | Name | Outcome |
|-------|------|---------|
| 0–4 | Core cleanup, API datasets/queries/SSE, Electron skeleton | Largely **done** per plan |
| 5 | Chat interface + dataset selector | **Implemented** (Vite renderer; matches Electron dev URL) |
| 6 | Inline result rendering in chat | **Partial** (tables + SSE path; rich intent renderers still gap per parity matrix) |
| 6b–6c | Enrichment + patch history in Electron | **Pending** |
| 7 | Session sidebar | **Pending** |
| 8 | Dataset upload in Electron | **Pending** |
| 9 | Playwright E2E (Chromium + Vite renderer; Electron shell harness deferred) | **Partial** |
| 10a–e | **Cutover**: delete Streamlit pages → components → `pyproject` → UI tests → final commit | **Pending** (hard gate) |

## Cutover: removing “all traces of Streamlit”

Execute **only after** parity matrix + Phase 9 E2E are satisfied.

### 10a — Pages and app entry

Remove or archive:

- `src/clinical_analytics/ui/app.py`
- `src/clinical_analytics/ui/pages/*.py` (all numbered pages)

### 10b — Streamlit-tied components

Remove modules that **`import streamlit`** or exist solely to support Streamlit widgets. **Exception:** Pure Python modules still required by FastAPI must be **moved** under `src/clinical_analytics/` outside `ui/` (e.g. a `presentation` or `services` package) and imports updated — do not delete logic still used by API.

**Known Streamlit imports today** (grep-driven inventory; re-verify before delete):

- `ui/components/question_engine.py`, `clarifying_ui.py`, `renderers.py`, `trust_ui.py`, `variable_mapper.py`, `result_interpreter.py`, `analysis_wizard.py`, `dataset_loader.py`
- `ui/helpers.py`, `ui/pages/*`, `ui/app.py`

### 10c — Dependencies and entrypoints

- Remove `streamlit` from `pyproject.toml`.
- Replace `Makefile` targets `run`, `STREAMLIT`, `run-app*` with documented **FastAPI** start (see Verification) + **Electron** dev instructions (optional future `make run-desktop` if added to the Makefile).
- Update `scripts/run_app.sh` and any CI that invokes Streamlit.

### 10d — Tests

- Remove or rewrite `tests/ui/` and unit tests that mock `streamlit` / full page flows.
- Keep **API tests** (`tests/api/`) and **core tests**; add Electron E2E coverage per plan Phase 9.

### 10e — Documentation

- Update `README.md`, `docs/getting-started/installation.md`, `docs/development/setup.md`, `docs/development/testing.md`, `tests/AGENTS.md` — **no** primary path referencing Streamlit.
- Add “Desktop (Electron)” as the default quick start.

## Verification commands

Verify the **target** stack:

```bash
# Backend
make run-api
# equivalent: uv run uvicorn clinical_analytics.api.main:app --reload --host 127.0.0.1 --port 8000

make test-api
make test-fast
make check
```

```bash
# Electron (from repo root)
cd electron && npm install && npm run dev
cd electron && npm test
cd electron && npm run test:e2e   # requires API running; see electron/package.json
```

## Acceptance criteria: “Streamlit fully removed”

1. **`grep -r "streamlit" pyproject.toml src/`** returns **no** production dependency and **no** `import streamlit` under `src/` (allowlist only if documented test stubs — prefer zero).
2. **Default user-facing runbook** describes **API + Electron** only.
3. **CI** does not start Streamlit; optional nightly can run Electron E2E against API.
4. **Parity matrix**: all **Required** rows implemented or **Won’t do** with sign-off.
5. **`make check`** (or project-agreed gate) passes on the branch that deletes Streamlit.

## Risks and rollback

| Risk | Mitigation |
|------|------------|
| Feature loss when deleting pages | Parity matrix + E2E before Phase 10 |
| Hidden `streamlit` imports in tests | Grep + `tests/` cleanup in 10d |
| Docs drift | Same PR as 10e or immediately follow-up PR |
| Rollback | Plan file recommends git revert; keep cutover in **dedicated commits** (10a–10e) |

## Maintenance

- When API contracts change, update **this spec’s parity matrix** and the **Cursor plan** in the same change set when possible.
- Prefer linking to **Makefile** targets and real paths; do not document commands that are not implemented.

## Related internal plans (do not merge scope blindly)

- [`staff_review_sql_perf_ui_followup.plan.md`](../../.cursor/plans/staff_review_sql_perf_ui_followup.plan.md) — SQL observability / benchmarks (orthogonal to UI cutover).
- [`ask_questions_solid_refactor_a57bf5ed.plan.md`](../../.cursor/plans/ask_questions_solid_refactor_a57bf5ed.plan.md) — Streamlit refactor; **decoupled** from Electron plan per migration plan notes.

---

*This file is a **specification** and checklist, not a post-incident solution doc. For compound-docs style “solved problem” captures after cutover is complete, add a short entry under `docs/solutions/` (if that directory is adopted) pointing back to this spec and the merge commit.*
