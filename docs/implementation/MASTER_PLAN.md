---
doc_type: master_plan
status: active
title: Master plan — consolidated roadmap
last_updated: 2026-03-23
---

# Master plan (single consolidated roadmap)

This document **replaces scattered “which plan are we on?” answers**. It inventories every major plan/spec, assigns **one primary execution track**, and lists **parallel / deferred** work without merging incompatible scope.

## What we are implementing (primary track)

**Primary:** **Desktop-first product shell** — **Electron + Vite renderer + local FastAPI**, then **remove Streamlit** from the default path and dependency set.

| Role | Canonical file |
|------|----------------|
| **Spec & parity gate** | [`docs/specs/omnibus_electron_streamlit_cutover.md`](../specs/omnibus_electron_streamlit_cutover.md) |
| **Executable detail (phases, APIs, TDD notes)** | [`.cursor/plans/electron_ui_migration_b421909d.plan.md`](../../.cursor/plans/electron_ui_migration_b421909d.plan.md) |
| **ADR (UI direction)** | [`ADR/ADR010.md`](ADR/ADR010.md) |

**Strategic intent (NL + semantic layer)** remains [`docs/vision/UNIFIED_VISION.md`](../vision/UNIFIED_VISION.md). The **delivery vehicle** for that intent is shifting from Streamlit to **Electron + API** per the omnibus spec.

### Primary track — consolidated phases (high level)

Statuses below mirror the **omnibus phased table** (authoritative for release gating). YAML todos inside the Cursor plan file may lag; **update that YAML when closing a phase**.

| Phase | Outcome | Status (2026-03-23) |
|-------|---------|---------------------|
| 0–4 | Core free of UI imports; `/api/datasets`, `/api/queries` + SSE; enrichment API; Electron skeleton | **Done** (subject to `make test-api` / `make test-fast`) |
| 5 | Chat + dataset selector | **Implemented** (Vite renderer; same URL Electron loads in dev) |
| 6 | Inline results in chat | **Partial** (generic table + SSE; parity rows P20–P24 still **gap**) |
| 6b–6c | Enrichment + patch history in Electron | **Pending** |
| 7 | Session sidebar | **Pending** |
| 8 | Dataset upload in Electron | **Pending** (parity **P01**) |
| 9 | Automated E2E | **Partial** (Playwright **Chromium + Vite**; **Electron harness** blocked by CLI `--remote-debugging-port` issue on macOS — document in omnibus / `electron/playwright.config.js`) |
| 10a–e | Delete Streamlit pages → components → `pyproject` → UI tests → docs | **Pending** — **hard gate** until parity + E2E criteria met |

**Cutover rule (non-negotiable):** Do **not** run Phase 10 until the omnibus **parity matrix** rows required for v1 are **Implemented** or **Won’t do** with sign-off.

---

## Plan registry — `.cursor/plans/` (active files)

| Plan | Purpose vs primary track |
|------|---------------------------|
| `electron_ui_migration_b421909d.plan.md` | **Main executable plan** for Track 1 |
| `ollama_health_electron_banner.plan.md` | **Shipped** slice: `/health` Ollama fields + Electron banner (OLL row) |
| `staff_review_sql_perf_ui_followup.plan.md` | **Parallel** — SQL observability, benchmarks, storage boundaries; **not** a substitute for UI cutover |
| `ask_questions_solid_refactor_a57bf5ed.plan.md` | **Optional / decoupled** — Streamlit `Ask_Questions` refactor; **deprioritize** if Streamlit removal is imminent |
| `adr011_metadata_enrichment_bb84ac23.plan.md` | **Largely complete** (enrichment API + core); Electron UI for ENR rows still open |
| `dry_solid_platform_refactoring_6b1c0a18.plan.md` | **Parallel / broad** — large refactor; avoid conflicting with Streamlit deletion; reconcile before wide UI refactors |
| `profile-first_performance_optimization_20f2917c.plan.md` | **Parallel** — performance program |
| `config-driven_semantic_layer_b71684b5.plan.md` | **Historical / verify** — much of semantic layer is as-built; use [`docs/architecture/AS_BUILT_ARCHITECTURE.md`](../architecture/AS_BUILT_ARCHITECTURE.md) before expanding scope |
| `reusable_visualization_framework_for_correlation_analysis_68d33cb3.plan.md` | **Future** — visualization; ties to parity **P24** etc. |
| `doctor-friendly_macos_dmg_installer_6f0341eb.plan.md` | **Future** — packaging / distribution |
| `chat_ux_fix_plan_cd0d58e8.plan.md` | **Superseded** for long-term UX by Electron track (Streamlit-only fixes) |
| `coverage_improvement.md` | **Parallel** — coverage roadmap; aligns with `.cursor/rules/108-coverage-enforcement.mdc` |

**Archive:** `.cursor/plans/Archive/` — completed or abandoned plans; **do not schedule** from Archive without explicit revival.

---

## Plan registry — `docs/implementation/plans/`

| Document | Purpose vs primary track |
|----------|---------------------------|
| `consolidate-docs-and-implement-question-driven-analysis.md` | **Historical** NL + docs program; NL engine **exists**; later “Phase 3/4” items (schema inference depth, MIMIC-scale multi-table) overlap **deferred** multi-table plan |
| `multi-table_handler_refactor_aggregate-before-join_architecture_b7ca2b5e.plan.md` | **DEFERRED V2** — explicit in frontmatter; not in the desktop cutover critical path |
| `quick-wins.md` | **Opportunistic** fixes (performance, logging, security hardening) — pick items without blocking Track 1 |
| `phase-1-security-fixes.md` | **Security** backlog items — parallel |
| `code-review-2025-12-27.md` | **Snapshot** review — mine for tickets; not a live phase plan |

---

## How to use this document

1. **Starting work:** Confirm it fits **Track 1** (omnibus + electron plan) or is explicitly **parallel/deferred** above.
2. **Status updates:** When a parity row or phase changes, update **[`omnibus_electron_streamlit_cutover.md`](../specs/omnibus_electron_streamlit_cutover.md)** in the **same PR** when possible, then refresh the table in **this file**.
3. **Executable tasks:** Large step-by-step content stays in **`.cursor/plans/electron_ui_migration_b421909d.plan.md`** to avoid duplicating thousands of lines.
4. **Git vs Cursor-only plans:** `.cursor/plans/` may be **gitignored** in some clones; **this file + omnibus** are the **portable** sources of truth for contributors.

---

## Explicit non-merge rules

- **Do not** fold `staff_review_sql_perf_ui_followup` or `dry_solid_platform_refactoring` into the Electron YAML todo list wholesale — they have different success criteria and risk profiles.
- **Do not** resurrect full **multi-table mart refactor** under the desktop cutover PR; keep **V2** boundary from the multi-table plan.
- **Streamlit SOLID refactor** is valuable only if Streamlit survives **months**; otherwise invest in **API + Electron** parity.

---

## Related entry points

| Topic | Document |
|-------|-----------|
| Testing commands | [`tests/AGENTS.md`](../../tests/AGENTS.md) |
| Makefile / uv workflow | [`.claude/CLAUDE.md`](../../.claude/CLAUDE.md) |
| Semantic layer ↔ API | [`docs/architecture/SEMANTIC_LAYER_FASTAPI_ADAPTER.md`](../architecture/SEMANTIC_LAYER_FASTAPI_ADAPTER.md) |
| As-built architecture | [`docs/architecture/AS_BUILT_ARCHITECTURE.md`](../architecture/AS_BUILT_ARCHITECTURE.md) |

---

*Maintainers: when adding a new “major” plan under `.cursor/plans/`, add one row to the registry here or retire an obsolete plan explicitly.*
