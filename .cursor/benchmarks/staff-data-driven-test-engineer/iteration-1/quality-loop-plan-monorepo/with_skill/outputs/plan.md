# Staff plan: 12 failing UI tests (Python monorepo, Makefile + pytest + Streamlit)

This plan follows the **test-quality-loop** workflow: orient → collect signals → parallel expert review → one consolidated report → targeted tests/fixtures → verify → loop or stop. Commands are referenced by **name only** (for example the UI module target); no invocation lines are included here.

---

## Agreed scope, stop criteria, and iteration cap

| Item | Agreement |
|------|-----------|
| **Scope** | All failures attributed to the **UI test target** (canonical: `make test-ui` or project-equivalent). Out of scope unless a failure is proven to be environment-only: unrelated suites, optional Electron work, and production data paths. |
| **Target bar (stop when met)** | UI target is **green** on CI (same configuration as the failing pipeline job): zero failures, zero errors; no new skips without an owner and a dated TODO. |
| **Secondary bar (if primary blocked)** | If after **two** full cycles a subset remains blocked by external dependency, **stop** with a written exception list (each item: blocker, owner, ETA, interim mitigation such as quarantine policy—**not** weakened assertions). |
| **Iteration cap** | **Maximum 3** improvement cycles (discover → report → implement → verify each). After cycle 3, produce a **remaining-risk** section and require explicit re-scoping before further work. |
| **Default if user gave no limit** | Per skill: complete **one** full cycle, then confirm whether to continue; this plan **raises** that default to the cap above for backlog closure. |

---

## Phase 0 — Orient on entrypoints

**Goal:** Map authoritative surfaces so every later step points at the same bars.

1. **Test entrypoints (read-only reconnaissance)**
   - `Makefile`: locate the UI target, how `PYTEST_ARGS` / markers are threaded, and any `test-fast` / `check` interactions.
   - CI workflow(s): identify the job that runs the UI suite, Python version, working directory, env vars, and artifact uploads (logs, junit).
   - `tests/AGENTS.md` (or equivalent): markers, fixture rules, slow/integration policy, Streamlit testing notes.
   - `pyproject.toml` / `pytest.ini`: default paths, filterwarnings, asyncio mode.

2. **Application surfaces**
   - Streamlit: how pages and session state are structured; where UI logic lives vs pure Python (test the latter first when possible).
   - Optional Electron later: note IPC and packaging paths only if failures reference them; do not expand scope preemptively.

3. **Data boundaries**
   - Uploads, temp dirs, DuckDB or file-backed state: list where tests must use `tmp_path` / isolated storage so parallel review does not confuse “product bug” with “shared cwd pollution.”

**Exit checklist for Phase 0:** One short **service and test surface** table (surface → how it is run in CI → notes) suitable for the consolidated report’s “Service and test surface map” section.

---

## Phase 1 — Collect failure fingerprints

**Goal:** Replace “12 failures” with a **deduplicated, ordered** inventory that enables parallel ownership.

1. **Pull canonical failing output**
   - From the failing CI run: job log, and any uploaded junit/xml or summary. Prefer the **same** job definition the team treats as source of truth.

2. **Build a failure fingerprint per failing test**
   For each failure, record at minimum:
   - **Test node id** (file, class, parametrized instance).
   - **Exception type and message** (first meaningful frame).
   - **Stable location**: file path and line from the test or from production code if the assertion is there.
   - **Owning suite / marker** (e.g. UI-only vs shared util).
   - **Flake hypothesis** (yes/no + one line: time, order, network, global state).

3. **Cluster**
   - Group fingerprints into **failure classes** (same root symptom, e.g. “mock of Streamlit API drift”, “semantic layer column rename”, “path assumes repo root”).
   - Order classes by **impact × fix confidence** (fix many tests with one change vs one-off).

4. **Optional signals** (if repo enforces them)
   - Coverage delta, recent commits touching UI modules, open issues tagged `ui` or `test`.

**Exit checklist for Phase 1:** A **failure inventory** table (IDs F1…F12 or fewer after dedupe) ready to drop into the consolidated report; raw artifact references (CI run URL, log path) listed under “Signals.”

---

## Phase 2 — Parallelize expert review

**Goal:** Reduce wall-clock and blind spots by splitting work with **non-overlapping briefs**; merge conflicts with reproduction.

Use **parallel** reviewers (human or agent), each **readonly** until a hypothesis is validated:

| Track | Focus | Deliverable |
|-------|--------|---------------|
| **A — Layout & fixtures** | `tests/ui` (and shared `conftest.py`): markers, duplicate setup, session-scoped state, Rule-of-two violations | Bullet list: risky fixtures, ordering assumptions |
| **B — Root cause** | Map each failure class to owning module(s) under `src/…/ui` and dependencies | Per-class: likely cause, minimal repro path |
| **C — Strategy / gates** | Alignment with `make test-fast`, skip/slow policy, Polars assertions, coverage hooks | Prioritized fix order; what must not be weakened |
| **D — Frontend / Streamlit** | Streamlit runtime boundaries, st.session_state, mocking strategy | Stable testing seams; avoid brittle full-layout assertions |

**Brief contents (each track):** goal, paths/modules, **fingerprints already collected**, and “return bullets only.”

**Merge rule:** If two tracks disagree, **reproduce** with a single agreed command (the UI target) or a **minimal** pytest node list; update the failure table with “confirmed cause.”

---

## Phase 3 — Consolidate one report

**Goal:** One Markdown document is the **system of record** for the cycle.

Use the repo’s standard template (aligned with **health-and-quality-report-template**):

- **Metadata:** branch/commit, scope, iteration number (1 of 3).
- **Facts vs recommendations:** CI outcome, fingerprint table, merged subagent bullets = facts; prioritized next actions = recommendations.
- **Failure inventory:** clustered IDs, status column (`open` / `in progress` / `fixed`).
- **Conflicts resolved:** how each was reproduced.
- **Work done this cycle:** leave empty until Phase 4; fill in Phase 5.

**Audience:** Staff + implementers; scannable in under five minutes via executive summary and failure table.

---

## Phase 4 — Add targeted tests and fixes

**Goal:** One **vertical slice** per cluster where possible: fix + regression guard.

1. **TDD discipline**
   - For behavior gaps: failing test → minimal production change → green.
   - For broken tests after intentional product change: update test to match **documented** behavior; if behavior wrong, fix product and keep assertion strict.

2. **Streamlit / UI specifics**
   - Prefer testing **pure functions** and presenters extracted from widgets; use session/mocking boundaries consistent with existing patterns.
   - Do not weaken assertions or accept partial pass rates; use **skip** only with concrete TODO and owner if truly blocked.

3. **Fixtures**
   - Small, deterministic data; shared setup extracted per project rules; no large production dumps.

4. **E2E (only if warranted)**
   - Keep count small; user-visible assertions; documented seeds/flags. Defer Playwright expansion unless failure class requires it.

**Exit checklist for Phase 4:** Each failure class has either a **merged PR-ready** fix or an explicit **blocked** row in the report (with external dependency called out).

---

## Phase 5 — Verify

**Goal:** Prove the bar against the **same** signals as Phase 1.

1. Re-run the **UI target** locally and confirm CI parity (Python version, env).
2. If the repo gates on **`make test-fast`** or **`make check`**, run those after UI is green to catch cross-module regressions.
3. Capture outcomes in the report: command names, pass/fail, link to CI run.
4. If **new** failures appear, append rows to the failure inventory; do not delete history—mark superseded rows `obsolete` with reason.

**Stop if:** UI target green + no new regressions on agreed secondary commands.

---

## Phase 6 — Loop or stop

| After cycle | Action |
|-------------|--------|
| **Target met** | Close loop; archive report path in PR description; optional coverage baseline update per project policy. |
| **Target not met, cycles < 3** | Narrow to **highest-risk** remaining classes; repeat Phases 1–5 only for those IDs. |
| **Cycles = 3** | **Stop** per iteration cap: deliver “remaining risk” (severity, likelihood, mitigation, suggested scope for cycle 4 if approved). |

---

## Anti-patterns to avoid (explicit)

- Unbounded “fix everything” without the scope and cap above.
- Lowering assertion strength or accepting fractional pass rates.
- Duplicating production datasets or adding flaky sleeps without a readiness contract.
- Splitting findings across chat threads without merging into the **single** report.

---

## Summary checklist (skill ticks)

- [ ] Phase 0: Entrypoints and surface map captured.
- [ ] Phase 1: Fingerprints clustered and ordered.
- [ ] Phase 2: Parallel reviews merged with reproduction.
- [ ] Phase 3: One consolidated report drafted/updated.
- [ ] Phase 4: Targeted tests/fixes per cluster.
- [ ] Phase 5: Verified against agreed commands.
- [ ] Phase 6: Stopped per criteria or began next cycle within cap.
