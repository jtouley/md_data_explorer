# Staff plan: triage and clear 12 failing UI tests (Python monorepo)

**Context:** Python monorepo with `Makefile`, `pytest`, Streamlit UI (optional Electron later). CI reports **12 failures** in the **UI test target** (e.g. `make test-ui`). This document is a written playbook only—no shell snippets.

---

## 1. Orient on entrypoints (first 30–60 minutes)

**Goal:** Know *where* UI tests live, *how* CI invokes them, and *what* “UI” means in this repo.

1. **Makefile contract**
   - Locate the `test-ui` (or equivalent) target and note: pytest path roots, `PYTEST_ARGS` passthrough, markers (`-m`), env vars, and whether coverage or xdist changes behavior vs CI.

2. **Pytest configuration**
   - Read `pyproject.toml` / `pytest.ini` for: `testpaths`, `python_files`, `addopts`, markers, asyncio/streamlit plugins, and any `conftest.py` hierarchy under `tests/ui` (or the path wired by `test-ui`).

3. **CI wiring**
   - Open the workflow job that runs the UI target. Record: Python version, dependency install path, working directory, exact `make` invocation, artifact retention (logs, junit), and whether failures are flaky-reported.

4. **Streamlit boundary**
   - Skim `tests/ui` layout: unit vs integration, use of `AppTest` / mocks vs full app, and shared fixtures in `tests/conftest.py`. Note anything that requires display, ports, or timing.

**Stop criterion for this phase:** You can answer, without guessing: (a) which directories and markers define “UI tests,” (b) how CI differs from local `make test-ui` if at all, and (c) whether failures are likely import/config vs behavioral.

---

## 2. Collect failure fingerprints (single structured pass)

**Goal:** Replace 12 opaque failures with a **deduplicated failure taxonomy** you can assign and parallelize.

For each failing test (from CI log or local `make test-ui` output when available), capture a **fingerprint record**:

| Field | Purpose |
|--------|--------|
| `nodeid` | Stable pytest identity |
| `error_class` | e.g. `AssertionError`, `FixtureLookupError`, `StreamlitAPIException` |
| `top_frame` | First project frame (file + function) |
| `one_line_signature` | Normalized message (strip paths, ids, timestamps) |
| `layer` | `fixture` / `mock` / `streamlit_runtime` / `data` / `async` / `unknown` |
| `suspected_owner` | Subsystem (e.g. NL query widget, session state, file upload) |

**Clustering rule:** Same `error_class` + same `top_frame` + same normalized signature ⇒ one **cluster**. Aim for **K clusters ≤ 12** (often much smaller).

**Artifacts to produce (conceptual):**

- A table: cluster id → member nodeids → hypothesized root cause in one sentence.
- A **priority order**: clusters blocking others (import/fixture/config) before assertion drift.

**Stop criterion:** Every failing nodeid maps to exactly one cluster; no orphan “misc” bucket without a named next investigative step.

---

## 3. Parallelize expert review (workstreams)

**Goal:** Minimize serial context-switching; maximize independent deep dives.

**Sizing:** With **12 failures**, use **2–3 parallel workstreams** (not 12).

**Assignment heuristic:**

| Workstream | Typical clusters |
|------------|------------------|
| **A — Harness & fixtures** | Missing fixtures, bad `conftest` scope, env, markers, import errors |
| **B — Streamlit / UI behavior** | `AppTest`, session state, widget keys, reruns, timing |
| **C — Data & backends** | DuckDB, semantic layer mocks, Polars schema drift in UI paths |

Each owner:

1. Owns 2–4 clusters end-to-end (repro → root cause → fix or test change).
2. Posts **one short update per cluster**: root cause, fix strategy, risk to other modules.
3. Escalates **cross-cutting** issues (e.g. shared fixture wrong for all UI tests) to a single **integrator** role.

**Stop criterion per workstream:** For every assigned cluster, either (a) merged fix + test adjustment with rationale, or (b) explicit “blocked on X” with owner and ETA.

---

## 4. Consolidate one report (integration gate)

**Goal:** One readable artifact for leads/CI—not 12 chat threads.

**Report sections (fixed outline):**

1. **Executive summary** — K clusters, 1–2 sentences each, green/red for CI UI target.
2. **Root causes** — Bullet per cluster; link to PR/commits by hash when applicable.
3. **Test changes** — New tests vs tightened assertions vs rewrites; call out any skip/xfail **only** with ticket + removal criteria (avoid silent weakening).
4. **Risks & follow-ups** — Flake sources, Streamlit upgrade sensitivity, Electron prep notes.
5. **Verification summary** — See §6; paste pass/fail matrix (UI target + spot checks).

**Owner:** Integrator (not parallelized). **Deadline:** Same sprint slice as fixes, before declaring “done.”

**Stop criterion:** Report is complete, internally consistent (nodeids match pytest), and lists no cluster as “fixed” without verification evidence.

---

## 5. Add targeted tests (surgical, not a rewrite)

**Goal:** Each fix is **defended by a test** that would fail if the regression returns.

**Rules:**

- Prefer **one focused test per distinct bug class**; avoid duplicating the same assertion across files (respect project fixture discipline).
- If the failure was **missing coverage** on a branch, add the smallest test that hits that branch without booting full Streamlit unless the Makefile already allows it.
- If the failure was **brittle UI coupling** (selector/key churn), stabilize via stable `key=` / helper, then assert behavior not layout trivia.
- Align with project norms: Polars assertions where DataFrames are compared; no broad exception swallowing.

**Stop criterion:** For each cluster, at least one of: (a) existing test updated with clear intent, or (b) new test with name following `test_<unit>_<scenario>_<expectedBehavior>`; no cluster closed on “fix only” without test delta unless explicitly documented as infra-only with alternate guard (e.g. CI config)—those should be rare.

---

## 6. Verify (definition of done)

**Minimum bar:**

1. **UI target green:** `make test-ui` (or CI-equivalent args) passes **10/10** consecutive runs locally or on CI for the merge candidate branch—or document **one** allowed flake with reproduction ticket (not multiple).
2. **Regression slice:** If changes touched shared fixtures or `conftest`, run the **next broader** target defined in the Makefile (e.g. `make test-fast` or full `make test`) per team policy—record result in the consolidated report.
3. **Lint/type gates:** Whatever the repo requires before merge (stated in report as pass/skip-with-reason).

**Stop criteria (“done”):**

- CI UI job green on the PR.
- Consolidated report merged or attached to the PR description.
- No open cluster without owner or without a tracked follow-up ticket.

---

## 7. Iteration cap and when to stop spinning

**Iteration cap:** **3 full triage cycles** per release slice.

- **Cycle 1:** Fingerprints + parallel fix + verify.
- **Cycle 2:** Only **new** failures or **same cluster recurring** (flake)—add instrumentation or stabilization; merge only with flake budget = **zero** or **one** documented issue.
- **Cycle 3:** Escalation: time-boxed **staff review** (architecture/session-state boundaries, split tests, or quarantine **with** explicit removal plan—**not** permanent silent skips).

**Hard stops (do not exceed without explicit leadership decision):**

- **More than 3 cycles** on the same PR for the same 12 tests → stop feature work; run a **blameless postmortem** on test design and CI parity.
- **Widening scope** (e.g. “rewrite all UI tests”) → out of scope for this plan; spawn a separate initiative.

---

## 8. Timeboxing (order-of-magnitude)

| Phase | Budget |
|--------|--------|
| Entrypoint orientation | 0.5–1 h |
| Fingerprints + clustering | 1–2 h |
| Parallel fix (2–3 streams) | 0.5–2 d wall-clock (depends on depth) |
| Consolidated report + verify | 2–4 h |

Adjust for monorepo size; Streamlit integration tests dominate the upper bound.

---

## Summary

Treat **12 UI failures** as **K small clusters**, assign **2–3 parallel owners**, merge fixes with **targeted tests**, integrate into **one report**, and verify with **`make test-ui` green + agreed broader slice**. **Stop** when CI is green, the report is complete, and you have not exceeded **three triage cycles** without escalation.
