# Reply to leadership (immediate)

**We are executing toward a fully green repo.** I will not block on scope negotiation: work proceeds in tight loops until canonical gates pass. **Source of truth for “green”** is this project’s **Makefile and CI targets** (e.g. `make test`, `make test-fast`, `make check`) per `tests/AGENTS.md`—not ad-hoc `pytest` invocations for merge-level verification. I will **not** weaken assertions or accept partial pass rates; failures get root-cause fixes or explicit skips with concrete TODOs only if truly unavoidable.

**First deliverable (end of Cycle 1):** one consolidated **health/quality report** (facts vs recommendations vs actions taken) with failure fingerprints, owning suites, and the exact commands run.

---

# Cycle 1 — structure

Aligned with the staff data-driven test engineer loop (orient → signals → parallel review → report → build → verify):

| Step | What |
|------|------|
| **0 — Orient** | Read `Makefile`, CI workflows, and `tests/AGENTS.md`; list runnable surfaces (`make test-*`, E2E if any) and data/env boundaries. |
| **1 — Collect signals** | Run the **broadest agreed canonical command** that matches leadership’s bar (full suite: `make test` / `make check` unless the Makefile documents a stricter “release” target). Capture raw output, counts, and **failure fingerprints** (error text, file:line, suite). |
| **2 — Parallel review** | Fan out readonly passes: test layout/fixtures, failure root-cause clusters, alignment with markers and fast vs slow suites. Reconcile conflicts by reproducing with a **single** Makefile-driven command. |
| **3 — Report** | One markdown report: failure table, flake suspects, coverage notes if enforced, and **prioritized fix order** (highest blast radius / unblockers first). |
| **4 — Build (start)** | Pick **one vertical slice** or failure class from the report; TDD where new behavior is needed; respect Polars/assertion rules. |
| **5 — Verify** | Re-run the **same** Makefile targets referenced in the report; append any new failures to the table. |
| **6 — Handoff** | Cycle 2 continues down the priority list until all targets in step 1 are green. |

**Note:** Leadership asked not to pause for scope; operationally that means **Cycle 1 still defines the measurable bar in writing** (exact commands + pass/fail snapshot) so the loop stays auditable and we avoid drift between “local green” and CI.
