# Staff recommendation: standardizing on bare pytest for UI-only merges

## Executive summary

**Do not** adopt “skip Make entirely” and **do not** treat a quiet, directory-scoped pytest invocation as equivalent to this repository’s canonical UI test entrypoint. The proposal optimizes *local keystrokes* at the cost of **verification drift**—the exact failure mode called out in our test-engineering playbook: local green, CI or teammates red, and PR templates that lie about what “done” means.

**Do** allow faster *local* loops with narrow pytest, **if** the team documents the equivalence rules (markers, parallelism, env) and **merge gates** stay aligned with CI and the Makefile’s contract.

## Why the proposed shortcut is not a drop-in replacement

In this codebase, the maintained UI target is not “everything under `tests/ui` with default pytest settings.” It wires in **venv discipline**, **verbosity**, **xdist parallelism**, and a **marker filter** so serial tests are not run in the parallel job. A minimal quiet run over the UI directory can:

- Execute tests meant to run **serially**, changing timing and shared-state behavior (flakes or false greens).
- Run **sequentially** on a large core machine, *increasing* wall time versus the parallel target—so “ship faster” is not guaranteed.
- Hide failures behind **quiet** output, slowing diagnosis and encouraging “rerun until green” culture.

So standardizing on that specific invocation as *the* team standard **widens** the gap between “what we say we ran” and “what the pipeline enforces,” which is an anti-pattern for staff-level quality.

## Staff-level recommendation

1. **Keep one source of truth for merge readiness**
   CI and the Makefile should remain the **authoritative** definition of green unless you intentionally change CI to match a new contract. If the release captain wants a simpler mental model, **rename or alias** in documentation (e.g. “UI suite = Makefile UI target”) rather than deleting it from the PR template.

2. **If the goal is developer speed, separate “inner loop” from “merge bar”**
   - **Inner loop:** narrow, fast feedback is fine (single file, single test, red-phase debugging)—already allowed by project norms when scoped.
   - **Merge bar:** must match **CI**: same markers, same parallelism strategy, same coverage or quality hooks the pipeline applies. If PR template text is noisy, shorten it to **“matches CI job(s): …”** with links to workflow names—not to a one-off pytest line.

3. **Do not remove Makefile-oriented guidance from the PR template without replacing parity**
   Omitting the canonical UI target from the template **does not** remove the Makefile from the repo or from CI; it only makes it easier to merge without running what automation actually runs. Prefer a **single line**: “Green on the same targets as `.github/workflows` for this change.”

4. **If you truly want one command for humans, make it *equivalent***
   Any “standard command” must encode: correct environment (uv/venv), **not-serial** (or explicit serial pass where required), and **xdist** if that is what CI uses for UI. A doc-only switch to “quiet UI-only pytest” without those knobs is a **process regression**, not a simplification.

## What I would still run before merge (written plan)

Order and scope should follow **what the diff touched**, but the **minimum** staff bar before merging a UI-leaning change is:

1. **Static quality gates** that CI (or pre-commit) enforces: formatter check, linter, type checker—so you do not burn cycle time on CI for avoidable noise.
2. **The same automated tests CI will run for that change class**—at least the UI module suite **with the same selection rules as CI** (including marker and parallelism semantics), not a looser subset.
3. **Cross-module regression when boundaries move**—if the change touches shared fixtures, session state, DuckDB usage, or imports used outside `tests/ui`, expand verification to the **fast full-repo** slice CI uses for PRs (e.g. excluding slow/integration markers if that is the agreed PR gate).
4. **Coverage policy**—if the repository enforces a threshold or regression check on push, satisfy that **before** merge; do not rely on a UI-only run that never exercises coverage configuration.
5. **Manual smoke** only where automation is known weak (e.g. Streamlit interaction nuances)—keep it **short and scripted** (checklist), not a substitute for (2).

## PR template wording (directional, no new canonical shell line)

Replace “mention this Makefile target” with **parity language**: contributors must attest that **local verification matches the CI workflow** for their change (UI-only vs full fast suite). Link to CI; optionally list **workflow job names** rather than duplicating evolving pytest flags in the template.

## Bottom line

**Reject** “skip Make entirely + standardize on quiet UI-only pytest” as the **team merge standard**. **Accept** faster local workflows **only** as a documented subset, with **merge gates** unchanged or **CI updated** to match. Otherwise you trade a small local convenience for **predictable integration pain**—the opposite of shipping faster.
