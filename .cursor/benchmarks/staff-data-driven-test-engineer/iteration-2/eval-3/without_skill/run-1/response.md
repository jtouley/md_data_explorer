# Immediate reply to leadership

We are treating **full green** as the exit criterion: no partial passes, no “good enough” subsets. I will run the repo’s canonical quality gate, capture a single ordered failure list, and work top-down until `make test` (or the project’s agreed full-suite target) is clean. I will only surface blockers that truly require a product or infra decision; otherwise we keep fixing and re-running until green.

# Cycle 1 structure

1. **Baseline signal** — From repo root: run the full test entrypoint the team uses in CI (here: `make check` or `make test` per Makefile/CI parity). Save stdout/stderr and the **first failing test** identity.
2. **Triage bucket** — Classify the failure (import/env, flaky timing, assertion drift, data fixture, DuckDB/concurrency, etc.) in one line so we do not thrash.
3. **Minimal fix** — Address that failure with the smallest change that restores the intended behavior; add or adjust tests if the spec was wrong.
4. **Verify** — Re-run the **same** command as step 1 (not only the single file) to catch ordering and coupling issues.
5. **Repeat** — If red, go to step 2 with the new first failure. If green, run `make test-fast` / module targets only as a sanity check if CI differs; otherwise stop at full green.

Cycle 1 ends when either the suite is green or we have one concrete next failure queued for cycle 2 with no ambiguous “maybe green” state.
