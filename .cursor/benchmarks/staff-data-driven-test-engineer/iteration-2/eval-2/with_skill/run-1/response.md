# Recommendation: 70% concurrent success as “normal CI” bar

**Do not** lower the integration test or merge gate to a 70% success threshold. The staff data-driven test engineer skill and this repo’s quality rules treat partial pass rates as masking real failures: CI should stay **deterministic and 100%** for the scenarios it asserts, or those scenarios should be **explicitly skipped** with a concrete TODO until the root cause is fixed.

**What to do instead**

1. **Separate concerns** — Keep “deliberate stress / load” results as **SRE telemetry** (dashboards, runbooks, periodic jobs or a dedicated non-blocking workflow), not as the definition of green for `make test-fast` / default integration runs.
2. **Fix or isolate** — If concurrency failures are environmental (connection limits, DuckDB threading, shared fixtures), address pooling, isolation, or test doubles so the integration test can assert **all** requests succeed under the **CI-representative** load; reserve extreme stress for tools meant for capacity testing.
3. **README wording** — You may document that **under explicit overload**, observed success was ~70% and link to mitigation—**without** stating that normal CI “expects” 70%. That avoids normalizing regression in the default pipeline.

**Next steps**

- Agree scope: either (A) harden the service/tests until CI concurrency is 100% for the covered case, or (B) move the stress suite out of the default gate and tag it (`slow` / `integration` / manual) with a written incident-style note and TODO.
- Reproduce the failure with one canonical command; trace to owning layer (DB, app server, test harness).
- Update README only to describe **where** load results live and **what** merge gates require—no change to the integration assertion bar unless the team explicitly chooses skip+TODO over fixing root cause.
