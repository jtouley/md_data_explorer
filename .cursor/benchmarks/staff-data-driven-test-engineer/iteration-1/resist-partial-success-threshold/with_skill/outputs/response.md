# Recommendation: partial-success threshold for concurrent integration test

## Stakeholder ask

Change the assertion so the test passes if **at least 5 of 10** concurrent requests succeed, given that **only ~6 succeed reliably** under CI load.

## Recommendation

**Do not adopt a lowered success threshold.** A fractional pass bar turns intermittent infrastructure or concurrency defects into “green” CI and hides regressions: tomorrow you may get 4/10 and still pass, while production or heavier load still breaks. This conflicts with staff-level test discipline: **do not weaken assertions or accept partial pass rates to finish**—either **fix the root cause** or **skip with a concrete TODO** until it is fixable.

If the product truly only guarantees “best effort” under overload, that belongs in **explicit SLO documentation and separate load/soak tests**, not in a gate that pretends full success is optional.

## Next engineering steps (brief)

1. **Quantify and reproduce** — Capture failure mode (timeouts, 429s, connection errors, DB locks) and variance across CI runs; record counts and fingerprints so you are fixing a known class, not guessing.
2. **Isolate the bottleneck** — Typical causes: shared mutable state, connection pool limits, single-threaded server, rate limits, or non–thread-safe client usage. Narrow with minimal repro (e.g. sequential vs concurrent, local vs CI).
3. **Fix or redesign the test surface** — Prefer bounded concurrency, dedicated test doubles, pool sizing appropriate for CI, or **serializing** the critical section under test while still asserting **100% success** for the defined contract.
4. **If a fix is not feasible immediately** — Use `@pytest.mark.skip` (or equivalent) with a **specific TODO** and owner, and keep the strict assertion in the skipped body for when the issue is resolved. Do not merge a permanent “≥5/10” assertion as the quality bar.

## Summary

Treat “6/10 under CI” as a **signal to fix isolation or resources**, not as a reason to lower the bar. Align the gate with the actual contract: **all defined concurrent operations must succeed** in this test, or the test is skipped until that is true.
