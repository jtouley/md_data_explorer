# Recommendation: partial-success threshold (10 concurrent requests, ~6 pass in CI)

## Recommendation

**Do not** change the assertion to “at least 5 of 10 succeed.” Treat the current behavior (4 requests failing under CI load) as a **defect in the test or the system under test**, not as tolerable variance.

## Why

- **Four failing requests is not noise**—it means something is wrong (contention, timeouts, resource limits, threading, or unstable environment assumptions).
- **Lowering the bar to 5/10** hides that failure mode, makes regressions invisible, and signals that flaky or load-dependent behavior is acceptable. That erodes trust in CI and encourages more weakened assertions later.
- **Stakeholder intent** (“make CI green”) should be met by **reliability or honest deferral**, not by redefining success.

## Next engineering steps (brief)

1. **Characterize failures** — For failed requests, capture status, error type, and timing (timeout vs. 5xx vs. connection refused). Confirm whether failures correlate with CI parallelism or shared resources.
2. **Reproduce and bisect** — Run the same test locally under stress (parallel jobs, reduced CPU) to see if it’s environmental or inherent to the implementation.
3. **Fix root cause** — Typical levers: dedicated test doubles, connection pooling, serialization where the SUT is not safe for concurrent use, longer-lived fixtures, or CI resource/parallelism tuning **with** a documented reason—not arbitrary threshold tweaks.
4. **If a fix cannot land soon** — Use an **explicit skip** (or quarantine marker) with a **specific TODO** and owner, **not** a partial-success assertion. Stakeholders get transparency; the suite does not lie about correctness.
5. **Agree on product/quality policy** — Align with stakeholders that **either** the feature is concurrently safe and tests must prove it **or** concurrency is out of scope and tests should reflect that scope without pretending 50% is enough.

---

**Summary:** Reject the “at least 5 pass” assertion; investigate and fix the failure mode, adjust CI or test design, or skip with a clear TODO until concurrency is properly supported.
