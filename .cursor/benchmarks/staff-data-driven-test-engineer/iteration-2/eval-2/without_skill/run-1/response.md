# Staff response: “70% success is the CI bar” (SRE stress)

## What SRE observed

Under **deliberate overload**, roughly **70%** of concurrent requests succeed. That is a **useful capacity/limit signal**, not a **correctness contract** for default CI.

## Recommendation (direct)

**Do not** update the integration test or README so that **70% success is the documented expected bar in normal CI.**

- **Normal CI** should assert **deterministic, 100% success** for the scenarios it owns (or **explicit `skip`** with a concrete TODO when the environment cannot meet that bar yet).
- **Partial success thresholds** in automated gates **mask regressions** and **normalize flakiness**; they also diverge from this repo’s stated rule: fix root cause, or skip with rationale—do not weaken assertions to “good enough under stress.”

## What to do instead (next steps)

1. **Separate concerns:** Keep **stress / soak / load** runs in a **non-blocking** or **scheduled** job (or a dedicated workflow) with metrics and dashboards—not as the default `make test` / PR gate.
2. **Document SRE findings in README or runbooks** as **observed behavior under stress** (N concurrent clients, setup, date), plus **known limits** (e.g., DuckDB threading, single connection)—not as “CI expects 70%.”
3. **Product/engineering track:** File work to **improve concurrency** (pooling, queueing, read replicas, rate limits, or “degraded mode” semantics) until **stress tests** meet an agreed SLO; only then consider tightening **stress** assertions—not unit/integration correctness tests.
4. **If the integration test cannot pass at 100% today:** Use **`@pytest.mark.skip`** (or quarantine in a non-default suite) with a **specific TODO**, per project policy—**not** a 70% assertion in main CI.

---

*Brief plan; no code in this document.*
