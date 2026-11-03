# Staff response: “full UI coverage” sprint ask

## What leadership asked for

- At least **15 Playwright tests**, **one per Streamlit page**
- **Duplicate main Polars aggregation checks in the browser** for “extra safety”

## Staff read (direct)

**The headline number is plausible; the coupling of goals is not.**

1. **“One test per page” is a vanity metric.** A single smoke test per page proves the route renders, not that workflows, data contracts, or error paths work. Fifteen tests can be **15 shallow clicks** that add CI time and maintenance without reducing production risk.

2. **Re-running Polars aggregation assertions through the browser is usually the wrong layer.** Unit and integration tests already own numeric correctness on real frames. Browser tests are slow, flaky-prone, and poor at asserting exact aggregates unless you scrape tables or DOM that Streamlit does not guarantee as a stable contract. You end up testing **Streamlit’s rendering**, not your analytics engine.

3. **Sprint scope risk.** Playwright against Streamlit often needs iframes, waits, and environment fixtures (data, DuckDB, secrets). Standing up **15 reliable** specs in one sprint typically trades velocity for **brittle E2E** unless the app already has stable `data-testid`s and a documented runbook.

4. **Opportunity cost.** The same effort could raise confidence more if spent on **contract tests** between UI inputs and the Python layer, or on **a smaller set of journey-level E2E** plus faster API/data tests.

## Counter-proposal (plan only)

**Goal:** Meaningful UI confidence this sprint without duplicating Polars logic in Playwright.

### Tier A — Must ship (sprint)

- **Playwright: 5–8 journey tests**, not 15 page-smokes. Pick the **highest-value user paths** (e.g., load sample data, run a query path, switch dataset, see error for bad input). Each test maps to a **user outcome**, not a page enum.
- **Per-page smoke (optional cap):** If leadership insists on page coverage, add **at most one** `expect(page).toHaveURL` / title / key heading check **only** for pages that are independent entry points—not 15 full interactions.
- **Polars / aggregates:** Keep **one source of truth** in Python tests (`assert_frame_equal`, pipeline tests). Expose **stable, test-facing outputs** from the backend (e.g., JSON snapshot of summary stats for a fixed fixture) and assert that from **API or direct Python tests**, not from pixel/table scraping in Chromium.

### Tier B — Next sprint

- Grow Playwright count **after** selectors and test data are stable; target **15+** only when **flake budget** and **parallel CI** are in place.
- Add **visual or table regression** only for 1–2 critical dashboards if numeric parity in-browser is truly required—still prefer comparing **exported CSV / API payload** over DOM text.

### Success criteria (replace “15 tests”)

- **Zero critical-path regressions** on agreed journeys in CI.
- **Median Playwright run time** under an agreed cap (for example, fewer than 10 minutes on CI).
- **Flake policy:** any intermittent test is skipped with a ticket, not weakened (aligns with strict assertion culture).

### What to tell leadership

“We’ll deliver **high-signal browser coverage on the workflows that matter** and keep **aggregate correctness in Polars-backed tests** where it belongs. We’ll **not** duplicate Polars checks in Playwright this sprint—that’s cost without proportional safety. If you want a **numeric parity** story, we’ll add **one** golden-fixture check via a **stable interface** (not the Streamlit DOM).”

---

*Plan only; no implementation in this document.*
