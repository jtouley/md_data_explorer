# Staff response: “Full UI coverage” sprint ask (plan only)

**Skill alignment:** This answer follows [test-quality-loop](file:///Users/jasontouleyrou/.cursor/skills/test-quality-loop/SKILL.md): orient with repo entrypoints, evidence before recommendations, **no weakened assertions**, and explicit rejection of shallow E2E that duplicates backend/data checks (see skill **Anti-patterns**).

---

## 0. Orient (facts about this repo)

- **Streamlit pages (source of truth):** `src/clinical_analytics/ui/pages/` currently has **8** page modules (e.g. Add Data, Your Dataset, Ask Questions, Descriptive Stats, Compare Groups, Risk Factors, Survival Analysis, Correlations)—not 15. A literal “one test per page” bar is **8 browser journeys**, not 15, unless leadership is counting something else (tabs, future pages, or Electron shell separately).
- **Existing UI test depth:** The project already invests heavily in **pytest-driven UI/page tests** under `tests/ui/pages/` and `tests/unit/ui/pages/` (state machines, caching, gating, uploads, etc.). Those run under canonical targets like `make test-ui` / `make test-fast` and are the right place for **most** behavior and Polars-adjacent outcomes that do not require a real browser.
- **`make test-e2e` today:** Names “E2E” but is **Python integration-style** against engines and flows, not Playwright in a browser. True browser E2E is a **new surface** (with Streamlit **iframe** scoping, server lifecycle, and CI cost)—consistent with prior internal benchmark notes in `.cursor/benchmarks/.../playwright-e2e-streamlit/`.
- **Quality rules:** Project standards require **100% deterministic** tests or explicit skips with TODOs—not partial pass thresholds. Duplicating “main Polars aggregation checks” in the browser risks **second implementations** of the same oracle, which tends to drift and flake without adding proportional safety.

---

## 1. Assessment of leadership’s ask

| Element | Verdict |
|--------|---------|
| **≥15 Playwright tests** | **Misaligned** with “one per page” given **8** pages; likely arbitrary count. Scaling Playwright count without risk ranking inflates CI time and maintenance for marginal signal. |
| **One per Streamlit page** | **Partially reasonable** as a *smoke* map (8 tests), if each test is **narrow** (load, no fatal error, critical chrome visible) and **does not** re-assert analytics numerics. |
| **Duplicate Polars aggregation checks in the browser** | **Not recommended.** It violates the skill’s anti-pattern of **shallow E2E duplicating unit/integration assertions**, especially for DataFrame logic. The browser sees rendered output, not the `LazyFrame`; you either assert fragile UI text of numbers or rebuild pipelines in test code—both are high-cost and low-trust compared to `pl.testing.assert_frame_equal` in Python. |

**Bottom line:** The intent (“more confidence in the product”) is valid; the proposed mechanism mixes **coverage theater** (test count) with **wrong layer** (browser as duplicate data oracle).

---

## 2. Staff counter-proposal (single sprint, bounded)

**Stop criteria for the sprint:** Agree up front on **one** of: (a) `make test-fast` green + new browser job green, or (b) **N named user journeys** (not N arbitrary tests), plus an **iteration cap** (e.g. one vertical slice per week after).

### A. Redefine “full UI coverage”

Define it as **risk-based critical paths**, not page count:

1. **Upload / dataset present** → **Your Dataset** visible and stable.
2. **Ask Questions** happy path (or minimal NL query smoke if env allows).
3. **One analytics page** (e.g. Descriptive Stats) **smoke only**: page renders after prerequisites; no numeric parity with Polars in browser.

Optional fourth/fifth journeys only if incident history or diff risk justifies them (e.g. Correlations if that page regressed recently).

**Playwright test count target:** **5–8** focused specs (aligned with **actual page count**), not 15—unless leadership funds **ongoing** CI and ownership (flakes, Streamlit upgrades, selector churn).

### B. Keep Polars / aggregation truth in Python

- **All aggregation correctness** stays in **unit/integration** tests with **`assert_frame_equal`** (project standard).
- If leadership wants “extra safety” at the boundary, add **contract tests**: e.g. stable **JSON/summary DTO** from the service layer that both Streamlit and tests consume—assert that contract once in Python, and in Playwright assert **presence/shape** of user-visible summary (labels, sections), **not** duplicated numeric recomputation.

### C. Engineering prerequisites (plan items, not code here)

- Document **server lifecycle** and **readiness** (Streamlit cold start); scope locators to the **Streamlit app frame** (`frame_locator`), not only top-level `page`.
- Add or extend **Makefile + CI** so browser E2E is **one optional job** (or tagged target), not silently divergent from what developers run—**verification parity** with documented commands.
- **Test data:** small, deterministic fixtures; no large production dumps.

### D. What we explicitly defer

- **15** Playwright tests as a KPI.
- **Browser-side replication** of Polars aggregation math.
- Putting browser E2E on the **same critical path as every PR** on day one without measuring runtime and flake budget.

---

## 3. Message back to leadership (one paragraph)

“We can materially improve UI confidence this sprint by shipping **5–8** Playwright **smokes** over the **8** Streamlit pages, tied to **named user journeys** and stable readiness patterns, plus **one** optional contract at the API/summary boundary. We should **not** duplicate Polars aggregation checks in the browser—that duplicates oracles, invites drift, and slows CI without matching risk reduction. If the goal is numeric assurance, we extend **Python tests with `assert_frame_equal`** and keep Playwright focused on **what only the browser can prove** (routing, rendering, critical UX, integration with a live app).”

---

## 4. Checklist (skill loop, condensed)

- [ ] **0. Orient** — Reconcile page count vs. “15”; confirm scope (Streamlit only vs. Electron).
- [ ] **1. Collect signals** — Incidents, flaky tests, slow CI; pick journeys from pain.
- [ ] **2. Parallel review** — Short pass on iframe strategy, selectors, and Makefile/CI alignment.
- [ ] **3. Report** — One short quality note: commands, risks, owners.
- [ ] **4. Build** — TDD in Python for data; minimal Playwright smokes.
- [ ] **5. Verify** — Same `make test-*` targets + browser job documented.
- [ ] **6. Loop** — Expand only where signals justify, not to hit an arbitrary count.

---

*Plan only; no implementation in this artifact.*
