# Plan: First Playwright E2E checks for Streamlit (`make run`, localhost)

## Context in this repo

- **`make run`** starts Streamlit with: `uv run streamlit run src/clinical_analytics/ui/app.py` (see `Makefile` `run` target). Default port is **8501** unless overridden by env/config.
- **`tests/e2e/`** today means **Python integration E2E** (e.g. NL query flow across components), invoked by **`make test-e2e`**. It is **not** browser automation. New browser checks should live under a **different path** (e.g. `tests/browser_e2e/` or `tests/e2e_browser/`) and optionally a **different Makefile target** (e.g. `make test-browser-e2e`) so naming stays honest and `make test-e2e` does not suddenly require Chromium.
- **Electron** already uses `@playwright/test` under `electron/`; Streamlit browser E2E is a separate surface (Python app + Playwright **Python** API is the natural fit for pytest alignment).
- The **webapp-testing** skill describes a **`scripts/with_server.py`** lifecycle helper; that script is **not present** in this repository. Either add a small repo-local equivalent or implement **pytest session-scoped server fixtures** with the same contract: start process → wait for TCP/HTTP ready → yield → teardown.

---

## 1. Layering: pytest vs browser

### Keep in **pytest** (orchestration and gates)

- **Process lifecycle**: spawn Streamlit subprocess with explicit `BASE_URL`, `STREAMLIT_SERVER_PORT`, `STREAMLIT_SERVER_HEADLESS=true` (or documented env), and working directory = repo root so imports match `make run`.
- **Readiness**: poll `http://127.0.0.1:<port>/_stcore/health` (Streamlit health) or TCP connect + first successful GET; **timeout with clear error** (not silent hang).
- **Markers and selection**: e.g. `@pytest.mark.browser_e2e` plus **`@pytest.mark.slow`** and **`@pytest.mark.integration`** so `make test-fast` stays fast. Register the marker in `pyproject.toml`.
- **Skipping**: `pytest.importorskip("playwright")`; skip if `PLAYWRIGHT_SKIP=1` or CI matrix leg without browsers; optional skip if port in use (with message to pick another port).
- **Artifacts on failure**: screenshot + HTML dump + trace path in `tmp_path` or `pytest` `request.node.name`-derived dir (configure in fixture finalizer).
- **Data/setup**: if a test needs a dataset, prefer **API/fixture-driven seeding** or a **dedicated test config** over manual UI upload in the first slice—unless the goal is explicitly upload UX.

### Keep in **Playwright** (browser-only)

- **Navigation and waits**: `page.goto(base_url)`, then waits tied to **app-specific signals** (e.g. role/text/locator for sidebar or main title), not arbitrary long `sleep`.
- **Assertions on user-visible behavior**: headings, buttons, navigation between pages, presence of key widgets—**minimal** first slice (e.g. app loads, sidebar renders, one page transition).
- **Selectors**: prefer **stable** attributes (roles, test ids if you add them in Streamlit components, or documented text). Avoid brittle CSS tied to Streamlit internals that change across versions.

### Boundary rule

- **Do not** re-assert business logic in the browser that is already covered by unit tests; browser tests should prove **wiring + rendering + critical paths** that pytest alone cannot see.

---

## 2. Flake guardrails

- **Serial execution**: run browser E2E **serially** (`-n 0` / no xdist for this folder, or dedicated `make test-browser-e2e-serial`) so two tests do not fight for the same port or global Streamlit state.
- **Dedicated port**: random free port or env `E2E_STREAMLIT_PORT` to avoid clashing with a developer’s manual `make run`.
- **Deterministic timeouts**: use Playwright `expect(...).to_be_visible()` with a **single** project-level timeout; avoid scattered magic numbers; **do not** “fix” flakes by loosening assertions (align with project test-quality rules—fix root cause or skip with TODO).
- **Streamlit specifics**:
  - The app may use **iframes** for embedded UIs; Playwright may not see inside iframes from the parent frame—first tests should target **top-level** UI only, or use `frame_locator` once a stable iframe selector exists.
  - **WebSocket/long-polling**: `networkidle` is often wrong for Streamlit; prefer **locator- or response-based** waits.
- **Cold start**: first test or session fixture may pay Streamlit import cost; either accept one slower test or warm up once in session setup.
- **Cleanup**: terminate the whole process group on teardown so orphan Streamlit processes do not leak between runs.

---

## 3. Alignment with webapp-testing / server lifecycle patterns

- **Mirror the skill’s intent** even without `with_server.py`:
  - **Outer**: one command that guarantees “server up before child, dead after” (Makefile target or pytest plugin-style fixture).
  - **Inner**: Playwright script/tests that assume **only** `BASE_URL` (no subprocess code duplicated per test file).
- **Makefile** (suggested):
  - `make test-browser-e2e`: `uv run pytest tests/browser_e2e -m browser_e2e -v` (serial).
  - Document **`E2E_BASE_URL`** override: if set, fixtures **skip starting** Streamlit and use the URL (matches “server already running” flow from webapp-testing).
- **Dependencies**: add `playwright` to dev deps and document **`uv run playwright install chromium`** (or CI step) once; fail fast in fixture if browser missing.
- **CI** (when ready): install browsers, run browser E2E on a **single** job, not on every parallel shard; cache Playwright browsers.

---

## 4. Suggested first slice (minimal vertical)

1. Session-scoped fixture: start Streamlit → health check → yield `base_url`.
2. One test: **home loads**—assert title or main heading text present.
3. One test: **navigate**—open sidebar page link or `st.navigation` equivalent and assert destination content stub.

Expand only after the slice is stable locally and in CI.

---

## 5. Naming and documentation

- Update **`tests/AGENTS.md`** (short subsection): distinction between `tests/e2e` (Python integration) and `tests/browser_e2e` (Playwright).
- Optionally cross-link **`docs/architecture/LIGHTWEIGHT_UI_ARCHITECTURE.md`** if it still mentions Playwright E2E—keep terminology consistent (Streamlit vs Electron).

---

## Summary

Introduce **Playwright-for-Streamlit** under a **new** test directory with **pytest-owned lifecycle**, **slow/integration markers**, **serial runs**, **health-based readiness**, and **Makefile + optional pre-started URL** aligned with the webapp-testing server pattern—without overloading existing **`tests/e2e`** or **`make test-e2e`**.
