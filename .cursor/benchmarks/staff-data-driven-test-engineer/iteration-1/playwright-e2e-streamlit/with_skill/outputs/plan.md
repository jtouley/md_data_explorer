# Plan: First Playwright E2E checks for Streamlit (`make run` on localhost)

**Context:** This repo already has `make run` → `uv run streamlit run src/clinical_analytics/ui/app.py` and `make test-e2e` → pytest on `tests/e2e/`. That folder today holds **Python integration-style** “E2E” (e.g. `NLQueryEngine` flows), **not** a real browser. Electron has separate Playwright under `electron/` (desktop shell), which is out of scope for this Streamlit-localhost slice.

**Skill alignment:** Follow the staff data-driven test engineer loop (orient → small vertical slice → verify) and [e2e-and-playwright-methodology.md](file:///Users/jasontouleyrou/.cursor/skills/test-quality-loop/references/e2e-and-playwright-methodology.md): Pareto E2E, determinism, user-visible assertions, no assertion weakening. For scripting and lifecycle patterns, align with the **webapp-testing** skill: prefer a **black-box server wrapper** and `sync_playwright`, explicit readiness waits—not fixed sleeps.

---

## 0. Orient (current surfaces)

| Surface | Command / path | Role |
|--------|----------------|------|
| Streamlit app | `make run` (port **8501** by default) | SUT for browser E2E |
| “E2E” pytest today | `make test-e2e` → `tests/e2e/` | **Not** browser; keep naming honest in docs |
| Fast gate | `make test-fast` | Excludes `@pytest.mark.slow` |
| Parallel pytest | `make test-e2e` uses `-n auto` | **Unsafe** for multiple browsers on one port unless isolated |

**Streamlit + Playwright caveat:** Streamlit serves the UI inside nested frames. Many automation tools only see the top document until you **target the app frame** (e.g. `frame_locator` for the Streamlit iframe). The first implementation task after “hello world” is to **record/locate the stable frame** and scope locators there; avoid assuming `page.getByRole` works on the root `page` without frame scoping.

---

## 1. Layering: what stays in pytest vs the browser

| Concern | Pytest / Python layer | Playwright (browser) layer |
|--------|------------------------|----------------------------|
| Business logic, NL parsing, QueryPlan, Polars | Unit + existing `tests/e2e` Python tests | **No** duplication |
| Streamlit session state, component render contracts | `tests/ui/` with mocks where possible | Only **critical** user-visible journeys |
| “App boots and primary chrome is reachable” | Fixture: server health (HTTP/TCP) | `goto` + frame + one stable heading/button |
| Data setup (cohorts, uploads, API) | Prefer **HTTP/API or filesystem fixtures** before `goto` | Use UI only for the **last mile** |
| Assertions | `pl.testing.assert_frame_equal`, etc. | `getByRole` / `getByLabel` / stable text; URL if meaningful |
| Flake diagnostics | pytest hooks, logs | Optional trace/screenshot **on failure** only (CI lean) |

**Rule:** One **vertical slice** first—for example: “Streamlit responds on localhost; main navigation or home title visible in app frame”—before adding NL-query-in-browser flows that need Ollama/LLM.

---

## 2. Dependencies and layout

1. Add **Playwright for Python** to dev dependencies (`pyproject.toml` optional group): `playwright` + run `playwright install chromium` in CI/docs (pin browser version in CI when possible).
2. **Directory choice** (avoid conflating with today’s `tests/e2e/`):
   - **Recommended:** `tests/e2e_streamlit/` or `tests/browser/` with module docstring explaining “browser E2E; not `tests/e2e`”.
   - **Alternative:** Keep `tests/e2e/` for Python-only and add `tests/e2e/playwright/` with a `conftest.py` that registers browser fixtures only under that subtree.
3. **Makefile:** Add a dedicated target (e.g. `make test-e2e-streamlit`) that runs **only** browser tests **serially** and does **not** use `-n auto`, **or** keep `make test-e2e` for legacy Python E2E and wire browser suite separately until renamed.

---

## 3. Server lifecycle (align with webapp-testing)

The webapp-testing skill documents `scripts/with_server.py --server ... --port ... -- python script.py`. **This repository does not currently ship that script.** Pick one pattern and standardize:

**Option A — Ported helper (closest to webapp-testing):** Add `scripts/with_server.py` (or a minimal `scripts/with_streamlit_e2e.py`) that:

- Spawns the **same command as `make run`**: `uv run streamlit run src/clinical_analytics/ui/app.py` (or invoke `make run` is awkward in subprocess—prefer explicit `uv run` to match venv).
- Accepts `--port` (default 8501) and optional `--bind 127.0.0.1`.
- Polls until `GET http://127.0.0.1:<port>/_stcore/health` (Streamlit’s health path) or TCP connect succeeds, with timeout and clear logs on failure.
- On exit, terminates the process tree (SIGTERM then kill).

**Option B — pytest fixture (no separate script):** Session-scoped fixture starts subprocess, yields `base_url`, tears down. Same health check. Good for `make test-e2e-streamlit`; slightly less reusable for ad-hoc scripts.

**Option C — pytest-playwright `web_server` config:** If using `pytest-playwright`, configure `web_server.command` and `web_server.url` to match Streamlit startup; ensure timeout accommodates cold start.

**Environment isolation for determinism:**

- Set `STREAMLIT_*` / cwd / `HOME` or `XDG_*` under `tmp_path` or a dedicated test workspace so runs do not read the developer’s real `~/.streamlit` or clash on uploads DB paths.
- Use a **random free port** if parallel browser workers are ever introduced (future); for iteration 1, **serial + fixed port** is simpler.

---

## 4. Flake guardrails (project rules–compatible)

1. **No weakened assertions** (see `.cursor/rules/106-test-quality-thresholds.mdc`): do not add “70% of steps passed” or CI retries to hide races—fix waits or skip with a concrete TODO.
2. **Markers:** `@pytest.mark.slow` + `@pytest.mark.integration` (and optionally a custom `@pytest.mark.browser`) so `make test-fast` stays fast; browser suite runs on demand / nightly / optional PR job.
3. **Serial execution:** `@pytest.mark.serial` (already respected by `make test-*` parallel modes) so xdist does not spawn multiple Streamlit instances on one port.
4. **Waits:** After navigation, prefer **app-stable** conditions: health OK, then `wait_for_load_state` as appropriate, then **frame** attached, then locator visible. Document any unavoidable `sleep` with reason (Streamlit websocket churn).
5. **External services:** If a journey needs Ollama, use `skip_if_ollama_unavailable` (pattern from `tests/AGENTS.md`) or a dedicated marker; do not block the **smoke** browser test on LLM.
6. **Artifacts:** Enable Playwright trace/video only in CI `on-first-retry` or local `--headed` debugging; default CI stays lean (methodology reference).
7. **Port conflicts:** Fail fast with a clear message if 8501 is in use; or use dynamic port + pass `base_url` into tests.

---

## 5. First tests (ordered backlog)

1. **Smoke:** Server up; HTTP health; browser opens `base_url`; **Streamlit iframe** found; visible text or landmark matches expected home state.
2. **Navigation (optional second):** Sidebar link to a known page title (use role/label; add `data-testid` only if Streamlit churn makes roles unstable—sparingly).
3. **Defer:** Full NL query in browser until smoke is stable and LLM/Ollama policy is explicit.

---

## 6. CI and `make check`

- **Do not** add browser E2E to `make check` on day one if it requires Playwright browsers and lengthens every PR; add a **workflow job** or optional target first.
- Document in `tests/AGENTS.md` (small section): how to run `make test-e2e-streamlit`, prerequisite `playwright install`, and difference vs `tests/e2e/` Python tests.

---

## 7. Verification checklist (exit criteria for iteration 1)

- [ ] `make test-e2e-streamlit` (or equivalent) passes locally headless with **no** manual `make run` in another terminal.
- [ ] `make test-fast` unchanged or still excludes browser tests via markers.
- [ ] One written **journey name** and user-visible assertions; no duplicate coverage of `NLQueryEngine` in Playwright.
- [ ] Failure mode is observable: server logs + optional trace on failure.

---

## 8. Relation to Electron Playwright

`electron/` uses Node `@playwright/test` for the desktop shell. **Reuse concepts** (trace, selectors discipline), **not** the same config or test folder. If the product later replaces Streamlit, browser E2E may move to the new renderer; keep **server lifecycle and health-check patterns** portable.
