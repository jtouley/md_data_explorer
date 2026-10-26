---
doc_type: review
status: active
title: Repository review and MVP roadmap
last_updated: 2026-10-26
---

# Repository review and MVP roadmap (2026-10-26)

Scope: `main` at `9eb4a9d`, every open PR (#59, #60, #61, #62), every open issue (#47, #51–#58), the CI history, and a scan of comparable open-source projects.
The single prompt that turns this review into a working MVP is in [`MVP_ONE_SHOT_PROMPT.md`](MVP_ONE_SHOT_PROMPT.md). In Claude Code it also runs as `/mvp`.

---

## 1. Verdict

The hard parts are already built. They are:

- a closed set of statistical intents: `DESCRIBE`, `COMPARE_GROUPS`, `FIND_PREDICTORS`, `SURVIVAL`, `CORRELATIONS`, `COUNT`;
- a semantic layer built on Ibis and DuckDB;
- a three-tier natural-language parser (pattern, embedding, optional local LLM);
- a `QueryService` with no Streamlit dependency;
- a 400-question golden suite checked against SQL oracles (on PR #60).

No project in the comparison set (§5) combines plain-English questions with *inferential* statistics. That is the moat.

What blocks an MVP is **not missing features**. It is **finish and focus**:

1. **Work is spread across four open PRs, with no merge order.** Two are green and mergeable (#59, #61). One is red on a single mypy error (#60). One is a 186-file branch that conflicts with `main` (#62).
2. **There are two UI tracks.** The "primary track" in `MASTER_PLAN.md` is Electron + FastAPI and deletes Streamlit. But `main` ships Streamlit, and Streamlit is the only UI that works end to end today. Trying to finish an MVP and a UI rewrite at the same time is how neither ships.
3. **There is no headless proof that the product works** (issue #51). Every API test mocks storage or the query service. PR #60 contains the missing piece (`scripts/headless_ask.py`, `eval/headless.py`) but has not merged.
4. **The trust step is missing.** A low-confidence parse still auto-executes on `main`. PR #59 fixes this. Answers start with a table rather than a claim (issue #52).
5. **The first-run experience has drifted badly.** The README describes built-in COVID-MS and Sepsis datasets that were removed. It says to run `pytest` directly, which this repo's own rules forbid. It lists pandas as the core stack. A new user cannot get from clone to first answer by following it.

**Recommendation:** Freeze Electron. Ship the MVP on Streamlit plus the headless CLI and FastAPI. Merge the open PRs in the order in §4. Treat Electron as v0.2, built on the same `QueryService`, so none of the MVP work is thrown away.

### Checks I ran locally for this review (2026-10-26)

| Check | Result |
|---|---|
| `uv sync --extra dev --group dev` | OK, but slow: it pulls CUDA torch wheels (§3.2 item 8) |
| `make check-fast` on `main` | ✅ lint and format clean, 341 files |
| `make test-fast` on `main` | ✅ **1758 passed, 26 skipped**, 3m15s (Python 3.13) |
| PR #60: `scripts/headless_ask.py --dataset gdsi --query "how many patients?"` | ✅ `success: true`, `count: 1141` (the table has 1,141 rows) |
| PR #60: mypy with the `isinstance(maximum, int \| float)` narrowing at `catalog.py:89` | ✅ `no issues found` |
| `data/analytics.duckdb` | Schema read only; see §3.1 item 1 |

---

## 2. What is on `main` today (verified)

| Area | State | Evidence |
|---|---|---|
| CI (`ci.yml`) | Green on `main` | Run #63 (`9eb4a9d`): success |
| Docs deploy (`docs.yml`) | **Red on every push to `main`** | It installs mkdocs on Python 3.10 without the package. mkdocstrings fails with `No module named 'clinical_analytics'`, and the project requires `>=3.11`. |
| Source size | 38.9k LOC in `src/`, 210 test files | Four god-files: `nl_query_engine.py` (3,600), `user_datasets.py` (2,803), `03_💬_Ask_Questions.py` (2,616), `semantic.py` (2,392) |
| UI | Streamlit, 8 pages | `make run` |
| API | FastAPI with sessions, datasets, queries (SSE), and enrichments | `make run-api`. The tests mock storage and the query service. |
| Electron | Skeleton and Playwright harness on `main`. Phase 10 (remove Streamlit) is only on #62. | `electron/` |
| Headless CLI | **None on `main`** | Issue #51. PR #60 adds one. |
| Confidence gate | **Does not hold execution** | `execute_query_plan` only attaches a warning. PR #59 fixes this. |
| Claude Code hooks | **Never run** | `.claude/settings.json` uses lowercase `preToolUse`/`postToolUse` and a `pattern`/`command` shape that Claude Code does not read. PR #62 fixes this, but that branch is blocked. |
| Committed data | `data/analytics.duckdb` (80 MB) is tracked in git | It holds two copies of the MIMIC-IV demo (#53) plus the `gdsi`, `dexa` and `statin` uploads. Their provenance and licence are undocumented (see §3.1). |

---

## 3. Findings, ranked

### 3.1 Must fix before calling anything an MVP

1. **Patient-level data is committed to git. This is the most serious finding.** `data/analytics.duckdb` is a binary blob in history, and anyone who clones the repo gets it. I read only the schema, never row values. It contains:
   - `…de_identified_dexa…`: 857 rows. Columns include Race, Gender, Age, CD4 count, viral load, prior tenofovir use, and the current ART regimen (Biktarvy, Symtuza, and others).
   - `…statin_use_deidentified…`: 807 rows. Race, Gender, Age, VL, CD4, ART regimen, and a lipid panel.
   - `…gdsi_opendataset_final…`: 1,141 rows. This is the public COVID-19 in MS dataset.
   - two copies of the MIMIC-IV Clinical Database Demo (32 tables each). It is open access under ODbL on PhysioNet, so it can be redistributed with attribution.

   **This also affects PR #60.** Its golden suite (`eval/catalog.py`, `tests/eval/golden/{dexa,statin}.yaml`) loads these tables and stores aggregates computed from them. Removing the database without replacing that data turns #60's suite red. The MVP prompt regenerates those two datasets as synthetic look-alikes.

   The DEXA and statin tables look like a **real HIV clinic extract**. "De-identified" in a file name is not a HIPAA Safe Harbor or Expert Determination. Race, gender and age combined with HIV status is a re-identification risk for a small clinic population. HIV status is also specially protected in many US states.
   - **Action (owner decision, today):**
     - Confirm where these two files came from and whether you are allowed to redistribute them.
     - If you cannot document that permission, rewrite history to remove `data/analytics.duckdb` (`git filter-repo --path data/analytics.duckdb --invert-paths`, then force-push and ask forks to re-clone).
     - The repo is public today, with 0 forks and 0 stars (checked 2026-10-26). That makes now the cheapest moment to rewrite history. Make it private until that is done.
     - The MIT relicense in #48 made this intentionally public OSS, which raises the stakes.
   - Move the database out of the tracked tree and add `data/*.duckdb` to `.gitignore`.
   - Ship a small **synthetic** demo CSV generated by a script (seeded and deterministic), and point the golden suite at it.
   - If any of the data is real, rewrite history (`git filter-repo`) before the repo gets more visibility. The MIT relicense in #48 made it intentionally public OSS.
2. **Low-confidence plans execute** (ADR001 gate, issue #52). PR #59 already implements the hold and has green CI. It is still a draft, so mark it ready and merge it.
3. **No uninterrupted upload → ask → answer test** (#51). Without it, every "it works" claim depends on mocks. PR #60 contains the headless path. Close #51 with:
   - a `make smoke` target that runs `scripts/headless_ask.py` against the synthetic CSV;
   - a pytest that runs that target as a subprocess.
4. **PR #60 is red** because of one mypy error at `src/clinical_analytics/eval/catalog.py:89`. `series.max()` returns a wide literal union, and the code compares it with `<= 20`. Fix it by narrowing the type (`isinstance(maximum, int) and maximum <= 20`). Do not add a `type: ignore`.
5. **The docs deploy fails on every push to `main`.** In `docs.yml`:
   - set Python to `3.11`;
   - install with `uv sync --group dev` (or `pip install -e .`) so mkdocstrings can import `clinical_analytics`;
   - optionally run it as a `mkdocs build --strict` check on PRs, so it cannot go silently red again.
6. **The README is not a quickstart.** Rewrite the top of it:
   - `uv sync`;
   - `make demo-data`;
   - `make run`;
   - "ask *Compare LDL by statin use*";
   - expected output.

   Remove the COVID-MS, Sepsis and MIMIC-III built-in claims and the `pytest` instruction.

### 3.2 High-leverage, small

7. **Claude Code guardrail hooks are inert** (see §2). Take the `.claude/settings.json` and hook-script fix out of #62 and land it as its own PR. Without it, the "HARD RULES" in `CLAUDE.md` are enforced only by convention.
8. **The heavy ML dependencies make first install slow.**
   - `sentence-transformers` pulls in PyTorch and, on Linux, about 3 GB of CUDA wheels.
   - The embedding tier only needs CPU. Pin the CPU wheel index for torch (`[tool.uv.sources]` / `[[tool.uv.index]]` with `pytorch-cpu`), or move the embedding tier into an optional extra (`[project.optional-dependencies] nlp = [...]`) and let the pattern tier work alone.
   - `langchain` and `langchain-community` appear to be used only for document loading or prompting, both of which the codebase already does directly. Audit them and drop them if unused.
9. **The project breaks its own rules.** There are 137 `except Exception` blocks in `src/` (CLAUDE.md: "Catch specific exceptions, never bare `Exception`"). There are 35 `print(` calls in `src/`, including in the FastAPI lifespan (CLAUDE.md requires `structlog`). Pandas appears in 27 source files. These are not MVP blockers. Add a ruff `BLE001` and `T20` gate with a baseline so the count can only go down.
10. **Duplicate MIMIC load** (#53). Drop one schema. This becomes moot if the database leaves git (item 1).
11. **Stale planning docs.** `docs/todos/001`–`010` are all `status: pending` from 2025-12, including SQL-injection and path-traversal items whose code paths have since moved. Close them, or re-verify them against the current code. Stale P1 security notes erode trust in the real ones.
12. **A misspelled folder: `data/dicitonaries/`** (#55). Rename it to `data/dictionaries/` and update `pyproject.toml` (`[tool.mutmut].also_copy`). Ingesting those PDFs is post-MVP.

### 3.3 Structural (post-MVP, but plan for it)

13. **The god-files.**
    - `nl_query_engine.py` holds pattern matching, embedding search, LLM prompting, DBA-style validation, filter extraction and fuzzy column resolution in one class. Split it along the tiers it already has: `parsing/patterns.py`, `parsing/semantic.py`, `parsing/llm.py`, `parsing/filters.py`, `parsing/resolve.py`. `NLQueryEngine` then becomes a thin coordinator.
    - `03_💬_Ask_Questions.py` should shrink to rendering only, once `QueryService` owns gating (PR #59 starts this).
14. **Mutable global state in `AsyncQueryService`.** `_queries` and `_events` dicts grow without bound and are lost on restart. That is fine for a local single-user MVP. It needs a TTL or a SQLite-backed store before any shared deployment.
15. **Authentication and audit** (todos 003 and 009). This is out of scope for a local, single-user MVP. Say so explicitly in the README ("local-only; do not expose port 8501/8000"). Bind uvicorn to `127.0.0.1`, not `0.0.0.0` as the `api/main.py` `__main__` block does today.

---

## 4. Open PRs and issues: disposition and merge order

| # | Title | CI | Mergeable | Disposition |
|---|---|---|---|---|
| **#61** | test: cover failure paths, collapse duplicates | ✅ | clean | **Merge first.** Tests only; it reduces noise for everything after it. |
| **#59** | fix(ask): hold low-confidence plans | ✅ | clean (draft) | **Mark ready and merge second.** It implements half of #52. Then rebase #60 onto it. |
| **#60** | feat(eval): grade golden questions against tables | ❌ mypy | unstable | **Fix `catalog.py:89`, merge third.** It brings the headless CLI, cohort constraints, and the 400-question SQL-oracle suite. Some of the golden questions are padding (#56). Keep them, and mark the padding as such. |
| **#62** | Phase 10 Electron/FastAPI + test-quality | — | **dirty** | **Do not merge as is.** It mixes four unrelated changes (Streamlit removal, agent and skill benchmarks, hook fixes, test fixes) across 186 files and deletes 18.8k lines. Take out: (a) the hook fix (§3.2 item 7); (b) the `api/dependencies.py` wiring and its tests. Park Phase 10 until after the MVP tag. |
| #47 | recursive health review | — | — | Close in favour of this document. |
| #51 | headless API/CLI path | — | — | Closed by #60 plus a `make smoke` target and a subprocess test. |
| #52 | claim-first + confirm low-confidence | — | — | The gate half is done by #59. **The claim-first render is the last MVP feature** (§6, step 6). |
| #53 | duplicate MIMIC load | — | — | Goes away with §3.1 item 1. |
| #54 | MIMIC multi-table cohort | — | — | Post-MVP. The multi-table handler exists; a cohort view is v0.2. |
| #55 | dictionary PDFs not queryable | — | — | Post-MVP. Rename the folder now. |
| #56 | paraphrase padding in golden set | — | — | Post-MVP. Resolve it with the synthetic demo dataset, which can be designed with enough columns. |
| #57 | cardinality off the pre-gate path | — | — | Post-MVP (different epic). |
| #58 | confirm flag on `QueryService.ask` | — | — | **Include it in the MVP.** It is about 20 lines on top of #59. Without it, API and CLI clients can never run a low-confidence plan. |

---

## 5. Comparable open-source projects

Star counts and activity come from the GitHub API on 2026-10-26.

| Project | What it is | Stars / activity | Licence | What to borrow |
|---|---|---|---|---|
| [WrenAI](https://github.com/Canner/WrenAI) | Text-to-SQL through a governed semantic layer (MDL) | 18.0k, active | Apache-2.0 (core) | The LLM may only reference declared metrics. Show the generated query and its reasoning next to each answer. |
| [Vanna](https://github.com/vanna-ai/vanna) | RAG over DDL and question→SQL pairs | 23.8k, **archived** | MIT | Confirmed answers become retrieval examples. Do not depend on it. |
| [pandas-ai](https://github.com/sinaptik-ai/pandas-ai) | LLM writes and runs pandas code | 23.9k, stalled since 2025-10 | MIT + `ee/` | A cautionary example. Running LLM-written code is the trust problem the closed intent set avoids. |
| [DB-GPT](https://github.com/eosphoros-ai/DB-GPT) | Agentic data assistant for local models | 20.2k, very active | MIT | A first-class local-model registry, which strengthens the PHI and offline story. |
| [Lightdash AI](https://github.com/lightdash/lightdash) | AI agents over a dbt semantic layer | 6.2k, very active | MIT + `ee/` | **Verified answers** (approved answers matched by embedding) and **evaluation suites** built from real threads. The strongest trust model in the set. |
| MotherDuck Dives | Natural-language question → saved, live DuckDB visualization | proprietary | — | The answer as a durable, re-runnable object |
| [LIDA](https://github.com/microsoft/lida) | LLM summarizer → goals → charts | 3.3k, inactive | MIT | A compact dataset profile at upload, used to seed suggested questions |
| [jamovi](https://github.com/jamovi/jamovi) / [JASP](https://github.com/jasp-stats/jasp-desktop) | Point-and-click stats (Electron + R) | 0.8k / 1.0k, active | AGPL | **Syntax mode**: emit the equivalent code for every analysis. Assumption checks with automatic non-parametric fallback. APA tables. |
| [OHDSI ATLAS](https://github.com/OHDSI/Atlas) | Cohort building on OMOP | 0.3k, active | Apache-2.0 | **Attrition tables**: how many patients remain after each filter |
| i2b2 / SHRINE | Federated cohort counts | — | — | **Small-cell suppression** on patient counts (under 11) |
| [Datasette](https://github.com/simonw/datasette) / [Rill](https://github.com/rilldata/rill) | Instant exploration of SQLite / DuckDB | 11.5k / 3.0k | Apache-2.0 | Automatic facets and profiles at file drop, so users get value before they ask anything |
| [Evidence](https://github.com/evidence-dev/evidence) / [marimo](https://github.com/marimo-team/marimo) | Reports or notebooks as code | 7.1k / 23.3k | MIT / Apache-2.0 | Export a session as a reproducible report or notebook |

**Positioning.** The natural-language tools stop at SQL aggregates. The statistics tools have no natural-language layer. This project is the only one aiming at *"ask in English, get a defensible inferential result."* It wins only if the parse is **visibly** correct, so the MVP features below are all about trust, not breadth.

**Five features to borrow, in priority order:**

1. **Plan card before execution**, with an editable plan (PR #59 plus a confirm control). After the result, show the equivalent Python (jamovi syntax mode).
2. **Claim first**: one sentence giving the effect, interval, N, test used, and what was not adjusted for (issue #52).
3. **Assumption checks with fallbacks** (JASP): normality, then Mann-Whitney; expected cell counts, then Fisher's exact. This prevents results that "work but shouldn't".
4. **Small-cell suppression and an attrition line** (i2b2, ATLAS): any count under 11 renders as `<11`, and the result shows `N after filters`.
5. **Verified answers** (Lightdash and Vanna): a confirmed plan is stored as a golden example and retrieved before the LLM tier. `golden_examples.yaml` and `prompt_optimizer.py` already exist. Wire them to the confirm button. **Post-MVP:** this is the first v0.2 item.

---

## 6. MVP definition and plan

**MVP (v0.1.0):** on a laptop with no internet, a clinician runs one command, and then:

- uploads a CSV or XLSX (or uses the bundled synthetic demo);
- asks any of the six intent types in English;
- sees the interpreted plan, and must confirm it when confidence is below 0.75;
- gets a **claim-first** answer with N, the test used, the effect and a 95% CI;
- sees small counts suppressed;
- can download the result as CSV.

The same question gives the same answer through the CLI (`scripts/headless_ask.py`), `POST /api/queries`, and the Streamlit UI. CI proves it with a smoke test and the golden suite. **No LLM is required:** Ollama is an optional upgrade.

**Out of MVP scope:** Electron (Phase 10), multi-table cohorts (#54), PDF dictionaries (#55), verified-answer retrieval, auth and multi-user, and the god-file splits.

**Steps.** These are what the one-shot prompt executes.

0. Baseline: `uv sync`, `make check-fast`, `make test-fast`. Record the pre-existing failures so they are not mistaken for regressions.
1. Merge PRs in this order: #61, then #59, then #60. Fix #60's mypy error, and fix any conflicts with a merge commit.
2. Data hygiene:
   - add `scripts/make_demo_data.py`, which writes a seeded synthetic cohort CSV to `data/demo/`;
   - add a `make demo-data` target;
   - stop tracking `data/analytics.duckdb` and ignore `data/*.duckdb`.
3. Headless proof (#51):
   - add a `make smoke` target that generates the demo data, registers it, and asks one question per intent through `QueryService`;
   - add a subprocess pytest for it;
   - add `confirm: bool = False` to `QueryService.ask` and to the API and CLI (#58).
4. Fix the hooks: correct `.claude/settings.json` to the real `PreToolUse`/`PostToolUse` schema (port from #62).
5. Fix CI and docs: `docs.yml` on Python 3.11 with the package installed; run the smoke target in `ci.yml`.
6. Claim-first answers (#52, the remaining half):
   - a pure function `build_claim(result) -> str` in `core/result_interpretation.py`, tested per intent;
   - rendered as the first element on Ask Questions, and returned as `claim` by the API and CLI;
   - small-cell suppression (counts under 11 shown as `<11`) in the same render path.
7. README quickstart rewrite. Mark the local-only security posture. Bind the API to `127.0.0.1`.
8. Tag `v0.1.0` after `make check` is green and `make smoke` passes.
