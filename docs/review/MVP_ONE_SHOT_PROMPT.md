# One-shot MVP prompt

Paste everything between the two `=====` lines into a fresh Claude Code session at the repo root. In Claude Code, `/mvp` loads the same text from `.claude/commands/mvp.md`.

The prompt assumes:

- `main` is at `9eb4a9d` or later;
- PRs #59, #60 and #61 are still open;
- the decisions in [`2026-10-26-repo-review.md`](2026-10-26-repo-review.md) stand: Streamlit is the MVP shell and Electron is frozen.

It works on one integration branch and opens one PR. It never merges into `main`, never rewrites history, and never deletes remote branches. Those steps are left to the owner.

=====

You are taking `md_data_explorer` (Clinical Analytics Platform) to a tagged-ready **v0.1.0 MVP** in one session. Read `.claude/CLAUDE.md`, `tests/AGENTS.md` and `docs/review/2026-10-26-repo-review.md` first. Their rules are binding:

- Polars, lazy by default; never `map_elements`;
- `structlog`, no `print`;
- specific exceptions only;
- AAA tests named `test_unit_scenario_expectedBehavior`;
- reuse the `tests/conftest.py` factory fixtures;
- run tests **only** through `make` targets, never `pytest` directly.

## MVP definition (the acceptance bar)

On a laptop with no internet and no LLM running:

1. `uv sync && make demo-data && make smoke` passes.
2. `make run` (Streamlit) lets a user upload a CSV or XLSX, or pick the bundled synthetic demo, and ask questions of all six intents: DESCRIBE, COMPARE_GROUPS, FIND_PREDICTORS, SURVIVAL, CORRELATIONS, COUNT.
3. A plan whose confidence is below `auto_execute_confidence_threshold` (0.75, from `config/nl_query.yaml`) shows a one-line plan with Confirm and Reject buttons, and does **not** execute until the user confirms. High-confidence plans run with no extra click.
4. Every successful result **leads with one plain-language claim**, built by one pure function and shared by the UI, the API and the CLI. The claim states:
   - the effect and its 95% CI where applicable;
   - the cohort N after filters;
   - the test used;
   - what was not adjusted for.
5. Any displayed patient count from 1 to 10 renders as `<11`. Zero stays 0.
6. `scripts/headless_ask.py --dataset demo --query "..." [--confirm]` and `POST /api/queries` return the same plan, the same claim and the same numbers as the UI for the same question.
7. `make check-fast` and `make test-fast` are green. Any failure already present on `main` at step 0 must either be fixed or be listed in the PR body with its root cause.

## Steps

Work through these in order. Commit after each one with a conventional-commit message. Run `make format && make lint-fix && make check-fast` before every commit, and run the relevant `make test-*` target as well.

### 0. Baseline

- Create the branch: `git checkout -b mvp/v0.1 origin/main`.
- Run `uv sync --extra dev --group dev`, then `make check-fast`, then `make test-fast`.
- Save the list of failing test IDs to `docs/review/mvp_baseline.txt`. These are pre-existing failures; do not mistake them for regressions later.

### 1. Bring in the open PRs, in this order

- `git merge --no-ff origin/chore/test-benchmark-dry-collisions` (PR #61).
- `git merge --no-ff origin/gh-52-claim-first-answers` (PR #59).
- `git merge --no-ff origin/review/repo-heal-golden-questions` (PR #60).

Resolve conflicts by keeping the behaviour of both sides. If a conflict forces you to choose one side's logic and lose the other's behaviour, stop and report it.

Then fix #60's CI failure. `mypy` fails at `src/clinical_analytics/eval/catalog.py:89`: `series.max()` returns a wide union that the code compares with `int`. Narrow it with `isinstance(maximum, int | float)`. Do not use `type: ignore`. Run `make type-check`.

**Do not** merge `feat/electron-ui-migration` (PR #62). From it, take only the `.claude/settings.json` and `.claude/hooks/*.sh` fix:

- `git checkout origin/feat/electron-ui-migration -- .claude/settings.json .claude/hooks/`.
- Check the result against the current Claude Code hooks schema: `PreToolUse`/`PostToolUse`, `matcher`, `hooks: [{type: "command", command}]`, and JSON on stdin.

### 2. Synthetic demo data, and take the patient data out of the tree

- Add `scripts/make_demo_data.py`. It writes deterministic, seeded synthetic cohorts with Polars to `data/demo/`:
  - `demo.csv`: about 600 rows, rich enough for all six intents. Include:
    - `patient_id`, age, sex;
    - a binary treatment;
    - two continuous labs (for example LDL and HDL), one of them correlated with treatment;
    - a 3-level categorical variable;
    - a binary outcome driven by age and treatment through a known logistic model;
    - time-to-event plus an event flag.

    Also include about 5% nulls in one lab, and one 1/2-coded yes/no column with the code in the header, like the real uploads.
  - `dexa_like.csv` and `statin_like.csv`: synthetic look-alikes of the two `data/analytics.duckdb` tables that PR #60's golden suite reads. Use the same column names and dtypes and plausible ranges. **Do not read or copy any row values from the real tables.** Use only `DESCRIBE` or the schema.
- Add a `demo-data` Makefile target.
- Repoint `src/clinical_analytics/eval/catalog.py` and the golden suite at the synthetic files:
  - keep `gdsi` (public dataset) and `mimic_patients` (MIMIC-IV demo, ODbL) where they are already sourced;
  - replace `dexa` and `statin` with the synthetic look-alikes;
  - regenerate `tests/eval/golden/{dexa,statin}.yaml` with the generator in PR #60, so the expected values come from the SQL oracle over the synthetic data.
- `git rm --cached data/analytics.duckdb`. Add `data/*.duckdb` and `data/demo/` to `.gitignore`; the data is regenerated, not committed. Make every test and the golden suite create the data they need, through `make demo-data` or a conftest session fixture.
- Rename `data/dicitonaries/` to `data/dictionaries/`, and update `[tool.mutmut].also_copy` and every reference to it.
- **Do not rewrite git history.** In the PR body, state that `data/analytics.duckdb` remains in history, and give the owner the `git filter-repo` command.

### 3. Headless proof (#51) and confirm flag (#58)

- Add a `confirm: bool = False` parameter to `QueryService.ask`. When it is `True`, a complete, valid, low-confidence plan executes and gets a `run_key`. Incomplete or invalid plans still refuse.
- Expose `confirm` on the API query request schema and route, and as a `--confirm` flag on `scripts/headless_ask.py`.
- Add `--dataset demo` support to the CLI. It must go through the **real upload path** (`UserDatasetStorage.save_upload` → `UploadedDataset.get_semantic_layer` → `QueryService.ask`), using a temporary upload dir. Do not use a hand-built `SemanticLayer` config.
- Add a `make smoke` target. It runs `make demo-data`, then one CLI question per intent against `demo`, and asserts that each one exits 0 and prints JSON containing `intent`, `claim` and `n`.
- Add `tests/e2e/test_smoke_headless.py`. It runs the CLI as a subprocess for each of the six intents, plus a low-confidence question run with and without `--confirm`. Mark it `@pytest.mark.integration`. Do not import Streamlit.
- Add an API test that uses FastAPI's `TestClient` against the same demo upload, with **no mocks** of storage or the query service.

### 4. Claim-first answers (#52, remaining half), plus small-cell suppression

- Add a pure function `build_claim(result: dict[str, Any], plan: QueryPlan) -> str` in `src/clinical_analytics/core/result_interpretation.py`, or a new `core/claim.py` if that file already does something else. It returns one sentence per intent. Examples:
  - **COMPARE_GROUPS:** "Mean LDL was 12.3 mg/dL lower with treatment (95% CI −15.1 to −9.5; Welch t-test; n=574 after excluding 26 with missing LDL; not adjusted for age or sex)."
  - **FIND_PREDICTORS:** the top odds ratio with its CI, the model N, and the covariates included.
  - **SURVIVAL:** the median survival per group and the log-rank p-value.
  - **CORRELATIONS:** the strongest r with its CI and n.
  - **DESCRIBE / COUNT:** n, and the statistic.

  Take every number from the result payload. Never recompute in the formatter. If a field is missing, leave out that clause rather than invent it.
- Add `suppress_small_counts(value: int, threshold: int = 11) -> int | str`. Put the threshold in `config/validation.yaml` or `config/ui.yaml`, not in code. Apply it in the claim and in the displayed count tables.
- Render the claim as the **first element** of a successful Ask Questions result, above the table and the chart. Return it as `claim` from `QueryService`, the API and the CLI.
- Write parametrized unit tests per intent. Include:
  - a missing CI;
  - zero exclusions;
  - n below 11;
  - a null effect.

  Also write a page test asserting that the claim renders before the table, reusing the patterns in `tests/unit/ui/pages/test_ask_questions_execution_hold.py`.

### 5. CI and docs

- `.github/workflows/docs.yml`:
  - Python `3.11`;
  - install with `astral-sh/setup-uv` + `uv sync --group dev`;
  - build with `uv run mkdocs gh-deploy --force`.
- Add a `docs-build` job to `ci.yml` that runs `uv run mkdocs build`, so docs breakage shows on PRs.
- Add a `smoke` job to `ci.yml` that runs `make smoke`, after `quality`.
- Bind the API to `127.0.0.1` in the `api/main.py` `__main__` block and in `make run-api`. Replace the `print` calls in the lifespan with `structlog`.

### 6. README quickstart

Rewrite the top of `README.md` as:

- **Quickstart:** `uv sync` → `make demo-data` → `make run` → three example questions with the claim each one should produce.
- **Headless:** `make smoke`, and one `headless_ask.py` example.
- **Security posture:** local, single user. Do not expose ports 8501 or 8000. Uploaded data stays in `data/uploads/`.
- **Status:** Electron is in progress (see `docs/specs/omnibus_electron_streamlit_cutover.md`); it is not part of v0.1.

Delete the stale built-in-dataset text (COVID-MS, Sepsis, MIMIC-III), the "pandas" requirement, and the `pytest` instruction. Keep the license section.

### 7. Verify, then open one PR

- Run `make check-fast`, `make test-fast`, `make smoke` and `make type-check`. Compare the results against `docs/review/mvp_baseline.txt`.
- Launch `make run` headlessly with `streamlit run ... --server.headless true`. Using Playwright with the pre-installed Chromium, upload `data/demo/demo.csv`, ask "compare LDL by treatment", and save a screenshot to `docs/review/mvp_screenshot.png` showing the claim above the table.
- Re-read your full diff adversarially for anything that would make CI red, then push `mvp/v0.1`.
- Open **one** PR into `main` titled `feat: v0.1.0 MVP — headless proof, confirm gate, claim-first answers`. In the body:
  - each issue it closes (#51, #52, #58) and supersedes (#59, #60, #61);
  - the baseline vs. final test counts;
  - the history-rewrite instruction for `data/analytics.duckdb`;
  - an explicit note that #62 stays open and is not merged.
- Do not merge it, tag it, or close the other PRs. The owner does that.

## Stop and ask (do not guess)

- A merge conflict where both sides changed the same logic.
- Any test that only passes once you weaken an assertion.
- Any need to read row values from `data/analytics.duckdb`.
- Any change to the closed intent set, or letting an LLM write SQL or Python.

=====
