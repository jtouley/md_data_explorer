# Contributing

Thanks for helping. Clinical Analytics runs on clinicians' own machines with their patients'
data, so two rules come before everything else:

1. **Synthetic data only.** Never commit, attach, paste or screenshot real patient data, in code,
   tests, issues, pull requests or logs. Use the synthetic kit (below) or the app's **Use the sample
   cohort** button.
2. **No small counts, no data leaving the machine.** Patient counts from 1 to 10 must never be
   shown or be recoverable, and nothing may send data to a non-loopback host. A change that weakens
   either is a security issue: report it as [SECURITY.md](SECURITY.md) describes, not in a public
   issue.

By contributing you agree to follow the [Code of Conduct](CODE_OF_CONDUCT.md). Contributions are
accepted under the project's [MIT licence](LICENSE).

## How changes get in

- **Bugs and ideas:** open an issue using the bug report or feature request template.
- **Pull requests:** for anything beyond a small fix, open an issue first so we can agree the
  approach. Branch from `main`, keep the PR to one concern, and fill in the pull request template.
  `main` only takes changes through pull requests; the maintainer ([MAINTAINERS.md](MAINTAINERS.md))
  reviews and merges.
- **CI must pass:** lint, format check, tracked-data guard, SPDX headers, type check, fast tests,
  coverage, the Electron quality gate and the installed-wheel smoke test (`.github/workflows/ci.yml`).

## Set up

Prerequisites are in the [README](README.md#prerequisites): [uv](https://docs.astral.sh/uv/),
Node.js 22.12 or later with npm, and `make`.

```bash
git clone https://github.com/jtouley/md_data_explorer.git
cd md_data_explorer
make install-dev                 # uv sync --extra dev --group dev
cd electron && npm ci && cd ..
make fetch-embedding-model       # already done by install-dev; re-run after a model bump
```

## Run the app

```bash
./scripts/run_app.sh             # macOS/Linux: Ollama (optional), the API and the desktop window
```

`cd electron && npm start` on its own also works on any OS: the desktop app starts its own
backend with a fresh API token.

Or in two terminals, against a backend that reloads on code changes:

```bash
make run                         # API on http://127.0.0.1:8000 with --reload, dev mode (no API token)
cd electron && CLINICAL_ANALYTICS_EXTERNAL_BACKEND=1 CLINICAL_ANALYTICS_DEV_NO_AUTH=1 npm start
```

`make run` turns off API authentication for development. Never use it with real patient data.
In the window, click **Use the sample cohort** and ask `compare outcome by diabetes`.

## Before you push

Pre-commit hooks are optional (`make install-pre-commit`). Whether or not you use them, run:

```bash
make format && make lint-fix     # ruff format + autofix
make check-fast                  # lint, format check, tracked-data guard, SPDX headers
make type-check                  # mypy over src/clinical_analytics
make test-fast                   # fast Python tests, in parallel
```

If you touched `electron/`:

```bash
cd electron
npm run lint                     # eslint src/
npm run format:check             # prettier
npm test                         # vitest
```

`make test-electron-quality` runs the full Electron gate CI runs (Vitest coverage plus the
Playwright suites; it needs `npx playwright install chromium` once).

Narrower Python targets: `make test-core`, `make test-analysis`, `make test-api`, `make test-ui`,
`make test-contracts`. Pass extra pytest flags with
`PYTEST_ARGS`, e.g. `make test-core PYTEST_ARGS='-k semantic -x'`. `make help` lists everything.

## Tests

The full rules are in [tests/AGENTS.md](tests/AGENTS.md) and
[.claude/CLAUDE.md](.claude/CLAUDE.md). The ones reviews check first:

- **Use `make` targets, never `pytest` or `uv run pytest` directly.**
- **Every change to behaviour comes with a test.** Coverage must stay at or above 67%
  (`make test-cov-check`) and must not regress against the baseline (`make test-cov-diff`).
- **Reuse fixtures.** Search `tests/conftest.py` before writing one. Use the factory fixtures
  `make_semantic_layer`, `make_cohort_with_categorical`, `make_multi_table_setup` and
  `mock_semantic_layer` rather than building a SemanticLayer, cohort or multi-table setup inline.
  `make pre-commit-check` flags violations.
- **Arrange / Act / Assert**, with the three sections marked, and names of the form
  `test_unit_scenario_expectedBehavior`.
- **Parametrize edge cases:** nulls, empty inputs and boundary values. For anything that
  displays counts, include the 10/11 boundary of small-count suppression.
- **Polars, not pandas:** `pl.testing.assert_frame_equal`, `df.height` (not `len(df)`).
- **Mark data-loading tests** `@pytest.mark.slow` and `@pytest.mark.integration`; fast unit tests
  use `get_sample_datasets()`.

## Test data

The only data files git may track are the synthetic kit written by
`scripts/make_synthetic_data.py` (`tests/fixtures/synthetic/` and the demo cohort). `make check-data`
(part of `make check-fast` and CI) fails if any other data file, contract, quality report, log or
local database is tracked. To add a dataset, follow
[tests/fixtures/synthetic/README.md](tests/fixtures/synthetic/README.md), then run `make golden`.

## Code style

- Python: Polars (lazy by default), full type hints, specific exceptions, `structlog` logging.
  See [.claude/CLAUDE.md](.claude/CLAUDE.md).
- Every file under `src/` (`.py`) and `electron/src/` (`.js`) starts with an SPDX licence line,
  `# SPDX-License-Identifier: MIT` or `// SPDX-License-Identifier: MIT`. `make check-spdx` checks
  it; `uv run python scripts/check_spdx_headers.py --fix` adds missing ones. Do not add it to
  third-party files (see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)).
- GitHub Actions are pinned to full commit SHAs with the version in a comment, and workflows
  default to `permissions: contents: read`. Dependabot keeps the pins current.
