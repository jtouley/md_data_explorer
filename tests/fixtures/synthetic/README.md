# Synthetic golden kit

The only data files this repo may track. Every one is written by
`scripts/make_synthetic_data.py` (seeded numpy draws or plain arithmetic, no real patients),
and regenerating it must reproduce the committed bytes exactly.

**Entry point:** `make golden`. It regenerates the kit in a temp directory, fails if any
committed file differs, then runs the generator and data-guard tests, the golden statistics tests and the
golden question suite (slow tests included).

## Datasets

| File | Rows | What it exercises | Golden tests |
| --- | --- | --- | --- |
| `cohort.csv` | 400 | Welch t-test with ~5% missing LDL (planted -18 drug effect), one-way ANOVA by site (planted +0/+4/+9), chi-square on a 1/2/3-coded DEXA header, 1=Yes/2=No statin coding, logistic regression with planted log-odds, Kaplan-Meier and log-rank on right-censored follow-up | `TestGoldenComparison`, `TestGoldenPredictor`, `TestGoldenSurvival`, coded-predictor cases in `TestEdgeInputs` |
| `unequal_groups.csv` | 45 | n=5 vs n=40 with unequal variances: Welch, not pooled | `test_compare_unequalVarianceSmallGroup_usesWelchNotPooled` |
| `sparse_cells.csv` | 20 | 2x2 table with expected counts below 5: Fisher's exact test, small-cell suppression | `test_compare_sparse2x2_*`, `TestGoldenPresentation` |
| `separation.csv` | 60 | A predictor that perfectly separates the outcome: logistic regression must refuse | `test_predict_perfectSeparation_refusesWithReason` |
| `../../../data/demo/sample_cohort.csv` | 80 | The app's "Load sample" cohort; 2x2 chi-square of outcome by hypertension. Two planted exceptions (P011, P020) keep the outcome from being a deterministic function of age and diabetes | `test_compare_demoOutcomeByHypertension_matchesContingencyReference` |

Tests are in `tests/analysis/test_golden_statistics.py` (load with the `load_synthetic_dataset`
fixture from `tests/conftest.py`) and `tests/scripts/test_make_synthetic_data.py`. The NL golden
questions (`tests/eval/golden_questions.yaml`, `config/golden_examples.yaml`) are hand-written
questions with expected intents and run against an in-memory synthetic semantic layer.

## How expected values are derived

Never from the code under test. Each golden recomputes its answer independently from the CSV:
counts and group splits with Polars, then the statistic with scipy, statsmodels or lifelines
called directly on numpy arrays. Where the generator planted an effect, the test also checks
that the reported confidence interval covers it.

## Adding a golden case

1. If no existing dataset has the property you need, add a `make_*` function to
   `scripts/make_synthetic_data.py` and register it in `build_kit()`, after the existing
   entries, so earlier seeded draws do not change.
2. Run `uv run python scripts/make_synthetic_data.py` and commit the new CSV.
3. Add its path to `SYNTHETIC_KIT` in `scripts/check_no_data_files.py` and a `!` line to the
   data-file block at the end of `.gitignore`. A test fails if the guard and generator disagree.
4. Write an AAA test that computes the expected value with an independent library, and add a
   row to the table above.
5. Run `make golden` and `make check-data`.

## Real-data goldens never go in the repo

Golden sets, aggregates or questions computed from real uploads stay in the per-user data home.
`scripts/check_no_data_files.py` (`make check-data`, run in CI) fails on any tracked golden
YAML/JSON outside this kit and the two hand-written question files, on any CSV here that the
generator does not write, and on overlays, settings, session databases, store backups, logs and
`.env` files, even when they were added with `git add -f`.
