# Clinical Analytics Platform

**Ask questions of your own clinical dataset in plain English and get a checked statistical answer, on your own computer, with a local AI model and no code.**

It is for anyone in medicine who has a spreadsheet of patients and a question: medical students, residents, fellows, attendings and faculty. You open a CSV, Excel or SPSS file, type something like `compare outcome by diabetes`, and the app picks the analysis, runs it and states the result in one sentence with its test, p-value and n. Nothing leaves your machine: the language model runs locally through [Ollama](https://ollama.com).

```text
You:  compare outcome by diabetes
App:  outcome=1 was 43% in diabetes=0 and 100% in diabetes=1; the difference was
      statistically significant (chi-square p<0.001; n=80 patients). Unadjusted.
```

## What it does

- **Reads your file and works out its shape.** It detects patient IDs, outcomes and variable types, and runs data-quality checks on every upload (missing IDs, duplicates, out-of-range values).
- **Picks the analysis for you.** Counts, descriptive statistics, group comparisons (chi-square, t-test), correlations and Kaplan–Meier survival. Logistic regression needs you to name the predictors (`do age and hypertension predict outcome`); it does not choose them for you.
- **Shows its working.** Every answer comes with the plan it ran (outcome, grouping, filters, test). When it isn't sure what you meant, it shows the plan and asks before running it, rather than guessing.
- **Protects small groups.** Counts of 1 to 10 patients are shown as `<11`, including where they could be worked out from other numbers.
- **Keeps a conversation.** Past questions are saved in a sidebar, and follow-up questions use what you asked before.
- **Runs offline after setup.** The analysis engine, the embedding model and the language model all run on your machine.

## Where your data lives & privacy

The app runs locally after setup. It is **not a compliance certification**: whether you may use it with a given dataset is up to your institution's rules.

- **Your data is stored outside this repository**, in a per-user folder (`src/clinical_analytics/core/data_home.py`):

  | OS | Default data folder |
  |----|---------------------|
  | macOS | `~/Library/Application Support/ClinicalAnalytics` |
  | Windows | `%LOCALAPPDATA%\ClinicalAnalytics` |
  | Linux | `$XDG_DATA_HOME/clinical-analytics` (default `~/.local/share/clinical-analytics`) |

  Uploads, the analytics database, chat sessions and prompt-learning files go under `<data folder>/data/`. Set `CLINICAL_ANALYTICS_DATA_HOME` to use another folder, or `DATABASE_URL` to keep chat sessions in a different database (default: SQLite at `<data folder>/data/clinical_analytics.db`).
- **Encrypt the disk.** The app does not encrypt data at rest. Turn on FileVault (macOS), BitLocker (Windows) or full-disk encryption (Linux).
- **The language model is local.** Questions the built-in matcher cannot handle go to Ollama at `OLLAMA_BASE_URL` (default `http://localhost:11434`). Prompts can include column names, category labels and summary results. **Do not point `OLLAMA_BASE_URL` at a remote host**: that would send them off your machine.
- **Other users and other machines can't reach it.** The backend listens on `127.0.0.1` only and needs a fresh random token for each launch, and the data folder is readable only by your account. Programs running as you still can read your data, and `make run-api` (development only) turns the token off.
- **Setup needs the internet; daily use does not.** Installing downloads Python and Node packages and the `all-MiniLM-L6-v2` embedding model (pinned revision, SHA-256 checked). Ollama models are a one-time download.

More detail: [Local data security](docs/security/LOCAL_DATA_SECURITY.md) and the [threat model](docs/security/THREAT_MODEL.md).

## Prerequisites

- macOS 13 (Ventura) or later on Apple silicon or Intel; Windows 10/11 x64; or Linux x86_64/aarch64 with glibc 2.28+. These are the platforms ONNX Runtime 1.23.2, the embedding runtime, ships wheels for ([packaging notes](docs/development/packaging.md)).
- [uv](https://docs.astral.sh/uv/). It installs a suitable Python (3.11 to 3.13) for you.
- Node.js 22.12 or later, with npm.
- `make` on macOS and Linux. Windows users can skip it (see below).
- Optional but recommended: [Ollama](https://ollama.com/download), for questions the built-in matcher cannot handle.

## Install

```bash
git clone https://github.com/jtouley/md_data_explorer.git
cd md_data_explorer
make install-dev                  # Python packages + the embedding model
cd electron && npm ci && cd ..    # desktop app packages
```

**Windows without `make`:** replace `make install-dev` with these two commands. The second one matters: without the embedding model the backend refuses to start.

```powershell
uv sync --extra dev --group dev
uv run python scripts/fetch_embedding_model.py
```

## Launch

**Any OS:**

```bash
cd electron
npm start
```

The desktop app starts its own backend on `127.0.0.1` (port 8000, or a free port if 8000 is taken) and stops it when you quit. Start Ollama yourself if you want it: `ollama serve`, then `ollama pull llama3.1:8b` once.

**macOS and Linux, one command:** `./scripts/run_app.sh` does the same and also manages Ollama. On macOS it installs Ollama with Homebrew if it is missing; on Linux, install Ollama yourself first. If Ollama has no models, the script downloads the smaller `llama3.2:3b`; the app's preferred model is `llama3.1:8b`, so run `ollama pull llama3.1:8b` for better results.

## Your first questions

1. In the desktop app, click **Use the sample cohort**. It loads a synthetic cohort of 80 made-up patients that ships with the app (`src/clinical_analytics/resources/demo/sample_cohort.csv`: patient ID, outcome, age, sex, diabetes, hypertension).
2. Try these, roughly simplest first:

   | Ask | The answer starts with |
   |-----|------------------------|
   | `how many patients` | Total count: 80 patients. |
   | `average age by sex` | Mean age by sex: F 50.75, M 52.00 (n=80 patients). Unadjusted. |
   | `compare outcome by diabetes` | outcome=1 was 43% in diabetes=0 and 100% in diabetes=1; … chi-square p<0.001 |
   | `compare age by outcome` | Mean age was 19.84 higher in outcome=1 than outcome=0 (95% CI 14.54 to 25.13; Welch t-test p<0.001; n=80 patients) |

   These answers came out the same with Ollama off and with Ollama running `llama3.2:3b`.
3. When you're ready, click **Upload a file…** and pick your own `.csv`, `.xlsx`, `.xls` or `.sav` file. See [Uploading data](docs/getting-started/uploading-data.md) for how to lay it out.

If a question uses a word the app can't tie to a column or value, it says which word and asks you to rephrase, rather than answering a different question. Using your column names helps.

## What it doesn't do (yet)

The aim is a general-purpose data assistant for clinicians, able to do what commercial natural-language data tools do, but on local models. The plan is in [Purpose and steel thread](docs/vision/PURPOSE.md) and [ADR013](docs/implementation/ADR/ADR013.md). It isn't there yet:

- **One table per upload.** Multi-table ZIP uploads exist in the storage layer but the desktop app rejects them; join tables into one file first.
- **No charts.** Answers are a sentence plus a table.
- **No free-form SQL or code.** Questions map to the analyses listed above.
- **Regression needs the predictors named, and is fragile.** `what predicts outcome` is refused with "No predictors specified for outcome. Name them, e.g. 'do age and treatment predict outcome?'", with Ollama off and with `llama3.2:3b` running. On the sample cohort, the named-predictor phrasings tried so far (`do age and hypertension predict outcome`, `do age and sex predict outcome`) were held, and refused after confirmation.

## Troubleshooting

| Problem | What to do |
|---------|------------|
| Backend says the embedding model is missing | Run `make install-dev` (or `uv run python scripts/fetch_embedding_model.py`). |
| `run_app.sh` says another program is using port 8000 | Close it, or use `cd electron && npm start`, which picks a free port. |
| Complex questions aren't understood | Start Ollama. `GET /health` on the backend reports `ollama_reachable` and `ollama_default_model_available`. |
| `Module not found` | `make install-dev` |

## Developers

```bash
make install-dev          # Python deps incl. dev tools and the embedding model
make test-fast            # Fast tests (the CI test job)
make check-fast           # Lint, format check, tracked-data guard, SPDX headers (no tests)
make check                # Lint, format check, type check, tests
make format && make lint-fix
make run-api              # API at http://127.0.0.1:8000 with --reload, NO auth: never use real patient data
```

To point the desktop app at a `make run-api` backend: `cd electron && CLINICAL_ANALYTICS_EXTERNAL_BACKEND=1 CLINICAL_ANALYTICS_DEV_NO_AUTH=1 npm start`. OpenAPI docs are at `http://127.0.0.1:8000/api/docs`.

Run tests through `make` targets, not `pytest` directly; pass extra flags with `PYTEST_ARGS`, e.g. `make test-core PYTEST_ARGS='-k foo -x'`. See [tests/AGENTS.md](tests/AGENTS.md) and [docs/development/testing.md](docs/development/testing.md). `make check-data` fails if git tracks data files: never copy patient files into this repo.

```
src/clinical_analytics/
├── api/            # FastAPI app, routes, auth
├── core/           # Semantic layer, question engine, query service, data folder
├── analysis/       # Statistical methods
├── contracts/      # Per-upload data contracts and data-quality checks
├── datasets/       # Uploaded-dataset definitions, registry, bundled sample cohort
├── storage/        # DuckDB/Parquet dataset store
├── eval/           # Question evaluation harness
├── cli/            # Headless upload preview
├── ui/             # Upload storage and enrichment helpers (no UI framework)
└── resources/      # Config, sample cohort, embedding model
electron/           # Desktop app (Vite + Electron)
tests/              # Pytest suites (see tests/AGENTS.md)
```

Architecture: [As-built architecture](docs/architecture/AS_BUILT_ARCHITECTURE.md) · [Module boundaries](docs/architecture/ARCHITECTURE_BOUNDARIES.md) · [Docs index](docs/index.md). Older docs mention a Streamlit UI and built-in COVID-MS, Sepsis and MIMIC-III datasets; both are gone ([omnibus spec](docs/specs/omnibus_electron_streamlit_cutover.md)).

## Contributing and support

- **Questions and bugs:** open a [GitHub issue](https://github.com/jtouley/md_data_explorer/issues) using the templates. Never include real patient data.
- **Security or privacy problems:** report them privately as [SECURITY.md](SECURITY.md) describes, not in a public issue.
- **Contributing:** see [CONTRIBUTING.md](CONTRIBUTING.md) and the [Code of Conduct](CODE_OF_CONDUCT.md).

## License

[MIT](LICENSE). Dataset files and uploaded content stay under their own licenses and terms; the application license does not relicense them.
