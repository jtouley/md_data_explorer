# Clinical Analytics Platform

Upload a clinical cohort and ask statistical questions in plain language.

**V1 surface (default):** **Add Your Data** + **Ask Questions** only (`V1_MVP_MODE`).
**Shell:** Streamlit via `make run`. Electron + FastAPI exist but are **experimental**.

---

## 60-second demo

```bash
make install-dev
make run
```

In the browser:

1. Open **Add Your Data** → click **Load sample dataset** (skips the mapping wizard).
2. Open **Ask Questions** and try:

| Ask | What you should see |
|-----|---------------------|
| `how many patients` | Count claim + table |
| `compare age by sex` | Claim with effect / interval / n above the table |
| `what predicts outcome` | Claim with top odds ratio above the predictors table |

Ollama is optional. Canned / pattern questions work without an LLM.

Headless check (no Streamlit):

```bash
make headless-preview
# or: uv run python scripts/headless_preview.py --csv data/demo/sample_cohort.csv
```

---

## Requirements

- **Python 3.11+**
- **[uv](https://docs.astral.sh/uv/)** package manager
- **make**

Do **not** use bare `pip` / `python` for project installs — use `make install-dev` / `uv`.

---

## Setup & run

```bash
make install-dev          # deps + editable install
make run                  # Streamlit on http://localhost:8501
make run-api              # FastAPI on http://127.0.0.1:8000 (experimental)
```

Convenience launcher (optional Ollama lifecycle):

```bash
./scripts/run_app.sh
```

---

## What this product is (AS_BUILT)

- **Uploads only** — there is no built-in COVID-MS / Sepsis sidebar dataset.
- Bundled **sample cohort** lives at `data/demo/sample_cohort.csv` for one-click demos.
- Uploaded files persist under `data/uploads/`.
- Ask Questions uses a closed intent set (DESCRIBE, COMPARE_GROUPS, FIND_PREDICTORS, SURVIVAL, CORRELATIONS, COUNT) through the semantic layer.
- Low-confidence plans show Confirm / Reject and do not auto-execute.
- Successful compare / predictors / survival answers lead with a plain-language **claim**.

Out of scope for this MVP: Electron Phase 10 cutover, OMOP/ATLAS parity, multi-table MIMIC cohorting as the primary path.

---

## Development

```bash
make test-fast            # skip slow tests
make check-fast           # lint + format check (no tests)
make check                # full gate
make headless-preview     # CSV → JSON preview without UI
```

Testing rules: see [`tests/AGENTS.md`](tests/AGENTS.md). Always use `make test-*`, not raw `pytest`.

---

## Docs

- [Installation](docs/getting-started/installation.md)
- [Quick start](docs/getting-started/quick-start.md)
- [Uploading data](docs/getting-started/uploading-data.md)

---

## License

See repository license file.
