# Quick start

## 1. Install and run

```bash
make install-dev
make run
```

Open `http://localhost:8501`.

## 2. Load data (one click)

On **Add Your Data**, click **Load sample dataset**.

That registers `data/demo/sample_cohort.csv` into upload storage with a complete variable mapping — no wizard.

Or upload your own CSV / Excel / SPSS and complete the mapping steps.

## 3. Ask questions

Open **Ask Questions**. Examples that work without Ollama via pattern / semantic matching:

- `how many patients`
- `compare age by sex`
- `what predicts outcome`

Successful answers lead with a **claim** (effect, interval when present, n, what was not adjusted). Low-confidence plans wait for **Confirm** / **Reject**.

## Headless path

```bash
make headless-preview
```

See [Installation](installation.md) and [Uploading data](uploading-data.md).

## Experimental

- **FastAPI:** `make run-api`
- **Electron:** see `electron/` — not required for the Streamlit MVP
