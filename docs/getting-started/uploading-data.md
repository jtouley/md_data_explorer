# Uploading data

## V1 product surface

The app is **uploads-only**. There is no built-in COVID-MS or Sepsis sidebar dataset.

## Fastest path: sample cohort

On **Add Your Data**, click **Load sample dataset**.

- Source file: `data/demo/sample_cohort.csv`
- Columns: `patient_id`, `outcome` (0/1), `age`, `sex`, `diabetes`, `hypertension`
- Mapping wizard is skipped — metadata is pre-wired for Ask Questions

## Supported formats (manual upload)

- CSV (`.csv`)
- Excel (`.xlsx`, `.xls`)
- SPSS (`.sav`)
- ZIP multi-table (when `MULTI_TABLE_ENABLED`)

## Minimum requirements

1. A unique patient / subject identifier
2. At least one outcome (binary 0/1 is typical for predictors)
3. One or more clinical / demographic variables

Example:

```text
patient_id,age,sex,diabetes,hypertension,outcome
P001,45,M,0,1,0
P002,67,F,1,1,1
```

## After upload

1. Review detected variable types
2. Confirm patient ID and outcome mapping (skipped for the sample cohort)
3. Save — data persists under `data/uploads/`
4. Open **Ask Questions**

## Headless upload + preview

Without Streamlit:

```bash
make headless-preview
# ARGS='--csv /path/to/file.csv' make headless-preview
```

Returns JSON with `columns`, `row_count`, and a short `preview` array.
