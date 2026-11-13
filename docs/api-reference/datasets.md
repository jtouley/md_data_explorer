# Datasets API Reference

## Uploaded Datasets

The platform supports user-uploaded datasets only. Built-in datasets (COVID-MS, Sepsis, MIMIC-III) have been removed in favor of a self-service upload model.

::: clinical_analytics.datasets.uploaded.definition
    options:
      show_root_heading: true
      show_source: true

## Storage & Versioning

::: clinical_analytics.storage.versioning
    options:
      show_root_heading: true
      show_source: true

**Key Functions:**
- `compute_dataset_version()` - Compute content hash of canonicalized tables

## Dataset Registration

Datasets are registered automatically when uploaded via the UI:

1. **Upload** - CSV, Excel (`.xlsx`, `.xls`) or SPSS (`.sav`) file sent to `POST /api/datasets/upload` (the Electron app's upload area). Other extensions, ZIP included, are refused with 400.
2. **Schema Inference** - Automatic detection of patient ID, outcomes, time columns. A patient ID is an existing `patient_id` column, else a copy of the most unique integer or text column that is not a measurement (never a float, fractional or estimate/bound column such as `mx`, `upper`, `lower` or a rate); the source column is kept. Else `patient_id` is a hash of a combination of columns unique on every row (location x cause x sex x year), else a `row_number` column is added. A synthesized ID (a hash, a row number, or a copy of a column not named as a patient ID) keys rows: answers count rows, never patients.
3. **Registry** - Dataset registered in `DatasetRegistry` with inferred config
4. **Semantic Layer** - `SemanticLayer` initialized for NL query support

See [Dataset Registry](../architecture/dataset-registry.md) for details.

## Upload responses

`POST /api/datasets/upload` answers:

- **200, `status: "ready"`** - stored, and every error-level data check passed.
- **200, `status: "needs_review"`** - stored and queryable, but an error-level data check failed (for example `patient_id_no_duplicates`). `failed_checks` lists the check ids and `message` names each failed check and what to do next.
- **400** - the file cannot be read as a table (for example an Excel file with no header row), with a plain message.
- **409** - a dataset with this name already exists. Upload under another name (`dataset_name`), or delete the existing dataset first. The route has no overwrite option.
- **400** - the extension or size is not accepted.

Data checks at upload are warnings except the identifier checks (`*_not_null`, `*_no_duplicates`). A column with any empty value is a warning (never "passed"), and when a value has interval bound columns (`lower`/`upper`, or `<value>_lower`/`<value>_upper`) every row with all three present must have lower <= value <= upper (`<value>_interval`, a warning).

`GET /api/datasets/{dataset_id}` reports each column's stored type in `schema` (`integer`, `number`, `string`, `boolean`, `date`, `timestamp`, ...).
