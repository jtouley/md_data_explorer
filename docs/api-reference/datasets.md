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

1. **Upload** - CSV, Excel (`.xlsx`, `.xls`), SPSS (`.sav`), SAS transport (`.xpt`) or ZIP file sent to `POST /api/datasets/upload`. A ZIP holds tables (`.csv`, `.tsv`, `.psv`, gzipped or not, `.xlsx`, `.xls`, `.xpt`, `.sav`, also inside one nested ZIP) and documentation (`.pdf`, `.md`, `.txt`), which becomes the dataset's doc context. Numbered files with the same columns (one per patient, as PhysioNet ships them) are read as one table with `patient_id` from the file name. Several tables are joined into one cohort when one of them has one row per patient or stay with an ID the others share; otherwise the upload fails and says so.
2. **Schema Inference** - Automatic detection of patient ID, outcomes, time columns
3. **Registry** - Dataset registered in `DatasetRegistry` with inferred config
4. **Semantic Layer** - `SemanticLayer` initialized for NL query support

See [Dataset Registry](../architecture/dataset-registry.md) for details.
