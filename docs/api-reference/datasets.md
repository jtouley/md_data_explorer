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

1. **Upload** - CSV, Excel, or ZIP file uploaded via Streamlit UI
2. **Schema Inference** - Automatic detection of patient ID, outcomes, time columns
3. **Registry** - Dataset registered in `DatasetRegistry` with inferred config
4. **Semantic Layer** - `SemanticLayer` initialized for NL query support

See [Dataset Registry](../architecture/dataset-registry.md) for details.
