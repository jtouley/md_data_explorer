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

1. **Upload** - CSV, Excel (`.xlsx`, `.xls`), SPSS (`.sav`), SAS transport (`.xpt`) or ZIP file sent to `POST /api/datasets/upload`. A ZIP holds tables (`.csv`, `.tsv`, `.psv`, gzipped or not, `.xlsx`, `.xls`, `.xpt`, `.sav`, also inside one nested ZIP) and documentation (`.pdf`, `.md`, `.txt`), which becomes the dataset's doc context. Numbered files with the same columns (one per patient, as PhysioNet ships them) are read as one table with `patient_id` from the file name. Several tables are joined into one cohort when one of them has one row per patient or stay with an ID the others share; otherwise the upload fails and says so. A file that cannot be used as sent (unreadable, damaged or password-protected, no patient ID, tables with no shared ID) is answered with 400 and a message that says what to fix; 500 is kept for the app's own faults.
2. **Schema Inference** - Automatic detection of patient ID, outcomes, time columns
3. **Registry** - Dataset registered in `DatasetRegistry` with inferred config
4. **Semantic Layer** - `SemanticLayer` initialized for NL query support

See [Dataset Registry](../architecture/dataset-registry.md) for details.

## Loading from a link

`POST /api/datasets/from-url` with `{"url": "https://…", "dataset_name": "optional"}` downloads one public data file (CDC, CMS, WHO, IHME exports) and hands the bytes to the upload route unchanged, so formats, limits and checks are the same as for a chosen file. The response is the upload response plus `source`: the link, the final address after redirects, the download time (UTC), the size, the SHA-256 and a `methods_text` sentence citing the link as given (a redirect target is often an expiring signed link), plus `source_recorded`, false if it could not be saved. The same record is saved on the dataset (metadata key `source`, shown in Cohort details).

The download is locked down and cannot be loosened by a caller (`storage/url_fetch.py`): HTTPS on port 443 only, certificate checked; every resolved address must be public (no loopback, private, link-local, carrier NAT, multicast or reserved ranges, including IPv6 forms that embed them), the connection goes to the checked address, and each of at most 5 redirects is checked again; no proxy, cookies, credentials or Referer; at most the upload size limit (100 MB), compressed transfers included; 15 s to connect, 30 s per read and 5 minutes overall (a timer cuts the connection at the deadline); web pages are refused as not data. Errors come back as 400 (bad or non-public link), 409 (a cohort with that name exists), 413 (too large), 415 (not a data file), 502 (the site failed or refused) or 504 (too slow), each with a sentence for the user. The app does not use a system proxy for these downloads; on a network that requires one, download the file in a browser and upload it.
