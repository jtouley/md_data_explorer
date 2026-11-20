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

1. **Upload** - CSV, tab-separated text (`.tsv`, or a tab-separated `.txt` such as a CDC WONDER export), Excel (`.xlsx`, `.xls`), SPSS (`.sav`), SAS transport (`.xpt`) or ZIP file sent to `POST /api/datasets/upload`. In delimited text, a notes block under the data (from a `---` line, as CDC WONDER writes) is kept as the dataset's documentation, never as rows; subtotal rows (a first column `Notes` holding `Total`) are left out and counted in `cohort_notes`; and values the source withheld (`Suppressed`, `Unreliable`) in a number column are stored as missing, never 0, and reported in `type_notes`. A ZIP holds tables (`.csv`, `.tsv`, `.psv`, gzipped or not, `.xlsx`, `.xls`, `.xpt`, `.sav`, also inside one nested ZIP) and documentation (`.pdf`, `.md`, `.txt`), which becomes the dataset's doc context. Numbered files with the same columns (one per patient, as PhysioNet ships them) are read as one table with `patient_id` from the file name. Several tables are joined into one cohort when one of them has one row per patient or stay with an ID the others share; otherwise the upload fails and says so. A file that cannot be used as sent (unreadable, damaged or password-protected, tables with no shared ID) is answered with 400 and a message that says what to fix; 500 is kept for the app's own faults.
2. **Schema Inference** - Automatic detection of patient ID, outcomes, time columns. A patient ID is an existing `patient_id` column, else a copy of the most unique integer or text column that is not a measurement (never a float, fractional or estimate/bound column such as `mx`, `upper`, `lower` or a rate); the source column is kept. Else `patient_id` is a hash of a combination of columns unique on every row (location x cause x sex x year), else a `row_number` column is added. A synthesized ID (a hash, a row number, or a copy of a column not named as a patient ID) keys rows: answers count rows, never patients.
3. **Registry** - Dataset registered in `DatasetRegistry` with inferred config
4. **Semantic Layer** - `SemanticLayer` initialized for NL query support

See [Dataset Registry](../architecture/dataset-registry.md) for details.

## Upload responses

`POST /api/datasets/upload` answers:

- **200, `status: "ready"`** - stored, and every error-level data check passed.
- **200, `status: "needs_review"`** - stored and queryable, but an error-level data check failed (for example `patient_id_no_duplicates`). `failed_checks` lists the check ids and `message` names each failed check and what to do next.
- **400** - the file cannot be used as sent (unreadable, damaged or password-protected, a ZIP whose tables share no ID, no tables), or it has a header and no data rows, with a plain message that says what to fix. An upload with no data rows is never stored.
- **409** - a dataset with this name already exists. Upload under another name (`dataset_name`), or delete the existing dataset first. The route has no overwrite option.
- **400** - the extension or size is not accepted.

A file with no column that identifies a patient still uploads (a `row_number` column is added, or a combination of columns tells the rows apart). Its 200 response carries `unit_note`, which says so in plain words: answers count rows, not patients, and adding a `patient_id` column with one ID per patient gives patient counts. `unit_note` is empty when `patient_id` names patients.

### "Is each row one patient?"

When the rows are told apart by one column that is unique on every row but not named as a patient ID (or by nothing, so a `row_number` was added), and no ID-like column repeats, the 200 response also carries `one_row_per_patient: {"status": "pending", "confirmation": "<token>"}`. It is `null` when the question is not asked: `patient_id` names patients, an ID-like column repeats (visit or test rows), rows are told apart only by a combination of columns, or the user already said yes for this version.

`POST /api/datasets/{dataset_id}/one-row-per-patient` with `{"confirmation": "<token>", "one_row_per_patient": true|false}` records the answer for the dataset's current version and returns `{"dataset_id": ..., "counts": "patients"|"rows"}`.

- `true` stores the version's one-row-per-patient grain, the same declaration governed SQL stores when its refusal is confirmed (same token, `contracts.declarations.confirmation_token`). Questions then count patients, and so does governed SQL.
- `false` keeps rows (and removes an earlier `true` on this version).
- Until either is sent, nothing is stored and answers count rows.

The token is bound to the dataset version and row count. A token for another version (a re-upload in between), a made-up one, or an answer on a dataset that is not asked gets **409** and nothing is stored. A body that is not a 32-character hex token and a JSON boolean gets **422**. A new version asks again.

Data checks at upload are warnings except the identifier checks (`*_not_null`, `*_no_duplicates`). A column with any empty value is a warning (never "passed"), and when a value has interval bound columns (`lower`/`upper`, or `<value>_lower`/`<value>_upper`) every row with all three present must have lower <= value <= upper (`<value>_interval`, a warning).

`GET /api/datasets/{dataset_id}` reports each column's stored type in `schema` (`integer`, `number`, `string`, `boolean`, `date`, `timestamp`, ...).

## Loading from a link

`POST /api/datasets/from-url` with `{"url": "https://…", "dataset_name": "optional"}` downloads one public data file (CDC, CMS, WHO, IHME exports) and hands the bytes to the upload route unchanged, so formats, limits and checks are the same as for a chosen file. The response is the upload response plus `source`: the link, the final address after redirects, the download time (UTC), the size, the SHA-256 and a `methods_text` sentence citing the link as given (a redirect target is often an expiring signed link), plus `source_recorded`, false if it could not be saved. The same record is saved on the dataset (metadata key `source`, shown in Cohort details).

The download is locked down and cannot be loosened by a caller (`storage/url_fetch.py`): HTTPS on port 443 only, certificate checked; every resolved address must be public (no loopback, private, link-local, carrier NAT, multicast or reserved ranges, including IPv6 forms that embed them), the connection goes to the checked address, and each of at most 5 redirects is checked again; no proxy, cookies, credentials or Referer; at most the upload size limit (100 MB), compressed transfers included; 15 s to connect, 30 s per read and 5 minutes overall (a timer cuts the connection at the deadline); web pages are refused as not data. Errors come back as 400 (bad or non-public link), 409 (a cohort with that name exists), 413 (too large), 415 (not a data file), 502 (the site failed or refused) or 504 (too slow), each with a sentence for the user. The app does not use a system proxy for these downloads; on a network that requires one, download the file in a browser and upload it.
