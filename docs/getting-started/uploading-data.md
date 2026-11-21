# Uploading Data

## Supported Formats

The desktop app's **Upload a file…** button accepts one file at a time (`ALLOWED_UPLOAD_EXTENSIONS` in `src/clinical_analytics/api/routes/datasets.py`):

- **CSV files** (.csv)
- **Tab-separated text** (.tsv, or .txt), such as a CDC WONDER export as downloaded. The notes block under the data (dataset, query, citation) is kept as the dataset's documentation, not as rows; "Total" subtotal rows are left out so nothing is counted twice; values WONDER withholds ("Suppressed", "Unreliable") are stored as missing, never 0, and the upload says how many in which column.
- **Excel files** (.xlsx, and legacy .xls). Only the first sheet is read; the upload names any other sheet that holds data.
- **SPSS files** (.sav)
- **SAS transport files** (.xpt), as NHANES publishes them. Whole-number columns (codes such as 1/2) are stored as integers.
- **ZIP files** (.zip) of tables and their documentation. See [Multi-table datasets](#multi-table-datasets).

To try the app without your own data, click **Use the sample cohort** instead. It loads a synthetic sample cohort that ships with the app (`src/clinical_analytics/resources/demo/sample_cohort.csv`, no real patients).

## Data Requirements

### Minimum Requirements

Your dataset should have:

1. **Unique identifier**: Patient ID, subject ID, or similar
2. **At least one outcome**: Binary variable (0/1) for mortality, readmission, etc.
3. **Variables**: Clinical measurements, demographics, treatments

### Recommended Structure

```
patient_id,age,sex,treatment,mortality,survival_days
001,45,M,A,0,365
002,67,F,B,1,120
003,52,M,A,0,365
```

## Automatic Schema Detection

The platform automatically detects:

### Patient ID

Columns named:
- `patient_id`, `patientid`, `id`
- `subject_id`, `subjectid`
- `mrn`, `study_id`

Or any column with >95% unique values.

### Outcomes

Binary columns (0/1) with outcome-related names:
- `mortality`, `death`, `died`
- `readmission`, `hospitalized`
- `complication`, `adverse_event`

### Time Variables

Columns with:
- Date/datetime types
- Names containing: `time`, `date`, `day`, `survival`, `followup`

### Variable Types

- **Categorical**: <20 unique values or string type
- **Continuous**: Numeric with >20 unique values

## Multi-table datasets

Upload a `.zip` that holds the data tables and their documentation, as PhysioNet and UCI package them:

- **Tables:** `.csv`, `.tsv` and `.psv` (each also as `.gz`), `.xlsx`, `.xls`, `.xpt` and `.sav`, in any folder of the ZIP. A ZIP inside the ZIP is opened too (one level).
- **Documentation:** `.pdf`, `.md` and `.txt` files (a README, a data dictionary). Their text is kept with the dataset as its documentation context.
- **One file per patient:** numbered files with the same columns (`p000001.psv`, `p000002.psv`, ...) are read as one table. The file name becomes `patient_id` (or `source_file` when the files already hold a patient ID column), and `source_folder` says which folder each row came from when there is more than one.
- **A codebook saved as CSV** that cannot be read as one table (several small tables separated by blank rows) is kept as documentation, and the upload message says so.
- **Several tables** become one cohort, anchored on a table with one row per patient or per stay (`subject_id`, `hadm_id`, `patientunitstayid`, NHANES `SEQN`, ...); the upload lists the tables left out of the cohort and why. If no table has an ID the others share, the upload fails with that reason: upload the table you want as its own file.

The upload limit is 100 MB for the ZIP file itself, and 1 GB for everything in it once unpacked. Documentation context is cut at 50,000 characters, in the order the files sit in the ZIP, so put the data dictionary before long files such as a licence or a paper.

## Manual Schema Override

If the platform doesn't automatically detect your patient ID, outcomes, or time variables correctly, you can manually specify them in the upload interface. The platform will guide you through selecting the correct columns.

## Your Data is Saved

**Important for your workflow:** Once you upload data, it's saved on your computer. You don't need to re-upload it every time.

- ✅ **Data persists**: Your uploaded datasets are saved and available when you restart the app
- ✅ **No re-uploading**: Close the app or restart your computer - your data is still there
- ✅ **Work across sessions**: Upload once, analyze multiple times over days or weeks
- ✅ **Local storage**: Data is kept in a per-user data home on your computer (see below)

**What this means for you:**
- Upload your dataset once
- Close the app and come back later - your data is still available
- Your analysis history is preserved (coming soon)

## Data Privacy

The app runs locally after setup. It is **not a compliance certification**; check your institution's rules before using it with patient data.

- **Where data is stored:** a per-user data home outside the repository - macOS `~/Library/Application Support/ClinicalAnalytics`, Windows `%LOCALAPPDATA%\ClinicalAnalytics`, Linux `$XDG_DATA_HOME/clinical-analytics`. Set `CLINICAL_ANALYTICS_DATA_HOME` to change it. See [Installation](installation.md#where-your-data-lives).
- **Encryption:** the app does not encrypt files at rest. Turn on FileVault (macOS), BitLocker (Windows) or full-disk encryption (Linux).
- **LLM:** questions the pattern matcher cannot handle go to Ollama at `OLLAMA_BASE_URL` (default `http://localhost:11434`). Prompts can include column names, category labels and summary results, so keep Ollama on your own machine; never point `OLLAMA_BASE_URL` at a remote server.
- **Small counts:** answers on screen show every count as it is; counts from 1 to 10 are written as `<11` only when an answer leaves your machine as a file.
- **Never copy patient files into the repository.** `make check-data` fails if git tracks data files.

## Troubleshooting

### "Answers count rows, not patients"

**What it means:** no column in the file identifies a patient, so the upload added a row number (or tells the rows apart by a combination of columns) and every answer counts rows. The upload says so, and answers say "rows", never "patients". If each row is one patient the numbers are right; if a patient can have several rows (visits, episodes), a row count is not a patient count.

**To count patients:**
- Add a column with one ID for each patient (patient ID, subject ID or MRN), with the same ID on each of that patient's rows, and upload the file again. A column named as a patient ID is used even when it repeats; repeated IDs are read as visits of one patient, patient counts count each ID once, and each answer shows what it counted ("Counted distinct patients (patient_id) ...").

### Other upload errors

A file the app cannot use as sent is refused with a message that says what to fix (the API answers 400): a damaged or password-protected ZIP, a ZIP packed with a compression method other than Deflate (Windows' Deflate64), more than 100,000 files in one ZIP, or two columns whose names differ only in upper and lower case (`Age` and `age`), which the query engine reads as one name.

### "No outcome variables detected"

**Problem:** The platform can't find any outcome variables (like mortality, readmission).

**Solution:**
- Your outcome columns should be binary (0/1 or Yes/No)
- If your outcomes are text (like "died"/"alive"), convert them to 0/1 in Excel before uploading
- Make sure outcome column names contain words like "mortality", "death", "readmission", etc.

## Next Steps

- Learn how to [ask questions](../user-guide/question-driven-analysis.md)
- Understand [statistical tests](../user-guide/statistical-tests.md)
- Explore the [architecture](../architecture/overview.md)
