# Uploading Data

## Supported Formats

The desktop app's **Upload dataset** button accepts one file at a time (`ALLOWED_UPLOAD_EXTENSIONS` in `src/clinical_analytics/api/routes/datasets.py`):

- **CSV files** (.csv)
- **Excel files** (.xlsx, .xls)
- **SPSS files** (.sav)

To try the app without your own data, click **Load sample** instead. It loads a synthetic sample cohort that ships with the app (`src/clinical_analytics/resources/demo/sample_cohort.csv`, no real patients).

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

ZIP uploads of several related tables (for example MIMIC-IV extracts) exist in the storage layer but are **not available in the desktop app**: the upload endpoint rejects `.zip`. Join the tables into one file before uploading.

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
- **Small counts:** patient counts from 1 to 10 are shown as `<11` in answers.
- **Never copy patient files into the repository.** `make check-data` fails if git tracks data files.

## Troubleshooting

### "Could not detect patient ID"

**Problem:** The platform can't find a unique identifier for each patient.

**Solution:**
- Make sure you have a column with a unique value for each patient (like patient ID, subject ID, or MRN)
- If your data doesn't have one, add a simple row number column in Excel before uploading

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
