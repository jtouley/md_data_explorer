# Uploading Data

## Supported Formats

- **CSV files** (.csv)
- **Excel files** (.xlsx, .xls)
- **ZIP archives** (multiple tables)

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

## Upload Types

You can upload data in two ways - both work the same way:

### Single File Upload

Upload one CSV or Excel file:

1. **Upload your file**: CSV, Excel, or SPSS format
2. **Automatic detection**: The platform finds patient IDs, outcomes, and time variables automatically
3. **Ready to analyze**: Start asking questions immediately

### Multiple Files Upload

For complex datasets with multiple related tables (like MIMIC-IV):

1. **Create a ZIP file** containing all your CSV files
2. **Upload the ZIP**: The platform automatically detects how tables relate to each other
3. **Review connections**: Check that the platform correctly identified relationships between tables
4. **Adjust if needed**: Manually specify connections if auto-detection missed something

### Example: MIMIC-IV

```
mimic-iv/
├── patients.csv        (patient_id, gender, age)
├── admissions.csv      (patient_id, admission_time)
└── diagnoses.csv       (patient_id, icd_code)
```

The platform automatically:

1. Finds the patient ID column in each table
2. Connects tables that share patient IDs
3. Combines all tables into a single view for analysis
4. Saves everything so you can analyze it later

## Manual Schema Override

If the platform doesn't automatically detect your patient ID, outcomes, or time variables correctly, you can manually specify them in the upload interface. The platform will guide you through selecting the correct columns.

## Your Data is Saved

**Important for your workflow:** Once you upload data, it's saved on your computer. You don't need to re-upload it every time.

- ✅ **Data persists**: Your uploaded datasets are saved and available when you restart the app
- ✅ **No re-uploading**: Close the browser or restart your computer - your data is still there
- ✅ **Work across sessions**: Upload once, analyze multiple times over days or weeks
- ✅ **Local storage only**: All data stays on your computer (never sent to external servers)

**What this means for you:**
- Upload your dataset once
- Close the app and come back later - your data is still available
- Refresh the page - your data doesn't disappear
- Your analysis history is preserved (coming soon)

## Data Privacy

**Your data stays on your computer.** The platform runs entirely locally:

- ✅ **Never leaves your computer**: All data processing happens on your machine
- ✅ **No cloud storage**: Your patient data is never uploaded to external servers
- ✅ **HIPAA-friendly**: Local-only storage avoids cloud vendor complications
- ⚠️ **Optional LLM fallback**: If enabled, only anonymized metadata (no patient data) is sent for complex queries

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

### "Failed to join tables"

**Problem:** When uploading multiple tables, the platform can't connect them together.

**Solution:**
- Make sure all tables have a matching patient ID column (same column name, same values)
- Check that patient IDs in child tables actually exist in the main patient table
- Column names are case-sensitive - "Patient_ID" and "patient_id" are treated as different

## Next Steps

- Learn how to [ask questions](../user-guide/question-driven-analysis.md)
- Understand [statistical tests](../user-guide/statistical-tests.md)
- Explore the [architecture](../architecture/overview.md)
