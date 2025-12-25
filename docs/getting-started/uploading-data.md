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

## Multi-Table Datasets

For complex datasets (like MIMIC-IV):

1. **Create a ZIP file** containing multiple CSV files
2. **Upload the ZIP**: The platform detects relationships automatically
3. **Review joins**: Check detected foreign keys in the UI
4. **Override if needed**: Manually specify joins if auto-detection fails

### Example: MIMIC-IV

```
mimic-iv/
├── patients.csv        (patient_id, gender, age)
├── admissions.csv      (patient_id, admission_time)
└── diagnoses.csv       (patient_id, icd_code)
```

The platform:

1. Detects `patient_id` as the primary key in patients.csv
2. Finds matching `patient_id` in other tables
3. Performs left joins automatically
4. Creates unified cohort view

## Manual Schema Override

If auto-detection fails, you can manually specify:

```python
from clinical_analytics.core.schema_inference import InferredSchema

schema = InferredSchema(
    patient_id_column='subject_id',
    outcome_columns=['mortality_28day'],
    time_columns=['admission_date'],
    time_zero='admission_date'
)
```

## Data Privacy

The platform runs **entirely locally**. Your data:

- ✅ Never leaves your machine
- ✅ Not sent to external APIs (for Tier 1/2 NL queries)
- ✅ Processed in-memory with DuckDB
- ⚠️ Tier 3 LLM fallback (optional) sends anonymized metadata only

## Troubleshooting

### "Could not detect patient ID"

Ensure you have a column with unique values per patient. If your data has no such column, add an index:

```python
df['patient_id'] = range(len(df))
```

### "No outcome variables detected"

Your outcome columns should be binary (0/1). Convert if needed:

```python
df['mortality'] = (df['status'] == 'died').astype(int)
```

### "Failed to join tables"

Check that:

1. Foreign key columns exist in child tables
2. Column names match (case-sensitive)
3. Values in child table exist in parent table

## Next Steps

- Learn how to [ask questions](../user-guide/question-driven-analysis.md)
- Understand [statistical tests](../user-guide/statistical-tests.md)
- Explore the [architecture](../architecture/overview.md)
