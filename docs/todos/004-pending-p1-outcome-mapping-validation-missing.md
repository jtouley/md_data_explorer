---
status: pending
priority: p1
issue_id: "004"
tags: [code-review, data-integrity, critical]
dependencies: []
estimated_effort: medium
created_date: 2025-12-24
---

# Missing Data Validation in Outcome Mappings

## Problem Statement

Outcome transformations in `mapper.py` default unmapped values to 0, causing **silent data corruption**. Values like "unknown", "pending", or NULL are incorrectly mapped to 0 (interpreted as negative outcome), skewing statistical analyses and research results.

**Why it matters:**
- Research integrity - incorrect outcome classifications
- Silent data corruption (no errors thrown)
- Publication validity compromised
- Regulatory compliance risk

**Impact:** Incorrect research conclusions, wasted research effort

## Findings

**Location:** `src/clinical_analytics/core/mapper.py:72-96`

**Vulnerable Code:**
```python
expr = pl.lit(0)  # DEFAULT - THIS IS DANGEROUS!
for key, value in mapping.items():
    if isinstance(key, str):
        expr = pl.when(
            pl.col(source_col).cast(pl.Utf8).str.to_lowercase() == key.lower()
        ).then(value).otherwise(expr)
```

**Problem:** If source column contains "unknown", "pending", "N/A", or NULL, they all become 0.

**Example Scenario:**
```yaml
# datasets.yaml
outcomes:
  outcome_hospitalized:
    source_column: covid19_admission_hospital
    mapping:
      yes: 1
      no: 0

# If data has:
# Row 1: covid19_admission_hospital = "yes" → outcome = 1 ✓
# Row 2: covid19_admission_hospital = "no" → outcome = 0 ✓
# Row 3: covid19_admission_hospital = "unknown" → outcome = 0 ✗ WRONG!
# Row 4: covid19_admission_hospital = NULL → outcome = 0 ✗ WRONG!
```

## Proposed Solutions

### Solution 1: Default to NULL + Validation (Recommended)
**Pros:**
- Preserves data integrity
- Makes unmapped values explicit
- Forces data quality review
- Statistical software handles NULLs correctly

**Cons:**
- May reduce sample size
- Requires handling missing data

**Effort:** Medium (4 hours)
**Risk:** Low

**Implementation:**
```python
def apply_outcome_transformations(self, df: pl.DataFrame) -> pl.DataFrame:
    for outcome_name, outcome_def in outcomes.items():
        source_col = outcome_def['source_column']
        mapping = outcome_def.get('mapping', {})

        # Default to NULL, not 0
        expr = pl.lit(None)

        for key, value in mapping.items():
            if isinstance(key, str):
                expr = pl.when(
                    pl.col(source_col).is_not_null() &
                    (pl.col(source_col).cast(pl.Utf8).str.to_lowercase() == key.lower())
                ).then(value).otherwise(expr)
            # ... other types

        df = df.with_columns([expr.alias(outcome_name)])

        # VALIDATE - Check for unmapped non-null values
        unmapped = df.filter(
            pl.col(source_col).is_not_null() & pl.col(outcome_name).is_null()
        )

        if len(unmapped) > 0:
            unmapped_values = unmapped[source_col].unique().to_list()
            raise ValueError(
                f"Unmapped values in outcome '{outcome_name}': {unmapped_values}. "
                f"Add to mapping in datasets.yaml or clean source data. "
                f"Affected records: {len(unmapped)}"
            )

    return df
```

### Solution 2: Strict Mode Flag
**Pros:**
- Backwards compatible
- Can warn instead of error
- Configurable behavior

**Cons:**
- Adds complexity
- Default should be safe

**Effort:** Medium (5 hours)
**Risk:** Low

## Recommended Action

**Implement Solution 1** with:
1. Change default from 0 to NULL
2. Add validation after mapping
3. Raise descriptive error with unmapped values list
4. Update documentation to explain requirement

## Technical Details

**Affected Files:**
- `src/clinical_analytics/core/mapper.py` (lines 72-96)
- All dataset YAML configs may need mapping updates

**Testing Requirements:**
```python
def test_outcome_mapping_rejects_unmapped_values():
    """Test that unmapped values raise errors."""
    df = pl.DataFrame({
        'outcome_col': ['yes', 'no', 'unknown', None]
    })

    config = {
        'outcomes': {
            'test_outcome': {
                'source_column': 'outcome_col',
                'type': 'binary',
                'mapping': {'yes': 1, 'no': 0}
            }
        }
    }

    mapper = ColumnMapper(config)

    with pytest.raises(ValueError, match="Unmapped values.*unknown"):
        mapper.apply_outcome_transformations(df)

def test_outcome_mapping_handles_null_correctly():
    """Test that NULL values stay NULL."""
    df = pl.DataFrame({
        'outcome_col': ['yes', 'no', None]
    })

    # Should complete without error (NULL is expected)
    result = mapper.apply_outcome_transformations(df)
    assert result.filter(pl.col('outcome_col').is_null())['test_outcome'].is_null().all()
```

## Acceptance Criteria

- [ ] Default outcome mapping is NULL, not 0
- [ ] Validation runs after all outcome transformations
- [ ] Error raised if unmapped non-null values exist
- [ ] Error message lists all unmapped values
- [ ] Error message shows count of affected records
- [ ] NULL values remain NULL (not treated as unmapped)
- [ ] Tests verify rejection of unmapped values
- [ ] Tests verify NULL handling
- [ ] Documentation updated with validation behavior
- [ ] All existing datasets verified to have complete mappings

## Work Log

### 2025-12-24
- **Action:** Data integrity review identified silent mapping failures
- **Learning:** Default values in data transformations should be NULL for safety
- **Next:** Implement NULL default + validation logic

## Resources

- **Data Integrity Review:** See comprehensive data integrity report
- **Polars Documentation:** Working with missing data
- **Related Finding:** Case-insensitive mapping NULL handling (lines 88-90)
