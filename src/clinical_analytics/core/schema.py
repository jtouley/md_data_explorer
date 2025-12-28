"""
Unified Cohort Schema Definition.

Ensures all datasets output data in a harmonized format for the generic analysis layer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # PANDAS EXCEPTION: ClinicalDataset.get_cohort() returns pd.DataFrame
    # This module validates that return type. Migration to Polars pending
    # refactor of all dataset implementations.
    # TODO: Remove when get_cohort() returns pl.DataFrame
    import pandas as pd


class UnifiedCohort:
    """
    Standard schema definition for cohort tables.

    All datasets must produce data conforming to this schema to enable
    consistent analysis across different data sources.
    """

    # Core Identifiers
    PATIENT_ID = "patient_id"  # String identifier

    # Temporal Anchors
    TIME_ZERO = "time_zero"  # Datetime of entry/infection/admission

    # Outcomes
    OUTCOME = "outcome"  # Binary (0/1) or Numeric target
    OUTCOME_LABEL = "outcome_label"  # String description (e.g., "mortality_30d", "sepsis_onset")

    # Flexible Features
    # Features can be flat columns in the dataframe, but this constant
    # reserves a name for a packed feature dictionary if needed.
    FEATURES_JSON = "features_json"

    REQUIRED_COLUMNS = [PATIENT_ID, TIME_ZERO, OUTCOME, OUTCOME_LABEL]


class SchemaValidationError(Exception):
    """
    Raised when a DataFrame does not conform to the UnifiedCohort schema.

    Contains details about which validations failed.
    """

    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__(f"Schema validation failed: {'; '.join(errors)}")


def validate_unified_cohort_schema(
    df: pd.DataFrame,
    strict: bool = False,
) -> tuple[bool, list[str]]:
    """
    Validate DataFrame conforms to UnifiedCohort schema contract.

    Args:
        df: DataFrame to validate
        strict: If True, raise SchemaValidationError on failure.
                If False (default), return (is_valid, errors) tuple.

    Returns:
        Tuple of (is_valid, list_of_errors)

    Raises:
        SchemaValidationError: If strict=True and validation fails
    """
    import pandas as pd_module

    errors: list[str] = []

    # Check required columns exist
    missing = set(UnifiedCohort.REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        errors.append(f"Missing required columns: {sorted(missing)}")

    # Check PATIENT_ID column
    if UnifiedCohort.PATIENT_ID in df.columns:
        patient_id_col = df[UnifiedCohort.PATIENT_ID]

        # No NULLs allowed in patient ID
        null_count = patient_id_col.isna().sum()
        if null_count > 0:
            errors.append(f"{UnifiedCohort.PATIENT_ID} contains {null_count} NULL value(s) - patient ID cannot be NULL")

        # Check uniqueness (one row per patient in patient-level cohort)
        # Note: This is a warning, not an error, as some datasets may have
        # admission-level or event-level granularity
        dup_count = patient_id_col.duplicated().sum()
        if dup_count > 0:
            # Just a warning - not an error for non-patient-level data
            pass  # Could add warning log here

    # Check OUTCOME column
    if UnifiedCohort.OUTCOME in df.columns:
        outcome_col = df[UnifiedCohort.OUTCOME]

        # Must be numeric
        if not pd_module.api.types.is_numeric_dtype(outcome_col):
            errors.append(f"{UnifiedCohort.OUTCOME} must be numeric (got {outcome_col.dtype})")
        else:
            # For binary outcomes, check values are 0 or 1 (allow NaN)
            non_null_values = outcome_col.dropna().unique()
            if len(non_null_values) <= 2:
                # Binary outcome - should be 0/1
                invalid_values = [v for v in non_null_values if v not in (0, 1, 0.0, 1.0)]
                if invalid_values:
                    errors.append(
                        f"{UnifiedCohort.OUTCOME} has invalid binary values: {invalid_values}. "
                        "Binary outcomes must be 0 or 1."
                    )

    # Check TIME_ZERO column
    if UnifiedCohort.TIME_ZERO in df.columns:
        time_col = df[UnifiedCohort.TIME_ZERO]

        # Should be datetime or convertible to datetime
        if not pd_module.api.types.is_datetime64_any_dtype(time_col):
            # Try to parse as datetime
            try:
                pd_module.to_datetime(time_col, errors="raise")
            except (ValueError, TypeError):
                errors.append(f"{UnifiedCohort.TIME_ZERO} is not a valid datetime (got {time_col.dtype})")

    # Check OUTCOME_LABEL column
    if UnifiedCohort.OUTCOME_LABEL in df.columns:
        label_col = df[UnifiedCohort.OUTCOME_LABEL]

        # Should be string/object type
        if not pd_module.api.types.is_object_dtype(label_col) and not pd_module.api.types.is_string_dtype(label_col):
            errors.append(f"{UnifiedCohort.OUTCOME_LABEL} should be string type (got {label_col.dtype})")

    is_valid = len(errors) == 0

    if strict and not is_valid:
        raise SchemaValidationError(errors)

    return is_valid, errors
