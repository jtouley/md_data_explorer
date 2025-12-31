"""
Schema conversion utilities for uploaded datasets.

Converts between variable_mapping and inferred_schema formats with defensive checks.
"""

from typing import Any

import polars as pl


def is_categorical(col: pl.Series) -> bool:
    """
    Improved categorical detection heuristic.

    Prevents patient IDs, lab values, and dates from being misclassified.
    Only string columns with low cardinality and low uniqueness ratio are categorical.

    Args:
        col: Polars Series to check

    Returns:
        True if column should be treated as categorical
    """
    # Defensive: handle empty series
    if len(col) == 0:
        return False

    unique_count = col.n_unique()
    total_count = len(col)

    # String columns with low cardinality and low uniqueness ratio
    if col.dtype == pl.Utf8:
        # Defensive: avoid division by zero (though total_count > 0 from check above)
        return unique_count <= 20 and unique_count / total_count < 0.5

    # Numeric columns need explicit annotation (never auto-categorical)
    # Prevents patient IDs, lab values, dates from being misclassified
    return False


def infer_granularities(df: pl.DataFrame) -> list[str]:
    """
    Infer supported granularities from column presence.

    Args:
        df: Polars DataFrame to analyze

    Returns:
        List of supported granularity levels
    """
    granularities = ["patient_level"]  # Always supported

    if "admission_id" in df.columns:
        granularities.append("admission_level")

    if "event_timestamp" in df.columns or "event_date" in df.columns:
        granularities.append("event_level")

    return granularities


def convert_schema(
    variable_mapping: dict[str, Any],
    df: pl.DataFrame,
    synthetic_id_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Convert variable_mapping format to inferred_schema format.

    CRITICAL: This function requires DataFrame access for type inference.
    Must be called AFTER table normalization (not before).

    Args:
        variable_mapping: Legacy variable_mapping format
        df: DataFrame for type inference
        synthetic_id_metadata: Optional metadata about synthetic ID generation

    Returns:
        inferred_schema format dict

    Raises:
        ValueError: If required columns are missing from DataFrame
    """
    inferred: dict[str, Any] = {
        "column_mapping": {},
        "outcomes": {},
        "time_zero": {},
        "analysis": {
            "default_outcome": None,
            "default_predictors": [],
            "categorical_variables": [],
        },
    }

    # Map patient_id
    if patient_id_col := variable_mapping.get("patient_id"):
        # Check if patient_id can be regenerated (synthetic_id_metadata present)
        can_regenerate = False
        if synthetic_id_metadata:
            patient_id_metadata = synthetic_id_metadata.get("patient_id", {})
            can_regenerate = patient_id_metadata.get("patient_id_source") in ["composite", "single_column"]

        # Defensive: check column exists
        if patient_id_col not in df.columns:
            # Check if "patient_id" exists as fallback (column was renamed during ingestion)
            if "patient_id" in df.columns:
                # Use patient_id directly (was renamed during ingestion)
                inferred["column_mapping"]["patient_id"] = "patient_id"
            # Allow missing patient_id if regeneration is possible
            elif can_regenerate:
                # Don't set column_mapping - get_cohort() will regenerate it
                pass
            else:
                available_cols = list(df.columns)
                raise ValueError(
                    f"Patient ID column '{patient_id_col}' not found in DataFrame. Available columns: {available_cols}"
                )
        else:
            inferred["column_mapping"][patient_id_col] = "patient_id"

    # Map outcome with type inference (defensive checks)
    if outcome_col := variable_mapping.get("outcome"):
        # Defensive: check column exists
        if outcome_col not in df.columns:
            raise ValueError(f"Outcome column '{outcome_col}' not found in DataFrame columns: {df.columns}")

        # Defensive: handle empty dataframe
        if df.height == 0:
            # Default to continuous for empty data
            outcome_type = "continuous"
        else:
            unique_count = df[outcome_col].n_unique()
            outcome_type = "binary" if unique_count == 2 else "continuous"

        inferred["outcomes"][outcome_col] = {
            "source_column": outcome_col,
            "type": outcome_type,
            "confidence": 0.9 if outcome_type == "binary" else 0.7,
        }
        inferred["analysis"]["default_outcome"] = outcome_col

    # Map time_zero
    if time_vars := variable_mapping.get("time_variables"):
        if time_zero_col := time_vars.get("time_zero"):
            # Defensive: check column exists
            if time_zero_col not in df.columns:
                raise ValueError(
                    f"Time zero column '{time_zero_col}' not found in DataFrame. Available columns: {list(df.columns)}"
                )
            inferred["time_zero"] = {"source_column": time_zero_col}

    # Map predictors and detect categoricals (better heuristic)
    predictors = variable_mapping.get("predictors", [])
    for col in predictors:
        if col in df.columns:
            inferred["analysis"]["default_predictors"].append(col)
            # Detect categorical using improved heuristic (handles empty df)
            if is_categorical(df[col]):
                inferred["analysis"]["categorical_variables"].append(col)
        # Skip missing columns silently (they may have been dropped)

    # Infer granularities from columns
    inferred["granularities"] = infer_granularities(df)

    return inferred
