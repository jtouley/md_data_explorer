"""
Patient ID Regeneration Logic - Single Source of Truth.

This module centralizes all patient ID regeneration decisions and validation.
Used by both schema_conversion.py and definition.py to ensure consistent behavior.

Staff Engineer Standards:
- Fail fast with explicit error messages
- Validate at boundaries (once during load, not lazily)
- Single source of truth for regeneration logic
"""

from dataclasses import dataclass
from typing import Any

import polars as pl


@dataclass(frozen=True)
class PatientIdRegenerationResult:
    """Result of patient ID regeneration validation."""

    can_regenerate: bool
    source_type: str | None  # "composite", "single_column", or None
    source_columns: list[str]  # Columns used for regeneration
    error_message: str | None = None  # Only set if validation fails


class PatientIdRegenerationError(ValueError):
    """Raised when patient ID regeneration is not possible but was expected."""

    def __init__(self, message: str, metadata: dict[str, Any] | None = None):
        self.metadata = metadata
        super().__init__(message)


def validate_synthetic_id_metadata(
    metadata: dict[str, Any] | None,
) -> dict[str, Any]:
    """
    Validate and normalize synthetic_id_metadata structure.

    Ensures the metadata has the expected shape. Called once during load/init,
    not lazily during cohort queries.

    Args:
        metadata: Raw synthetic_id_metadata from upload

    Returns:
        Normalized metadata dict with guaranteed structure

    Raises:
        PatientIdRegenerationError: If metadata structure is invalid
    """
    if metadata is None:
        return {}

    if not isinstance(metadata, dict):
        raise PatientIdRegenerationError(
            f"synthetic_id_metadata must be a dict, got {type(metadata).__name__}",
            metadata={"raw": metadata},
        )

    # Validate patient_id sub-structure if present
    patient_id_meta = metadata.get("patient_id", {})
    if not isinstance(patient_id_meta, dict):
        raise PatientIdRegenerationError(
            f"synthetic_id_metadata['patient_id'] must be a dict, got {type(patient_id_meta).__name__}",
            metadata=metadata,
        )

    # Validate patient_id_source if present
    valid_sources = {"composite", "single_column", "existing"}
    source = patient_id_meta.get("patient_id_source")
    if source is not None and source not in valid_sources:
        raise PatientIdRegenerationError(
            f"Invalid patient_id_source: '{source}'. Must be one of: {valid_sources}",
            metadata=metadata,
        )

    # Validate patient_id_columns if present
    columns = patient_id_meta.get("patient_id_columns")
    if columns is not None and not isinstance(columns, list):
        raise PatientIdRegenerationError(
            f"patient_id_columns must be a list, got {type(columns).__name__}",
            metadata=metadata,
        )

    return metadata


def can_regenerate_patient_id(
    df: pl.DataFrame,
    synthetic_id_metadata: dict[str, Any] | None,
) -> PatientIdRegenerationResult:
    """
    Determine if patient_id can be regenerated and validate all requirements.

    This is the SINGLE SOURCE OF TRUTH for regeneration decisions.
    Used by both convert_schema() and get_cohort().

    Args:
        df: DataFrame to check columns against
        synthetic_id_metadata: Metadata about synthetic ID generation

    Returns:
        PatientIdRegenerationResult with validation details

    Note:
        Does NOT perform regeneration - only validates it's possible.
        Actual regeneration is done by regenerate_patient_id().
    """
    if synthetic_id_metadata is None:
        return PatientIdRegenerationResult(
            can_regenerate=False,
            source_type=None,
            source_columns=[],
            error_message=None,
        )

    patient_id_meta = synthetic_id_metadata.get("patient_id", {})
    source_type = patient_id_meta.get("patient_id_source")

    if source_type not in ("composite", "single_column"):
        return PatientIdRegenerationResult(
            can_regenerate=False,
            source_type=source_type,
            source_columns=[],
            error_message=None,
        )

    source_columns = patient_id_meta.get("patient_id_columns", [])
    df_columns = set(df.columns)

    if source_type == "composite":
        if not source_columns:
            return PatientIdRegenerationResult(
                can_regenerate=False,
                source_type=source_type,
                source_columns=[],
                error_message="Composite regeneration requires patient_id_columns but none specified",
            )

        missing_columns = [col for col in source_columns if col not in df_columns]
        if missing_columns:
            return PatientIdRegenerationResult(
                can_regenerate=False,
                source_type=source_type,
                source_columns=source_columns,
                error_message=(
                    f"Cannot regenerate composite patient_id: "
                    f"columns {missing_columns} not found in DataFrame. "
                    f"Available columns: {sorted(df_columns)}"
                ),
            )

        return PatientIdRegenerationResult(
            can_regenerate=True,
            source_type=source_type,
            source_columns=source_columns,
        )

    elif source_type == "single_column":
        if not source_columns:
            return PatientIdRegenerationResult(
                can_regenerate=False,
                source_type=source_type,
                source_columns=[],
                error_message="Single-column regeneration requires patient_id_columns but none specified",
            )

        source_col = source_columns[0]
        if source_col not in df_columns:
            return PatientIdRegenerationResult(
                can_regenerate=False,
                source_type=source_type,
                source_columns=source_columns,
                error_message=(
                    f"Cannot regenerate patient_id from column '{source_col}': "
                    f"column not found in DataFrame. Available columns: {sorted(df_columns)}"
                ),
            )

        return PatientIdRegenerationResult(
            can_regenerate=True,
            source_type=source_type,
            source_columns=source_columns,
        )

    # Should not reach here given earlier checks
    return PatientIdRegenerationResult(
        can_regenerate=False,
        source_type=source_type,
        source_columns=[],
    )


def regenerate_patient_id(
    df: pl.DataFrame,
    synthetic_id_metadata: dict[str, Any],
) -> tuple[pl.DataFrame, dict[str, Any]]:
    """
    Regenerate patient_id column based on metadata specification.

    MUST call can_regenerate_patient_id() first to validate.

    Args:
        df: DataFrame to add patient_id to
        synthetic_id_metadata: Validated metadata with regeneration instructions

    Returns:
        Tuple of (DataFrame with patient_id, metadata dict matching ensure_patient_id format)

    Raises:
        PatientIdRegenerationError: If regeneration fails
    """
    # Import here to avoid circular imports
    from clinical_analytics.ui.components.variable_detector import VariableTypeDetector

    patient_id_meta = synthetic_id_metadata.get("patient_id", {})
    source_type = patient_id_meta.get("patient_id_source")
    source_columns = patient_id_meta.get("patient_id_columns", [])

    if source_type == "composite":
        df_with_id = VariableTypeDetector.create_synthetic_patient_id(df, source_columns)
        id_metadata = {
            "patient_id_source": "composite",
            "patient_id_columns": source_columns,
        }
        return df_with_id, id_metadata

    elif source_type == "single_column":
        source_col = source_columns[0]
        df_with_id = df.with_columns(pl.col(source_col).alias("patient_id"))
        id_metadata = {
            "patient_id_source": "single_column",
            "patient_id_columns": [source_col],
        }
        return df_with_id, id_metadata

    else:
        raise PatientIdRegenerationError(
            f"Cannot regenerate patient_id: unsupported source_type '{source_type}'",
            metadata=synthetic_id_metadata,
        )
