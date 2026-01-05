"""
Data Quality Validator

Observes data quality characteristics for uploaded datasets.
All observations are NON-BLOCKING - they are stored as metadata for the
semantic layer to surface at query time when relevant.

Philosophy:
- Accept data as-is at upload time
- Record quality observations as metadata
- Surface warnings at query time when user tries to use problematic columns
- Let users make informed decisions about their data
"""

from __future__ import annotations

import logging
from typing import Any

import polars as pl

logger = logging.getLogger(__name__)


def _ensure_polars(df: Any) -> pl.DataFrame:
    """Convert pandas DataFrame to polars if needed."""
    if isinstance(df, pl.DataFrame):
        return df
    # Assume pandas DataFrame - convert at boundary
    try:
        return pl.from_pandas(df)
    except Exception as e:
        # Handle type conversion errors (e.g., numeric columns with string values)
        # This can happen when Excel files have mixed types in columns
        if "ArrowInvalid" in str(type(e).__name__) or "Could not convert" in str(e):
            logger.warning(
                f"Type conversion error during pandas->polars conversion: {e}. "
                "Attempting to fix by converting problematic columns to string."
            )
            # Convert all columns to string first, then let Polars infer types
            # This is safer but less efficient - we'll optimize later if needed
            df_str = df.astype(str)
            # Replace 'nan' strings with actual nulls
            df_str = df_str.replace("nan", None)
            df_str = df_str.replace("", None)
            # Convert to polars
            result = pl.from_pandas(df_str)
            # Try to infer better types where possible
            # (Polars will handle this automatically, but we can be explicit)
            return result
        else:
            # Re-raise if it's a different error
            raise


class DataQualityValidator:
    """
    Observe data quality characteristics for uploaded datasets.

    All quality issues are WARNINGS, not blocking errors.
    Only structural issues (empty dataset, no columns) are errors.

    Quality observations are stored as metadata and surfaced by the
    semantic layer at query time.
    """

    # Thresholds for quality observations (configurable via class attributes)
    HIGH_MISSING_THRESHOLD = 30  # % missing per column - flag for attention
    VERY_HIGH_MISSING_THRESHOLD = 80  # % missing per column - likely unusable

    @classmethod
    def observe_id_column(cls, df: pl.DataFrame, id_column: str) -> list[dict[str, Any]]:
        """
        Observe characteristics of a patient ID column.

        All observations are warnings - user decides how to handle.

        Args:
            df: Polars DataFrame
            id_column: Name of ID column

        Returns:
            List of quality observations (all warnings)
        """
        observations = []

        if id_column not in df.columns:
            observations.append(
                {
                    "severity": "warning",
                    "type": "column_not_found",
                    "message": f"Column '{id_column}' not found in data",
                    "column": id_column,
                }
            )
            return observations

        id_series = df[id_column]
        n_rows = df.height

        # Observe missing IDs
        n_missing = id_series.null_count()
        if n_missing > 0:
            observations.append(
                {
                    "severity": "warning",
                    "type": "missing_ids",
                    "message": f"{n_missing} missing values in ID column '{id_column}'",
                    "column": id_column,
                    "count": str(int(n_missing)),
                    "pct": str(float(n_missing / n_rows * 100) if n_rows > 0 else 0),
                }
            )

        # Observe duplicates
        n_unique = id_series.n_unique()
        n_duplicates = n_rows - n_unique
        if n_duplicates > 0:
            # Get sample of duplicates
            duplicate_counts = df.group_by(id_column).len().filter(pl.col("len") > 1)
            sample_ids = duplicate_counts.head(5)[id_column].to_list()
            observations.append(
                {
                    "severity": "warning",
                    "type": "duplicate_ids",
                    "message": (
                        f"{n_duplicates} duplicate values in '{id_column}' (may indicate multi-row-per-patient data)"
                    ),
                    "column": id_column,
                    "count": str(int(n_duplicates)),
                    "examples": str(sample_ids) if isinstance(sample_ids, list) else sample_ids,
                }
            )

        # Observe uniqueness ratio
        n_non_null = n_rows - n_missing
        if n_non_null > 0:
            uniqueness_ratio = n_unique / n_non_null
            observations.append(
                {
                    "severity": "info",
                    "type": "id_uniqueness",
                    "message": f"ID column '{id_column}' is {uniqueness_ratio * 100:.1f}% unique",
                    "column": id_column,
                    "uniqueness": str(float(uniqueness_ratio)),
                    "unique_count": str(int(n_unique)),
                    "total_count": str(int(n_non_null)),
                }
            )

        return observations

    @classmethod
    def observe_missing_data(cls, df: pl.DataFrame) -> list[dict[str, Any]]:
        """
        Observe missing data patterns across all columns.

        All observations are warnings - user decides how to handle.

        Args:
            df: Polars DataFrame

        Returns:
            List of quality observations (all warnings)
        """
        observations = []
        column_stats = []
        n_rows = df.height

        for col in df.columns:
            n_missing = df[col].null_count()
            pct_missing = (n_missing / n_rows * 100) if n_rows > 0 else 0

            column_stats.append(
                {
                    "column": col,
                    "missing_count": int(n_missing),
                    "missing_pct": float(pct_missing),
                }
            )

            if pct_missing >= cls.VERY_HIGH_MISSING_THRESHOLD:
                observations.append(
                    {
                        "severity": "warning",
                        "type": "very_high_missing",
                        "message": (f"Column '{col}' has {pct_missing:.1f}% missing - may be unreliable for analysis"),
                        "column": col,
                        "missing_count": int(n_missing),
                        "missing_pct": float(pct_missing),
                    }
                )
            elif pct_missing >= cls.HIGH_MISSING_THRESHOLD:
                observations.append(
                    {
                        "severity": "warning",
                        "type": "high_missing",
                        "message": f"Column '{col}' has {pct_missing:.1f}% missing data",
                        "column": col,
                        "missing_count": int(n_missing),
                        "missing_pct": float(pct_missing),
                    }
                )

        # Overall missing data observation
        total_cells = df.height * df.width
        missing_cells = sum(df[col].null_count() for col in df.columns)
        overall_missing_pct = (missing_cells / total_cells * 100) if total_cells > 0 else 0

        observations.append(
            {
                "severity": "info",
                "type": "overall_missing",
                "message": f"Dataset has {overall_missing_pct:.1f}% missing values overall",
                "missing_pct": float(overall_missing_pct),
                "missing_cells": int(missing_cells),
                "total_cells": int(total_cells),
                "column_stats": column_stats,
            }
        )

        return observations

    @classmethod
    def observe_outcome_column(cls, df: pl.DataFrame, outcome_column: str) -> list[dict[str, Any]]:
        """
        Observe characteristics of an outcome column.

        All observations are warnings - user decides how to handle.

        Args:
            df: Polars DataFrame
            outcome_column: Name of outcome column

        Returns:
            List of quality observations (all warnings)
        """
        observations = []

        if outcome_column not in df.columns:
            observations.append(
                {
                    "severity": "warning",
                    "type": "column_not_found",
                    "message": f"Outcome column '{outcome_column}' not found in data",
                    "column": outcome_column,
                }
            )
            return observations

        outcome_series = df[outcome_column]
        n_rows = df.height

        # Observe missing outcomes
        n_missing = outcome_series.null_count()
        pct_missing = (n_missing / n_rows * 100) if n_rows > 0 else 0

        if n_missing > 0:
            observations.append(
                {
                    "severity": "warning" if pct_missing > 20 else "info",
                    "type": "missing_outcome",
                    "message": f"Outcome has {pct_missing:.1f}% missing values",
                    "column": outcome_column,
                    "missing_count": str(int(n_missing)),
                    "missing_pct": str(float(pct_missing)),
                }
            )

        # Observe variation
        n_unique = outcome_series.n_unique()
        observations.append(
            {
                "severity": "info",
                "type": "outcome_distribution",
                "message": f"Outcome has {n_unique} unique value(s)",
                "column": outcome_column,
                "unique_count": str(int(n_unique)),
            }
        )

        if n_unique < 2:
            observations.append(
                {
                    "severity": "warning",
                    "type": "no_variation",
                    "message": f"Outcome has only {n_unique} unique value(s) - no variation for analysis",
                    "column": outcome_column,
                    "unique_count": str(int(n_unique)),
                }
            )

        # For binary outcomes, observe balance
        n_non_null = n_rows - n_missing
        if n_unique == 2 and n_non_null > 0:
            value_counts = outcome_series.drop_nulls().value_counts()
            min_count_val = value_counts["count"].min()
            min_count = (
                int(min_count_val) if min_count_val is not None and isinstance(min_count_val, (int, float)) else 0
            )
            minority_pct = (min_count / n_non_null * 100) if n_non_null > 0 else 0.0

            if minority_pct < 5.0:
                observations.append(
                    {
                        "severity": "warning",
                        "type": "imbalanced_outcome",
                        "message": f"Outcome is very imbalanced ({minority_pct:.1f}% minority class)",
                        "column": outcome_column,
                        "minority_pct": str(float(minority_pct)),
                    }
                )

        return observations

    @classmethod
    def validate_complete(
        cls,
        df: Any,
        id_column: str | None = None,
        outcome_column: str | None = None,
        granularity: str = "unknown",
    ) -> dict[str, Any]:
        """
        Run complete data quality observation.

        Philosophy: Accept data, observe quality, store metadata.
        Only structural issues (empty dataset) are blocking errors.
        All quality issues are warnings for the semantic layer.

        Args:
            df: DataFrame to observe (pandas or polars - converted internally)
            id_column: Patient ID column (optional - from user mapping)
            outcome_column: Outcome column (optional - from user mapping)
            granularity: Data granularity hint (for context)

        Returns:
            Observation results dictionary with:
            {
                'is_valid': bool (True unless structurally broken),
                'schema_errors': list (only structural errors),
                'quality_warnings': list (all quality observations),
                'quality_metadata': dict (for semantic layer to use),
                'issues': list (combined for UI display),
                'summary': dict (summary statistics)
            }
        """
        # Convert pandas to polars at boundary
        df = _ensure_polars(df)

        schema_errors: list[str] = []
        quality_warnings = []
        all_issues = []

        logger.info(
            f"DataQualityValidator.validate_complete: "
            f"id_column={id_column}, outcome_column={outcome_column}, "
            f"df.shape={df.shape}, granularity={granularity}"
        )

        # === STRUCTURAL VALIDATION (blocking errors) ===
        if df.height == 0:
            return {
                "is_valid": False,
                "schema_errors": ["Dataset is empty (no rows)"],
                "quality_warnings": [],
                "quality_metadata": {},
                "issues": [{"severity": "error", "type": "empty_dataset", "message": "Dataset is empty (no rows)"}],
                "summary": {"total_rows": 0, "total_columns": 0, "errors": 1, "warnings": 0},
            }

        if df.width == 0:
            return {
                "is_valid": False,
                "schema_errors": ["Dataset has no columns"],
                "quality_warnings": [],
                "quality_metadata": {},
                "issues": [{"severity": "error", "type": "no_columns", "message": "Dataset has no columns"}],
                "summary": {"total_rows": df.height, "total_columns": 0, "errors": 1, "warnings": 0},
            }

        # === QUALITY OBSERVATIONS (non-blocking warnings) ===

        # Observe ID column if user has mapped one
        if id_column:
            id_observations = cls.observe_id_column(df, id_column)
            for obs in id_observations:
                if obs["severity"] in ("warning", "info"):
                    quality_warnings.append(obs)
                all_issues.append(obs)

        # Observe missing data patterns
        missing_observations = cls.observe_missing_data(df)
        for obs in missing_observations:
            if obs["severity"] in ("warning", "info"):
                quality_warnings.append(obs)
            all_issues.append(obs)

        # Observe outcome column if user has mapped one
        if outcome_column:
            outcome_observations = cls.observe_outcome_column(df, outcome_column)
            for obs in outcome_observations:
                if obs["severity"] in ("warning", "info"):
                    quality_warnings.append(obs)
                all_issues.append(obs)

        # === BUILD QUALITY METADATA (for semantic layer) ===
        quality_metadata = {
            "columns_with_high_missing": [
                obs["column"] for obs in quality_warnings if obs.get("type") in ("high_missing", "very_high_missing")
            ],
            "id_column_observations": [obs for obs in quality_warnings if obs.get("column") == id_column]
            if id_column
            else [],
            "outcome_observations": [obs for obs in quality_warnings if obs.get("column") == outcome_column]
            if outcome_column
            else [],
            "overall_missing_pct": next(
                (obs["missing_pct"] for obs in missing_observations if obs["type"] == "overall_missing"), 0.0
            ),
        }

        # === SUMMARY ===
        total_cells = df.height * df.width
        missing_cells = sum(df[col].null_count() for col in df.columns)

        summary = {
            "total_rows": df.height,
            "total_columns": df.width,
            "total_cells": total_cells,
            "missing_cells": int(missing_cells),
            "missing_pct": float(missing_cells / total_cells * 100) if total_cells > 0 else 0,
            "errors": len(schema_errors),
            "warnings": len([w for w in quality_warnings if w["severity"] == "warning"]),
        }

        # is_valid = True unless structural errors
        is_valid = len(schema_errors) == 0

        logger.info(
            f"Validation complete: is_valid={is_valid}, errors={len(schema_errors)}, warnings={len(quality_warnings)}"
        )

        return {
            "is_valid": is_valid,
            "schema_errors": schema_errors,
            "quality_warnings": quality_warnings,
            "quality_metadata": quality_metadata,
            "issues": all_issues,
            "summary": summary,
        }
