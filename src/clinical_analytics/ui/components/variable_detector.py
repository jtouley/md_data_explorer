"""
Variable Type Detector

Detects variable types from uploaded data using DATA-DRIVEN heuristics only.
No hardcoded column name patterns - the user makes explicit mapping decisions.

Types detected:
- Continuous (numeric with high cardinality)
- Categorical (limited unique values)
- Binary (exactly 2 unique values)
- Datetime (date/time types)
- High-cardinality identifier (potential ID column)
"""

from __future__ import annotations

from typing import Any

import polars as pl


def _ensure_polars_df(df: Any) -> pl.DataFrame:
    """Convert pandas DataFrame to polars if needed."""
    if isinstance(df, pl.DataFrame):
        return df
    return pl.from_pandas(df)


class VariableTypeDetector:
    """
    Detect variable types from uploaded data using data characteristics only.

    No hardcoded column name patterns - relies purely on:
    - Data types
    - Cardinality (unique value counts)
    - Value distributions
    """

    # Configurable thresholds
    CATEGORICAL_THRESHOLD = 20  # If unique values <= this, likely categorical
    ID_UNIQUENESS_THRESHOLD = 0.95  # If >95% unique, likely an identifier

    @classmethod
    def detect_variable_type(cls, series: pl.Series, column_name: str) -> tuple[str, dict]:
        """
        Detect variable type for a single column.

        Uses DATA characteristics only, not column names.

        Args:
            series: Polars Series
            column_name: Column name (for metadata only, not pattern matching)

        Returns:
            Tuple of (variable_type, metadata_dict)
            where variable_type is one of:
            - 'binary': exactly 2 unique values
            - 'categorical': limited unique values
            - 'continuous': numeric with high cardinality
            - 'datetime': date/time type
            - 'identifier': high cardinality, likely unique ID
            - 'text': high cardinality string
        """
        n_total = series.len()
        n_null = series.null_count()
        n_non_null = n_total - n_null
        n_unique = series.n_unique()

        # Uniqueness ratio (excluding nulls)
        uniqueness_ratio = n_unique / n_non_null if n_non_null > 0 else 0

        # Check for datetime types first
        if series.dtype in (pl.Date, pl.Datetime, pl.Time):
            return "datetime", {
                "dtype": str(series.dtype),
                "unique_count": n_unique,
            }

        # Check for binary (exactly 2 unique non-null values)
        if n_unique == 2:
            # Get the actual values
            unique_vals = series.drop_nulls().unique().to_list()
            return "binary", {
                "unique_count": 2,
                "values": unique_vals,
            }

        # Check for high-cardinality identifier (potential ID column)
        # Criteria: >95% unique AND no/few nulls
        if uniqueness_ratio > cls.ID_UNIQUENESS_THRESHOLD and n_null == 0:
            return "identifier", {
                "unique_count": n_unique,
                "uniqueness": float(uniqueness_ratio),
                "potential_id": True,
            }

        # Check for numeric types
        if series.dtype.is_numeric():
            # Low cardinality numeric = categorical (ordinal)
            if n_unique <= cls.CATEGORICAL_THRESHOLD:
                return "categorical", {
                    "unique_count": n_unique,
                    "numeric": True,
                    "values": sorted(series.drop_nulls().unique().to_list()),
                }
            # High cardinality numeric = continuous
            return "continuous", {
                "min": float(series.min()) if series.min() is not None else None,
                "max": float(series.max()) if series.max() is not None else None,
                "mean": float(series.mean()) if series.mean() is not None else None,
            }

        # String/other types
        if n_unique <= cls.CATEGORICAL_THRESHOLD:
            # Low cardinality string = categorical
            values = series.drop_nulls().unique().to_list()
            return "categorical", {
                "unique_count": n_unique,
                "numeric": False,
                "values": sorted(str(v) for v in values),
            }

        # High cardinality string
        return "text", {
            "unique_count": n_unique,
            "high_cardinality": True,
            "sample_values": [str(v) for v in series.drop_nulls().head(5).to_list()],
        }

    @classmethod
    def detect_all_variables(cls, df: Any) -> dict[str, dict]:
        """
        Detect variable types for all columns in a DataFrame.

        Args:
            df: DataFrame (pandas or polars - converted internally)

        Returns:
            Dictionary mapping column names to type info:
            {
                'column_name': {
                    'type': 'continuous',
                    'metadata': {...},
                    'missing_count': int,
                    'missing_pct': float,
                }
            }
        """
        # Convert pandas to polars at boundary
        df = _ensure_polars_df(df)

        results = {}
        n_rows = df.height

        for col in df.columns:
            series = df[col]
            var_type, metadata = cls.detect_variable_type(series, col)

            n_missing = series.null_count()

            results[col] = {
                "type": var_type,
                "metadata": metadata,
                "missing_count": int(n_missing),
                "missing_pct": float(n_missing / n_rows * 100) if n_rows > 0 else 0,
            }

        return results

    @classmethod
    def suggest_schema_mapping(cls, df: Any) -> dict[str, str | None]:
        """
        Suggest mapping to UnifiedCohort schema based on DATA characteristics.

        No hardcoded column name patterns - uses only:
        - Uniqueness ratio (for ID detection)
        - Binary type (for outcome detection)
        - Datetime type (for time_zero detection)

        Args:
            df: DataFrame (pandas or polars - converted internally)

        Returns:
            Dictionary with suggested mappings (hints only, user decides):
            {
                'patient_id': 'suggested_column' or None,
                'outcome': 'suggested_column' or None,
                'time_zero': 'suggested_column' or None
            }
        """
        variable_info = cls.detect_all_variables(df)

        suggestions = {"patient_id": None, "outcome": None, "time_zero": None}

        # Find best ID candidate: highest uniqueness, identifier type, no nulls
        best_id_col = None
        best_uniqueness = 0.0

        for col, info in variable_info.items():
            if info["type"] == "identifier":
                uniqueness = info["metadata"].get("uniqueness", 0)
                if uniqueness > best_uniqueness:
                    best_uniqueness = uniqueness
                    best_id_col = col

        suggestions["patient_id"] = best_id_col

        # Find best outcome candidate: binary type
        for col, info in variable_info.items():
            if info["type"] == "binary":
                suggestions["outcome"] = col
                break  # Take first binary column

        # Find best time_zero candidate: datetime type
        for col, info in variable_info.items():
            if info["type"] == "datetime":
                suggestions["time_zero"] = col
                break  # Take first datetime column

        return suggestions
