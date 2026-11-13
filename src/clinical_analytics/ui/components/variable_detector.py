# SPDX-License-Identifier: MIT
"""
Variable Type Detector

Detects variable types from uploaded data using DATA-DRIVEN heuristics only.
No hardcoded column name patterns - the user makes explicit mapping decisions.

Types detected:
- Continuous (numeric with high cardinality)
- Categorical (limited unique values)
- Binary (exactly 2 unique values)
- Datetime (date/time types)
- High-cardinality identifier (potential ID column): integers or text only, never a measurement
  (``core.identifier_rules``)
"""

from __future__ import annotations

import logging
from itertools import combinations
from typing import Any

import polars as pl

from clinical_analytics.core.identifier_rules import ROW_NUMBER_COLUMN, can_identify_patients

logger = logging.getLogger(__name__)


def _is_measurement(df: pl.DataFrame, column: str) -> bool:
    """A measurement (a rate, a bound, a lab value) is never an identifier: an ID built from it
    changes when the value is re-estimated, and renaming it to patient_id deletes the measurement
    from the dataset. Floats, fractional numbers stored as text, and columns named as an estimate,
    bound or rate (``core.identifier_rules``) all count, so a rate rounded to integers does too."""
    return not can_identify_patients(df.get_column(column))


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
    def detect_variable_type(cls, series: pl.Series, column_name: str) -> tuple[str, dict[str, Any]]:
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
        # Criteria: >95% unique, no nulls, and integers or text that is not a measurement: a unique
        # float (a rate to 6 decimals) is a value, never an ID.
        if uniqueness_ratio > cls.ID_UNIQUENESS_THRESHOLD and n_null == 0 and can_identify_patients(series):
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
            min_val = series.min()
            max_val = series.max()
            mean_val = series.mean()
            return "continuous", {
                "min": float(min_val) if min_val is not None and isinstance(min_val, (int, float)) else None,
                "max": float(max_val) if max_val is not None and isinstance(max_val, (int, float)) else None,
                "mean": float(mean_val) if mean_val is not None and isinstance(mean_val, (int, float)) else None,
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
    def detect_all_variables(cls, df: Any) -> dict[str, dict[str, Any]]:
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
    def suggest_schema_mapping(cls, df: Any) -> dict[str, str | None | dict[str, Any]]:
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

        from clinical_analytics.core.type_aliases import OptionalStr

        suggestions: dict[str, OptionalStr | dict[str, Any]] = {"patient_id": None, "outcome": None, "time_zero": None}

        # Check for patient_id - use ensure_patient_id logic but return original column name for suggestions
        if "patient_id" in df.columns:
            suggestions["patient_id"] = "patient_id"
        else:
            # Try to find single column identifier first
            best_id_col = None
            best_uniqueness = 0.0

            for col, info in variable_info.items():
                if info["type"] == "identifier":
                    uniqueness = info["metadata"].get("uniqueness", 0)
                    if uniqueness > best_uniqueness:
                        best_uniqueness = uniqueness
                        best_id_col = col

            if best_id_col:
                suggestions["patient_id"] = best_id_col
            else:
                # Try composite identifier (a row key: answers on it count rows)
                df_polars = _ensure_polars_df(df)
                composite_cols = cls.find_composite_identifier_candidates(df_polars)
                if composite_cols:
                    suggestions["patient_id"] = "patient_id"  # Synthetic ID
                    suggestions["_patient_id_metadata"] = {
                        "patient_id_source": "composite",
                        "patient_id_columns": composite_cols,
                    }

        # Find best outcome candidate among binary columns: the canonical name wins (mirrors the
        # patient_id check above), then numeric columns (the schema requires a numeric 0/1 outcome,
        # so a string binary like M/F is a last resort), then the first binary column.
        binary_cols = [col for col, info in variable_info.items() if info["type"] == "binary"]
        df_polars = _ensure_polars_df(df)
        numeric_binary = [col for col in binary_cols if df_polars[col].dtype.is_numeric()]
        if "outcome" in binary_cols:
            suggestions["outcome"] = "outcome"
        elif numeric_binary:
            suggestions["outcome"] = numeric_binary[0]
        elif binary_cols:
            suggestions["outcome"] = binary_cols[0]

        # Find best time_zero candidate: datetime type
        for col, info in variable_info.items():
            if info["type"] == "datetime":
                suggestions["time_zero"] = col
                break  # Take first datetime column

        return suggestions

    @classmethod
    def find_composite_identifier_candidates(cls, df: pl.DataFrame) -> list[str] | None:
        """
        Find combination of columns that together create a unique identifier.

        Uses data-driven heuristics only - no hardcoded values.
        Tries combinations of 2-4 columns. Parts are categorical, continuous (a year) or
        identifier columns with low missing values (<10%), never a measurement (float,
        fractional, or named as an estimate, bound or rate: a rate with 6 decimals makes any
        combination unique). The combination must be unique on every row: a key with repeats
        would merge different rows into one.

        Args:
            df: Polars DataFrame

        Returns:
            List of column names that together form unique key, or None if not found
        """
        n_rows = df.height
        if n_rows == 0:
            return None

        variable_info = cls.detect_all_variables(df)

        # Get candidate columns (low missing, reasonable types)
        candidates = []
        for col, info in variable_info.items():
            missing_pct = info.get("missing_pct", 100)
            var_type = info.get("type", "")

            # Skip if too many missing values
            if missing_pct > 10:
                continue
            if _is_measurement(df, col):
                continue

            # Prefer categorical, continuous, or identifier types
            # Exclude datetime, binary, and text (too high cardinality alone)
            if var_type in ("categorical", "continuous", "identifier"):
                candidates.append(col)

        if len(candidates) < 2:
            return None

        # Try combinations of increasing size
        for combo_size in range(2, min(5, len(candidates) + 1)):
            for combo in combinations(candidates, combo_size):
                # Check uniqueness of combination using Polars
                combo_cols = list(combo)

                # Create composite key by concatenating values
                composite = df.select(
                    pl.concat_str(
                        [pl.col(col).cast(pl.Utf8).fill_null("") for col in combo_cols],
                        separator="|",
                    ).alias("_composite_key")
                )

                # Exactly unique, or not a key
                if composite["_composite_key"].n_unique() == n_rows:
                    return combo_cols

        return None

    @classmethod
    def create_synthetic_patient_id(cls, df: pl.DataFrame, source_columns: list[str]) -> pl.DataFrame:
        """
        Create synthetic patient_id column from source columns using hash.

        Uses Polars expressions only - deterministic hash function.

        Args:
            df: Polars DataFrame
            source_columns: List of column names to combine

        Returns:
            DataFrame with added 'patient_id' column
        """
        # Create composite key string
        composite_expr = pl.concat_str(
            [pl.col(col).cast(pl.Utf8).fill_null("") for col in source_columns],
            separator="|",
        )

        # Hash using Polars's hash function (deterministic)
        # Use hash with seed=0 for consistency
        synthetic_id = composite_expr.hash(seed=0).cast(pl.Utf8)

        # Add as patient_id column
        return df.with_columns([synthetic_id.alias("patient_id")])

    @classmethod
    def ensure_patient_id(cls, df: Any) -> tuple[pl.DataFrame, dict[str, Any]]:
        """
        Ensure the DataFrame has a patient_id column, or a row number when nothing identifies rows.

        In order:
        - An existing 'patient_id' column is kept.
        - Else the most unique identifier column (integers or text, never a measurement) is
          copied to 'patient_id'; the source column stays, so it can still be queried.
        - Else a combination of columns unique on every row is hashed into 'patient_id'
          (source 'composite'). It is a row key (location x cause x sex x year), not a patient:
          answers on it count rows.
        - Else a row number column (``ROW_NUMBER_COLUMN``, 1..n) is added and no 'patient_id'
          exists (source 'row_index'): answers count rows.

        Args:
            df: DataFrame (pandas or polars - converted internally)

        Returns:
            Tuple of (DataFrame, metadata dict)
            Metadata includes:
            - 'patient_id_source': 'existing' | 'single_column' | 'composite' | 'row_index'
            - 'patient_id_columns': list of source columns (None for 'existing' and 'row_index')
            - 'row_number_column': the added row number column ('row_index' only)
        """
        df = _ensure_polars_df(df)
        from clinical_analytics.core.type_aliases import OptionalList, OptionalStr

        metadata: dict[str, OptionalStr | OptionalList] = {"patient_id_source": None, "patient_id_columns": None}
        n_rows = df.height

        logger.debug(f"ensure_patient_id: Starting with {n_rows} rows, {df.width} columns")

        # CONDITION 1: Check if patient_id already exists
        if "patient_id" in df.columns:
            logger.info("patient_id already exists in DataFrame")
            metadata["patient_id_source"] = "existing"
            return df, metadata  # Early return - no synthetic ID needed

        # CONDITION 2: Try to find single column identifier
        logger.debug("No existing patient_id, searching for single-column identifier...")
        variable_info = cls.detect_all_variables(df)
        best_id_col = None
        best_uniqueness = 0.0

        for col, info in variable_info.items():
            if info["type"] == "identifier" and not _is_measurement(df, col):
                uniqueness = info["metadata"].get("uniqueness", 0)
                if uniqueness > best_uniqueness:
                    best_uniqueness = uniqueness
                    best_id_col = col

        # Copied, never renamed: the source column stays queryable under its own name.
        if best_id_col:
            logger.info(f"Found single-column identifier with {best_uniqueness * 100:.1f}% uniqueness")
            metadata["patient_id_source"] = "single_column"
            metadata["patient_id_columns"] = [best_id_col]
            return df.with_columns(pl.col(best_id_col).alias("patient_id")), metadata

        # CONDITION 3: a combination unique on every row: a synthesized row key
        composite_cols = cls.find_composite_identifier_candidates(df)
        if composite_cols:
            logger.info(f"Creating composite row key from {len(composite_cols)} columns ({n_rows} rows)")
            metadata["patient_id_source"] = "composite"
            metadata["patient_id_columns"] = composite_cols
            return cls.create_synthetic_patient_id(df, composite_cols), metadata

        # CONDITION 4: Nothing identifies a row: number the rows (unit "rows", not "patients")
        row_column = ROW_NUMBER_COLUMN
        while row_column in df.columns:
            row_column = f"_{row_column}"
        logger.info(f"No identifier among {df.width} columns; added row number column for {n_rows} rows")
        metadata["patient_id_source"] = "row_index"
        metadata["row_number_column"] = row_column
        return df.with_row_index(row_column, offset=1).with_columns(pl.col(row_column).cast(pl.Int64)), metadata
