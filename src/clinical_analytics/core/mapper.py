"""
Column Mapping Engine - Config-driven data transformation.

This module provides generic utilities for applying column mappings,
data type conversions, and transformations based on YAML configuration.
"""

from pathlib import Path
from typing import Any

import polars as pl
import yaml

from clinical_analytics.core.schema import DataQualityError, UnifiedCohort


class ColumnMapper:
    """
    Generic column mapping engine that applies transformations
    based on configuration rather than hardcoded logic.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """
        Initialize mapper with dataset configuration.

        Args:
            config: Dataset configuration dictionary from datasets.yaml
        """
        self.config = config
        self.column_mapping = config.get("column_mapping", {})
        self.outcomes = config.get("outcomes", {})
        self.analysis_config = config.get("analysis", {})

    @classmethod
    def from_yaml(cls, dataset_name: str, config_path: Path | None = None) -> "ColumnMapper":
        """
        Create mapper from YAML config file.

        Args:
            dataset_name: Name of dataset in config
            config_path: Path to datasets.yaml

        Returns:
            ColumnMapper instance
        """
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent.parent / "config" / "datasets.yaml"

        with open(config_path) as f:
            all_configs = yaml.safe_load(f)

        if dataset_name not in all_configs:
            raise KeyError(f"Dataset '{dataset_name}' not found in config")

        return cls(all_configs[dataset_name])

    def apply_outcome_transformations(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Apply outcome transformations based on config.

        Args:
            df: Input Polars DataFrame

        Returns:
            DataFrame with outcome columns added

        Raises:
            DataQualityError: If unmapped values are found in the source column
        """
        for outcome_name, outcome_spec in self.outcomes.items():
            source_col = outcome_spec["source_column"]
            outcome_type = outcome_spec["type"]

            if outcome_type == "binary" and "mapping" in outcome_spec:
                # Apply binary mapping (e.g., yes/no -> 1/0)
                mapping = outcome_spec["mapping"]

                # Validate: check for unmapped values before transformation
                # Get all unique non-null values in source column
                if source_col in df.columns:
                    # Get unique values (excluding NULLs)
                    unique_values = df.select(pl.col(source_col)).unique().drop_nulls()

                    # Normalize mapping keys for comparison (lowercase strings)
                    normalized_mapping: dict[str | bool, Any] = {}
                    for key in mapping.keys():
                        if isinstance(key, bool):
                            normalized_mapping[key] = key
                        elif isinstance(key, str):
                            normalized_mapping[key.lower()] = key
                        else:
                            normalized_mapping[key] = key

                    # Find unmapped values
                    unmapped = []
                    for row in unique_values.iter_rows():
                        value = row[0]
                        # Normalize value for comparison
                        if isinstance(value, str):
                            normalized_value: str | bool | Any = value.lower()
                        elif isinstance(value, bool):
                            normalized_value = value
                        else:
                            normalized_value = value

                        # Check if value has a mapping
                        if normalized_value not in normalized_mapping:
                            unmapped.append(value)

                    # Raise error if unmapped values found
                    if unmapped:
                        raise DataQualityError(
                            f"Unmapped values found in column '{source_col}' for outcome '{outcome_name}': "
                            f"{unmapped[:10]}{'...' if len(unmapped) > 10 else ''}. "
                            f"Available mappings: {list(mapping.keys())}",
                            unmapped_values=unmapped,
                        )

                # Build Polars expression for mapping
                # Default to NULL (not 0) to preserve data integrity:
                # - Unmapped values remain NULL (explicit missing data)
                # - Downstream code uses .drop_nulls() before aggregations (see compute.py)
                # - Statistical analyses handle NULLs correctly (logistic regression, etc.)
                # - Validation raises DataQualityError if unmapped non-null values exist
                expr = pl.lit(None)  # default to NULL, not 0
                for key, value in mapping.items():
                    # Handle both string and boolean keys
                    if isinstance(key, bool):
                        # Boolean key: compare directly
                        expr = pl.when(pl.col(source_col) == key).then(value).otherwise(expr)
                    elif isinstance(key, str):
                        # String key: convert to lowercase for comparison
                        # Handle case where source column might be string or boolean
                        expr = (
                            pl.when(pl.col(source_col).cast(pl.Utf8).str.to_lowercase() == key.lower())
                            .then(value)
                            .otherwise(expr)
                        )
                    else:
                        # Numeric or other types: direct comparison
                        expr = pl.when(pl.col(source_col) == key).then(value).otherwise(expr)

                df = df.with_columns([expr.alias(outcome_name)])

            elif outcome_type == "binary" and "aggregation" in outcome_spec:
                # For aggregated data (handled in loader)
                pass

        return df

    def apply_filters(self, df: pl.DataFrame, filters: dict[str, Any]) -> pl.DataFrame:
        """
        Apply filters to DataFrame based on config-defined filter types.

        Args:
            df: Input Polars DataFrame
            filters: Dictionary of filter_name -> filter_value

        Returns:
            Filtered DataFrame
        """
        # Get filter definitions from config
        filter_definitions = self.config.get("filters", {})

        for filter_name, filter_value in filters.items():
            if filter_value is None:
                continue

            # Get filter definition from config
            filter_def = filter_definitions.get(filter_name, {})
            filter_type = filter_def.get("type", "equals")
            column = filter_def.get("column", filter_name)

            # Skip if column doesn't exist
            if column not in df.columns:
                continue

            if filter_type == "equals":
                # Simple equality filter
                if isinstance(filter_value, bool):
                    # Boolean filter - check for yes/no strings or boolean values
                    # Need to handle different column types separately to avoid type errors
                    col_dtype = df[column].dtype

                    # Define type-specific truthy/falsy value mappings
                    int_types = [
                        pl.Int8,
                        pl.Int16,
                        pl.Int32,
                        pl.Int64,
                        pl.UInt8,
                        pl.UInt16,
                        pl.UInt32,
                        pl.UInt64,
                    ]
                    float_types = [pl.Float32, pl.Float64]
                    string_types = [pl.Utf8, pl.Categorical]

                    # Get the comparison value based on column type and filter value
                    if col_dtype in string_types:
                        comparison_value_str = "yes" if filter_value else "no"
                        df = df.filter(pl.col(column).str.to_lowercase() == comparison_value_str)
                    elif col_dtype == pl.Boolean:
                        df = df.filter(pl.col(column) == filter_value)
                    elif col_dtype in int_types:
                        comparison_value_int: int = 1 if filter_value else 0
                        df = df.filter(pl.col(column) == comparison_value_int)
                    elif col_dtype in float_types:
                        comparison_value_float: float = 1.0 if filter_value else 0.0
                        df = df.filter(pl.col(column) == comparison_value_float)
                    else:
                        # Unknown type - try direct comparison
                        df = df.filter(pl.col(column) == filter_value)
                else:
                    df = df.filter(pl.col(column) == filter_value)

            elif filter_type == "in":
                # Value in list
                if isinstance(filter_value, list):
                    # Ensure column and filter values have compatible types
                    # Cast filter_value list elements to match column dtype
                    col_dtype = df[column].dtype
                    try:
                        # If column is string, ensure filter values are strings
                        if col_dtype == pl.Utf8:
                            filter_value = [str(v) for v in filter_value]
                        # If column is numeric, try to convert filter values
                        elif col_dtype in [pl.Int8, pl.Int16, pl.Int32, pl.Int64]:
                            filter_value = [int(v) for v in filter_value if v is not None]
                        elif col_dtype in [pl.Float32, pl.Float64]:
                            filter_value = [float(v) for v in filter_value if v is not None]

                        df = df.filter(pl.col(column).is_in(filter_value))
                    except Exception as e:
                        import logging

                        logger = logging.getLogger(__name__)
                        logger.error(
                            f"Error in 'in' filter for column {column} (dtype={col_dtype}): "
                            f"{type(e).__name__}: {str(e)}"
                        )
                        filter_type = type(filter_value[0]) if filter_value else "empty"
                        logger.error(f"Filter values: {filter_value[:10]}... (type: {filter_type})")
                        raise

            elif filter_type == "range":
                # Range filter (min, max)
                if isinstance(filter_value, dict):
                    if "min" in filter_value:
                        df = df.filter(pl.col(column) >= filter_value["min"])
                    if "max" in filter_value:
                        df = df.filter(pl.col(column) <= filter_value["max"])

            elif filter_type == "exists":
                # Check if value exists (not null)
                if filter_value:
                    df = df.filter(pl.col(column).is_not_null())
                else:
                    df = df.filter(pl.col(column).is_null())
            else:
                # Default: simple equality
                df = df.filter(pl.col(column) == filter_value)

        return df

    def apply_aggregations(
        self, df: pl.DataFrame, group_by: str, aggregation_config: dict[str, Any] | None = None
    ) -> pl.DataFrame:
        """
        Apply aggregations to DataFrame based on config.

        Args:
            df: Input Polars DataFrame (time-series data)
            group_by: Column to group by (typically patient_id)
            aggregation_config: Optional aggregation config (defaults to self.config['aggregation'])

        Returns:
            Aggregated DataFrame (patient-level)
        """
        if aggregation_config is None:
            aggregation_config = self.config.get("aggregation", {})

        if not aggregation_config:
            return df

        # Build aggregation expressions
        agg_exprs = []

        # Handle static features
        static_features = aggregation_config.get("static_features", [])
        for feature_spec in static_features:
            source_col = feature_spec["column"]
            method = feature_spec.get("method", "first")
            target_col = feature_spec.get("target", source_col.lower())

            if source_col not in df.columns:
                continue

            # Map method to Polars aggregation
            if method == "first":
                agg_exprs.append(pl.col(source_col).first().alias(target_col))
            elif method == "last":
                agg_exprs.append(pl.col(source_col).last().alias(target_col))
            elif method == "max":
                agg_exprs.append(pl.col(source_col).max().alias(target_col))
            elif method == "min":
                agg_exprs.append(pl.col(source_col).min().alias(target_col))
            elif method == "mean":
                agg_exprs.append(pl.col(source_col).mean().alias(target_col))
            elif method == "sum":
                agg_exprs.append(pl.col(source_col).sum().alias(target_col))
            elif method == "count":
                agg_exprs.append(pl.col(source_col).count().alias(target_col))

        # Handle outcome aggregation
        outcome_spec = aggregation_config.get("outcome", {})
        if outcome_spec:
            source_col = outcome_spec["column"]
            method = outcome_spec.get("method", "max")
            target_col = outcome_spec.get("target", "outcome")

            if source_col in df.columns:
                if method == "max":
                    agg_exprs.append(pl.col(source_col).max().alias(target_col))
                elif method == "min":
                    agg_exprs.append(pl.col(source_col).min().alias(target_col))
                elif method == "first":
                    agg_exprs.append(pl.col(source_col).first().alias(target_col))
                elif method == "last":
                    agg_exprs.append(pl.col(source_col).last().alias(target_col))
                elif method == "mean":
                    agg_exprs.append(pl.col(source_col).mean().alias(target_col))

        # Add count of records if not already included
        if "num_hours" not in [expr.meta.output_name() if hasattr(expr, "meta") else None for expr in agg_exprs]:
            agg_exprs.append(pl.len().alias("num_hours"))

        # Group by and aggregate
        if group_by in df.columns and agg_exprs:
            return df.group_by(group_by).agg(agg_exprs)
        else:
            return df

    def map_to_unified_cohort(
        self,
        df: pl.DataFrame,
        time_zero_value: str | None = None,
        outcome_col: str | None = None,
        outcome_label: str | None = None,
    ) -> pl.DataFrame:
        """
        Map DataFrame columns to UnifiedCohort schema based on config.

        Args:
            df: Input Polars DataFrame
            time_zero_value: Override for time_zero (if not in data)
            outcome_col: Which outcome column to use (if multiple)
            outcome_label: Label for the outcome

        Returns:
            DataFrame conforming to UnifiedCohort schema
        """
        # Determine which outcome to use
        if outcome_col is None:
            outcome_col = self.analysis_config.get("default_outcome", "outcome")

        # Build select expressions
        select_exprs = []

        # Map each required column
        for target_col in UnifiedCohort.REQUIRED_COLUMNS:
            if target_col == UnifiedCohort.PATIENT_ID:
                # Find source column for patient_id
                source_col = self._find_source_column(target_col)
                if source_col and source_col in df.columns:
                    select_exprs.append(pl.col(source_col).alias(target_col))

            elif target_col == UnifiedCohort.TIME_ZERO:
                # Time zero - use provided value or find in mapping
                if time_zero_value:
                    select_exprs.append(pl.lit(time_zero_value).str.strptime(pl.Datetime, "%Y-%m-%d").alias(target_col))
                else:
                    source_col = self._find_source_column(target_col)
                    if source_col and source_col in df.columns:
                        select_exprs.append(pl.col(source_col).alias(target_col))

            elif target_col == UnifiedCohort.OUTCOME:
                # Outcome column
                if outcome_col in df.columns:
                    select_exprs.append(pl.col(outcome_col).alias(target_col))

            elif target_col == UnifiedCohort.OUTCOME_LABEL:
                # Outcome label
                if outcome_label:
                    select_exprs.append(pl.lit(outcome_label).alias(target_col))
                else:
                    # Use outcome column name as label
                    select_exprs.append(pl.lit(outcome_col).alias(target_col))

        # Add feature columns based on mapping
        feature_mapping = {
            source: target
            for source, target in self.column_mapping.items()
            if target not in UnifiedCohort.REQUIRED_COLUMNS
        }

        for source_col, target_col in feature_mapping.items():
            if source_col in df.columns:
                select_exprs.append(pl.col(source_col).alias(target_col))

        # Select and return
        return df.select(select_exprs)

    def _find_source_column(self, target_col: str) -> str | None:
        """
        Find source column name for a target column.

        Args:
            target_col: Target column name (e.g., 'patient_id')

        Returns:
            Source column name or None
        """
        for source, target in self.column_mapping.items():
            if target == target_col:
                return str(source)  # Explicitly convert to str
        return None

    def get_default_predictors(self) -> list[str]:
        """Get list of default predictor variables from config."""
        result = self.analysis_config.get("default_predictors", [])
        return list(result) if result else []

    def get_categorical_variables(self) -> list[str]:
        """Get list of categorical variables from config."""
        result = self.analysis_config.get("categorical_variables", [])
        return list(result) if result else []

    def get_default_outcome(self) -> str:
        """Get default outcome column name from config."""
        result = self.analysis_config.get("default_outcome", "outcome")
        return str(result)

    def get_default_filters(self) -> dict[str, Any]:
        """Get default filter settings from config."""
        result = self.config.get("default_filters", {})
        return dict(result) if result else {}

    def get_time_zero_value(self) -> str | None:
        """
        Get time_zero value from config.

        Returns:
            Time zero value as string (YYYY-MM-DD) or None
        """
        time_zero_config = self.config.get("time_zero", {})
        if isinstance(time_zero_config, str):
            return time_zero_config
        elif isinstance(time_zero_config, dict):
            return time_zero_config.get("value")
        return None

    def get_default_outcome_label(self, outcome_col: str) -> str:
        """
        Get default outcome label from config for a given outcome column.

        Args:
            outcome_col: Name of the outcome column

        Returns:
            Outcome label string
        """
        # Check for explicit outcome label mapping in config
        outcome_labels = self.config.get("outcome_labels", {})
        if outcome_col in outcome_labels:
            return str(outcome_labels[outcome_col])

        # Check if outcome has a label in its definition
        outcome_def = self.outcomes.get(outcome_col, {})
        if "label" in outcome_def:
            return str(outcome_def["label"])

        # Default: use outcome column name
        return outcome_col


def load_dataset_config(dataset_name: str, config_path: Path | None = None) -> dict[str, Any]:
    """
    Load configuration for a specific dataset.

    Args:
        dataset_name: Name of dataset
        config_path: Path to datasets.yaml

    Returns:
        Dataset configuration dictionary
    """
    if config_path is None:
        config_path = Path(__file__).parent.parent.parent.parent / "data" / "configs" / "datasets.yaml"

    with open(config_path) as f:
        all_configs = yaml.safe_load(f)

    if dataset_name not in all_configs:
        raise KeyError(f"Dataset '{dataset_name}' not found in config")

    result = all_configs[dataset_name]
    if isinstance(result, dict):
        return {k: v for k, v in result.items()}
    return {}


def get_global_config(config_path: Path | None = None) -> dict:
    """
    Load global configuration settings.

    Args:
        config_path: Path to datasets.yaml

    Returns:
        Global configuration dictionary
    """
    if config_path is None:
        config_path = Path(__file__).parent.parent.parent.parent / "data" / "configs" / "datasets.yaml"

    with open(config_path) as f:
        all_configs = yaml.safe_load(f)

    result = all_configs.get("global", {}) if isinstance(all_configs, dict) else {}
    return dict(result) if isinstance(result, dict) else {}
