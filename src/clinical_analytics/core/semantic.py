"""
Semantic Layer - Dynamic SQL Generation via Ibis.

This module provides a DRY, config-driven semantic layer that generates SQL
behind the scenes based on dataset configurations. No more custom Python
mapping logic - just define your logic in YAML and let Ibis compile to SQL.
"""

import hashlib
import json
import logging
import re
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import ibis
import pandas as pd
import polars as pl
from ibis import _

from clinical_analytics.core.mapper import load_dataset_config
from clinical_analytics.core.schema import UnifiedCohort

if TYPE_CHECKING:
    from clinical_analytics.core.dataset import Granularity
    from clinical_analytics.core.query_plan import QueryPlan

# Phase 3.3: Import chart_spec generation function
from clinical_analytics.core.query_plan import generate_chart_spec

logger = logging.getLogger(__name__)


class TypeValidationError(Exception):
    """
    Exception raised when filter type validation fails.

    This indicates a validation bug - LLM validation layers should have
    caught the type mismatch before execution.

    Attributes:
        column: Column name where type mismatch occurred
        expected_type: Expected column type (e.g., "float64", "int64")
        actual_type: Actual type of the filter value (e.g., "str")
        message: Optional custom message
    """

    def __init__(
        self,
        column: str,
        expected_type: str,
        actual_type: str,
        message: str | None = None,
    ) -> None:
        """
        Initialize TypeValidationError.

        Args:
            column: Column name where type mismatch occurred
            expected_type: Expected column type
            actual_type: Actual type of the filter value
            message: Optional custom message (defaults to generated message)
        """
        self.column = column
        self.expected_type = expected_type
        self.actual_type = actual_type

        if message is None:
            message = (
                f"Type validation failed for column '{column}': "
                f"expected {expected_type}, got {actual_type}. "
                "Pre-execution validation should have caught this - this is a validation bug."
            )

        super().__init__(message)


def validate_query_against_schema(plan: "QueryPlan", active_version: dict[str, Any]) -> list[str]:
    """
    Validate QueryPlan column references against active version schema (Phase 8).

    Checks if metric, group_by, and filter columns exist in the active version's schema.
    Returns warnings for missing columns (non-blocking - execution proceeds).

    Args:
        plan: QueryPlan to validate
        active_version: Active version entry dict with schema information

    Returns:
        List of warning messages (empty if all columns exist)
    """
    warnings: list[str] = []

    # Extract schema from active version
    schema = active_version.get("schema", {}).get("inferred_schema", {})
    if not schema:
        # No schema available, skip validation
        return warnings

    # Get table name from schema (first table, or use table name from query context)
    table_name = list(schema.keys())[0] if schema else None
    if not table_name:
        return warnings

    # Get available columns from schema
    table_schema = schema.get(table_name, {})
    columns = table_schema.get("columns", {})
    available_columns = set(columns.keys())

    # Check metric column
    if plan.metric and plan.metric not in available_columns:
        warnings.append(
            f"Metric column '{plan.metric}' not in active version schema. "
            "Query may fail at runtime. Consider rolling back to a version with this column."
        )

    # Check group_by column
    if plan.group_by and plan.group_by not in available_columns:
        warnings.append(
            f"Group-by column '{plan.group_by}' not in active version schema. "
            "Query may fail at runtime. Consider rolling back to a version with this column."
        )

    # Check filter columns
    for filter_spec in plan.filters:
        if filter_spec.column not in available_columns:
            warnings.append(
                f"Filter column '{filter_spec.column}' not in active version schema. "
                "Query may fail at runtime. Consider rolling back to a version with this column."
            )

    return warnings


def _safe_identifier(name: str, max_len: int = 50) -> str:
    """
    Generate a SQL-safe identifier from a dataset name.

    Handles any user-provided name (hyphens, dots, spaces, emojis, etc.)
    by sanitizing and adding a hash for uniqueness.

    Args:
        name: Original dataset name (can contain any characters)
        max_len: Maximum length for base name (before hash)

    Returns:
        SQL-safe identifier (e.g., "mimic_iv_clinical_demo_2_2_a1b2c3d4")
    """
    # Replace non-alphanumeric (except underscore) with underscore
    base = re.sub(r"[^0-9a-zA-Z_]+", "_", name).strip("_").lower()

    # Ensure it starts with letter or underscore (not a number)
    if not base or base[0].isdigit():
        base = f"t_{base}" if base else "t"

    # Add hash suffix for uniqueness and collision prevention
    h = hashlib.sha256(name.encode("utf-8")).hexdigest()[:8]

    # Limit base length
    base = base[:max_len]

    return f"{base}_{h}"


# SQL identifier validation pattern: must start with letter or underscore,
# followed by letters, digits, or underscores
_SQL_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _validate_table_identifier(name: str) -> str:
    """
    Validate table identifier against SQL identifier pattern. Fail closed.

    This is a security function - if the identifier doesn't match the allowlist
    pattern, we reject it entirely rather than trying to sanitize.

    Args:
        name: Table identifier to validate

    Returns:
        The validated name (unchanged if valid)

    Raises:
        ValueError: If identifier is invalid (contains special chars, etc.)
    """
    if not name:
        raise ValueError("Table identifier cannot be empty")

    if not _SQL_IDENTIFIER_PATTERN.match(name):
        raise ValueError(
            f"Invalid table identifier: '{name}'. "
            f"Must match pattern: ^[A-Za-z_][A-Za-z0-9_]*$ "
            f"(start with letter or underscore, contain only letters, digits, underscores)"
        )

    # Additional safety: check length (DuckDB has limits)
    if len(name) > 255:
        raise ValueError(f"Table identifier too long: {len(name)} chars (max 255)")

    return name


class SemanticLayer:
    """
    Base semantic layer class that generates SQL dynamically from config.

    DRY Principle: All datasets use this same class, just with different configs.
    """

    def __init__(
        self,
        dataset_name: str,
        config: dict[str, Any] | None = None,
        workspace_root: Path | None = None,
        upload_id: str | None = None,
        dataset_version: str | None = None,
    ):
        """
        Initialize semantic layer for a dataset.

        Args:
            dataset_name: Name of dataset (e.g., 'covid_ms', 'sepsis')
            config: Optional config dict (if None, loads from datasets.yaml)
            workspace_root: Optional workspace root path (if None, auto-detects)
            upload_id: Optional upload ID for user-uploaded datasets (Phase 2: alias persistence)
            dataset_version: Optional dataset version for user-uploaded datasets (Phase 2: alias persistence)
        """
        if config is None:
            config = load_dataset_config(dataset_name)

        self.config = config
        self.dataset_name = dataset_name
        self.upload_id = upload_id
        self.dataset_version = dataset_version

        # Detect workspace root
        self.workspace_root = self._detect_workspace_root(workspace_root)
        logger.info(
            f"Initializing semantic layer for dataset '{dataset_name}'",
            extra={"workspace_root": str(self.workspace_root)},
        )

        # Connect to DuckDB (in-memory for now, can be file-based later)
        self.con = ibis.duckdb.connect()

        # Register data source
        self._register_source()

        # Build base semantic view (lazy - no SQL executed yet)
        self._base_view = None

        # Alias index (lazy initialization)
        self._alias_index: dict[str, str] | None = None  # normalized_alias -> canonical_name
        self._alias_to_canonicals: dict[str, set[str]] | None = None  # For collision detection
        self._collision_warnings: set[str] | None = None  # Aliases that map to multiple canonicals

        # Phase 2: Load user aliases if upload_id and dataset_version provided
        if upload_id and dataset_version:
            self._load_user_aliases(upload_id, dataset_version)

    def _detect_workspace_root(self, provided_root: Path | None = None) -> Path:
        """
        Detect workspace root using priority order:
        1. Provided workspace_root parameter
        2. config.get("workspace_root")
        3. Walk up from __file__ until finding marker (.git or pyproject.toml)
        4. Fallback to Path.cwd() with warning

        Args:
            provided_root: Optional workspace root path provided at init

        Returns:
            Path to workspace root
        """
        # Priority 1: Use provided parameter
        if provided_root is not None:
            logger.debug(f"Using provided workspace root: {provided_root}")
            return provided_root.resolve()

        # Priority 2: Check config
        config_root = self.config.get("workspace_root")
        if config_root:
            root_path = Path(config_root).resolve()
            logger.debug(f"Using workspace root from config: {root_path}")
            return root_path

        # Priority 3: Walk up from __file__ looking for markers
        # Markers: .git (directory) or pyproject.toml (file)
        current = Path(__file__).parent
        while current != current.parent:  # Stop at filesystem root
            # Check for .git directory
            if (current / ".git").exists() and (current / ".git").is_dir():
                logger.debug(f"Detected workspace root via .git marker: {current}")
                return current.resolve()
            # Check for pyproject.toml file
            if (current / "pyproject.toml").exists() and (current / "pyproject.toml").is_file():
                logger.debug(f"Detected workspace root via pyproject.toml marker: {current}")
                return current.resolve()
            current = current.parent

        # Priority 4: Fallback to cwd()
        fallback_root = Path.cwd()
        logger.warning(
            f"Could not detect workspace root via markers (.git or pyproject.toml). "
            f"Falling back to current working directory: {fallback_root}. "
            f"This may cause path resolution issues if running from a different directory."
        )
        return fallback_root.resolve()

    def _register_source(self) -> None:
        """Register the raw data source (CSV, table, etc.) with DuckDB."""
        init_params = self.config.get("init_params", {})

        if "source_path" in init_params:
            original_path = Path(init_params["source_path"])

            # Resolve relative paths relative to workspace root
            if original_path.is_absolute():
                source_path = original_path
                logger.debug(
                    f"Using absolute source path: {source_path}",
                    extra={"original_path": str(original_path), "resolved_path": str(source_path)},
                )
            else:
                # Resolve relative to workspace root
                source_path = self.workspace_root / original_path
                logger.debug(
                    "Resolved relative source path",
                    extra={
                        "original_path": str(original_path),
                        "resolved_path": str(source_path),
                        "workspace_root": str(self.workspace_root),
                    },
                )

            if not source_path.exists():
                raise FileNotFoundError(
                    f"Source file not found: {source_path} "
                    f"(original path: {original_path}, workspace root: {self.workspace_root})"
                )

            # Generate SQL-safe table name from dataset name
            safe_name = _safe_identifier(self.dataset_name)
            table_name = f"{safe_name}_raw"

            # Check if it's a directory (for multi-file datasets like Sepsis)
            if source_path.is_dir():
                # For directory-based sources, we need to aggregate first
                # This is handled by dataset-specific logic
                # For now, we'll use DuckDB's ability to read multiple files
                # But Sepsis needs special handling - see SepsisDataset
                raise NotImplementedError(
                    f"Directory sources need dataset-specific handling. See {self.dataset_name} dataset implementation."
                )
            else:
                # Single file (CSV)
                abs_path = str(source_path.resolve())
                logger.debug(f"Registering source file with DuckDB: {abs_path}")

                # Get underlying DuckDB connection
                duckdb_con = self.con.con

                # SECURITY: Validate identifier before SQL interpolation
                # table_name comes from _safe_identifier which already produces valid
                # identifiers, but we validate as defense-in-depth
                validated_name = _validate_table_identifier(table_name)

                # Use quoted identifiers for extra safety + parameter binding for file path
                # DuckDB supports ? placeholders for parameterized queries
                quoted_table = f'"{validated_name}"'
                duckdb_con.execute(
                    f"CREATE OR REPLACE TABLE {quoted_table} AS SELECT * FROM read_csv_auto(?)",
                    [abs_path],
                )

                # Now reference it via Ibis
                self.raw = self.con.table(validated_name)
                logger.info(f"Successfully registered source table '{validated_name}' from {abs_path}")

        elif "db_table" in init_params:
            # Database table source (already registered)
            # SECURITY: db_table could come from config which may be user-influenced
            # Validate the identifier before using it
            table_name = init_params["db_table"]

            # Validate: fail closed if identifier is not safe
            validated_name = _validate_table_identifier(table_name)
            logger.debug(f"Using database table source: {validated_name}")
            self.raw = self.con.table(validated_name)
            logger.info(f"Successfully registered database table '{validated_name}'")
        else:
            raise ValueError(f"No valid source found in config for {self.dataset_name}")

    def get_base_view(self) -> Any:
        """
        Build the semantic view from config - defines logic, doesn't execute.

        This is where the "SQL behind the scenes" magic happens. We read the
        YAML config and build Ibis expressions that compile to SQL.
        """
        if self._base_view is not None:
            return self._base_view

        mutations = {}

        # 1. Build outcome columns from config
        outcomes = self.config.get("outcomes", {})
        for outcome_name, outcome_def in outcomes.items():
            source_col = outcome_def["source_column"]

            if outcome_def.get("type") == "binary" and "mapping" in outcome_def:
                # Build CASE WHEN expression for binary mapping using modern Ibis API
                mapping = outcome_def["mapping"]

                # Build nested ifelse expression (reverse order for proper chaining)
                result = ibis.null()  # Default value
                for key, value in reversed(list(mapping.items())):
                    # Handle string keys (case-insensitive comparison)
                    if isinstance(key, str):
                        condition = _[source_col].cast(str).lower() == key.lower()
                    else:
                        # Handle boolean/numeric keys
                        condition = _[source_col] == key

                    result = condition.ifelse(value, result)

                mutations[outcome_name] = result.cast("int64")
            else:
                # Simple column reference
                mutations[outcome_name] = _[source_col]

        # 2. Apply column mappings (rename columns)
        column_mapping = self.config.get("column_mapping", {})
        for source_col, target_col in column_mapping.items():
            if source_col in self.raw.columns:
                mutations[target_col] = _[source_col]
            elif source_col in mutations:
                # Column was created in step 1 (outcomes) - can rename it
                mutations[target_col] = mutations[source_col]
            elif source_col == target_col:
                # Column already has correct name
                if source_col in self.raw.columns:
                    mutations[target_col] = _[source_col]

        # 3. Add time_zero (from config)
        time_zero_config = self.config.get("time_zero", {})
        if isinstance(time_zero_config, dict) and "value" in time_zero_config:
            # Static time_zero value
            time_zero_val = time_zero_config["value"]
            mutations[UnifiedCohort.TIME_ZERO] = ibis.literal(time_zero_val).cast("date")
        elif isinstance(time_zero_config, dict) and "source_column" in time_zero_config:
            # Time_zero from a column
            source_col = time_zero_config["source_column"]
            if source_col in self.raw.columns:
                mutations[UnifiedCohort.TIME_ZERO] = _[source_col]
        elif isinstance(time_zero_config, str):
            # Simple string value
            mutations[UnifiedCohort.TIME_ZERO] = ibis.literal(time_zero_config).cast("date")

        # 4. Ensure required UnifiedCohort columns exist
        # patient_id should already be mapped, outcome will be set later

        # Build the view with all mutations
        if mutations:
            self._base_view = self.raw.mutate(**mutations)
        else:
            self._base_view = self.raw

        return self._base_view

    def apply_filters(self, view, filters: dict[str, Any]) -> ibis.Table:
        """
        Apply filters dynamically to the view.

        Args:
            view: Ibis table expression
            filters: Dictionary of filter_name -> filter_value

        Returns:
            Filtered Ibis table expression
        """
        if not filters:
            return view

        filter_definitions = self.config.get("filters", {})

        for filter_name, filter_value in filters.items():
            if filter_value is None:
                continue

            # Get filter definition from config
            filter_def = filter_definitions.get(filter_name, {})
            filter_type = filter_def.get("type", "equals")
            column = filter_def.get("column", filter_name)

            # Check if column exists in view
            if column not in view.columns:
                continue

            if filter_type == "equals":
                if isinstance(filter_value, bool):
                    # Boolean filter - handle different column types
                    view = view.filter(_[column] == filter_value)
                else:
                    view = view.filter(_[column] == filter_value)
            elif filter_type == "in":
                if isinstance(filter_value, list):
                    view = view.filter(_[column].isin(filter_value))
            elif filter_type == "range":
                if isinstance(filter_value, dict):
                    if "min" in filter_value:
                        view = view.filter(_[column] >= filter_value["min"])
                    if "max" in filter_value:
                        view = view.filter(_[column] <= filter_value["max"])
            elif filter_type == "exists":
                if filter_value:
                    view = view.filter(~_[column].isnull())
                else:
                    view = view.filter(_[column].isnull())

        return view

    def build_cohort_query(
        self,
        granularity: "Granularity" = "patient_level",
        outcome_col: str | None = None,
        outcome_label: str | None = None,
        filters: dict[str, Any] | None = None,
    ) -> ibis.Table:
        """
        Build a query that returns UnifiedCohort-compliant data.

        Args:
            granularity: Grain level (patient_level, admission_level, event_level).
                        Accepted but not validated (validation done by dataset classes).
                        For single-table datasets, ignored (always patient-level queries).
                        For multi-table datasets (future), selects appropriate materialized mart.
            outcome_col: Which outcome column to use (defaults to config)
            outcome_label: Label for outcome (defaults to config)
            filters: Optional filters to apply

        Returns:
            Ibis table expression (lazy - SQL not executed yet)
        """
        # NOTE: SemanticLayer is permissive. Dataset classes validate granularity support.
        # For single-table datasets, granularity is currently ignored (patient-level query only).
        view = self.get_base_view()

        # Apply default filters from config
        default_filters = self.config.get("default_filters", {})
        all_filters = {**default_filters, **(filters or {})}

        # Remove target_outcome from filters (it's not a data filter)
        filter_only = {k: v for k, v in all_filters.items() if k != "target_outcome"}
        view = self.apply_filters(view, filter_only)

        # Determine outcome column
        if outcome_col is None:
            analysis_config = self.config.get("analysis", {})
            outcome_col = analysis_config.get("default_outcome", "outcome")

        # Determine outcome label
        if outcome_label is None:
            outcome_labels = self.config.get("outcome_labels", {})
            outcome_label = outcome_labels.get(outcome_col, outcome_col)

        # Build select to ensure UnifiedCohort schema
        selects = {}

        # Required columns
        # Check if patient_id is already in view (after column mapping in get_base_view)
        if UnifiedCohort.PATIENT_ID in view.columns:
            selects[UnifiedCohort.PATIENT_ID] = _[UnifiedCohort.PATIENT_ID]
        else:
            # Fallback: find source column from mapping
            patient_id_col = self._find_mapped_column(UnifiedCohort.PATIENT_ID)
            if patient_id_col and patient_id_col in view.columns:
                selects[UnifiedCohort.PATIENT_ID] = _[patient_id_col]

        if UnifiedCohort.TIME_ZERO in view.columns:
            selects[UnifiedCohort.TIME_ZERO] = _[UnifiedCohort.TIME_ZERO]

        # Outcome column
        if outcome_col in view.columns:
            selects[UnifiedCohort.OUTCOME] = _[outcome_col]
        else:
            # Fallback: use first outcome if available
            outcomes = self.config.get("outcomes", {})
            if outcomes:
                first_outcome = list(outcomes.keys())[0]
                if first_outcome in view.columns:
                    selects[UnifiedCohort.OUTCOME] = _[first_outcome]

        # Outcome label (literal)
        selects[UnifiedCohort.OUTCOME_LABEL] = ibis.literal(outcome_label)

        # Add all other columns as features
        for col in view.columns:
            if col not in selects and col not in UnifiedCohort.REQUIRED_COLUMNS:
                selects[col] = _[col]

        # Pass the expressions dict, not just the keys
        return view.select(**selects)

    def _find_mapped_column(self, target_col: str) -> str | None:
        """Find source column name for a target column."""
        column_mapping = self.config.get("column_mapping", {})
        for source, target in column_mapping.items():
            if target == target_col:
                return str(source)  # Explicitly convert to str
        return target_col  # Fallback: assume same name

    def get_data_quality_warnings(self) -> list[dict]:
        """
        Get data quality warnings from config.

        Reads from self.config only (not from metadata or external sources).
        Canonical warnings should be stored at config["validation"]["quality_warnings"].

        Returns:
            List of quality warning dictionaries
        """
        validation = self.config.get("validation", {})
        quality_warnings = validation.get("quality_warnings", [])
        return list(quality_warnings) if quality_warnings else []

    def get_cohort(
        self,
        granularity: "Granularity" = "patient_level",
        outcome_col: str | None = None,
        outcome_label: str | None = None,
        filters: dict[str, Any] | None = None,
        show_sql: bool = False,
    ) -> pd.DataFrame:
        """
        Execute the cohort query and return Pandas DataFrame.

        This is the main entry point - generates SQL behind the scenes and executes it.

        Args:
            granularity: Grain level (patient_level, admission_level, event_level).
                        Accepted but not validated (validation done by dataset classes).
                        For single-table datasets, ignored (always patient-level queries).
                        For multi-table datasets (future), selects appropriate materialized mart.
            outcome_col: Which outcome to use
            outcome_label: Label for outcome
            filters: Optional filters
            show_sql: If True, log the generated SQL (for debugging)

        Returns:
            Pandas DataFrame conforming to UnifiedCohort schema
        """
        query = self.build_cohort_query(
            granularity=granularity,
            outcome_col=outcome_col,
            outcome_label=outcome_label,
            filters=filters,
        )

        if show_sql:
            dataset_name = getattr(self, "dataset_name", self.__class__.__name__)
            try:
                sql = query.compile()
            except Exception as e:
                sql = f"<failed to compile SQL: {e}>"
            logger.info("Generated SQL for %s:\n%s", dataset_name, sql)

        # Execute query (this is where SQL actually runs)
        result = query.execute()

        return result

    def get_available_metrics(self) -> dict[str, dict[str, Any]]:
        """
        Get all available metrics from config.

        Returns:
            Dictionary mapping metric names to their definitions
        """
        result = self.config.get("metrics", {})
        return dict(result) if isinstance(result, dict) else {}

    def get_available_dimensions(self) -> dict[str, dict[str, Any]]:
        """
        Get all available dimensions from config.

        Returns:
            Dictionary mapping dimension names to their definitions
        """
        result = self.config.get("dimensions", {})
        return dict(result) if isinstance(result, dict) else {}

    def get_available_filters(self) -> dict[str, dict[str, Any]]:
        """
        Get all available filter definitions from config.

        Returns:
            Dictionary mapping filter names to their definitions
        """
        result = self.config.get("filters", {})
        return dict(result) if isinstance(result, dict) else {}

    def get_dataset_info(self) -> dict[str, Any]:
        """
        Get dataset metadata from config.

        Returns:
            Dictionary with dataset metadata (name, display_name, description, etc.)
        """
        return {
            "name": self.config.get("name", self.dataset_name),
            "display_name": self.config.get("display_name", self.dataset_name),
            "description": self.config.get("description", ""),
            "source": self.config.get("source", ""),
            "status": self.config.get("status", "unknown"),
            "metrics": self.get_available_metrics(),
            "dimensions": self.get_available_dimensions(),
            "filters": self.get_available_filters(),
        }

    def get_column_metadata(self, column_name: str) -> dict[str, Any] | None:
        """
        Get metadata for a specific column if available.

        For uploaded datasets, this accesses variable_types metadata from the upload metadata.
        Falls back to None if metadata is not available.

        Args:
            column_name: Canonical column name

        Returns:
            Dictionary with column metadata (type, numeric, values, etc.) or None if not available
        """
        # Try to access variable_types from upload metadata
        # This is stored in the config for uploaded datasets
        variable_types = self.config.get("variable_types")
        if variable_types and column_name in variable_types:
            result = variable_types[column_name]
            return dict(result) if isinstance(result, dict) else None

        # Fallback: check if metadata is stored elsewhere in config
        metadata = self.config.get("metadata")
        if metadata:
            variable_types = metadata.get("variable_types")
            if variable_types and column_name in variable_types:
                result = variable_types[column_name]
                return dict(result) if isinstance(result, dict) else None

        return None

    def extract_column_metadata(self, column_name: str) -> dict[str, Any] | None:
        """
        Extract column metadata compatible with ColumnContext construction.

        Uses existing get_column_metadata() internally but formats for AutoContext use.

        Args:
            column_name: Canonical column name

        Returns:
            Dict compatible with ColumnContext construction, or None if metadata unavailable

        Example return structure:
            {
                "name": "Current Regimen",
                "normalized_name": "current_regimen",  # lowercase, underscore-normalized
                "dtype": "coded",  # Maps from variable_types["type"]: "numeric"|"categorical"|"coded"|"datetime"|"text"
                "units": None,  # From variable_types["units"] if available
                "codebook": {"1": "Biktarvy", "2": "Symtuza", ...},  # From variable_types["codebook"] if available
            }

        Dtype mapping:
            - "numeric" → "numeric"
            - "categorical" (with numeric=True) → "coded"
            - "categorical" (with numeric=False) → "categorical"
            - "datetime" → "datetime"
            - "text" → "categorical" (fallback)
            - "id" → "id" (for patient_id columns)

        Normalized name: lowercase, replace spaces/special chars with underscores
        """
        import re

        # Get base metadata
        metadata = self.get_column_metadata(column_name)
        if not metadata:
            return None

        # Normalize column name
        normalized_name = column_name.lower()
        normalized_name = re.sub(r"[^\w]+", "_", normalized_name)
        normalized_name = re.sub(r"_+", "_", normalized_name).strip("_")

        # Map dtype
        var_type = metadata.get("type", "categorical")
        metadata_info = metadata.get("metadata", {})
        is_numeric = metadata_info.get("numeric", False)

        if var_type == "numeric" or var_type == "continuous":
            dtype = "numeric"
        elif var_type == "datetime":
            dtype = "datetime"
        elif var_type in ("categorical", "binary") and is_numeric:
            dtype = "coded"
        elif "id" in column_name.lower():
            dtype = "id"
        else:
            dtype = "categorical"

        # Build result dict
        result = {
            "name": column_name,
            "normalized_name": normalized_name,
            "dtype": dtype,
        }

        # Add optional fields
        if "units" in metadata:
            result["units"] = metadata["units"]
        if "codebook" in metadata:
            result["codebook"] = metadata["codebook"]

        return result

    def _build_metric_expression(self, metric_name: str, view: ibis.Table) -> ibis.Expr:
        """
        Build an Ibis expression for a metric from config.

        Args:
            metric_name: Name of metric (must be in config)
            view: Ibis table to build expression on

        Returns:
            Ibis expression for the metric
        """
        metrics = self.get_available_metrics()
        if metric_name not in metrics:
            raise ValueError(f"Metric '{metric_name}' not found in config")

        metric_def = metrics[metric_name]
        expression = metric_def.get("expression", "")

        # Parse expression (simple cases for now)
        # Format: column.aggregation() or aggregation()  # noqa: ERA001
        if "." in expression:
            col_name, agg_func = expression.rsplit(".", 1)
            if col_name in view.columns:
                col_expr = _[col_name]
                if agg_func == "mean()":
                    return col_expr.mean()
                elif agg_func == "sum()":
                    return col_expr.sum()
                elif agg_func == "max()":
                    return col_expr.max()
                elif agg_func == "min()":
                    return col_expr.min()
                elif agg_func == "count()":
                    return col_expr.count()
        elif expression == "count()":
            return view.count()

        # Fallback: try to use column directly
        if metric_name in view.columns:
            return _[metric_name]

        raise ValueError(f"Could not build expression for metric '{metric_name}'")

    def query(
        self,
        metrics: list[str] | None = None,
        dimensions: list[str] | None = None,
        filters: dict[str, Any] | None = None,
        show_sql: bool = False,
    ) -> pd.DataFrame:
        """
        Build and execute a custom query with metrics and dimensions.

        This is for more advanced use cases (aggregations, group bys).
        All metrics and dimensions come from config - fully extensible!

        Args:
            metrics: List of metric names (from config) to aggregate
            dimensions: List of dimension names (from config) to group by
            filters: Optional filters
            show_sql: If True, print generated SQL

        Returns:
            Pandas DataFrame with query results
        """
        view = self.get_base_view()

        # Apply filters
        default_filters = self.config.get("default_filters", {})
        all_filters = {**default_filters, **(filters or {})}
        view = self.apply_filters(view, all_filters)

        if metrics or dimensions:
            # Build aggregations from config
            aggs = []

            if metrics:
                for metric_name in metrics:
                    try:
                        metric_expr = self._build_metric_expression(metric_name, view)
                        metric_def = self.get_available_metrics().get(metric_name, {})
                        label = metric_def.get("label", metric_name)
                        aggs.append(metric_expr.name(label))
                    except ValueError as e:
                        print(f"Warning: {e}, skipping metric '{metric_name}'")

            if dimensions:
                # Group by dimensions (validate they exist in config)
                available_dims = self.get_available_dimensions()
                group_cols = []
                for dim in dimensions:
                    if dim in available_dims and dim in view.columns:
                        group_cols.append(dim)
                    else:
                        print(f"Warning: Dimension '{dim}' not found, skipping")

                if aggs:
                    result = view.group_by(group_cols).aggregate(aggs)
                elif group_cols:
                    result = view.select(group_cols)
                else:
                    result = view
            else:
                if aggs:
                    result = view.aggregate(aggs)
                else:
                    result = view
        else:
            result = view

        if show_sql:
            sql = result.compile()
            print(f"Generated SQL:\n{sql}\n")

        return result.execute()

    def _normalize_alias(self, text: str) -> str:
        """
        Normalize text for alias matching.

        Args:
            text: Text to normalize

        Returns:
            Normalized string (lowercase, punctuation removed, whitespace collapsed)
        """
        # Lowercase
        normalized = text.lower()

        # Remove punctuation (keep alphanumeric and spaces)
        normalized = re.sub(r"[^\w\s]", "", normalized)

        # Collapse whitespace
        normalized = re.sub(r"\s+", " ", normalized).strip()

        return normalized

    def _build_alias_index(self) -> None:
        """
        Build alias index for column matching.

        Builds:
        - _alias_index: normalized_alias -> canonical_name (single mapping, collisions dropped)
        - _alias_to_canonicals: normalized_alias -> set[canonical_names] (for collision detection)
        - _collision_warnings: set of aliases that map to multiple canonicals

        Strategy:
        1. Always index full normalized display_name
        2. Index selected aliases (domain terms, acronyms)
        3. Build word frequency FIRST
        4. Conditionally index single words only if rare (appears in < 3 columns)
        """
        from clinical_analytics.core.column_parser import parse_column_name

        if self._alias_index is not None:
            return  # Already built

        view = self.get_base_view()
        columns = view.columns

        # Initialize structures
        alias_index: dict[str, str] = {}  # normalized_alias -> canonical_name (single mapping)
        alias_to_canonicals: dict[str, set[str]] = {}  # normalized_alias -> set[canonical_names]

        # Step 1: Parse all columns and build word frequency
        column_metadata = {}
        word_frequency: dict[str, int] = {}  # word -> count across columns

        for col in columns:
            meta = parse_column_name(col)
            column_metadata[col] = meta

            # Extract words from display_name for frequency counting
            display_normalized = self._normalize_alias(meta.display_name)
            words = display_normalized.split()

            for word in words:
                if len(word) > 2:  # Skip very short words
                    word_frequency[word] = word_frequency.get(word, 0) + 1

        # Step 2: Index full normalized display_name (always)
        for canonical, meta in column_metadata.items():
            display_normalized = self._normalize_alias(meta.display_name)

            # Add to alias_to_canonicals (for collision detection)
            if display_normalized not in alias_to_canonicals:
                alias_to_canonicals[display_normalized] = set()
            alias_to_canonicals[display_normalized].add(canonical)

            # Add to alias_index (single mapping, last write wins for now)
            alias_index[display_normalized] = canonical

        # Step 3: Index selected aliases (domain terms, acronyms)
        # Extract acronyms (all uppercase words > 1 char)
        for canonical, meta in column_metadata.items():
            display_words = meta.display_name.split()
            for word in display_words:
                if word.isupper() and len(word) > 1:
                    # Acronym (e.g., "DEXA", "CD4")
                    alias_normalized = self._normalize_alias(word)
                    if alias_normalized not in alias_to_canonicals:
                        alias_to_canonicals[alias_normalized] = set()
                    alias_to_canonicals[alias_normalized].add(canonical)
                    alias_index[alias_normalized] = canonical

        # Step 4: Conditionally index single words (only if rare)
        for canonical, meta in column_metadata.items():
            display_normalized = self._normalize_alias(meta.display_name)
            words = display_normalized.split()

            for word in words:
                if len(word) > 2:  # Skip very short words
                    # Only index if word appears in < 3 columns (rare)
                    if word_frequency.get(word, 0) < 3:
                        if word not in alias_to_canonicals:
                            alias_to_canonicals[word] = set()
                        alias_to_canonicals[word].add(canonical)
                        # Only add to alias_index if not already there (avoid overwriting)
                        if word not in alias_index:
                            alias_index[word] = canonical

        # Step 5: Merge user aliases (Phase 2: user aliases override system aliases)
        user_aliases = getattr(self, "_user_aliases", {})
        for normalized_term, column in user_aliases.items():
            # User aliases override system aliases for same normalized key
            if normalized_term in alias_index:
                logger.debug(
                    f"User alias '{normalized_term}' overriding system alias "
                    f"'{alias_index[normalized_term]}' -> '{column}'"
                )

            # Update alias_index (user alias takes precedence)
            alias_index[normalized_term] = column

            # Update alias_to_canonicals (for collision detection)
            if normalized_term not in alias_to_canonicals:
                alias_to_canonicals[normalized_term] = set()
            alias_to_canonicals[normalized_term].add(column)

        # Step 6: Detect collisions and drop ambiguous aliases
        collision_warnings: set[str] = set()
        for alias, canonicals in alias_to_canonicals.items():
            if len(canonicals) > 1:
                # Collision detected
                collision_warnings.add(alias)
                # Drop from alias_index (ambiguous)
                if alias in alias_index:
                    del alias_index[alias]

        # Store results
        self._alias_index = alias_index
        self._alias_to_canonicals = alias_to_canonicals
        self._collision_warnings = collision_warnings

        user_count = len(user_aliases)
        logger.info(
            f"Built alias index: {len(alias_index)} unique aliases "
            f"({user_count} user, {len(alias_index) - user_count} system), "
            f"{len(collision_warnings)} collisions detected"
        )

    def get_column_alias_index(self) -> dict[str, str]:
        """
        Get alias index for column matching.

        Returns:
            Dict mapping normalized_alias -> canonical_name
            (collisions are dropped, so each alias maps to at most one canonical)

        Note:
            This is a pure function (no Streamlit dependencies).
            Index is built lazily on first call.
        """
        self._build_alias_index()
        return self._alias_index.copy() if self._alias_index else {}

    def get_collision_warnings(self) -> set[str]:
        """
        Get set of aliases that map to multiple canonical names.

        Returns:
            Set of normalized aliases that have collisions
        """
        self._build_alias_index()
        return self._collision_warnings.copy() if self._collision_warnings else set()

    def get_collision_suggestions(self, query_term: str) -> list[str] | None:
        """
        Get collision suggestions for a query term.

        If the normalized query term matches a dropped alias (collision),
        return list of canonical names that could match.

        Args:
            query_term: Query term from user

        Returns:
            List of canonical names if collision detected, None otherwise
        """
        self._build_alias_index()
        normalized = self._normalize_alias(query_term)

        from clinical_analytics.core.type_guards import safe_get

        if (
            self._collision_warnings is not None
            and normalized in self._collision_warnings
            and self._alias_to_canonicals is not None
        ):
            # Return all canonical names that match this alias
            canonicals = safe_get(self._alias_to_canonicals, normalized, set())
            return sorted(list(canonicals))
        return None

    # ============================================================================
    # Phase 2: User Alias Persistence (ADR003)
    # ============================================================================

    def _get_metadata_path(self, upload_id: str) -> Path:
        """
        Get path to metadata JSON file for upload.

        Args:
            upload_id: Upload identifier

        Returns:
            Path to metadata JSON file
        """
        # Metadata stored in data/uploads/metadata/{upload_id}.json
        metadata_dir = self.workspace_root / "data" / "uploads" / "metadata"
        return metadata_dir / f"{upload_id}.json"

    def _load_metadata(self, upload_id: str) -> dict[str, Any] | None:
        """
        Load metadata JSON for upload.

        Args:
            upload_id: Upload identifier

        Returns:
            Metadata dict if found, None otherwise
        """
        metadata_path = self._get_metadata_path(upload_id)
        if not metadata_path.exists():
            return None

        try:
            import json

            with open(metadata_path) as f:
                result = json.load(f)
                return dict(result) if isinstance(result, dict) else None
        except Exception as e:
            logger.warning(f"Failed to load metadata for {upload_id}: {e}")
            return None

    def _save_metadata(self, upload_id: str, metadata: dict[str, Any]) -> None:
        """
        Save metadata JSON for upload (atomic update).

        Args:
            upload_id: Upload identifier
            metadata: Metadata dict to save
        """
        import json

        metadata_path = self._get_metadata_path(upload_id)
        metadata_dir = metadata_path.parent
        metadata_dir.mkdir(parents=True, exist_ok=True)

        # Atomic write: write to temp file, then rename
        temp_path = metadata_path.with_suffix(".json.tmp")
        try:
            with open(temp_path, "w") as f:
                json.dump(metadata, f, indent=2)
            temp_path.replace(metadata_path)
        except Exception as e:
            logger.error(f"Failed to save metadata for {upload_id}: {e}")
            if temp_path.exists():
                temp_path.unlink()
            raise

    def _load_user_aliases(self, upload_id: str, dataset_version: str) -> None:
        """
        Load user aliases from metadata and merge into alias index.

        Args:
            upload_id: Upload identifier
            dataset_version: Dataset version (for scope validation)
        """
        metadata = self._load_metadata(upload_id)
        if not metadata:
            logger.debug(f"No metadata found for {upload_id}, skipping user alias load")
            return

        # Verify dataset version matches (scope isolation)
        stored_version = metadata.get("dataset_version")
        if stored_version != dataset_version:
            logger.warning(
                f"Dataset version mismatch for {upload_id}: "
                f"stored={stored_version}, requested={dataset_version}. "
                "Skipping user alias load."
            )
            return

        # Extract user aliases
        alias_mappings = metadata.get("alias_mappings", {})
        user_aliases = alias_mappings.get("user_aliases", {})

        if not user_aliases:
            logger.debug(f"No user aliases found for {upload_id}")
            return

        # Get base view to validate columns exist
        try:
            base_view = self.get_base_view()
            available_columns = set(base_view.columns)
        except Exception as e:
            logger.warning(f"Failed to get base view for alias validation: {e}")
            available_columns = set()

        # Store user aliases (will be merged when alias index is built)
        self._user_aliases: dict[str, str] = {}
        orphaned_count = 0

        for term, column in user_aliases.items():
            # Validate column exists (handle orphaned aliases)
            if column not in available_columns:
                logger.warning(f"Orphaned alias for {upload_id}: '{term}' -> '{column}' (column not found in schema)")
                orphaned_count += 1
                continue

            # Normalize term
            normalized_term = self._normalize_alias(term)
            self._user_aliases[normalized_term] = column

        if orphaned_count > 0:
            logger.info(f"Marked {orphaned_count} orphaned aliases for {upload_id}")

        logger.info(f"Loaded {len(self._user_aliases)} user aliases for {upload_id}")

    def add_user_alias(
        self,
        term: str,
        column: str,
        upload_id: str,
        dataset_version: str,
    ) -> None:
        """
        Add user alias and persist to metadata.

        Args:
            term: Alias term (e.g., "VL")
            column: Canonical column name (e.g., "viral_load")
            upload_id: Upload identifier
            dataset_version: Dataset version (for scope isolation)

        Raises:
            ValueError: If column doesn't exist or collision detected
        """
        # Validate column exists
        try:
            base_view = self.get_base_view()
            available_columns = set(base_view.columns)
        except Exception as e:
            raise ValueError(f"Failed to validate column: {e}") from e

        if column not in available_columns:
            raise ValueError(f"Column '{column}' not found in dataset schema")

        # Normalize term
        normalized_term = self._normalize_alias(term)

        # Check for collisions (multiple columns match same alias)
        # Build alias index if not already built
        self._build_alias_index()

        # Check if normalized term already maps to a different column
        if self._alias_index is not None and normalized_term in self._alias_index:
            existing_column = self._alias_index[normalized_term]
            if existing_column != column:
                # Collision detected - surface in UI, don't silently remap
                raise ValueError(
                    f"Alias '{term}' already maps to '{existing_column}'. "
                    f"Cannot remap to '{column}' without explicit confirmation."
                )

        # Load existing metadata
        metadata = self._load_metadata(upload_id)
        if not metadata:
            # Create new metadata structure
            metadata = {
                "upload_id": upload_id,
                "dataset_version": dataset_version,
                "alias_mappings": {
                    "user_aliases": {},
                    "system_aliases": {},
                },
            }

        # Verify dataset version matches (scope isolation)
        stored_version = metadata.get("dataset_version")
        if stored_version and stored_version != dataset_version:
            raise ValueError(
                f"Dataset version mismatch: stored={stored_version}, "
                f"requested={dataset_version}. Aliases are scoped per dataset version."
            )

        # Ensure alias_mappings structure exists
        if "alias_mappings" not in metadata:
            metadata["alias_mappings"] = {"user_aliases": {}, "system_aliases": {}}

        if "user_aliases" not in metadata["alias_mappings"]:
            metadata["alias_mappings"]["user_aliases"] = {}

        # Add alias (use original term, not normalized, for storage)
        metadata["alias_mappings"]["user_aliases"][term] = column

        # Persist to metadata JSON
        self._save_metadata(upload_id, metadata)

        # Update in-memory user aliases
        if not hasattr(self, "_user_aliases"):
            self._user_aliases = {}
        self._user_aliases[normalized_term] = column

        # Invalidate alias index to force rebuild with new user alias
        self._alias_index = None
        self._alias_to_canonicals = None
        self._collision_warnings = None

        logger.info(f"Added user alias '{term}' -> '{column}' for {upload_id}")

    def _execute_plan_with_retry(
        self, plan: "QueryPlan", max_retries: int = 3, initial_delay: float = 0.5
    ) -> pd.DataFrame:
        """
        Execute query plan with exponential backoff retry for transient errors (Phase 2.5.2).

        Handles backend initialization errors like AttributeError: '_record_batch_readers_consumed',
        connection errors, and other transient execution failures.

        Args:
            plan: QueryPlan to execute
            max_retries: Maximum retry attempts (default: 3)
            initial_delay: Initial delay in seconds before first retry (default: 0.5s)

        Returns:
            pd.DataFrame: Query results

        Raises:
            Exception: Re-raises final exception if all retries exhausted
        """
        delay = initial_delay
        last_exception: Exception | None = None

        # Known transient error patterns for DuckDB/Ibis backends
        # Narrow patterns to avoid retrying on deterministic failures
        transient_patterns = [
            "temporary",
            "timeout",
            "connection refused",
            "connection reset",
            "connection closed",
            "database is locked",
            "deadlock",
            "service unavailable",
            "temporarily unavailable",
        ]

        for attempt in range(max_retries + 1):  # +1 for initial attempt
            try:
                return self._execute_plan(plan)
            except AttributeError as e:
                # Backend initialization error: '_record_batch_readers_consumed'
                last_exception = e
                if "_record_batch_readers_consumed" in str(e) and attempt < max_retries:
                    logger.warning(
                        "backend_initialization_error_retry: attempt=%d/%d error_type=%s delay=%.1fs - %s",
                        attempt + 1,
                        max_retries + 1,
                        type(e).__name__,
                        delay,
                        str(e),
                        exc_info=True,
                    )
                    time.sleep(delay)
                    delay *= 2  # Exponential backoff
                else:
                    # Not a backend error or final retry - re-raise
                    logger.error(
                        "backend_initialization_error_final: attempt=%d error_type=%s - %s",
                        attempt + 1,
                        type(e).__name__,
                        str(e),
                        exc_info=True,
                    )
                    raise
            except (ConnectionError, OSError, TimeoutError) as e:
                # Connection/network errors - retry
                last_exception = e
                if attempt < max_retries:
                    logger.warning(
                        "connection_error_retry: attempt=%d/%d error_type=%s delay=%.1fs - %s",
                        attempt + 1,
                        max_retries + 1,
                        type(e).__name__,
                        delay,
                        str(e),
                        exc_info=True,
                    )
                    time.sleep(delay)
                    delay *= 2  # Exponential backoff
                else:
                    # Final retry - re-raise
                    logger.error(
                        "connection_error_final: attempt=%d error_type=%s - %s",
                        attempt + 1,
                        type(e).__name__,
                        str(e),
                        exc_info=True,
                    )
                    raise
            except Exception as e:
                # Check if it's a known transient error pattern (narrow matching)
                error_msg = str(e).lower()
                error_type = type(e).__name__
                is_transient = any(pattern in error_msg for pattern in transient_patterns)

                if is_transient and attempt < max_retries:
                    last_exception = e
                    matched = next((p for p in transient_patterns if p in error_msg), None)
                    logger.warning(
                        "transient_error_retry: attempt=%d/%d error_type=%s matched='%s' delay=%.1fs - %s",
                        attempt + 1,
                        max_retries + 1,
                        error_type,
                        matched,
                        delay,
                        str(e),
                        exc_info=True,
                    )
                    time.sleep(delay)
                    delay *= 2  # Exponential backoff
                else:
                    # Not transient or final retry - re-raise
                    if is_transient:
                        logger.error(
                            "transient_error_final: attempt=%d error_type=%s - %s",
                            attempt + 1,
                            error_type,
                            str(e),
                            exc_info=True,
                        )
                    else:
                        logger.debug(
                            "non_transient_error_no_retry: error_type=%s - %s",
                            error_type,
                            str(e),
                            exc_info=True,
                        )
                    raise

        # Should never reach here, but just in case
        if last_exception:
            logger.error(
                "retry_exhausted",
                extra={
                    "max_attempts": str(max_retries + 1),
                    "error_type": type(last_exception).__name__,
                    "error_message": str(last_exception),
                },
            )
            raise last_exception
        raise RuntimeError("Unexpected retry loop exit")

    def _validate_filter_types(self, filters: list) -> list[str]:
        """
        Lightweight pre-execution sanity check for filter types.

        This is NOT the primary validation - LLM validation layers should have
        already caught type mismatches. This is a final safety check.

        Args:
            filters: List of FilterSpec objects to validate

        Returns:
            List of error messages (empty if all valid)
        """
        errors: list[str] = []

        try:
            base_view = self.get_base_view()
            schema = base_view.schema()
            available_columns = set(base_view.columns)

            for filter_spec in filters:
                column = filter_spec.column
                value = filter_spec.value

                # Check if column exists
                if column not in available_columns:
                    errors.append(
                        f"Column '{column}' not found in schema. " "Pre-execution validation should have caught this."
                    )
                    continue

                # Get column dtype
                dtype = schema[column]
                dtype_str = str(dtype).lower()

                # Check type compatibility
                numeric_types = ["int", "float", "decimal", "double", "numeric"]
                is_numeric_column = any(t in dtype_str for t in numeric_types)

                if is_numeric_column:
                    # Numeric column should have numeric value (int/float)
                    if isinstance(value, str) and not self._is_numeric_string(value):
                        errors.append(
                            f"Type mismatch for column '{column}': "
                            f"expected numeric value, got string '{value}'. "
                            "LLM validation should have caught this."
                        )

                logger.debug(
                    "filter_type_validation_checked: column=%s, dtype=%s, value_type=%s, is_numeric=%s",
                    column,
                    dtype_str,
                    type(value).__name__,
                    is_numeric_column,
                )

        except Exception as e:
            logger.warning("filter_type_validation_error: %s", str(e))
            # Don't block on validation errors - just log
            pass

        return errors

    def _is_numeric_string(self, value: str) -> bool:
        """Check if a string can be converted to a number."""
        try:
            float(value)
            return True
        except (ValueError, TypeError):
            return False

    def execute_query_plan(
        self, plan: "QueryPlan", confidence_threshold: float = 0.75, query_text: str | None = None
    ) -> dict[str, Any]:
        """
        Execute a QueryPlan with warnings for observability (ADR003 Phase 3 + Phase 2.2).

        Phase 2.1: Added warnings infrastructure for observability.
        Phase 2.2: Removed gating logic - always execute, collect warnings only.
        Phase 2.5.1: Returns step information for progressive thinking indicator (core layer).

        This method no longer blocks execution. Instead, it:
        - Collects warnings for low confidence, incompleteness, validation issues
        - Always attempts execution
        - Returns success=False only for actual execution errors
        - Returns step information for UI rendering (Phase 2.5.1)

        Args:
            plan: QueryPlan to execute
            confidence_threshold: Minimum confidence for warning threshold (default: 0.75)
            query_text: Optional query text for run_key generation

        Returns:
            dict with keys:
            - "success": bool - Whether execution succeeded
            - "result": pd.DataFrame | None - Query results if successful
            - "run_key": str - Deterministic run key for idempotency
            - "warnings": list[str] - Warnings collected during execution
            - "steps": list[dict] - Step information for progressive thinking indicator (Phase 2.5.1)
            - "chart_spec": dict | None - Chart specification for visualization (Phase 3.3)
        """
        # Phase 3.2: Log execution start for contract enforcement and observability
        logger.debug(
            f"execute_query_plan called: intent={plan.intent}, metric={plan.metric}, "
            f"group_by={plan.group_by}, confidence={plan.confidence:.2f}"
        )

        # Phase 2.5.1: Initialize steps list (core layer generates step data)
        steps: list[dict[str, Any]] = []

        # Phase 2.2: Initialize warnings list for observability
        warnings: list[str] = []

        # Phase 8: Validate query against active version schema (non-blocking)
        if self.upload_id:
            # For user-uploaded datasets, validate against active version schema
            metadata = self._load_metadata(self.upload_id)
            if metadata:
                version_history = metadata.get("version_history", [])
                if version_history:
                    # Find active version
                    active_versions = [v for v in version_history if v.get("is_active", False)]
                    if active_versions:
                        active_version = active_versions[0]
                        # Validate query against active version schema
                        schema_warnings = validate_query_against_schema(plan, active_version)
                        warnings.extend(schema_warnings)

        # Step 1: Interpreting query (Phase 2.5.1)
        step_details = {
            "intent": plan.intent,
            "metric": plan.metric,
            "group_by": plan.group_by,
            "filter_count": len(plan.filters) if plan.filters else 0,
        }
        steps.append(
            {
                "status": "processing",
                "text": "Interpreting query",
                "details": step_details,
            }
        )

        # Step 2: Confidence Check (warning only, no blocking)
        if plan.confidence < confidence_threshold:
            warning_msg = (
                f"Low confidence: {plan.confidence:.2f} (threshold: {confidence_threshold:.2f}). "
                f"Query interpretation may be ambiguous or uncertain."
            )
            warnings.append(warning_msg)

        # Step 3: Completeness Check (warning only, no blocking)
        is_complete, completeness_error = self._check_plan_completeness(plan)
        if not is_complete:
            warning_msg = f"Incomplete plan: {completeness_error}"
            warnings.append(warning_msg)

        # Step 4: Validation Check (warning only, no blocking)
        validation_result = self._validate_query_plan(plan)
        if not validation_result["valid"]:
            warning_msg = f"Validation failed: {validation_result['error']}"
            warnings.append(warning_msg)

        # Step 5: Validating plan (Phase 2.5.1)
        steps.append(
            {
                "status": "processing",
                "text": "Validating plan",
                "details": {
                    "has_warnings": len(warnings) > 0,
                    "warning_count": len(warnings),
                },
            }
        )

        # Step 6: Generate run_key (always generate)
        run_key = self._generate_run_key(plan, query_text)

        # Phase 3.3: Generate chart_spec from QueryPlan (deterministic, tied to plan)
        chart_spec_obj = generate_chart_spec(plan)
        # Convert ChartSpec dataclass to dict (or None if not visualizable)
        chart_spec = (
            {
                "type": chart_spec_obj.type,
                "x": chart_spec_obj.x,
                "y": chart_spec_obj.y,
                "group_by": chart_spec_obj.group_by,
                "title": chart_spec_obj.title,
            }
            if chart_spec_obj
            else None
        )

        # Step 7: Executing query (Phase 2.5.1)
        steps.append(
            {
                "status": "processing",
                "text": "Executing query",
                "details": {"run_key": run_key},
            }
        )

        # Step 8: Execute query with retry logic (Phase 2.5.2)
        try:
            result_df = self._execute_plan_with_retry(plan)
            # Step 9: Complete (Phase 2.5.1)
            # Handle both DataFrame and scalar results
            if result_df is not None:
                if hasattr(result_df, "__len__"):
                    result_rows = len(result_df)
                else:
                    # Scalar result (e.g., COUNT returns int)
                    result_rows = 1
            else:
                result_rows = 0
            steps.append(
                {
                    "status": "completed",
                    "text": "Query complete",
                    "details": {"result_rows": result_rows},
                }
            )
            return {
                "success": True,
                "result": result_df,
                "run_key": run_key,
                "warnings": warnings,
                "steps": steps,  # Phase 2.5.1: Core provides step data
                "chart_spec": chart_spec,  # Phase 3.3: Chart specification for visualization
            }
        except Exception as e:
            logger.error(f"Query execution failed: {e}", exc_info=True)
            warning_msg = f"Execution error: {str(e)}"
            warnings.append(warning_msg)
            # Step 9: Failed (Phase 2.5.1)
            steps.append(
                {
                    "status": "error",
                    "text": "Query failed",
                    "details": {"error": str(e)},
                }
            )
            return {
                "success": False,
                "result": None,
                "run_key": run_key,
                "warnings": warnings,
                "steps": steps,  # Phase 2.5.1: Core provides step data
                "chart_spec": chart_spec,  # Phase 3.3: Chart specification (still included on error)
            }

    def format_execution_result(self, execution_result: dict[str, Any], context: Any) -> dict[str, Any]:
        """
        Format execution result from execute_query_plan() for UI consumption (Phase 3.1).

        This is a transitional method that bridges QueryPlan execution to legacy result format.
        Eventually this will be replaced by QueryPlan-driven formatting (Phase 3.3).

        Args:
            execution_result: Result dict from execute_query_plan()
            context: AnalysisContext with intent and variables (for legacy compatibility)

        Returns:
            Formatted result dict compatible with render_analysis_by_type()
            Phase 3.3: Now includes chart_spec from execution_result
        """
        # Phase 3.3: Extract chart_spec from execution_result
        chart_spec = execution_result.get("chart_spec")

        if not execution_result.get("success"):
            return {
                "type": "error",
                "error": execution_result.get("warnings", ["Execution failed"])[0]
                if execution_result.get("warnings")
                else "Execution failed",
                "chart_spec": chart_spec,  # Phase 3.3: Include chart_spec even on error
            }

        result_df = execution_result.get("result")
        if result_df is None:
            return {"type": "error", "error": "No result data", "chart_spec": chart_spec}

        # Normalize result to Polars DataFrame (handle scalar edge cases)
        # _execute_plan always returns pd.DataFrame, but normalize for safety
        if isinstance(result_df, pd.DataFrame):
            result_df_pl = pl.from_pandas(result_df)
        elif isinstance(result_df, int | float):
            # Scalar count result - convert to DataFrame for consistent handling
            result_df_pl = pl.DataFrame({"count": [int(result_df)]})
        elif isinstance(result_df, pl.DataFrame):
            result_df_pl = result_df
        else:
            # Unknown type - try to convert
            result_df_pl = pl.DataFrame({"value": [result_df]})

        # Phase 3.1: Format result DataFrame directly (it's already aggregated from execute_query_plan)
        # Do NOT call compute_analysis_by_type() - it expects raw cohort, not aggregated results
        query_plan = getattr(context, "query_plan", None)
        if not query_plan:
            return {"type": "error", "error": "QueryPlan required for formatting", "chart_spec": chart_spec}

        # Format based on intent
        # Phase 3.3: Pass chart_spec to formatting methods
        if query_plan.intent == "COUNT":
            formatted = self._format_count_result(result_df_pl, query_plan, context)
        elif query_plan.intent == "DESCRIBE":
            formatted = self._format_describe_result(result_df_pl, query_plan, context)
        else:
            # For other intents, return basic format (will be enhanced in Phase 3.3)
            formatted = {
                "type": "unknown",
                "error": f"Formatting not yet implemented for intent: {query_plan.intent}",
            }

        # Phase 3.3: Add chart_spec to formatted result
        formatted["chart_spec"] = chart_spec
        return formatted

    def _format_count_result(self, result_df: pl.DataFrame, query_plan: "QueryPlan", context: Any) -> dict[str, Any]:
        """Format COUNT result DataFrame to result dict format."""
        # Note: result_df is always a DataFrame (normalized in format_execution_result)

        # DataFrame result - check if grouped or simple count
        if result_df.height == 1 and result_df.width == 1:
            # Single value (total count as DataFrame)
            total_count = int(result_df.item())
            return {
                "type": "count",
                "total_count": total_count,
                "headline": f"Total count: **{total_count}**",
            }

        # Grouped count: result_df has [group_by_column, "count"]
        if query_plan.group_by and query_plan.group_by in result_df.columns:
            group_col = query_plan.group_by
            # Convert to list of dicts
            group_counts = result_df.to_dicts()
            total_count = sum(item.get("count", 0) for item in group_counts)

            # Check for "most" query pattern
            query_text = getattr(context, "query_text", "") or getattr(context, "research_question", "")
            is_most_query = False
            if query_text:
                query_lower = query_text.lower()
                is_most_query = "most" in query_lower and ("which" in query_lower or "what" in query_lower)

            # Create headline
            if is_most_query and len(group_counts) > 0:
                top_group = group_counts[0]
                headline = f"**{top_group[group_col]}** with {top_group['count']} patients"
            else:
                headline = f"Total count: **{total_count}**"
                if len(group_counts) > 0:
                    top_group = group_counts[0]
                    headline += f" (largest group: {top_group[group_col]} with {top_group['count']})"

            return {
                "type": "count",
                "total_count": total_count,
                "grouped_by": group_col,
                "group_counts": group_counts,
                "headline": headline,
                "is_most_query": is_most_query,
            }
        else:
            # Simple count (no grouping)
            total_count = result_df.height if result_df.height > 0 else 0
            return {
                "type": "count",
                "total_count": total_count,
                "headline": f"Total count: **{total_count}**",
            }

    def _format_describe_result(self, result_df: pl.DataFrame, query_plan: "QueryPlan", context: Any) -> dict[str, Any]:
        """Format DESCRIBE result DataFrame to result dict format."""
        # Phase 3.1: Basic formatting for DESCRIBE (will be enhanced in Phase 3.3)
        if result_df.height == 0:
            return {"type": "error", "error": "No data to describe"}

        # Convert to dict format
        result_dict = result_df.to_dicts()[0] if result_df.height == 1 else result_df.to_dicts()

        return {
            "type": "descriptive",
            "summary": result_dict,
            "headline": f"Descriptive statistics for {query_plan.metric or 'data'}",
        }

    def _check_plan_completeness(self, plan: "QueryPlan") -> tuple[bool, str]:
        """Check if QueryPlan has all required fields for its intent."""
        if plan.intent == "COUNT":
            # COUNT requires entity_key OR grouping_variable
            if not plan.entity_key and not plan.group_by:
                return False, "COUNT intent requires entity_key or grouping_variable"
        elif plan.intent == "DESCRIBE":
            # DESCRIBE requires metric (primary_variable)
            if not plan.metric:
                return False, "DESCRIBE intent requires metric (primary_variable)"
        elif plan.intent == "COMPARE_GROUPS":
            # COMPARE_GROUPS requires both metric and group_by
            if not plan.metric or not plan.group_by:
                return False, "COMPARE_GROUPS intent requires both metric and group_by"
        # FIND_PREDICTORS and CORRELATIONS have no specific requirements for now
        return True, ""

    def _validate_query_plan(self, plan: "QueryPlan") -> dict[str, Any]:
        """Validate QueryPlan contract (columns exist, operators valid, types compatible)."""
        view = self.get_base_view()
        available_columns = set(view.columns)

        # Check metric exists
        if plan.metric and plan.metric not in available_columns:
            return {"valid": False, "error": f"Column '{plan.metric}' not found in dataset"}

        # Check group_by exists
        if plan.group_by and plan.group_by not in available_columns:
            return {"valid": False, "error": f"Column '{plan.group_by}' not found in dataset"}

        # Check filter columns exist and operators are valid
        valid_operators = {"==", "!=", ">", ">=", "<", "<=", "IN", "NOT_IN"}
        for filter_spec in plan.filters:
            if filter_spec.column not in available_columns:
                return {"valid": False, "error": f"Filter column '{filter_spec.column}' not found"}
            if filter_spec.operator not in valid_operators:
                return {"valid": False, "error": f"Invalid operator '{filter_spec.operator}'"}

        # COUNT-specific validation
        if plan.intent == "COUNT":
            # Refuse scope="all" with filters
            if plan.scope == "all" and plan.filters:
                return {
                    "valid": False,
                    "error": "Cannot use scope='all' with filters. Use scope='filtered' or remove filters.",
                }
            # Refuse scope="filtered" with empty filters (if requires_filters=True)
            if plan.scope == "filtered" and not plan.filters and plan.requires_filters:
                return {
                    "valid": False,
                    "error": "scope='filtered' requires filters, but no filters provided.",
                }
            # Require entity_key or group_by for COUNT
            if not plan.entity_key and not plan.group_by:
                return {
                    "valid": False,
                    "error": "COUNT intent requires entity_key or group_by. Please specify what to count.",
                }

        # Breakdown validation
        if plan.group_by:
            # Refuse grouping by entity_key (would create near-unique groups)
            if plan.entity_key and plan.group_by == plan.entity_key:
                return {
                    "valid": False,
                    "error": (
                        f"Cannot group by entity key '{plan.entity_key}'. "
                        "Grouping by entity key yields near-unique groups."
                    ),
                }
            # Check for high cardinality (warn but don't block - let user decide)
            try:
                distinct_count = view[plan.group_by].nunique().execute()
                if distinct_count > 100:  # High cardinality threshold
                    logger.warning(f"High cardinality grouping: '{plan.group_by}' has {distinct_count} distinct values")
            except Exception:
                pass  # Cardinality check failed, continue

        # Refuse requires_grouping=True with group_by=None
        if plan.requires_grouping and not plan.group_by:
            return {"valid": False, "error": "This query requires grouping, but no grouping_variable provided."}

        # Filter deduplication: detect redundant filters (filtering and grouping on same field)
        if plan.group_by:
            for filter_spec in plan.filters:
                if filter_spec.column == plan.group_by:
                    logger.warning(
                        f"Redundant filter detected: filtering and grouping on same field '{filter_spec.column}'. "
                        "Filter will be ignored (grouping takes precedence)."
                    )

        return {"valid": True, "error": None}

    def _generate_run_key(self, plan: "QueryPlan", query_text: str | None = None) -> str:
        """
        Generate deterministic run_key from canonical plan + query text.

        Runtime guardrails (Phase 8 - Staff Feedback):
        - Validates query_text is normalized (if provided)
        - Enforces contract that query text must be pre-normalized by caller

        Args:
            plan: QueryPlan to generate key for
            query_text: Optional query text (must be normalized if provided)

        Returns:
            str: Deterministic hash key for caching

        Raises:
            ValueError: If query_text is provided but not normalized
        """
        # Build canonical plan JSON (sorted for determinism)
        canonical_plan = {
            "intent": plan.intent,
            "metric": plan.metric,
            "group_by": plan.group_by,
            "filters": sorted(
                [
                    {
                        "column": f.column,
                        "operator": f.operator,
                        "value": f.value,
                    }
                    for f in plan.filters
                ],
                key=lambda x: (x["column"], x["operator"]),
            ),
            "entity_key": plan.entity_key,
            "scope": plan.scope,
        }
        canonical_json = json.dumps(canonical_plan, sort_keys=True)

        # Runtime guardrails: validate query_text is normalized (if provided)
        query_signature = ""
        if query_text:
            # Check for normalization violations
            if query_text != query_text.strip():
                raise ValueError(
                    f"Query text not normalized: has leading/trailing spaces. "
                    f"Got: {repr(query_text)}. "
                    f"Call normalize_query() before passing to _generate_run_key()"
                )
            if "  " in query_text:
                raise ValueError(
                    f"Query text not normalized: contains double spaces. "
                    f"Got: {repr(query_text)}. "
                    f"Call normalize_query() before passing to _generate_run_key()"
                )
            if query_text != query_text.lower():
                raise ValueError(
                    f"Query text not normalized: not lowercase. "
                    f"Got: {repr(query_text)}. "
                    f"Call normalize_query() before passing to _generate_run_key()"
                )

            # Query is normalized - generate signature
            query_signature = " ".join(query_text.lower().split())

        # Build hash
        dataset_version = self.dataset_version or "unknown"
        hash_input = f"{dataset_version}|{canonical_json}|{query_signature}"
        run_key = hashlib.sha256(hash_input.encode("utf-8")).hexdigest()[:16]

        return run_key

    def _execute_plan(self, plan: "QueryPlan") -> pd.DataFrame:
        """Execute QueryPlan and return results DataFrame."""
        view = self.get_base_view()

        # Filter deduplication: remove redundant filters (filtering and grouping on same field)
        effective_filters = []
        for filter_spec in plan.filters:
            if plan.group_by and filter_spec.column == plan.group_by:
                # Skip redundant filter (grouping takes precedence)
                logger.debug(f"Skipping redundant filter on '{filter_spec.column}' (grouped by same field)")
                continue
            effective_filters.append(filter_spec)

        # Apply filters directly using Ibis expressions (not through apply_filters which expects config-based filters)
        for filter_spec in effective_filters:
            column = filter_spec.column
            if column not in view.columns:
                logger.warning(f"Filter column '{column}' not found in view, skipping filter")
                continue

            col_expr = view[column]

            if filter_spec.operator == "==":
                view = view.filter(col_expr == filter_spec.value)
            elif filter_spec.operator == "!=":
                view = view.filter(col_expr != filter_spec.value)
            elif filter_spec.operator == "IN":
                # Handle IN operator with proper type casting
                if isinstance(filter_spec.value, list):
                    # For IN filters, we need to ensure type compatibility
                    # Get column type by executing a minimal query to check dtype
                    # Then cast filter values to match
                    try:
                        # First, try direct isin (works in most cases)
                        view = view.filter(col_expr.isin(filter_spec.value))
                    except Exception as first_error:
                        # Type mismatch - need to cast values to match column type
                        try:
                            # Get a sample row to determine column dtype
                            sample_df = view.select(column).limit(1).execute()
                            if len(sample_df) > 0:
                                col_dtype = str(sample_df[column].dtype)
                                # Normalize dtype string (pandas uses 'int64', polars might use 'Int64')
                                col_dtype_lower = col_dtype.lower()

                                # Cast filter values to match column dtype
                                if "int" in col_dtype_lower:
                                    # Integer types - ensure all values are Python ints (which are int64)
                                    if isinstance(filter_spec.value, list):
                                        cast_values_int: list[int] = [
                                            int(v) for v in filter_spec.value if isinstance(v, int | float | str)
                                        ]
                                    else:
                                        cast_values_int = [int(filter_spec.value)]
                                    # Cast column to int64 to ensure compatibility
                                    view = view.filter(col_expr.cast("int64").isin(cast_values_int))
                                elif "float" in col_dtype_lower:
                                    if isinstance(filter_spec.value, list):
                                        cast_values_float: list[float] = [
                                            float(v) for v in filter_spec.value if isinstance(v, int | float | str)
                                        ]
                                    else:
                                        cast_values_float = [float(filter_spec.value)]
                                    view = view.filter(col_expr.cast("float64").isin(cast_values_float))
                                else:
                                    # String or other types - try direct isin
                                    view = view.filter(col_expr.isin(filter_spec.value))
                            else:
                                # Empty table - apply filter anyway (will return empty)
                                view = view.filter(col_expr.isin(filter_spec.value))
                        except Exception as cast_error:
                            # If casting also fails, log and re-raise original error
                            logger.error(
                                f"Failed to apply IN filter for {column}: "
                                f"direct attempt: {first_error}, cast attempt: {cast_error}"
                            )
                            raise first_error from cast_error
                else:
                    # Single value - treat as equality
                    view = view.filter(col_expr == filter_spec.value)
            elif filter_spec.operator == "NOT_IN":
                if isinstance(filter_spec.value, list):
                    # Similar casting logic as IN
                    try:
                        view = view.filter(~col_expr.isin(filter_spec.value))
                    except Exception:
                        # If that fails, try with type casting
                        try:
                            sample = view.select(column).limit(1).execute()
                            if len(sample) > 0:
                                col_dtype = sample[column].dtype
                                if col_dtype in ["int8", "int16", "int32", "int64", "Int8", "Int16", "Int32", "Int64"]:
                                    if isinstance(filter_spec.value, list):
                                        cast_values_not_int: list[int] = [
                                            int(v) for v in filter_spec.value if isinstance(v, int | float | str)
                                        ]
                                    else:
                                        cast_values_not_int = [int(filter_spec.value)]
                                    view = view.filter(~col_expr.isin(cast_values_not_int))
                                elif col_dtype in ["float32", "float64", "Float32", "Float64"]:
                                    if isinstance(filter_spec.value, list):
                                        cast_values_not_float: list[float] = [
                                            float(v) for v in filter_spec.value if isinstance(v, int | float | str)
                                        ]
                                    else:
                                        cast_values_not_float = [float(filter_spec.value)]
                                    view = view.filter(~col_expr.isin(cast_values_not_float))
                                else:
                                    view = view.filter(~col_expr.isin(filter_spec.value))
                            else:
                                view = view.filter(~col_expr.isin(filter_spec.value))
                        except Exception as e:
                            logger.warning(f"Failed to apply NOT_IN filter for {column}: {e}")
                            raise
                else:
                    view = view.filter(col_expr != filter_spec.value)
            elif filter_spec.operator == ">":
                view = view.filter(col_expr > filter_spec.value)
            elif filter_spec.operator == ">=":
                view = view.filter(col_expr >= filter_spec.value)
            elif filter_spec.operator == "<":
                view = view.filter(col_expr < filter_spec.value)
            elif filter_spec.operator == "<=":
                view = view.filter(col_expr <= filter_spec.value)
            else:
                logger.warning(f"Unsupported filter operator: {filter_spec.operator}, skipping")

            # Handle exclude_nulls
            if filter_spec.exclude_nulls:
                view = view.filter(~col_expr.isnull())

        # Execute based on intent
        if plan.intent == "COUNT":
            # COUNT: Use entity_key or group_by
            if plan.group_by:
                # Count by group
                result = view.group_by(plan.group_by).aggregate(_.count().name("count"))
            elif plan.entity_key:
                # Count distinct entities
                result = view.select(plan.entity_key).distinct().count()
            else:
                # Total count
                result = view.count()
        elif plan.intent == "DESCRIBE":
            # DESCRIBE: Type-aware aggregation
            if plan.metric:
                # Detect if categorical or numeric
                is_categorical = self._detect_categorical_encoding(view[plan.metric])
                if is_categorical:
                    # Frequency table
                    result = view.group_by(plan.metric).aggregate(_.count().name("count"))
                else:
                    # Descriptive statistics
                    metric_col = view[plan.metric]
                    result = view.aggregate(
                        [
                            metric_col.mean().name("mean"),
                            metric_col.median().name("median"),
                            metric_col.std().name("std"),
                            metric_col.min().name("min"),
                            metric_col.max().name("max"),
                        ]
                    )
            else:
                result = view
        else:
            # Other intents: return base view for now
            result = view

        return result.execute()

    def _detect_categorical_encoding(self, column: Any) -> bool:
        """
        Detect if column is categorical (encoded as '1: Yes 2: No' or limited distinct values).

        Args:
            column: Ibis column expression or column name

        Returns:
            True if categorical, False if numeric
        """
        # If column is a string (column name), get the actual column from base view
        if isinstance(column, str):
            view = self.get_base_view()
            if column not in view.columns:
                return False  # Column doesn't exist, default to numeric
            column_expr = view[column]
        else:
            column_expr = column

        # Execute a sample query to check data characteristics
        # Get distinct count and sample values
        try:
            # Get distinct count (efficient check)
            distinct_count = column_expr.nunique().execute()

            # If very few distinct values (< 20), likely categorical
            if distinct_count <= 20:
                # Check if values look like codes (numeric codes with labels)
                # Sample a few values to check pattern
                sample = column_expr.head(100).execute()
                if len(sample) > 0:
                    # Check if values are strings containing ":" pattern (encoded labels)
                    if hasattr(sample, "str") and sample.str.contains(":").any():
                        return True
                    # Check if all values are small integers (likely codes)
                    if sample.dtype in ["int8", "int16", "int32", "int64"]:
                        if sample.min() >= 0 and sample.max() <= 20:
                            return True

            # Check column name/alias for encoding patterns
            column_name = column if isinstance(column, str) else str(column)
            alias_index = self.get_column_alias_index()
            for alias, canonical in alias_index.items():
                if canonical == column_name:
                    # Check if alias contains encoding pattern (e.g., "1: Yes 2: No")
                    if ":" in alias and any(char.isdigit() for char in alias):
                        return True
                    break

            # Check if column metadata indicates categorical
            try:
                metadata = self.get_column_metadata(column_name)
                if metadata:
                    var_type = metadata.get("type")
                    if var_type in ("categorical", "binary"):
                        return True
            except Exception:
                pass  # Metadata check failed, continue with other checks

            # Default: assume numeric if high cardinality
            return False

        except Exception:
            # If detection fails, default to numeric (safer for statistical operations)
            logger.warning(f"Failed to detect categorical encoding for column {column}, defaulting to numeric")
            return False
