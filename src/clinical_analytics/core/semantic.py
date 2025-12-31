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
from pathlib import Path
from typing import TYPE_CHECKING, Any

import ibis
import pandas as pd
from ibis import _

from clinical_analytics.core.mapper import load_dataset_config
from clinical_analytics.core.schema import UnifiedCohort

if TYPE_CHECKING:
    from clinical_analytics.core.dataset import Granularity
    from clinical_analytics.core.query_plan import QueryPlan

logger = logging.getLogger(__name__)


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

    def _register_source(self):
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

    def get_base_view(self):
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
                return source
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
        return quality_warnings

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
        return self.config.get("metrics", {})

    def get_available_dimensions(self) -> dict[str, dict[str, Any]]:
        """
        Get all available dimensions from config.

        Returns:
            Dictionary mapping dimension names to their definitions
        """
        return self.config.get("dimensions", {})

    def get_available_filters(self) -> dict[str, dict[str, Any]]:
        """
        Get all available filter definitions from config.

        Returns:
            Dictionary mapping filter names to their definitions
        """
        return self.config.get("filters", {})

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
            return variable_types[column_name]

        # Fallback: check if metadata is stored elsewhere in config
        metadata = self.config.get("metadata")
        if metadata:
            variable_types = metadata.get("variable_types")
            if variable_types and column_name in variable_types:
                return variable_types[column_name]

        return None

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
        # Format: "column.aggregation()" or "aggregation()"
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
            return ibis.count()

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

        if normalized in self._collision_warnings and self._alias_to_canonicals:
            # Return all canonical names that match this alias
            return sorted(list(self._alias_to_canonicals.get(normalized, set())))
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
                return json.load(f)
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
        if normalized_term in self._alias_index:
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

    def execute_query_plan(
        self, plan: "QueryPlan", confidence_threshold: float = 0.75, query_text: str | None = None
    ) -> dict[str, Any]:  # type: ignore[valid-type]
        """
        Execute a QueryPlan with confidence and completeness gating (ADR003 Phase 3 + Phase 2.1 Observability).

        Phase 2.1: Added warnings infrastructure for observability before removing gates.

        This method enforces hard gates before execution:
        - Confidence must be >= threshold
        - Plan must be complete (all required fields present)
        - Plan must pass validation (columns exist, operators valid, types compatible)

        Args:
            plan: QueryPlan to execute
            confidence_threshold: Minimum confidence required (default: 0.75)
            query_text: Optional query text for run_key generation

        Returns:
            dict with keys:
            - "success": bool - Whether execution succeeded
            - "requires_confirmation": bool - True if gate failed (user must confirm)
            - "failure_reason": str - Explanation if gate failed
            - "result": pd.DataFrame | None - Query results if successful
            - "run_key": str | None - Deterministic run key for idempotency
            - "warnings": list[str] - Warnings collected during execution (Phase 2.1)
        """
        # Phase 2.1: Initialize warnings list for observability
        warnings: list[str] = []

        # Step 1: Confidence Gating (with warning collection)
        if plan.confidence < confidence_threshold:
            warning_msg = (
                f"Low confidence: {plan.confidence:.2f} (threshold: {confidence_threshold:.2f}). "
                f"Query interpretation may be ambiguous or uncertain."
            )
            warnings.append(warning_msg)
            return {
                "success": False,
                "requires_confirmation": True,
                "failure_reason": f"Confidence {plan.confidence:.2f} below threshold {confidence_threshold:.2f}",
                "result": None,
                "run_key": None,
                "warnings": warnings,
            }

        # Step 2: Completeness Gating (with warning collection)
        is_complete, completeness_error = self._check_plan_completeness(plan)
        if not is_complete:
            warning_msg = f"Incomplete plan: {completeness_error}"
            warnings.append(warning_msg)
            return {
                "success": False,
                "requires_confirmation": True,
                "failure_reason": completeness_error,
                "result": None,
                "run_key": None,
                "warnings": warnings,
            }

        # Step 3: Validation Gating (with warning collection)
        validation_result = self._validate_query_plan(plan)
        if not validation_result["valid"]:
            warning_msg = f"Validation failed: {validation_result['error']}"
            warnings.append(warning_msg)
            return {
                "success": False,
                "requires_confirmation": True,
                "failure_reason": validation_result["error"],
                "result": None,
                "run_key": None,
                "warnings": warnings,
            }

        # Step 4: Generate run_key
        run_key = self._generate_run_key(plan, query_text)

        # Step 5: Execute query
        try:
            result_df = self._execute_plan(plan)
            # Success: no warnings
            return {
                "success": True,
                "requires_confirmation": False,
                "failure_reason": None,
                "result": result_df,
                "run_key": run_key,
                "warnings": warnings,
            }
        except Exception as e:
            logger.error(f"Query execution failed: {e}", exc_info=True)
            warning_msg = f"Execution error: {str(e)}"
            warnings.append(warning_msg)
            return {
                "success": False,
                "requires_confirmation": True,
                "failure_reason": f"Execution error: {str(e)}",
                "result": None,
                "run_key": run_key,
                "warnings": warnings,
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
        """Generate deterministic run_key from canonical plan + query text."""
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

        # Normalize query text
        query_signature = ""
        if query_text:
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
                                    cast_values = [int(v) for v in filter_spec.value]
                                    # Cast column to int64 to ensure compatibility
                                    view = view.filter(col_expr.cast("int64").isin(cast_values))
                                elif "float" in col_dtype_lower:
                                    cast_values = [float(v) for v in filter_spec.value]
                                    view = view.filter(col_expr.cast("float64").isin(cast_values))
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
                                    cast_values = [int(v) for v in filter_spec.value]
                                    view = view.filter(~col_expr.isin(cast_values))
                                elif col_dtype in ["float32", "float64", "Float32", "Float64"]:
                                    cast_values = [float(v) for v in filter_spec.value]
                                    view = view.filter(~col_expr.isin(cast_values))
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
