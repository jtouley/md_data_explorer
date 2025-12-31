"""
User-Uploaded Dataset

Dynamic dataset implementation for user-uploaded files.
Integrates uploaded data with the existing registry system.
"""

import logging
from typing import Any

import pandas as pd
import polars as pl

from clinical_analytics.core.dataset import ClinicalDataset, Granularity
from clinical_analytics.core.schema import UnifiedCohort
from clinical_analytics.datasets.uploaded.patient_id_regeneration import (
    PatientIdRegenerationError,
    can_regenerate_patient_id,
    regenerate_patient_id,
    validate_synthetic_id_metadata,
)
from clinical_analytics.ui.storage.user_datasets import UserDatasetStorage

logger = logging.getLogger(__name__)

# Centralize categorical threshold (matches VariableTypeDetector)
CATEGORICAL_THRESHOLD = 20  # If unique values <= this, likely categorical


def _to_lazy(df_or_lf: pl.LazyFrame | pl.DataFrame | pd.DataFrame) -> pl.LazyFrame:
    """
    Normalize any data representation to LazyFrame.

    This ensures internal representation is always lazy, regardless of IO boundary.
    CSV files use pl.scan_csv() (true lazy IO).
    Excel files are eagerly loaded via pandas, then converted to LazyFrame.
    """
    if isinstance(df_or_lf, pl.LazyFrame):
        return df_or_lf
    if isinstance(df_or_lf, pl.DataFrame):
        return df_or_lf.lazy()
    # pandas - convert eagerly then make lazy
    return pl.from_pandas(df_or_lf).lazy()


class UploadedDataset(ClinicalDataset):
    """
    Dynamic dataset implementation for user uploads.

    This class allows uploaded datasets to work seamlessly with
    the existing registry system and analysis infrastructure.
    """

    def __init__(self, upload_id: str, storage: UserDatasetStorage | None = None):
        """
        Initialize uploaded dataset.

        Args:
            upload_id: Unique upload identifier
            storage: Storage manager (optional, will create if not provided)
        """
        if storage is None:
            storage = UserDatasetStorage()

        self.storage = storage
        self.upload_id = upload_id
        self.metadata = None
        self.data = None
        self._semantic_initialized = False  # Track lazy init

        # Load metadata
        self.metadata = storage.get_upload_metadata(upload_id)
        if not self.metadata:
            raise ValueError(f"Upload {upload_id} not found")

        # Validate synthetic_id_metadata once at load time (not lazily during cohort)
        # Staff Engineer Standard: Validate at boundaries
        raw_synthetic_id_metadata = self.metadata.get("synthetic_id_metadata")
        try:
            self._validated_synthetic_id_metadata = validate_synthetic_id_metadata(raw_synthetic_id_metadata)
        except PatientIdRegenerationError as e:
            logger.warning(
                "Invalid synthetic_id_metadata for upload %s: %s",
                upload_id,
                str(e),
            )
            self._validated_synthetic_id_metadata = {}

        # Initialize with upload info
        dataset_name = self.metadata.get("dataset_name", upload_id)
        super().__init__(name=dataset_name, source_path=None)

    def validate(self) -> bool:
        """
        Check if upload data exists and is valid.

        Returns:
            True if data is accessible
        """
        # Primary: upload_id is the immutable storage key
        csv_path = self.storage.raw_dir / f"{self.upload_id}.csv"

        # Backward compatibility: check for old friendly-name files
        if not csv_path.exists() and self.metadata:
            if "stored_relpath" in self.metadata:
                legacy_path = self.storage.raw_dir / self.metadata["stored_relpath"]
                if legacy_path.exists():
                    csv_path = legacy_path
            elif "stored_filename" in self.metadata:
                legacy_path = self.storage.raw_dir / self.metadata["stored_filename"]
                if legacy_path.exists():
                    csv_path = legacy_path

        return csv_path.exists()

    def load(self) -> None:
        """Load uploaded data, normalized to LazyFrame internally."""
        if not self.validate():
            raise FileNotFoundError(f"Upload data not found: {self.upload_id}")

        raw_data = self.storage.get_upload_data(self.upload_id, lazy=True)
        if raw_data is None:
            logger.error(f"Failed to load upload data for upload_id: {self.upload_id}")
            raise ValueError(f"Failed to load upload data: {self.upload_id}")

        # Normalize to LazyFrame (handles both CSV lazy scan and Excel eager→lazy conversion)
        self.data = _to_lazy(raw_data)
        logger.info(f"Loaded upload data as LazyFrame: {self.upload_id}")

    def get_cohort(self, granularity: Granularity = "patient_level", **filters) -> pd.DataFrame:
        """
        Return analysis cohort mapped to UnifiedCohort schema.

        Maps user columns to UnifiedCohort schema based on either:
        - variable_mapping (from single-table upload wizard)
        - inferred_schema (from multi-table ZIP upload)

        Args:
            granularity: Grain level (patient_level, admission_level, event_level)
                        All uploads support granularities based on their columns
            **filters: Optional filters (not yet implemented)

        Returns:
            DataFrame conforming to UnifiedCohort schema (outcome column optional)
        """
        # Phase 8: Log active version being used for query
        active_version = self.storage.get_active_version(self.upload_id)
        if active_version:
            # Guard against None values in version field
            version_str = active_version.get("version") or "unknown"
            version_display = version_str[:8] + "..." if len(version_str) > 8 else version_str
            logger.info(
                f"Query using dataset {self.upload_id}, active version: {version_display}, "
                f"event_type: {active_version.get('event_type', 'unknown')}"
            )
        else:
            logger.warning(f"No active version found for dataset {self.upload_id}")

        # Runtime validation: check if requested granularity is supported
        if granularity != "patient_level":
            # Get inferred_schema (or convert variable_mapping if needed)
            inferred_schema = self.metadata.get("inferred_schema")
            if not inferred_schema and self.metadata.get("variable_mapping"):
                # Convert variable_mapping to inferred_schema for granularity check
                if self.data is None:
                    self.load()
                from clinical_analytics.datasets.uploaded.schema_conversion import convert_schema

                if isinstance(self.data, pl.LazyFrame):
                    try:
                        schema = self.data.collect_schema()  # Preferred (Polars 0.19+)
                    except AttributeError:
                        schema = self.data.schema  # Fallback (may be incomplete but works for column names)
                    # Create empty DataFrame with schema for convert_schema()
                    data_for_schema = pl.DataFrame(schema={k: v for k, v in schema.items()})
                elif isinstance(self.data, pd.DataFrame):
                    data_for_schema = pl.from_pandas(self.data)
                else:
                    data_for_schema = self.data

                inferred_schema = convert_schema(
                    self.metadata["variable_mapping"],
                    data_for_schema,
                )

            supported = (
                inferred_schema.get("granularities", ["patient_level"]) if inferred_schema else ["patient_level"]
            )

            if granularity not in supported:
                raise ValueError(
                    f"Dataset does not support {granularity} granularity. "
                    f"Supported: {supported}. "
                    f"Hint: Granularities are inferred from columns (admission_id, event_timestamp)."
                )

        if self.data is None:
            self.load()

        # Ensure LazyFrame (normalize if somehow not)
        lf = _to_lazy(self.data)

        # Apply filters in Polars (lazy)
        if filters:
            filter_exprs = []
            for key, value in filters.items():
                if isinstance(value, list):
                    filter_exprs.append(pl.col(key).is_in(value))
                else:
                    filter_exprs.append(pl.col(key) == value)

            if filter_exprs:
                combined_filter = filter_exprs[0]
                for expr in filter_exprs[1:]:
                    combined_filter = combined_filter & expr
                lf = lf.filter(combined_filter)

        # Get variable mapping from metadata (single-table uploads)
        variable_mapping = self.metadata.get("variable_mapping", {})

        # If no variable mapping, try to build it from inferred schema (ZIP uploads)
        if not variable_mapping:
            inferred_schema = self.metadata.get("inferred_schema", {})

            if inferred_schema:
                # Convert inferred_schema to variable_mapping format
                variable_mapping = self._convert_inferred_schema_to_mapping(inferred_schema)
            else:
                raise ValueError("Neither variable_mapping nor inferred_schema found in upload metadata")

        # Extract mapping fields
        patient_id_col = variable_mapping.get("patient_id")
        outcome_col = variable_mapping.get("outcome")
        predictors = variable_mapping.get("predictors", [])
        time_vars = variable_mapping.get("time_variables", {})

        # Get schema to check for columns (lazy)
        try:
            schema = lf.collect_schema()
        except AttributeError:
            schema = lf.schema
        schema_names = set(schema.keys())

        # Handle patient_id regeneration if needed (materialize once, then continue lazy)
        if patient_id_col:
            if patient_id_col not in schema_names:
                # Handle case where column was renamed to 'patient_id' during ingestion
                if "patient_id" in schema_names:
                    logger.warning(
                        f"Mapped ID column '{patient_id_col}' not found, but 'patient_id' exists. "
                        f"Using 'patient_id' (column was likely renamed during ingestion)."
                    )
                    patient_id_col = "patient_id"
                else:
                    logger.error(
                        f"Patient ID column '{patient_id_col}' not found in data. "
                        f"Available columns: {list(schema_names)}. "
                        f"Metadata: {self.metadata.get('synthetic_id_metadata', {})}"
                    )
                    # If patient_id was created synthetically, it should be in the CSV
                    # Check if it exists with different casing or was lost
                    if patient_id_col == "patient_id" and "patient_id" not in schema_names:
                        # Try to regenerate using centralized helper (DRY: single source of truth)
                        logger.warning("Synthetic patient_id not found in loaded data, regenerating...")

                        df_materialized = lf.collect()

                        # Use validated metadata from init (already validated at boundary)
                        regen_result = can_regenerate_patient_id(df_materialized, self._validated_synthetic_id_metadata)

                        if regen_result.can_regenerate:
                            logger.info(
                                "Regenerating patient_id: source=%s, columns=%s",
                                regen_result.source_type,
                                regen_result.source_columns,
                            )
                            df_with_id, id_metadata = regenerate_patient_id(
                                df_materialized, self._validated_synthetic_id_metadata
                            )
                        elif regen_result.error_message:
                            # Fail fast with explicit error
                            raise PatientIdRegenerationError(
                                f"Cannot regenerate patient_id: {regen_result.error_message}",
                                metadata=self._validated_synthetic_id_metadata,
                            )
                        else:
                            # No regeneration metadata - fall back to auto-detection
                            from clinical_analytics.ui.components.variable_detector import (
                                VariableTypeDetector,
                            )

                            logger.warning("No regeneration metadata, using auto-detection")
                            df_with_id, id_metadata = VariableTypeDetector.ensure_patient_id(df_materialized)

                        logger.info(
                            "Regenerated patient_id: source=%s, columns=%s",
                            id_metadata["patient_id_source"],
                            id_metadata.get("patient_id_columns"),
                        )

                        # Persist as LazyFrame for future calls (avoids re-materialization)
                        self.data = df_with_id.lazy()
                        lf = self.data  # Update lf to use corrected LazyFrame

                        if "patient_id" not in df_with_id.columns:
                            raise ValueError(
                                f"Failed to create patient_id. Available columns: {list(df_with_id.columns)}"
                            )
                        patient_id_col = "patient_id"
                    else:
                        raise KeyError(
                            f"Patient ID column '{patient_id_col}' not found in data. "
                            f"Available columns: {list(schema_names)}"
                        )

        # Build select expressions for UnifiedCohort schema (all lazy)
        select_exprs = []

        # Map patient ID
        if patient_id_col:
            select_exprs.append(pl.col(patient_id_col).alias(UnifiedCohort.PATIENT_ID))
        else:
            # Generate sequential IDs if not provided
            logger.warning("No patient_id column specified, generating sequential IDs")
            # Use row number for sequential IDs (lazy) - int_range with len() creates 0..len-1 range
            select_exprs.append(
                (pl.lit("patient_") + pl.int_range(pl.len()).cast(pl.Utf8)).alias(UnifiedCohort.PATIENT_ID)
            )

        # Map outcome (optional - semantic layer pattern)
        if outcome_col and outcome_col in schema_names:
            select_exprs.append(pl.col(outcome_col).alias(UnifiedCohort.OUTCOME))
            # Add outcome_label as literal (metadata)
            outcome_label = variable_mapping.get("outcome_label", "outcome")
            select_exprs.append(pl.lit(outcome_label).alias(UnifiedCohort.OUTCOME_LABEL))

        # Map time zero (use upload date if not provided)
        upload_timestamp = pd.Timestamp(self.metadata["upload_timestamp"])
        if time_vars and time_vars.get("time_zero"):
            time_col = time_vars["time_zero"]
            if time_col in schema_names:
                select_exprs.append(pl.col(time_col).cast(pl.Datetime).alias(UnifiedCohort.TIME_ZERO))
            else:
                select_exprs.append(pl.lit(upload_timestamp).alias(UnifiedCohort.TIME_ZERO))
        else:
            # Use upload timestamp as time zero
            select_exprs.append(pl.lit(upload_timestamp).alias(UnifiedCohort.TIME_ZERO))

        # Add predictor variables (keep original names)
        for pred in predictors:
            if pred in schema_names:
                select_exprs.append(pl.col(pred))

        # For descriptive analysis, include ALL columns from original data
        # (not just predictors) so users can analyze any variable
        excluded_from_all = {
            patient_id_col,
            outcome_col,
            UnifiedCohort.PATIENT_ID,
            UnifiedCohort.OUTCOME,
            UnifiedCohort.OUTCOME_LABEL,
            UnifiedCohort.TIME_ZERO,
        }
        # Track which columns we've already added (by checking what we're selecting)
        # Get updated schema after potential patient_id regeneration
        try:
            current_schema = lf.collect_schema()
        except AttributeError:
            current_schema = lf.schema

        # Build set of columns already in select_exprs (by checking aliases and root names)
        added_columns = set()
        for expr in select_exprs:
            try:
                # Try to get output name (alias)
                if hasattr(expr, "meta") and hasattr(expr.meta, "output_name"):
                    added_columns.add(expr.meta.output_name())
                # Try to get root column names
                if hasattr(expr, "meta") and hasattr(expr.meta, "root_names"):
                    added_columns.update(expr.meta.root_names())
            except Exception:
                pass  # If we can't extract, continue - will check by column name below

        # Add remaining columns
        for col in current_schema.keys():
            if col not in excluded_from_all and col not in added_columns:
                select_exprs.append(pl.col(col))

        # Apply all transformations lazily
        lf = lf.select(select_exprs)

        # EXACTLY ONE COLLECT at return boundary
        result_df = lf.collect().to_pandas()

        return result_df

    def _convert_inferred_schema_to_mapping(self, inferred_schema: dict[str, Any]) -> dict[str, Any]:
        """
        Convert inferred_schema format (from ZIP uploads) to variable_mapping format.

        Args:
            inferred_schema: Schema from schema inference engine

        Returns:
            variable_mapping dictionary compatible with get_cohort()
        """
        variable_mapping = {
            "patient_id": None,
            "outcome": None,
            "time_variables": {},
            "predictors": [],
        }

        # Extract patient ID from column_mapping
        column_mapping = inferred_schema.get("column_mapping", {})
        for col, role in column_mapping.items():
            if role == "patient_id":
                variable_mapping["patient_id"] = col
                break

        # Extract first outcome
        outcomes = inferred_schema.get("outcomes", {})
        if outcomes:
            # Use first outcome as primary
            first_outcome = list(outcomes.keys())[0]
            variable_mapping["outcome"] = first_outcome

        # Extract time_zero
        time_zero_config = inferred_schema.get("time_zero", {})
        if "source_column" in time_zero_config:
            variable_mapping["time_variables"]["time_zero"] = time_zero_config["source_column"]

        # Add all other columns as predictors (exclude patient_id and outcome)
        # Use schema to get column names without collecting (avoid materializing large datasets)
        if self.data is not None:
            # Get column names efficiently
            if isinstance(self.data, pl.LazyFrame):
                # Use collect_schema() to get column names without collecting data
                all_cols = set(self.data.collect_schema().names())
            elif isinstance(self.data, pd.DataFrame):
                all_cols = set(self.data.columns)
            else:
                # Fallback for other types (e.g., Polars DataFrame)
                all_cols = set(self.data.columns)

            excluded = {variable_mapping["patient_id"], variable_mapping["outcome"]}
            variable_mapping["predictors"] = [col for col in all_cols if col not in excluded and col not in {None}]

        return variable_mapping

    def get_semantic_layer(self):
        """
        Get semantic layer, lazy-initializing from unified cohort if available.

        Overrides base class to support all uploads (single-table and multi-table).
        """
        # Lazy initialize if not already done
        if not self._semantic_initialized:
            self._maybe_init_semantic()
            self._semantic_initialized = True

        # Call parent to get semantic (or raise if None)
        return super().get_semantic_layer()

    def _maybe_init_semantic(self) -> None:
        """Lazy initialization of semantic layer for all uploads (single-table and multi-table)."""
        # Ensure metadata is not None (should be set in __init__)
        if self.metadata is None:
            raise ValueError("Metadata not initialized")

        # Check for inferred_schema first (multi-table path)
        inferred_schema = self.metadata.get("inferred_schema")
        if inferred_schema:
            config = self._build_config_from_inferred_schema(inferred_schema)
            upload_type = "multi-table"
        else:
            # Check for variable_mapping (single-table path)
            variable_mapping = self.metadata.get("variable_mapping")
            if variable_mapping:
                config = self._build_config_from_variable_mapping(variable_mapping)
                upload_type = "single-table"
            else:
                # Explicit error: raise instead of silent return
                raise ValueError(
                    f"No schema or mapping found for upload {self.upload_id}. "
                    "Upload must have either 'inferred_schema' (multi-table) or 'variable_mapping' (single-table)."
                )

        csv_path = self.storage.raw_dir / f"{self.upload_id}.csv"
        if not csv_path.exists():
            logger.warning(f"Unified cohort CSV not found for upload {self.upload_id}")
            return

        try:
            from clinical_analytics.core.semantic import SemanticLayer, _safe_identifier

            logger.info(f"Initializing semantic layer for {upload_type} upload: {self.upload_id}")
            logger.info(f"Built semantic layer config from {upload_type} schema")

            workspace_root = self.storage.upload_dir.parent.parent

            # Use absolute path - SemanticLayer handles absolute paths correctly
            config["init_params"] = {"source_path": str(csv_path.resolve())}

            self.semantic = SemanticLayer(dataset_name=self.name, config=config, workspace_root=workspace_root)

            # Register all individual tables from metadata (deterministic, not directory listing)
            table_names = self.metadata.get("tables", [])
            if not table_names:
                logger.info(f"No tables list in metadata for {self.upload_id}, skipping table registration")
                return

            tables_dir = self.storage.raw_dir / f"{self.upload_id}_tables"
            if not tables_dir.exists():
                logger.warning(
                    f"Tables directory missing for {self.upload_id}: {tables_dir}. "
                    "Running in legacy single-table mode (cohort only)."
                )
                return

            logger.info(f"Registering {len(table_names)} individual tables from metadata")

            duckdb_con = self.semantic.con.con
            safe_dataset_name = _safe_identifier(self.name)

            # Phase 2: Try loading from persistent DuckDB first, fallback to CSV

            db_path = self.storage.upload_dir.parent / "analytics.duckdb"
            dataset_version = self.metadata.get("dataset_version")
            use_persistent_duckdb = db_path.exists() and dataset_version

            if use_persistent_duckdb:
                logger.info(f"Loading tables from persistent DuckDB: {db_path}")
                # Attach persistent DuckDB to semantic layer's in-memory DuckDB
                duckdb_con.execute(f"ATTACH '{db_path}' AS persistent_db")

            for table_name in table_names:  # Use metadata list, not directory listing
                safe_table_name = _safe_identifier(f"{safe_dataset_name}_{table_name}")

                # Phase 2: Try loading from persistent DuckDB first
                if use_persistent_duckdb:
                    # Table name in persistent DB: {upload_id}_{table_name}_{dataset_version}
                    persistent_table_name = f"{self.upload_id}_{table_name}_{dataset_version}"

                    # Check if table exists in persistent DB
                    try:
                        duckdb_con.execute(
                            f"CREATE TABLE IF NOT EXISTS {safe_table_name} AS "
                            f"SELECT * FROM persistent_db.{persistent_table_name}"
                        )
                        logger.info(f"Registered table '{safe_table_name}' from persistent DuckDB")
                        continue  # Successfully loaded from DuckDB, skip CSV fallback
                    except Exception as e:
                        logger.warning(
                            f"Failed to load '{persistent_table_name}' from persistent DuckDB: {e}. "
                            "Falling back to CSV."
                        )

                # Fallback: Load from CSV (migration path for old uploads)
                table_path = tables_dir / f"{table_name}.csv"
                if not table_path.exists():
                    raise FileNotFoundError(
                        f"Table file missing for upload {self.upload_id}, table '{table_name}': expected {table_path}"
                    )

                abs_path = str(table_path.resolve())

                # CRITICAL FIX: Use IF NOT EXISTS instead of OR REPLACE
                # Prevents data loss on semantic layer re-init within same session
                duckdb_con.execute(
                    f"CREATE TABLE IF NOT EXISTS {safe_table_name} AS SELECT * FROM read_csv_auto(?)",
                    [abs_path],
                )
                logger.info(f"Registered table '{safe_table_name}' from CSV (fallback): {abs_path}")

            logger.info(f"Created semantic layer for uploaded dataset '{self.name}' with {len(table_names)} tables")
        except Exception as e:
            logger.warning(f"Failed to create semantic layer: {e}")
            import traceback

            logger.debug(traceback.format_exc())
            # Leave self.semantic as None (class attribute default)

    def _build_config_from_inferred_schema(self, inferred_schema: dict[str, Any]) -> dict[str, Any]:
        """Build semantic layer config from inferred schema."""
        outcomes = inferred_schema.get("outcomes", {})
        default_outcome = list(outcomes.keys())[0] if outcomes else None

        config = {
            "name": self.name,
            "display_name": self.metadata.get("original_filename", self.name),
            "status": "available",
            "init_params": {},  # Will be set to absolute CSV path
            "column_mapping": inferred_schema.get("column_mapping", {}),
            "outcomes": outcomes,
            "time_zero": inferred_schema.get("time_zero", {}),
            "analysis": {
                "default_outcome": default_outcome,
                "default_predictors": inferred_schema.get("predictors", []),
                "categorical_variables": inferred_schema.get("categorical_columns", []),
            },
            # Include variable_types metadata for column type detection
            "variable_types": self.metadata.get("variable_types"),
        }
        return config

    def _build_config_from_variable_mapping(self, variable_mapping: dict[str, Any]) -> dict[str, Any]:
        """
        Build semantic layer config from variable_mapping (single-table uploads).

        Converts variable_mapping format to semantic layer config format.
        Matches structure of _build_config_from_inferred_schema() for consistency.

        Args:
            variable_mapping: Dictionary with patient_id, outcome, time_variables, predictors

        Returns:
            Semantic layer config dictionary compatible with SemanticLayer
        """
        # Load data if needed for categorical detection and outcome type inference
        if self.data is None:
            self.load()

        # After load(), self.data should not be None
        if self.data is None:
            raise ValueError("Failed to load data for semantic layer config")

        # Convert to Polars DataFrame for efficient processing (materialize LazyFrame if needed)
        if isinstance(self.data, pl.LazyFrame):
            df_polars = self.data.collect()
        elif isinstance(self.data, pd.DataFrame):
            df_polars = pl.from_pandas(self.data)
        else:
            df_polars = self.data

        # Build column_mapping
        column_mapping = {}
        patient_id_col = variable_mapping.get("patient_id")
        if patient_id_col:
            column_mapping[patient_id_col] = "patient_id"

        # Build outcomes with type inference
        outcomes = {}
        outcome_col = variable_mapping.get("outcome")
        if outcome_col:
            # Infer outcome type from data (don't hardcode "binary")
            outcome_type = self._infer_outcome_type(df_polars, outcome_col)
            outcomes[outcome_col] = {
                "source_column": outcome_col,
                "type": outcome_type,
            }

        # Build time_zero (must match multi-table format exactly: {"source_column": str})
        time_zero = {}
        time_vars = variable_mapping.get("time_variables", {})
        time_zero_col = time_vars.get("time_zero")
        if time_zero_col:
            # Match multi-table format: {"source_column": str}
            time_zero["source_column"] = time_zero_col

        # Detect categorical variables from predictors
        # TODO: Consider sampling strategy for large columns (series.n_unique() can be expensive)
        predictors = variable_mapping.get("predictors", [])
        categorical_variables = []

        for col in predictors:
            if col not in df_polars.columns:
                continue

            series = df_polars.select(col).to_series()
            dtype = series.dtype

            # String type → categorical
            if dtype == pl.Utf8 or dtype == pl.Categorical:
                categorical_variables.append(col)
            # Boolean → categorical
            elif dtype == pl.Boolean:
                categorical_variables.append(col)
            # Numeric with ≤CATEGORICAL_THRESHOLD unique values → categorical
            elif dtype.is_numeric():
                # For large columns, consider sampling (TODO: implement if performance issues)
                unique_count = series.n_unique()
                if unique_count > 100_000:
                    logger.warning(
                        f"Column '{col}' has {unique_count:,} unique values. "
                        "Categorical detection may be slow. Consider sampling strategy."
                    )
                if unique_count <= CATEGORICAL_THRESHOLD:
                    categorical_variables.append(col)

        config = {
            "name": self.name,
            "display_name": self.metadata.get("original_filename", self.name),
            "status": "available",
            "init_params": {},  # Will be set to absolute CSV path in _maybe_init_semantic()
            "column_mapping": column_mapping,
            "outcomes": outcomes,
            "time_zero": time_zero,
            "analysis": {
                "default_outcome": outcome_col,
                "default_predictors": predictors,
                "categorical_variables": categorical_variables,
            },
            # Include variable_types metadata for column type detection
            "variable_types": self.metadata.get("variable_types"),
        }
        return config

    def _infer_outcome_type(self, df_polars: pl.DataFrame, outcome_col: str) -> str:
        """
        Infer outcome type from data characteristics.

        Args:
            df_polars: Polars DataFrame (must be materialized, not LazyFrame)
            outcome_col: Outcome column name

        Returns:
            Outcome type: "binary", "continuous", or "time_to_event"

        Note:
            Defaults to "binary" if ambiguous, but logs warning.
        """
        if outcome_col not in df_polars.columns:
            logger.warning(f"Outcome column '{outcome_col}' not found in data, defaulting to 'binary'")
            return "binary"

        series = df_polars.select(outcome_col).to_series()
        dtype = series.dtype
        unique_count = series.n_unique()

        # Binary: exactly 2 unique values
        if unique_count == 2:
            return "binary"

        # Time-to-event: datetime type or name suggests time
        if dtype in (pl.Date, pl.Datetime, pl.Time):
            return "time_to_event"

        # Continuous: numeric with >2 unique values
        if dtype.is_numeric() and unique_count > 2:
            logger.warning(
                f"Outcome '{outcome_col}' has {unique_count} unique values. "
                "Inferring 'continuous' type. If this is binary, ensure data is encoded as 0/1."
            )
            return "continuous"

        # Default to binary with warning
        logger.warning(
            f"Outcome '{outcome_col}' type ambiguous (dtype={dtype}, unique={unique_count}). "
            "Defaulting to 'binary'. Verify outcome type is correct."
        )
        return "binary"

    def get_info(self) -> dict[str, Any]:
        """
        Get dataset information.

        Returns:
            Dictionary with dataset metadata
        """
        # Ensure metadata is not None (should be set in __init__)
        if self.metadata is None:
            raise ValueError("Metadata not initialized")

        return {
            "upload_id": self.upload_id,
            "name": self.name,
            "uploaded_at": self.metadata.get("upload_timestamp"),
            "original_filename": self.metadata.get("original_filename"),
            "row_count": self.metadata.get("row_count"),
            "column_count": self.metadata.get("column_count"),
            "columns": self.metadata.get("columns"),
            "variable_mapping": self.metadata.get("variable_mapping"),
        }


class UploadedDatasetFactory:
    """
    Factory for creating UploadedDataset instances.

    Provides methods to list and create uploaded datasets
    for integration with the registry system.
    """

    @staticmethod
    def list_available_uploads() -> list[dict[str, Any]]:
        """
        List all available uploaded datasets.

        Returns:
            List of upload metadata dictionaries
        """
        storage = UserDatasetStorage()
        return storage.list_uploads()

    @staticmethod
    def create_dataset(upload_id: str) -> UploadedDataset:
        """
        Create dataset instance for an upload.

        Args:
            upload_id: Upload identifier

        Returns:
            UploadedDataset instance
        """
        return UploadedDataset(upload_id=upload_id)

    @staticmethod
    def register_all_uploads() -> dict[str, UploadedDataset]:
        """
        Create dataset instances for all uploads.

        Returns:
            Dictionary mapping upload_id to dataset instances
        """
        uploads = UploadedDatasetFactory.list_available_uploads()
        datasets = {}

        for upload in uploads:
            upload_id = upload["upload_id"]
            try:
                datasets[upload_id] = UploadedDataset(upload_id=upload_id)
            except Exception as e:
                logger.warning(f"Failed to load upload {upload_id}: {e}")

        return datasets
