# SPDX-License-Identifier: MIT
"""
Multi-Table Handler - Automatic Relationship Detection and Joins

This module enables handling of multi-table datasets (like MIMIC-IV) by automatically:
- Detecting primary keys in each table
- Discovering foreign key relationships
- Building a join graph
- Executing joins to create unified cohort views

Key Principles:
- Use Polars for all DataFrame operations and for the unified cohort pipeline
  (`build_unified_cohort`: `_build_dimension_mart`, `_aggregate_fact_tables`, lazy joins, `collect()`).
- DuckDB (`conn.register`) holds registered tables for compatibility and optional paths;
  the hot join path for the unified cohort is Polars, not DuckDB SQL.
- Privacy-preserving (all local computation)
- Fail gracefully with user override options
"""

import fnmatch
import hashlib
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import duckdb
import polars as pl
import structlog

from clinical_analytics.core.nl_query_config import SMALL_CELL_THRESHOLD

try:
    import ibis
except ImportError:
    ibis = None  # Optional dependency

logger = structlog.get_logger(__name__)

# Schema version for caching - increment when schema changes break compatibility
SCHEMA_VERSION = "1.0"

# Classification heuristic only: a table whose rows / distinct keys is at most this is treated as
# "meant to be one row per key" (dimension). Whether it IS unique is judged exactly, never with this.
NEAR_UNIQUE_RATIO = 1.05

# Type aliases for clarity
AggregationOp = Literal["count", "count_distinct", "min", "max", "mean", "sum", "last"]
CodeColumnPattern = str  # Regex or glob pattern for code columns
TableClass = Literal["dimension", "fact", "event", "bridge", "reference"]

_NUMERIC_DTYPES = (
    pl.Int8,
    pl.Int16,
    pl.Int32,
    pl.Int64,
    pl.UInt8,
    pl.UInt16,
    pl.UInt32,
    pl.UInt64,
    pl.Float32,
    pl.Float64,
)
_TEMPORAL_DTYPES = (pl.Datetime, pl.Date)


class AggregationPolicyError(ValueError):
    """Raised when aggregation policy is violated (e.g., mean on code column)."""

    pass


class DataQualityError(ValueError):
    """Raised when data breaks a grain invariant the cohort depends on (e.g. duplicate anchor keys)."""


def describe_count(n: int) -> str:
    """Return ``n`` as text, or "some" when it is below the small-cell threshold."""
    return str(n) if n >= SMALL_CELL_THRESHOLD else "some"


def duplicate_key_error(table_name: str, key: str, n_duplicates: int, context: str) -> DataQualityError:
    """Build the error for a table that has repeated values in a key that must be unique."""
    return DataQualityError(
        f"Table '{table_name}' has {describe_count(n_duplicates)} duplicate rows on key '{key}' "
        f"({context}). Each '{key}' must appear once, otherwise counts would include the same "
        f"entity more than once. Remove the duplicate rows or choose a different key."
    )


@dataclass
class AggregationPolicy:
    """
    Policy for safe aggregation of fact/event tables.

    Enforces constraints to prevent incorrect aggregations:
    - Default safe aggregations (always allowed)
    - Opt-in aggregations (require explicit permission)
    - Code column protection (prevent mean/avg on codes)

    Attributes:
        default_numeric: Aggregations always safe for numeric columns
        allow_mean: Enable mean/avg aggregations (requires normalized units)
        allow_last: Permit last() aggregation. It is only emitted for tables with a Date/Datetime
            time column, and rows are sorted by that column first; without one it is off,
            so file order never stands in for time.
        code_column_patterns: Patterns identifying code/identifier columns (no mean/min/max)
    """

    default_numeric: list[AggregationOp] = field(default_factory=lambda: ["min", "max"])
    allow_mean: bool = False
    allow_last: bool = True
    code_column_patterns: list[CodeColumnPattern] = field(
        default_factory=lambda: [
            "icd_code",
            "itemid",
            "ndc",
            "cpt_code",
            "drg",
            "hcpcs",
            "*_code",
            "*_id",  # Glob patterns
        ]
    )


@dataclass
class TableClassification:
    """
    Classification metadata for a table in a multi-table dataset.

    Used to determine how a table should be handled in aggregate-before-join pipeline:
    - Dimensions: Small, unique on grain, safe to join directly
    - Facts: High cardinality, must be pre-aggregated before joining
    - Events: Time-series facts, high cardinality, temporal dimension
    - Bridges: Many-to-many relationships (composite unique), excluded from auto-joins
    - Reference: Code mappings, lookup tables with duplicates allowed

    Attributes:
        table_name: Name of the table
        classification: Category based on cardinality and structure
        grain: Level of granularity (patient, admission, event)
        grain_key: Column name of the detected grain key
        cardinality_ratio: non-null rows / distinct non-null grain_key values (exact, whole table),
            >1.1 indicates facts/events
        is_unique_on_grain: True only if every row has its own grain_key value (exact, whole
            table; two rows with a missing key are not unique)
        estimated_bytes: Total table size in bytes (rows * avg_row_bytes)
        relationship_degree: Number of foreign keys detected
        has_time_column: True if table has non-constant time column
        time_column_name: Name of time column if detected
        is_n_side_of_anchor: True if table is on N-side of relationship to anchor
        null_rate_in_grain: % of NULLs in grain_key column (0-1)
    """

    table_name: str
    classification: Literal["dimension", "fact", "event", "bridge", "reference"]
    grain: Literal["patient", "admission", "event"]
    grain_key: str
    cardinality_ratio: float
    is_unique_on_grain: bool
    estimated_bytes: int
    relationship_degree: int
    has_time_column: bool
    time_column_name: str | None
    is_n_side_of_anchor: bool
    null_rate_in_grain: float


@dataclass
class CohortMetadata:
    """Metadata for materialized cohort mart."""

    grain: Literal["patient", "admission", "event"]
    grain_key: str
    anchor_table: str
    output_path: Path  # Base output directory
    parquet_path: Path  # Single path to mart Parquet (file or directory)
    schema: dict[str, str]  # column_name -> dtype (e.g., "patient_id" -> "Utf8")
    row_count: int
    materialized_at: datetime
    run_id: str  # Deterministic hash: sha256(config + schema_version + dataset_fingerprint)


@dataclass(frozen=True)
class ExcludedTable:
    """A source table that is not part of the unified cohort, and why."""

    table_name: str
    classification: str
    reason: str


@dataclass(frozen=True)
class CohortBuildResult:
    """Unified cohort plus what was left out, so callers can tell the user."""

    data: pl.DataFrame
    anchor_table: str
    grain_key: str
    excluded_tables: list[ExcludedTable]


def is_code_column(col: str, patterns: list[CodeColumnPattern]) -> bool:
    """True if ``col`` matches a code/identifier pattern (case-insensitive glob)."""
    return any(fnmatch.fnmatch(col.lower(), pattern.lower()) for pattern in patterns)


def aggregate_fact_table(
    df: pl.DataFrame,
    grain_key: str,
    *,
    time_column: str | None,
    policy: AggregationPolicy,
    table_name: str,
) -> pl.LazyFrame:
    """
    Aggregate one fact/event table to one row per ``grain_key``.

    Output columns, in order: ``grain_key``, ``count``, then per column ``{col}_count_distinct``,
    and for numeric/temporal columns ``{col}_min``/``{col}_max`` (not for code/id columns),
    ``{col}_mean`` (numeric, if allowed) and ``{col}_last``.

    ``{col}_last`` is the value on the row with the latest ``time_column`` (nulls rank earliest;
    ties keep file order). It is only produced when ``policy.allow_last`` is set and
    ``time_column`` is a Date/Datetime column; otherwise there is no ordering to trust.

    Raises:
        AggregationPolicyError: mean requested on a code column.
        ValueError: ``grain_key`` or ``time_column`` missing from ``df``.
    """
    if grain_key not in df.columns:
        raise ValueError(f"Column '{grain_key}' not found in table '{table_name}'")
    if time_column is not None and time_column not in df.columns:
        raise ValueError(f"Time column '{time_column}' not found in table '{table_name}'")

    sort_column = time_column if time_column is not None and df[time_column].dtype in _TEMPORAL_DTYPES else None
    emit_last = policy.allow_last and sort_column is not None

    agg_exprs: list[pl.Expr] = [pl.len().alias("count")]
    for col in df.columns:
        if col == grain_key:
            continue
        code_col = is_code_column(col, policy.code_column_patterns)
        dtype = df[col].dtype
        agg_exprs.append(pl.col(col).n_unique().alias(f"{col}_count_distinct"))

        is_numeric = dtype in _NUMERIC_DTYPES
        if not (is_numeric or dtype in _TEMPORAL_DTYPES):
            continue  # strings: count distinct only

        # min/max of an identifier or code is not a meaningful number
        if not code_col:
            agg_exprs.append(pl.col(col).min().alias(f"{col}_min"))
            agg_exprs.append(pl.col(col).max().alias(f"{col}_max"))

        if is_numeric and policy.allow_mean:
            if code_col:
                raise AggregationPolicyError(
                    f"Cannot compute mean on code column '{col}' in table '{table_name}'. "
                    f"Column matches pattern: {policy.code_column_patterns}. "
                    f"Code columns should not be averaged."
                )
            agg_exprs.append(pl.col(col).mean().alias(f"{col}_mean"))

        if emit_last:
            agg_exprs.append(pl.col(col).sort_by(sort_column, maintain_order=True).last().alias(f"{col}_last"))

    return df.lazy().group_by(grain_key).agg(agg_exprs)


@dataclass
class TableRelationship:
    """
    Detected relationship between two tables.

    Represents a foreign key relationship discovered through pattern matching
    and referential integrity analysis.

    Attributes:
        parent_table: Table with the primary key
        child_table: Table with the foreign key
        parent_key: Primary key column name
        child_key: Foreign key column name
        relationship_type: Type of relationship (one-to-many, etc.)
        confidence: Detection confidence score (0-1)
        match_ratio: % of child values that exist in parent
    """

    parent_table: str
    child_table: str
    parent_key: str
    child_key: str
    relationship_type: str  # "one-to-many", "many-to-one", "one-to-one"
    confidence: float
    match_ratio: float = 0.0

    def __str__(self) -> str:
        return (
            f"{self.parent_table}.{self.parent_key} → {self.child_table}.{self.child_key} "
            f"({self.relationship_type}, conf={self.confidence:.2f}, match={self.match_ratio:.2f})"
        )


class MultiTableHandler:
    """
    Handle multi-table datasets with automatic relationship detection.

    Workflow:
    1. Load all tables (Dict[table_name, Polars DataFrame])
    2. Detect primary keys in each table
    3. Discover foreign key relationships
    4. Build join graph
    5. Execute joins to create unified cohort view

    Example:
        >>> tables = {
        ...     'patients': pl.read_csv('patients.csv'),
        ...     'admissions': pl.read_csv('admissions.csv'),
        ...     'diagnoses': pl.read_csv('diagnoses.csv')
        ... }
        >>> handler = MultiTableHandler(tables)
        >>> relationships = handler.detect_relationships()
        >>> cohort = handler.build_unified_cohort()
    """

    def __init__(
        self,
        tables: dict[str, pl.DataFrame],
        max_dimension_bytes: int = 250_000_000,  # 250 MB default
    ):
        """
        Initialize with dictionary of table_name -> Polars DataFrame.

        Args:
            tables: Dict mapping table names to Polars DataFrames
            max_dimension_bytes: Max size for dimension tables (default 250 MB)
        """
        self.tables = tables
        self.relationships: list[TableRelationship] = []
        self.primary_keys: dict[str, str] = {}
        self.classifications: dict[str, TableClassification] = {}
        self.max_dimension_bytes = max_dimension_bytes
        self._ibis_connection: Any | None = None  # Cached Ibis DuckDB connection
        self.last_build_result: CohortBuildResult | None = None
        # Reasons for tables left out of the most recent build, keyed by table name
        self._exclusion_reasons: dict[str, str] = {}
        self._joined_dimensions: set[str] = set()

        # Normalize key column types across all tables
        self._normalize_key_columns()

        # Initialize DuckDB connection for SQL-based joins
        self.conn = duckdb.connect(":memory:")

        # Register all tables in DuckDB
        for table_name, df in self.tables.items():
            self.conn.register(table_name, df)

    def _normalize_key_columns(self) -> None:
        """
        Normalize data types for common key columns across all tables.

        This prevents type mismatch errors when comparing keys across tables
        (e.g., subject_id as int64 in one table, string in another).

        Strategy:
        1. Identify columns that appear in multiple tables (potential join keys)
        2. Normalize them to string type for consistent comparisons
        """
        logger.debug("Starting key column normalization")

        # Count how many tables each column appears in
        column_counts: dict[str, int] = {}
        for df in self.tables.values():
            for col in df.columns:
                column_counts[col] = column_counts.get(col, 0) + 1

        # Identify columns that appear in multiple tables (potential join keys)
        shared_columns = {col for col, count in column_counts.items() if count > 1}
        logger.debug(f"Found {len(shared_columns)} shared columns: {list(shared_columns)[:10]}")

        # Also include common key patterns
        key_patterns = [
            "_id",
            "subject_id",
            "hadm_id",
            "stay_id",
            "itemid",
            "icustay_id",
            "transfer_id",
            "caregiver_id",
            "charttime",
        ]

        key_columns = shared_columns.copy()
        for df in self.tables.values():
            for col in df.columns:
                col_lower = col.lower()
                if any(pattern in col_lower for pattern in key_patterns):
                    key_columns.add(col)

        logger.debug(f"Total key columns to normalize: {len(key_columns)}")

        # Normalize each key column to string type across all tables
        for table_name, df in self.tables.items():
            cols_to_cast = [col for col in key_columns if col in df.columns]

            if cols_to_cast:
                logger.debug(f"Normalizing {len(cols_to_cast)} columns in table '{table_name}': {cols_to_cast}")
                # Log original dtypes
                original_dtypes = {col: df[col].dtype for col in cols_to_cast}
                logger.debug(f"Original dtypes for {table_name}: {original_dtypes}")

                try:
                    # Cast all key columns in one operation
                    cast_exprs = [pl.col(col).cast(pl.Utf8, strict=False).alias(col) for col in cols_to_cast]
                    self.tables[table_name] = df.with_columns(cast_exprs)

                    # Verify cast succeeded
                    new_dtypes = {col: self.tables[table_name][col].dtype for col in cols_to_cast}
                    logger.debug(f"New dtypes for {table_name}: {new_dtypes}")

                    # Check for any that didn't convert
                    failed_casts = [col for col in cols_to_cast if self.tables[table_name][col].dtype != pl.Utf8]
                    if failed_casts:
                        logger.warning(f"Failed to cast columns in {table_name}: {failed_casts}")

                except Exception as e:
                    logger.error(f"Batch cast failed for {table_name}: {type(e).__name__}")
                    # Try one by one if batch fails
                    for col in cols_to_cast:
                        try:
                            original_dtype = df[col].dtype
                            self.tables[table_name] = self.tables[table_name].with_columns(
                                pl.col(col).cast(pl.Utf8, strict=False)
                            )
                            new_dtype = self.tables[table_name][col].dtype
                            logger.debug(f"Cast {table_name}.{col}: {original_dtype} -> {new_dtype}")
                        except Exception as col_e:
                            logger.error(
                                f"Failed to cast {table_name}.{col} from {df[col].dtype}: "
                                f"{type(col_e).__name__}: {str(col_e)}"
                            )
                            pass  # Skip if casting fails

    def _sample_df(self, df: pl.DataFrame, n: int = 10_000) -> pl.DataFrame:
        """
        Deterministic sample using head (fastest, stable).

        Bounds classification cost to O(sample_size) regardless of table size.
        Uses head() for simplicity and stability (no randomness).

        Args:
            df: DataFrame to sample
            n: Maximum sample size (default 10k rows)

        Returns:
            Sampled DataFrame with at most n rows
        """
        return df.head(min(n, df.height))

    def _compute_sampled_uniqueness(self, df: pl.DataFrame, col: str) -> tuple[int, float]:
        """
        Compute uniqueness and null rate on sampled data only.

        Critical: This method ensures all uniqueness computations are bounded
        by sample size, preventing expensive full table scans.

        Args:
            df: DataFrame to sample
            col: Column name

        Returns:
            Tuple of (unique_count, null_rate)
        """
        s = self._sample_df(df)

        if col not in s.columns:
            return (0, 1.0)

        non_null = s[col].drop_nulls()
        if non_null.len() == 0:
            return (0, 1.0)

        unique_count = non_null.n_unique()
        null_rate = s[col].null_count() / max(s.height, 1)

        return (unique_count, null_rate)

    def _is_probably_id_col(self, col_name: str) -> bool:
        """
        Check if column name suggests ID column using tight pattern matching.

        Tight pattern: exact 'id' or endswith('_id')
        This avoids false positives like: valid, fluid, paid, acid

        Args:
            col_name: Column name to check

        Returns:
            True if column name matches tight ID pattern
        """
        col_lower = col_name.lower()
        return col_lower == "id" or col_lower.endswith("_id")

    def detect_relationships(self) -> list[TableRelationship]:
        """
        Auto-detect foreign key relationships between tables.

        Strategy:
        1. Detect primary keys for each table
        2. For each table pair, check if child has column matching parent PK
        3. Verify referential integrity (% of child values in parent)
        4. Score confidence based on name similarity + integrity

        Returns:
            List of detected relationships sorted by confidence
        """
        # 1. Detect primary keys for each table
        for table_name, df in self.tables.items():
            pk = self._detect_primary_key(df)
            if pk:
                self.primary_keys[table_name] = pk

        # 2. Detect foreign key relationships
        for parent_table, parent_df in self.tables.items():
            parent_pk = self.primary_keys.get(parent_table)

            if not parent_pk:
                continue  # Skip tables without primary keys

            for child_table, child_df in self.tables.items():
                if parent_table == child_table:
                    continue  # Skip self-joins

                # Check if child has column matching parent's primary key
                for child_col in child_df.columns:
                    if self._is_foreign_key_candidate(parent_pk, child_col):
                        # Verify referential integrity
                        match_ratio = self._verify_referential_integrity(parent_df, parent_pk, child_df, child_col)

                        if match_ratio > 0.8:  # Threshold for FK confidence
                            # Calculate confidence based on name match + integrity
                            name_conf = 1.0 if parent_pk.lower() == child_col.lower() else 0.9
                            confidence = (name_conf + match_ratio) / 2

                            self.relationships.append(
                                TableRelationship(
                                    parent_table=parent_table,
                                    child_table=child_table,
                                    parent_key=parent_pk,
                                    child_key=child_col,
                                    relationship_type="one-to-many",
                                    confidence=confidence,
                                    match_ratio=match_ratio,
                                )
                            )

        # Sort by confidence (descending)
        self.relationships.sort(key=lambda r: r.confidence, reverse=True)

        return self.relationships

    def classify_tables(self, anchor_table: str | None = None) -> dict[str, TableClassification]:
        """
        Classify all tables as dimension/fact/event/bridge/reference.

        Classification rules:
        - Dimension: cardinality_ratio <= 1.1, unique on grain, bytes < max_dimension_bytes
        - Fact: cardinality_ratio > 1.1, NOT unique on grain, NOT bridge
        - Event: has_time_column, cardinality_ratio > 1.1, time not constant, N-side of anchor
        - Bridge: 2+ foreign keys, neither unique, but composite near-unique
        - Reference: code mappings, small size with duplicates allowed

        Args:
            anchor_table: Optional anchor table name (for is_n_side_of_anchor detection)

        Returns:
            Dict mapping table names to TableClassification objects
        """
        logger.info("Classifying tables for aggregate-before-join pipeline")

        # Ensure relationships are detected first
        if not self.relationships:
            self.detect_relationships()

        # Count foreign keys per table
        fk_counts = self._count_foreign_keys()

        for table_name, df in self.tables.items():
            # Detect grain key
            grain_key = self._detect_grain_key(df)
            if not grain_key:
                logger.warning(f"No grain key detected for table '{table_name}', skipping classification")
                continue

            # Grain metrics are exact over the whole table: a sample cannot prove uniqueness,
            # and one repeated key is enough to count an entity twice.
            total_rows = df.height
            key_col = df[grain_key]
            null_count = key_col.null_count()
            null_rate = null_count / total_rows if total_rows else 1.0

            if null_count < total_rows:
                non_null_keys = key_col.drop_nulls()
                cardinality_ratio = non_null_keys.len() / non_null_keys.n_unique()
                # Exact: every row has its own key. Two rows with a missing key are not unique.
                is_unique = key_col.n_unique() == total_rows
            else:
                cardinality_ratio = float("inf")
                is_unique = False
            # Role heuristic: a table that is unique, or nearly so, is meant to be one row per key.
            # Its duplicates are then reported as a data-quality error rather than hidden.
            near_unique = cardinality_ratio <= NEAR_UNIQUE_RATIO

            # Estimate bytes
            estimated_bytes = self._estimate_table_bytes(df)

            # Detect time column
            time_col, has_time = self._detect_time_column(df)

            # Detect grain level
            grain_level = self._detect_grain_level(grain_key)

            # Detect bridge table
            is_bridge = self._detect_bridge_table(table_name, df, fk_counts.get(table_name, 0))

            # Determine if table is on N-side of anchor relationship
            is_n_side = self._is_n_side_of_anchor(table_name, anchor_table) if anchor_table else False

            # Classify based on rules
            classification = self._classify_table_type(
                cardinality_ratio=cardinality_ratio,
                is_unique=near_unique,
                estimated_bytes=estimated_bytes,
                has_time=has_time,
                is_bridge=is_bridge,
                is_n_side=is_n_side,
                time_col=time_col,
                df=df,
            )

            # Store classification
            self.classifications[table_name] = TableClassification(
                table_name=table_name,
                classification=classification,
                grain=grain_level,
                grain_key=grain_key,
                cardinality_ratio=cardinality_ratio,
                is_unique_on_grain=is_unique,
                estimated_bytes=estimated_bytes,
                relationship_degree=fk_counts.get(table_name, 0),
                has_time_column=has_time,
                time_column_name=time_col,
                is_n_side_of_anchor=is_n_side,
                null_rate_in_grain=null_rate,
            )

            logger.debug(
                f"Classified '{table_name}': {classification} "
                f"(grain={grain_level}, key={grain_key}, card_ratio={cardinality_ratio:.2f}, "
                f"bytes={estimated_bytes:,}, fks={fk_counts.get(table_name, 0)})"
            )

        return self.classifications

    def _detect_grain_key(self, df: pl.DataFrame) -> str | None:
        """
        Detect grain key column using explicit scoring formula.

        Prioritizes explicit keys (patient_id, subject_id, hadm_id) over
        row-level IDs (event_id, row_id, uuid). Uses sampled uniqueness
        to prevent expensive full table scans.

        Scoring formula (on sample):
        - -2.0 * uniq_ratio (penalize row-level IDs with uniq ~ 1.0)
        - -1.0 * null_rate
        - +1.0 if col ends with _id
        - +2.0 if col is explicit key (patient_id, subject_id, etc.)
        - -5.0 if col contains event/row/uuid/guid tokens

        Args:
            df: Polars DataFrame

        Returns:
            Grain key column name or None
        """
        # 1. Check explicit patient grain patterns first (highest priority)
        patient_patterns = ["patient_id", "subject_id", "patientid", "subjectid"]
        for col in df.columns:
            if col.lower() in patient_patterns:
                return str(col)

        # 2. Check explicit admission grain patterns
        admission_patterns = ["hadm_id", "encounter_id", "visit_id", "admissionid", "encounterid"]
        for col in df.columns:
            if col.lower() in admission_patterns:
                return str(col)

        # 3. Fallback: score ID columns using sampled data
        id_cols = [col for col in df.columns if self._is_probably_id_col(col)]

        if not id_cols:
            return None

        # Sample for scoring
        s = self._sample_df(df)

        # Score each ID column
        scores = {}
        for col in id_cols:
            # Compute sampled uniqueness
            unique_count, null_rate = self._compute_sampled_uniqueness(df, col)

            # Calculate uniqueness ratio on sample
            non_null_count = max(s.height - int(null_rate * s.height), 1)
            uniq_ratio = unique_count / non_null_count

            # Apply scoring formula
            score = 0.0
            score -= 2.0 * uniq_ratio  # Penalize row-level IDs (uniq ~ 1.0)
            score -= 1.0 * null_rate  # Penalize NULLs
            score += 1.0 if self._is_probably_id_col(col) else 0.0
            score += 2.0 if col.lower() in patient_patterns + admission_patterns else 0.0

            # Hard penalty for row-level ID tokens
            col_lower = col.lower()
            if any(tok in col_lower for tok in ["event", "row", "uuid", "guid"]):
                score -= 5.0

            scores[col] = score

            logger.debug(
                f"Grain key candidate '{col}': score={score:.2f} "
                f"(uniq_ratio={uniq_ratio:.2f}, null_rate={null_rate:.2f})"
            )

        # Return highest scoring column
        if scores:
            best_col = max(scores.keys(), key=lambda k: scores[k])
            logger.debug(f"Selected grain key: '{best_col}' (score={scores[best_col]:.2f})")
            return str(best_col)

        return None

    def _detect_grain_level(self, grain_key: str) -> Literal["patient", "admission", "event"]:
        """
        Detect grain level from grain key name.

        Args:
            grain_key: Name of grain key column

        Returns:
            Grain level (patient, admission, or event)
        """
        grain_key_lower = grain_key.lower()

        if any(p in grain_key_lower for p in ["patient", "subject"]):
            return "patient"
        elif any(p in grain_key_lower for p in ["hadm", "encounter", "visit", "admission"]):
            return "admission"
        else:
            return "event"

    def _estimate_table_bytes(self, df: pl.DataFrame, sample_size: int = 1000) -> int:
        """
        Estimate total table size in bytes.

        Strategy:
        1. Sample up to sample_size rows
        2. Calculate average bytes per row
        3. Multiply by total rows

        Args:
            df: Polars DataFrame
            sample_size: Number of rows to sample for estimation

        Returns:
            Estimated total bytes
        """
        if df.height == 0:
            return 0

        # Sample rows
        sample = df.head(min(sample_size, df.height))

        # Estimate bytes per row (rough approximation)
        bytes_per_row = 0
        for col in sample.columns:
            dtype = sample[col].dtype

            if dtype in [pl.Int8, pl.UInt8]:
                bytes_per_row += 1
            elif dtype in [pl.Int16, pl.UInt16]:
                bytes_per_row += 2
            elif dtype in [pl.Int32, pl.UInt32, pl.Float32]:
                bytes_per_row += 4
            elif dtype in [pl.Int64, pl.UInt64, pl.Float64]:
                bytes_per_row += 8
            elif dtype == pl.Utf8:
                # Estimate string column bytes
                avg_str_len = sample[col].drop_nulls().str.len_chars().mean()
                if avg_str_len is not None and isinstance(avg_str_len, int | float):
                    bytes_per_row += int(avg_str_len)
            elif dtype == pl.Boolean:
                bytes_per_row += 1
            else:
                # Default estimate for unknown types
                bytes_per_row += 8

        # Total estimate
        return int(bytes_per_row * int(df.height))

    def _detect_time_column(self, df: pl.DataFrame) -> tuple[str | None, bool]:
        """
        Detect time column in DataFrame using sampled uniqueness.

        A valid time column must:
        1. Have datetime/date type OR name contains time/date patterns
        2. Not be constant (all same value in sample)

        Note: Removed 'dt' pattern to avoid false positives (dt_code, mdt_flag).
        Uses sampled uniqueness to prevent expensive full table scans.

        Args:
            df: Polars DataFrame

        Returns:
            Tuple of (column_name, has_valid_time_column)
        """
        time_patterns = ["time", "date", "timestamp", "datetime"]

        # Prioritize dtype check first (more reliable)
        for col in df.columns:
            # Check dtype first
            is_time_type = df[col].dtype in [pl.Datetime, pl.Date]

            if is_time_type:
                # Verify not constant using sampled data
                unique_count, _ = self._compute_sampled_uniqueness(df, col)
                if unique_count > 1:
                    return (col, True)

        # Fallback: check name patterns
        for col in df.columns:
            col_lower = col.lower()
            has_time_pattern = any(p in col_lower for p in time_patterns)

            if has_time_pattern:
                # Verify not constant using sampled data
                unique_count, _ = self._compute_sampled_uniqueness(df, col)
                if unique_count > 1:
                    return (col, True)

        return (None, False)

    def _count_foreign_keys(self) -> dict[str, int]:
        """
        Count number of foreign key relationships per table.

        Returns:
            Dict mapping table name to number of FKs
        """
        fk_counts: dict[str, int] = {}

        for rel in self.relationships:
            # Child table has a foreign key
            fk_counts[rel.child_table] = fk_counts.get(rel.child_table, 0) + 1

        return fk_counts

    def _detect_bridge_table(self, table_name: str, df: pl.DataFrame, fk_count: int) -> bool:
        """
        Detect if table is a bridge (many-to-many) table using sampled data.

        Bridge table characteristics:
        1. Two or more foreign keys to different parent tables
        2. Neither FK is unique individually
        3. Composite key (fk1, fk2) is near-unique (>95% unique on sample)
        4. Narrow table (width < 15 columns)

        Uses sampled composite uniqueness to prevent expensive full table scans.

        Args:
            table_name: Name of table
            df: Polars DataFrame
            fk_count: Number of foreign keys

        Returns:
            True if bridge table detected
        """
        # Structural heuristic: bridges are narrow (< 15 columns)
        if df.width >= 15:
            return False

        # Must have 2+ foreign keys
        if fk_count < 2:
            return False

        # Get FK columns for this table
        fk_cols = [rel.child_key for rel in self.relationships if rel.child_table == table_name]

        if len(fk_cols) < 2:
            return False

        # Sample for uniqueness checks
        s = self._sample_df(df)

        # Check if individual FKs are non-unique (on sample)
        fk_unique_flags = []
        for col in fk_cols:
            if col in s.columns:
                unique_count, _ = self._compute_sampled_uniqueness(df, col)
                is_unique = unique_count == s.height
                fk_unique_flags.append(is_unique)

        if all(fk_unique_flags):
            # If all FKs are unique, not a bridge
            return False

        # Check if composite key is near-unique (on sample)
        try:
            # Use first two FK columns for composite check
            fk1, fk2 = fk_cols[0], fk_cols[1]
            if fk1 in s.columns and fk2 in s.columns:
                composite_unique = s.select([fk1, fk2]).n_unique()
                composite_ratio = composite_unique / s.height if s.height > 0 else 0.0

                # Near-unique threshold: 95%
                is_composite_unique = composite_ratio > 0.95

                return is_composite_unique

        except Exception as e:
            logger.debug(f"Error detecting bridge for {table_name}: {e}")
            return False

        return False

    def _is_n_side_of_anchor(self, table_name: str, anchor_table: str | None) -> bool:
        """
        Check if table is on N-side of relationship to anchor.

        Args:
            table_name: Table to check
            anchor_table: Anchor table name

        Returns:
            True if table is child (N-side) of anchor in relationship
        """
        if not anchor_table:
            return False

        for rel in self.relationships:
            if rel.parent_table == anchor_table and rel.child_table == table_name:
                return True

        return False

    def _classify_table_type(
        self,
        cardinality_ratio: float,
        is_unique: bool,
        estimated_bytes: int,
        has_time: bool,
        is_bridge: bool,
        is_n_side: bool,
        time_col: str | None,
        df: pl.DataFrame,
    ) -> Literal["dimension", "fact", "event", "bridge", "reference"]:
        """
        Classify table based on characteristics.

        Classification priority (first match wins):
        1. Bridge: detected as bridge table
        2. Reference: small size (<10 MB) with duplicates allowed
        3. Event: has time column, high cardinality, N-side of anchor
        4. Dimension: low cardinality, unique on grain, size < max_dimension_bytes
        5. Fact: high cardinality, not unique

        Args:
            cardinality_ratio: rows / unique(grain_key)
            is_unique: True if unique on grain
            estimated_bytes: Table size in bytes
            has_time: True if has time column
            is_bridge: True if bridge table
            is_n_side: True if N-side of anchor
            time_col: Time column name
            df: DataFrame

        Returns:
            Classification type
        """
        # Priority 1: Bridge
        if is_bridge:
            return "bridge"

        # Priority 2: Reference (small lookup tables)
        if estimated_bytes < 10_000_000 and not is_unique:  # < 10 MB
            return "reference"

        # Priority 3: Event (time-series facts)
        if has_time and cardinality_ratio > 1.1 and is_n_side:
            # Verify time column is not constant
            if time_col and df[time_col].n_unique() > 1:
                return "event"

        # Priority 4: Dimension
        if cardinality_ratio <= 1.1 and is_unique and estimated_bytes < self.max_dimension_bytes:
            return "dimension"

        # Priority 5: Fact (default for high cardinality non-events)
        return "fact"

    def _build_dimension_mart(self, anchor_table: str | None = None, join_type: str = "left") -> pl.LazyFrame:
        """
        Build dimension mart by joining only 1:1 and small dimensions to anchor.

        Critical invariants enforced:
        1. Mart rowcount equals anchor unique grain count
        2. No joins where RHS key is non-unique
        3. Bridges, facts, and events are EXCLUDED

        Strategy:
        1. Find anchor table using centrality-based selection
        2. Filter to only dimension tables (exclude fact/event/bridge)
        3. Verify RHS join keys are unique (enforce 1:1 or many:1)
        4. Execute joins in BFS order from anchor
        5. Return LazyFrame (not materialized)

        Args:
            anchor_table: Optional anchor table name (auto-detected if None)
            join_type: Type of join (default "left")

        Returns:
            LazyFrame with dimension mart (anchor + joined dimensions)

        Raises:
            ValueError: If anchor has non-unique grain key or no dimensions found
        """
        logger.info("Building dimension mart with aggregate-before-join architecture")

        # Ensure classifications exist
        if not self.classifications:
            self.classify_tables()

        # Ensure relationships exist
        if not self.relationships:
            self.detect_relationships()

        # Auto-detect anchor if not specified
        if anchor_table is None:
            anchor_table = self._find_anchor_by_centrality()
            logger.info(f"Auto-detected anchor table: {anchor_table}")

        if anchor_table not in self.tables:
            raise ValueError(f"Anchor table '{anchor_table}' not found in tables")

        anchor_class = self.classifications.get(anchor_table)
        if not anchor_class:
            raise ValueError(f"Anchor table '{anchor_table}' not classified")

        # Verify anchor is unique on grain (exact, whole table)
        anchor_df = self.tables[anchor_table]
        anchor_key = anchor_df[anchor_class.grain_key]
        n_duplicates = anchor_df.height - anchor_key.n_unique()
        if n_duplicates > 0:
            context = "the cohort's anchor table"
            if anchor_key.null_count() > 1:
                context += f"; {describe_count(anchor_key.null_count())} rows have no key and count as repeats"
            raise duplicate_key_error(anchor_table, anchor_class.grain_key, n_duplicates, context)

        # Start with anchor table as LazyFrame
        mart = anchor_df.lazy()
        joined_tables = {anchor_table}

        logger.debug(f"Anchor '{anchor_table}': {anchor_df.height} rows, grain_key={anchor_class.grain_key}")

        # Build join graph using BFS from anchor
        queue = deque([anchor_table])
        join_count = 0

        while queue:
            current = queue.popleft()

            # Find relationships where current table is involved
            # Sort for deterministic traversal order
            for rel in sorted(
                self.relationships,
                key=lambda r: (r.parent_table, r.child_table, r.parent_key, r.child_key),
            ):
                next_table = None
                join_key_left = None
                join_key_right = None

                # Check if we can join a new table to current table
                if rel.parent_table == current and rel.child_table not in joined_tables:
                    next_table = rel.child_table
                    join_key_left = rel.parent_key
                    join_key_right = rel.child_key

                elif rel.child_table == current and rel.parent_table not in joined_tables:
                    next_table = rel.parent_table
                    join_key_left = rel.child_key
                    join_key_right = rel.parent_key

                if next_table:
                    # Check if next_table is a dimension (exclude facts, events, bridges)
                    next_class = self.classifications.get(next_table)

                    if not next_class:
                        logger.debug(f"Skipping '{next_table}': not classified")
                        continue

                    if next_class.classification != "dimension":
                        logger.debug(
                            f"Skipping '{next_table}': classification={next_class.classification} "
                            f"(only dimensions allowed in mart)"
                        )
                        continue

                    # CRITICAL: Verify RHS join key is unique (enforce 1:1 or many:1), exactly
                    next_df = self.tables[next_table]
                    if join_key_right is None:
                        continue
                    rhs_key = next_df[join_key_right].drop_nulls()
                    rhs_duplicates = rhs_key.len() - rhs_key.n_unique()
                    if rhs_duplicates > 0:
                        reason = (
                            f"key '{join_key_right}' repeats on {describe_count(rhs_duplicates)} rows, "
                            f"so joining it would duplicate cohort rows"
                        )
                        self._exclusion_reasons[next_table] = reason
                        logger.warning(f"Skipping join to '{next_table}': {reason}")
                        continue

                    # Join next dimension table
                    next_lazy = next_df.lazy()

                    # Type-safe join type
                    join_type_safe: str = join_type if isinstance(join_type, str) else "left"
                    mart = mart.join(
                        next_lazy,
                        left_on=join_key_left,
                        right_on=join_key_right,
                        how=join_type_safe,  # type: ignore[arg-type]
                        suffix=f"_{next_table}",
                        validate="m:1",  # fail fast if RHS key is not unique
                    )

                    joined_tables.add(next_table)
                    queue.append(next_table)
                    join_count += 1

                    logger.debug(f"Joined dimension '{next_table}' on {join_key_left}={join_key_right}")

        logger.info(
            f"Dimension mart built: anchor='{anchor_table}', "
            f"joined_dimensions={join_count}, total_tables={len(joined_tables)}"
        )
        self._joined_dimensions = joined_tables

        return mart

    def _aggregate_fact_tables(
        self, grain_key: str, policy: AggregationPolicy | None = None
    ) -> dict[str, pl.LazyFrame]:
        """
        Aggregate fact/event tables by grain key with policy enforcement.

        Returns feature tables with safe aggregations applied. Enforces aggregation
        policy to prevent incorrect operations (e.g., mean on code columns).

        Strategy:
        1. Filter to only fact/event tables (exclude dimensions/bridges/reference)
        2. For each fact/event table:
           - Verify grain_key exists in table
           - Check columns against code patterns
           - Build aggregation expressions per policy
           - Group by grain_key and aggregate
        3. Return Dict[table_name + "_features", LazyFrame]

        Args:
            grain_key: Column to group by (e.g., "patient_id")
            policy: Aggregation policy with safety rules (default: safe policy)

        Returns:
            Dict mapping "{table_name}_features" to aggregated LazyFrames

        Raises:
            AggregationPolicyError: If policy violation detected (e.g., mean on code)

        Fact tables without ``grain_key`` are skipped and recorded as excluded (see
        ``build_unified_cohort_with_report``).
        """
        # Use default policy if not specified
        if policy is None:
            policy = AggregationPolicy()

        log = logger.bind(grain_key=grain_key)
        log.info("Aggregating fact/event tables")

        # Ensure classifications exist
        if not self.classifications:
            self.classify_tables()

        # Filter to fact/event tables only
        fact_event_tables = {
            name: cls for name, cls in self.classifications.items() if cls.classification in ["fact", "event"]
        }

        if not fact_event_tables:
            log.warning("No fact/event tables found for aggregation")
            return {}

        feature_tables = {}

        for table_name, cls in fact_event_tables.items():
            df = self.tables[table_name]

            # A fact table without the anchor key cannot be rolled up to the cohort grain
            if grain_key not in df.columns:
                reason = (
                    f"it has no '{grain_key}' column, so its rows cannot be linked to the cohort "
                    f"(its own key is '{cls.grain_key}')"
                )
                self._exclusion_reasons[table_name] = reason
                log.warning("Fact table left out of cohort", table_name=table_name, reason=reason)
                continue

            feature_table_name = f"{table_name}_features"
            feature_tables[feature_table_name] = aggregate_fact_table(
                df,
                grain_key,
                time_column=cls.time_column_name,
                policy=policy,
                table_name=table_name,
            )
            log.debug("Aggregated fact table", table_name=table_name, feature_table=feature_table_name)

        log.info(f"Created {len(feature_tables)} feature tables from fact/event tables")

        return feature_tables

    def build_unified_cohort(
        self,
        anchor_table: str | None = None,
        join_type: str = "left",
    ) -> pl.DataFrame:
        """
        Build the unified cohort and return only the data.

        Same as ``build_unified_cohort_with_report(...).data``. The full result, including the
        tables that were left out and why, is kept on ``self.last_build_result``.
        """
        return self.build_unified_cohort_with_report(anchor_table=anchor_table, join_type=join_type).data

    def build_unified_cohort_with_report(
        self,
        anchor_table: str | None = None,
        join_type: str = "left",
    ) -> CohortBuildResult:
        """
        Build unified cohort using aggregate-before-join architecture.

        Strategy:
          1) classify + detect relationships
          2) choose anchor (centrality)
          3) build dimension mart (1:1 joins only; anchor must be exactly unique on its key)
          4) aggregate fact/event tables to grain (one row per grain)
          5) join aggregated feature tables to mart (validate 1:1)
          6) collect, then assert one row per anchor key

        Args:
            anchor_table: Root table for joins (auto-detected if None)
            join_type: Type of join (left, inner, outer) - only affects dimension mart

        Returns:
            CohortBuildResult with the cohort and every source table left out, with a reason

        Raises:
            DataQualityError: Anchor key has duplicates, or the result is not one row per key
            ValueError: If anchor not found or invalid join_type
            AggregationPolicyError: If aggregation policy violation detected
        """
        log = logger.bind(anchor_table=anchor_table, join_type=join_type)
        log.info("Building unified cohort")

        # Normalize and validate join_type (fail-fast on invalid input)
        join_type = join_type.lower().strip()
        if join_type not in {"left", "inner", "outer"}:
            raise ValueError(f"Unsupported join_type: {join_type}")

        if not self.classifications:
            self.classify_tables()
        if not self.relationships:
            self.detect_relationships()
        self._exclusion_reasons = {}
        self._joined_dimensions = set()

        if anchor_table is None:
            anchor_table = self._find_anchor_by_centrality()
            log = log.bind(anchor_table=anchor_table)
            log.info("Auto-detected anchor table")

        if anchor_table not in self.tables:
            raise ValueError(f"Anchor table '{anchor_table}' not found in tables")

        anchor_class = self.classifications.get(anchor_table)
        if anchor_class is None:
            raise ValueError(f"Anchor table '{anchor_table}' not classified")

        grain_key = anchor_class.grain_key
        if not grain_key:
            raise ValueError(f"Anchor table '{anchor_table}' has no detected grain_key")
        log = log.bind(grain_key=grain_key)

        # 1) Dimensions only (safe joins); raises DataQualityError on duplicate anchor keys
        mart = self._build_dimension_mart(anchor_table=anchor_table, join_type=join_type)

        # 2) Aggregate facts/events to grain
        feature_tables = self._aggregate_fact_tables(grain_key=grain_key)

        # 3) Join features (must be 1 row per grain on both sides)
        for feature_name in sorted(feature_tables.keys()):
            feature_lazy = feature_tables[feature_name]

            # Enforce deterministic RHS column order (grain_key first, then sorted)
            names = feature_lazy.collect_schema().names()
            rhs_cols = [grain_key] + sorted([c for c in names if c != grain_key])
            feature_lazy = feature_lazy.select(rhs_cols)

            # Extract base table name from feature_name (format: "{table_name}_features")
            base_table_name = feature_name.removesuffix("_features")

            mart = mart.join(
                feature_lazy,
                on=grain_key,
                how="left",
                validate="1:1",  # mart unique on grain, aggregated features unique on grain
                suffix=f"_{base_table_name}",  # Unique suffix per feature table
            )

        result = mart.collect()
        self._assert_cohort_grain(result, anchor_table, grain_key, join_type)

        joined = self._joined_dimensions | {name.removesuffix("_features") for name in feature_tables}
        excluded = self._describe_excluded_tables(joined)
        for table in excluded:
            log.warning(
                "Table left out of unified cohort",
                table_name=table.table_name,
                classification=table.classification,
                reason=table.reason,
            )

        log.info(
            "Unified cohort built",
            rows=result.height,
            cols=result.width,
            features=len(feature_tables),
            excluded=len(excluded),
        )
        self.last_build_result = CohortBuildResult(
            data=result, anchor_table=anchor_table, grain_key=grain_key, excluded_tables=excluded
        )
        return self.last_build_result

    def _assert_cohort_grain(self, result: pl.DataFrame, anchor_table: str, grain_key: str, join_type: str) -> None:
        """Fail loudly unless the cohort is exactly one row per anchor key."""
        anchor_keys = self.tables[anchor_table][grain_key].n_unique()
        result_duplicates = result.height - result[grain_key].n_unique()
        if result_duplicates > 0:
            raise duplicate_key_error(anchor_table, grain_key, result_duplicates, "in the built cohort after joins")
        # Left/outer joins keep every anchor row; an inner join may legitimately drop some
        if join_type != "inner" and result.height != anchor_keys:
            difference = abs(result.height - anchor_keys)
            raise DataQualityError(
                f"The cohort built from '{anchor_table}' has {describe_count(difference)} "
                f"{'more' if result.height > anchor_keys else 'fewer'} rows than distinct '{grain_key}' "
                f"values in the anchor table. Expected exactly one row per '{grain_key}'."
            )

    def _describe_excluded_tables(self, joined: set[str]) -> list[ExcludedTable]:
        """List every source table not in the cohort, with the reason it was left out."""
        default_reasons: dict[str, str] = {
            "bridge": "it links two tables many-to-many, which cannot be joined at the cohort grain",
            "reference": (
                "its key repeats, so it was treated as a lookup table and not joined automatically "
                "(small tables with repeated keys are not rolled up)"
            ),
            "dimension": "no detected relationship connects it to the anchor table",
        }
        excluded: list[ExcludedTable] = []
        for table_name in sorted(self.tables):
            if table_name in joined:
                continue
            cls = self.classifications.get(table_name)
            classification = cls.classification if cls else "unclassified"
            reason = self._exclusion_reasons.get(table_name) or default_reasons.get(
                classification, "no key column was detected, so it cannot be linked to the cohort"
            )
            excluded.append(ExcludedTable(table_name=table_name, classification=classification, reason=reason))
        return excluded

    def _get_ibis_connection(self) -> Any:
        """
        Get or create Ibis DuckDB connection (cached on instance).

        Returns:
            Ibis DuckDB connection
        """
        if ibis is None:
            raise ImportError("Ibis is required for plan_mart(). Install with: pip install ibis-framework[duckdb]")

        # Reuse cached connection if available
        if self._ibis_connection is None:
            self._ibis_connection = ibis.duckdb.connect()

        return self._ibis_connection

    def _add_hash_bucket_column(
        self,
        df: pl.DataFrame,
        grain_key: str,
        num_buckets: int,
    ) -> pl.DataFrame:
        """
        Add deterministic hash bucket column for partitioning.

        Uses stable hash seed for reproducibility across runs.

        Args:
            df: Polars DataFrame
            grain_key: Column to hash
            num_buckets: Number of buckets

        Returns:
            DataFrame with 'bucket' column added
        """
        bucket_col = (
            pl.col(grain_key)
            .hash(seed=0)  # Stable seed for reproducibility
            .mod(num_buckets)
            .cast(pl.UInt16)
            .alias("bucket")
        )
        return df.with_columns(bucket_col)

    def _compute_dataset_fingerprint(self) -> str:
        """
        Compute deterministic fingerprint of dataset for caching.

        Fingerprint includes: table names, row counts, column counts, and content hash.
        Content hash uses a small deterministic sample per table to detect value changes
        even when shape remains the same.

        Returns:
            SHA256 hash as hex string
        """
        # Build fingerprint from table metadata + content sample
        fingerprint_parts = []
        for table_name in sorted(self.tables.keys()):
            df = self.tables[table_name]

            # Shape metadata
            shape_info = f"{table_name}:{df.height}:{df.width}:{','.join(sorted(df.columns))}"

            # Content hash: sample first 100 rows (or all if smaller) deterministically
            # This detects value changes even when shape is unchanged
            sample_size = min(100, df.height)
            if sample_size > 0:
                # Sample deterministically (first N rows, sorted by all columns for stability)
                sample_df = df.head(sample_size)
                # Hash the sample data (convert to string representation)
                sample_str = str(sample_df.to_dict(as_series=False))
                content_hash = hashlib.sha256(sample_str.encode()).hexdigest()[:16]  # First 16 chars
                fingerprint_parts.append(f"{shape_info}:{content_hash}")
            else:
                fingerprint_parts.append(f"{shape_info}:empty")

        fingerprint_str = "|".join(fingerprint_parts)
        return hashlib.sha256(fingerprint_str.encode()).hexdigest()

    def _compute_run_id(
        self,
        grain: str,
        anchor_table: str,
        grain_key: str,
        join_type: str,
    ) -> str:
        """
        Compute deterministic run_id for caching.

        Hash of: config (grain, anchor, grain_key, join_type) + schema_version + dataset fingerprint.

        Args:
            grain: Grain level
            anchor_table: Anchor table name
            grain_key: Grain key column
            join_type: Join type

        Returns:
            SHA256 hash as hex string
        """
        # Build config string
        config_str = f"{grain}:{anchor_table}:{grain_key}:{join_type}"

        # Add schema version (module-level constant)
        # SCHEMA_VERSION is defined at module level in this file

        # Add dataset fingerprint (hash of table names + row counts + content sample)
        dataset_fingerprint = self._compute_dataset_fingerprint()

        # Combine and hash
        combined = f"{config_str}:{SCHEMA_VERSION}:{dataset_fingerprint}"
        return hashlib.sha256(combined.encode()).hexdigest()

    def materialize_mart(
        self,
        output_path: Path,
        grain: str = "patient",
        anchor_table: str | None = None,
        join_type: str = "left",
        num_buckets: int = 64,
        force_recompute: bool = False,
    ) -> CohortMetadata:
        """
        Compute mart using Polars pipeline and write to partitioned Parquet.

        Strategy:
          1) Compute run_id (deterministic hash of config + schema + dataset)
          2) Check cache: if {output_path}/{run_id}/ exists and not force_recompute, return existing metadata
          3) Use existing Polars pipeline (_build_dimension_mart, _aggregate_fact_tables)
          4) Collect Polars LazyFrames to DataFrames (bounded: facts pre-aggregated to grain)
          5) Write to Parquet:
             - Patient/admission level: single file at {output_path}/{run_id}/{grain}/mart.parquet
             - Event level: hash-bucketed directory at {output_path}/{run_id}/{grain}/mart/ (bucket=0, bucket=1, ...)
          6) Return metadata with paths, schema, row counts, run_id

        Memory Constraint:
          Materialization is bounded because facts/events are pre-aggregated to grain
          (one row per grain). Final mart should be manageable. If _aggregate_fact_tables()
          produces a wide mart that's still large, this should be documented as a limitation.

        Args:
            output_path: Base directory for Parquet output
            grain: Grain level (patient, admission, event)
            anchor_table: Optional anchor table (auto-detected if None)
            join_type: Join type for dimension mart
            num_buckets: Number of hash buckets for event-level partitioning
            force_recompute: If True, recompute even if cached version exists

        Returns:
            CohortMetadata with paths, schema, row counts, and run_id
        """
        import time

        start_time = time.perf_counter()
        logger.info(f"Materializing mart (grain={grain}, anchor_table={anchor_table}, join_type={join_type})")

        # Normalize and validate join_type
        join_type = join_type.lower().strip()
        if join_type not in {"left", "inner", "outer"}:
            raise ValueError(f"Unsupported join_type: {join_type}")

        # Ensure classifications and relationships exist
        if not self.classifications:
            self.classify_tables()
        if not self.relationships:
            self.detect_relationships()

        # Find anchor table
        if anchor_table is None:
            anchor_table = self._find_anchor_by_centrality()
            logger.info(f"Auto-detected anchor table: {anchor_table}")

        if anchor_table not in self.tables:
            raise ValueError(f"Anchor table '{anchor_table}' not found in tables")

        anchor_class = self.classifications.get(anchor_table)
        if anchor_class is None:
            raise ValueError(f"Anchor table '{anchor_table}' not classified")

        grain_key = anchor_class.grain_key
        if not grain_key:
            raise ValueError(f"Anchor table '{anchor_table}' has no detected grain_key")

        # Compute run_id
        run_id = self._compute_run_id(grain, anchor_table, grain_key, join_type)
        logger.info(f"Computed run_id: {run_id[:16]}...")

        # Check cache
        run_dir = output_path / run_id
        metadata_file = run_dir / f"{grain}_metadata.json"

        if run_dir.exists() and metadata_file.exists() and not force_recompute:
            logger.info(f"Cache hit: reusing existing materialized mart at {run_dir}")
            import json

            with open(metadata_file) as f:
                metadata_dict = json.load(f)

            # Convert string paths back to Path objects
            metadata_dict["output_path"] = Path(metadata_dict["output_path"])
            metadata_dict["parquet_path"] = Path(metadata_dict["parquet_path"])
            metadata_dict["materialized_at"] = datetime.fromisoformat(metadata_dict["materialized_at"])

            metadata = CohortMetadata(**metadata_dict)
            logger.info(f"Reusing cached mart: {metadata.row_count:,} rows, path={metadata.parquet_path}")
            return metadata

        # Build mart using Polars pipeline
        logger.info("Building dimension mart...")
        mart = self._build_dimension_mart(anchor_table=anchor_table, join_type=join_type)

        logger.info("Aggregating fact/event tables...")
        feature_tables = self._aggregate_fact_tables(grain_key=grain_key)

        # Join feature tables to mart
        if feature_tables:
            logger.info(f"Joining {len(feature_tables)} feature tables to mart...")
            for feature_name in sorted(feature_tables.keys()):
                feature_lazy = feature_tables[feature_name]

                schema = feature_lazy.collect_schema()
                names = schema.names()
                if grain_key not in names:
                    logger.warning(f"Skipping '{feature_name}': grain_key '{grain_key}' not in schema (cols={names})")
                    continue

                # Enforce deterministic RHS column order
                rhs_cols = [grain_key] + sorted([c for c in names if c != grain_key])
                feature_lazy = feature_lazy.select(rhs_cols)

                mart = mart.join(
                    feature_lazy,
                    on=grain_key,
                    how="left",
                    validate="1:1",
                )

        # Collect to DataFrame
        logger.info("Collecting mart...")
        mart_df = mart.collect()
        row_count = mart_df.height

        # Determine parquet path
        parquet_base = run_dir / grain
        parquet_base.mkdir(parents=True, exist_ok=True)

        # Write Parquet based on grain level
        if grain == "event":
            # Event level: hash bucket partitioning
            logger.info(f"Writing event-level mart with hash bucket partitioning ({num_buckets} buckets)...")
            mart_df = self._add_hash_bucket_column(mart_df, grain_key, num_buckets)
            parquet_path = parquet_base / "mart"
            mart_df.write_parquet(
                str(parquet_path),
                partition_by=["bucket"],
            )
        else:
            # Patient/admission level: single file
            logger.info(f"Writing {grain}-level mart as single Parquet file...")
            parquet_path = parquet_base / "mart.parquet"
            mart_df.write_parquet(str(parquet_path))

        # Build schema dict (column_name -> dtype string)
        schema_dict = {col: str(dtype) for col, dtype in zip(mart_df.columns, mart_df.dtypes)}

        # Create metadata
        grain_safe: str = str(grain) if grain is not None else "patient"
        metadata = CohortMetadata(
            grain=grain_safe,  # type: ignore[arg-type]
            grain_key=grain_key,
            anchor_table=anchor_table,
            output_path=output_path,
            parquet_path=parquet_path,
            schema=schema_dict,
            row_count=row_count,
            materialized_at=datetime.now(),
            run_id=run_id,
        )

        # Save metadata to file for caching
        import json

        metadata_file.parent.mkdir(parents=True, exist_ok=True)
        with open(metadata_file, "w") as f:
            json.dump(
                {
                    "grain": metadata.grain,
                    "grain_key": metadata.grain_key,
                    "anchor_table": metadata.anchor_table,
                    "output_path": str(metadata.output_path),
                    "parquet_path": str(metadata.parquet_path),
                    "schema": metadata.schema,
                    "row_count": metadata.row_count,
                    "materialized_at": metadata.materialized_at.isoformat(),
                    "run_id": metadata.run_id,
                },
                f,
                indent=2,
            )

        duration = time.perf_counter() - start_time
        parquet_size = (
            sum(f.stat().st_size for f in parquet_path.rglob("*.parquet") if f.is_file())
            if parquet_path.is_dir()
            else parquet_path.stat().st_size
        )

        logger.info(
            f"Mart materialized: {row_count:,} rows, {len(schema_dict)} cols, "
            f"{parquet_size / 1024 / 1024:.2f} MB, duration={duration:.2f}s, "
            f"path={parquet_path}"
        )

        return metadata

    def plan_mart(
        self,
        metadata: CohortMetadata | None = None,
        parquet_path: Path | None = None,
    ) -> Any:
        """
        Return lazy Ibis expression over materialized Parquet mart.

        Strategy:
          1) Resolve parquet_path from metadata or parameter
          2) Create/get DuckDB Ibis connection
          3) Use con.read_parquet() (DuckDB handles partitioned directories natively)
          4) Return lazy Ibis.Table (no computation, SQL generation only)
          5) Enables semantic layer queries to compile to SQL lazily

        Args:
            metadata: CohortMetadata (preferred, includes schema info)
            parquet_path: Path to materialized Parquet (file or directory)

        Returns:
            Lazy Ibis expression over Parquet (SQL not executed)

        Raises:
            ValueError: If neither metadata nor parquet_path provided
            FileNotFoundError: If Parquet path doesn't exist
        """
        if ibis is None:
            raise ImportError("Ibis is required for plan_mart(). Install with: pip install ibis-framework[duckdb]")

        # Resolve parquet_path
        if metadata is not None:
            parquet_path = metadata.parquet_path
        elif parquet_path is None:
            raise ValueError("Either metadata or parquet_path must be provided")

        if not parquet_path.exists():
            raise FileNotFoundError(f"Parquet path does not exist: {parquet_path}")

        logger.info(f"Planning mart query over Parquet: {parquet_path}")

        # Create DuckDB Ibis connection
        con = self._get_ibis_connection()

        # Read Parquet (DuckDB handles partitioned directories natively)
        if parquet_path.is_dir():
            # Partitioned directory: DuckDB read_parquet handles this natively
            # Use glob pattern to read all parquet files in directory tree
            # DuckDB read_parquet accepts directory paths and reads recursively
            table = con.read_parquet(str(parquet_path))
        else:
            # Single file
            table = con.read_parquet(str(parquet_path))

        # Drop bucket column if present (internal partition column, not for consumers)
        # Event-level marts use hash bucket partitioning, which adds a 'bucket' column
        # Check metadata schema if available, otherwise check if it's a partitioned directory
        should_drop_bucket = False
        if metadata is not None:
            # Check metadata schema for bucket column
            if "bucket" in metadata.schema:
                should_drop_bucket = True
        elif parquet_path.is_dir():
            # Partitioned directory likely has bucket column (event-level partitioning)
            should_drop_bucket = True

        if should_drop_bucket:
            logger.debug("Dropping internal 'bucket' partition column from planned table")
            table = table.drop("bucket")

        logger.debug("Created lazy Ibis table over Parquet (no computation triggered)")

        return table

    def _detect_primary_key(self, df: pl.DataFrame) -> str | None:
        """
        Detect primary key column in a table using Polars.

        Strategy:
        1. Column must be 100% unique (no duplicates)
        2. Column must have no null values
        3. Prefer columns with "id" in name

        Args:
            df: Polars DataFrame

        Returns:
            Primary key column name or None
        """
        id_pattern_cols = [col for col in df.columns if "id" in col.lower()]

        # Check ID pattern columns first
        for col in id_pattern_cols:
            if df[col].n_unique() == df.height and df[col].null_count() == 0:
                return str(col)

        # Fallback: check all columns
        for col in df.columns:
            if df[col].n_unique() == df.height and df[col].null_count() == 0:
                return str(col)

        return None

    def _is_foreign_key_candidate(self, parent_key: str, child_col: str) -> bool:
        """
        Check if column name suggests foreign key relationship.

        Strategy:
        - Exact match (case-insensitive)
        - Common FK naming patterns (parent_key_id, fk_parent_key, etc.)

        Args:
            parent_key: Parent table's primary key name
            child_col: Candidate foreign key column name

        Returns:
            True if naming suggests FK relationship
        """
        parent_lower = parent_key.lower()
        child_lower = child_col.lower()

        # Exact match
        if parent_lower == child_lower:
            return True

        # Common FK patterns
        if child_lower in [
            parent_lower,
            f"{parent_lower}_id",
            f"fk_{parent_lower}",
            f"{parent_lower}_fk",
        ]:
            return True

        # Handle cases like "patient_id" matching "patientid"
        parent_no_underscore = parent_lower.replace("_", "")
        child_no_underscore = child_lower.replace("_", "")

        if parent_no_underscore == child_no_underscore:
            return True

        return False

    def _verify_referential_integrity(
        self, parent_df: pl.DataFrame, parent_key: str, child_df: pl.DataFrame, child_col: str
    ) -> float:
        """
        Verify referential integrity using Polars operations.

        Calculates what percentage of child FK values exist in parent PK.

        Args:
            parent_df: Parent table DataFrame
            parent_key: Parent primary key column
            child_df: Child table DataFrame
            child_col: Child foreign key column

        Returns:
            Match ratio (0-1) representing referential integrity
        """
        logger.debug(f"Verifying referential integrity: {parent_key} -> {child_col}")

        # Get non-null child values
        child_values = child_df[child_col].drop_nulls()
        logger.debug(
            f"Child values dtype: {child_values.dtype}, len: {child_values.len()}, "
            f"sample: {child_values.head(3).to_list()}"
        )

        if child_values.len() == 0:
            logger.debug("Child values empty, returning 0.0")
            return 0.0

        # Get parent values
        parent_values = parent_df[parent_key].drop_nulls()
        logger.debug(
            f"Parent values dtype: {parent_values.dtype}, len: {parent_values.len()}, "
            f"sample: {parent_values.head(3).to_list()}"
        )

        # Cast both to string for type-safe comparison using pure Polars operations
        # This handles cases where one table has int64 and another has string
        try:
            # Cast both to Utf8 for consistent comparison
            logger.debug(f"Casting child from {child_values.dtype} to Utf8")
            child_str = child_values.cast(pl.Utf8, strict=False).drop_nulls().unique()
            logger.debug(
                f"Child after cast: dtype={child_str.dtype}, len={child_str.len()}, "
                f"sample: {child_str.head(3).to_list()}"
            )

            logger.debug(f"Casting parent from {parent_values.dtype} to Utf8")
            parent_str = parent_values.cast(pl.Utf8, strict=False).drop_nulls().unique()
            logger.debug(
                f"Parent after cast: dtype={parent_str.dtype}, len={parent_str.len()}, "
                f"sample: {parent_str.head(3).to_list()}"
            )

            # Verify both are Utf8 before join
            if child_str.dtype != pl.Utf8:
                logger.warning(f"Child cast failed: expected Utf8, got {child_str.dtype}, recasting...")
                child_str = child_str.cast(pl.Utf8, strict=False)
            if parent_str.dtype != pl.Utf8:
                logger.warning(f"Parent cast failed: expected Utf8, got {parent_str.dtype}, recasting...")
                parent_str = parent_str.cast(pl.Utf8, strict=False)

            # Use join-based approach to avoid is_in() deprecation warning
            # Create DataFrames for join operation
            logger.debug("Creating DataFrames for join operation")
            child_df_join = pl.DataFrame({"value": child_str})
            parent_df_join = pl.DataFrame({"value": parent_str})

            logger.debug(f"Child DF schema: {child_df_join.schema}, Parent DF schema: {parent_df_join.schema}")

            # Inner join to find matches
            logger.debug("Performing inner join to find matches")
            matches_df = child_df_join.join(parent_df_join, on="value", how="inner")
            matches = matches_df.height
            logger.debug(f"Found {matches} matches out of {child_str.len()} child values")

            ratio = float(matches) / float(child_str.len()) if child_str.len() > 0 else 0.0
            logger.debug(f"Match ratio: {ratio:.4f}")
            return ratio

        except Exception as e:
            # If casting fails, skip this relationship
            logger.error(
                f"Error in referential integrity check ({parent_key} -> {child_col}): {type(e).__name__}: {str(e)}"
            )
            logger.error(f"Child dtype: {child_values.dtype}, Parent dtype: {parent_values.dtype}")
            import traceback

            logger.error(f"Traceback: {traceback.format_exc()}")
            return 0.0

    def _find_anchor_table(self) -> str:
        """
        Find anchor table (most central in join graph).

        DEPRECATED: Use _find_anchor_by_centrality() instead.

        Strategy:
        - Table with most relationships (parent or child)
        - Prefer tables with "patient" or "subject" in name

        Returns:
            Anchor table name
        """
        logger.warning("_find_anchor_table() is deprecated, use _find_anchor_by_centrality() instead")

        # Count relationships for each table
        table_counts: dict[str, int] = {}

        for rel in self.relationships:
            table_counts[rel.parent_table] = table_counts.get(rel.parent_table, 0) + 1
            table_counts[rel.child_table] = table_counts.get(rel.child_table, 0) + 1

        # Prefer tables with "patient" or "subject" in name
        patient_tables = [t for t in self.tables.keys() if "patient" in t.lower() or "subject" in t.lower()]

        if patient_tables:
            # Pick patient table with most relationships
            return max(patient_tables, key=lambda t: table_counts.get(t, 0))

        # Fallback: table with most relationships
        if table_counts:
            return max(table_counts.keys(), key=lambda k: table_counts[k])

        # Last resort: first table
        return list(self.tables.keys())[0]

    def _find_anchor_by_centrality(self) -> str:
        """
        Find anchor table using graph centrality with hard exclusions and tie-breakers.

        Hard Exclusions (never anchor on):
        - Classification in {event, fact, bridge}
        - Tables without unique grain key
        - Tables with >50% NULLs in grain key

        Scoring Rules:
        - +10 if has hadm_id or encounter_id column
        - +5 if has patient_id or subject_id column
        - +1 per relationship (incoming + outgoing)
        - +3 if classified as dimension with patient grain

        Tie-breakers (deterministic, in order):
        1. Prefer fewer NULLs in grain key (lower null_rate)
        2. Prefer unique grain key (is_unique_on_grain = True)
        3. Prefer smaller estimated_bytes
        4. Prefer patient grain over admission grain

        Returns:
            Anchor table name

        Raises:
            ValueError: If no suitable anchor table found
        """
        # Ensure classifications exist
        if not self.classifications:
            self.classify_tables()

        # Hard exclusion: filter to only dimensions
        dimension_tables = {
            name: cls for name, cls in self.classifications.items() if cls.classification == "dimension"
        }

        if not dimension_tables:
            raise ValueError(
                "No dimension tables found for anchor selection. "
                "All tables are classified as fact/event/bridge/reference."
            )

        # Hard exclusion: filter out tables with >50% NULLs or non-unique grain
        candidates = {}
        for name, cls in dimension_tables.items():
            if cls.null_rate_in_grain > 0.5:
                logger.debug(f"Excluding {name} from anchor candidates: null_rate={cls.null_rate_in_grain:.2%}")
                continue

            # A near-unique table with duplicate keys stays a candidate: if it wins, building the
            # mart fails loudly. Dropping it here would silently anchor on a different table.
            candidates[name] = cls

        if not candidates:
            raise ValueError("No suitable anchor table found. All dimensions have >50% NULLs in their grain key.")

        # Count relationships per table
        relationship_counts: dict[str, int] = {}
        for rel in self.relationships:
            relationship_counts[rel.parent_table] = relationship_counts.get(rel.parent_table, 0) + 1
            relationship_counts[rel.child_table] = relationship_counts.get(rel.child_table, 0) + 1

        # Score each candidate
        scores = {}
        for name, cls in candidates.items():
            score = 0

            # Check for key columns
            df = self.tables[name]
            col_lower = [c.lower() for c in df.columns]

            if any(p in col_lower for p in ["hadm_id", "encounter_id"]):
                score += 10

            if any(p in col_lower for p in ["patient_id", "subject_id"]):
                score += 5

            # Relationship count
            score += relationship_counts.get(name, 0)

            # Bonus for dimension with patient grain
            if cls.grain == "patient":
                score += 3

            scores[name] = score

        # Find max score
        max_score = max(scores.values())
        top_candidates = [name for name, score in scores.items() if score == max_score]

        # If single winner, return it
        if len(top_candidates) == 1:
            winner = top_candidates[0]
            logger.info(
                f"Selected anchor table '{winner}' (score={max_score}, "
                f"grain={candidates[winner].grain}, "
                f"null_rate={candidates[winner].null_rate_in_grain:.2%})"
            )
            return winner

        # Apply tie-breakers
        logger.debug(f"Tie detected among {len(top_candidates)} tables, applying tie-breakers")

        # Tie-breaker 1: Lower null rate
        min_null_rate = min(candidates[name].null_rate_in_grain for name in top_candidates)
        top_candidates = [name for name in top_candidates if candidates[name].null_rate_in_grain == min_null_rate]

        if len(top_candidates) == 1:
            winner = top_candidates[0]
            logger.info(f"Selected anchor table '{winner}' (tie-breaker: null_rate={min_null_rate:.2%})")
            return winner

        # Tie-breaker 2: Unique on grain (should all be True at this point, but check anyway)
        unique_candidates = [name for name in top_candidates if candidates[name].is_unique_on_grain]

        if unique_candidates:
            top_candidates = unique_candidates

        if len(top_candidates) == 1:
            winner = top_candidates[0]
            logger.info(f"Selected anchor table '{winner}' (tie-breaker: unique on grain)")
            return winner

        # Tie-breaker 3: Smaller estimated bytes
        min_bytes = min(candidates[name].estimated_bytes for name in top_candidates)
        top_candidates = [name for name in top_candidates if candidates[name].estimated_bytes == min_bytes]

        if len(top_candidates) == 1:
            winner = top_candidates[0]
            logger.info(f"Selected anchor table '{winner}' (tie-breaker: bytes={min_bytes:,})")
            return winner

        # Tie-breaker 4: Patient grain over admission grain
        patient_grain_candidates = [name for name in top_candidates if candidates[name].grain == "patient"]

        if patient_grain_candidates:
            top_candidates = patient_grain_candidates

        # Final tie-breaker: alphabetical order (deterministic)
        winner = sorted(top_candidates)[0]
        logger.info(f"Selected anchor table '{winner}' (tie-breaker: alphabetical, grain={candidates[winner].grain})")

        return winner

    def get_relationship_summary(self) -> str:
        """Generate human-readable summary of detected relationships."""
        lines = ["=== Detected Table Relationships ==="]

        if not self.relationships:
            lines.append("No relationships detected")
            return "\n".join(lines)

        for rel in self.relationships:
            lines.append(f"  {rel}")

        lines.append(f"\nTotal: {len(self.relationships)} relationships")

        return "\n".join(lines)

    def close(self) -> None:
        """Close DuckDB connection."""
        if self.conn:
            self.conn.close()

    def __del__(self) -> None:
        """Cleanup DuckDB connection on deletion."""
        self.close()
