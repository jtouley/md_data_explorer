"""
Multi-Table Handler - Automatic Relationship Detection and Joins

This module enables handling of multi-table datasets (like MIMIC-IV) by automatically:
- Detecting primary keys in each table
- Discovering foreign key relationships
- Building a join graph
- Executing joins to create unified cohort views

Key Principles:
- Use Polars for all DataFrame operations
- Use DuckDB for SQL-based joins
- Privacy-preserving (all local computation)
- Fail gracefully with user override options
"""

from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Set, Literal
import polars as pl
import duckdb
import logging

logger = logging.getLogger(__name__)


# Type aliases for clarity
AggregationOp = Literal["count", "count_distinct", "min", "max", "mean", "sum", "last"]
CodeColumnPattern = str  # Regex or glob pattern for code columns


class AggregationPolicyError(ValueError):
    """Raised when aggregation policy is violated (e.g., mean on code column)."""
    pass


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
        allow_last: Enable last() aggregation (requires stable ordering)
        code_column_patterns: Patterns identifying code columns (no mean/avg)
    """
    default_numeric: List[AggregationOp] = field(default_factory=lambda: ["min", "max"])
    allow_mean: bool = False
    allow_last: bool = True
    code_column_patterns: List[CodeColumnPattern] = field(default_factory=lambda: [
        "icd_code", "itemid", "ndc", "cpt_code", "drg", "hcpcs",
        "*_code", "*_id"  # Glob patterns
    ])


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
        cardinality_ratio: rows / unique(grain_key), >1.1 indicates facts/events
        is_unique_on_grain: True if grain_key is unique (cardinality_ratio ≈ 1.0)
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
    time_column_name: Optional[str]
    is_n_side_of_anchor: bool
    null_rate_in_grain: float


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
        return f"{self.parent_table}.{self.parent_key} → {self.child_table}.{self.child_key} " \
               f"({self.relationship_type}, conf={self.confidence:.2f}, match={self.match_ratio:.2f})"


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
        tables: Dict[str, pl.DataFrame],
        max_dimension_bytes: int = 250_000_000  # 250 MB default
    ):
        """
        Initialize with dictionary of table_name -> Polars DataFrame.

        Args:
            tables: Dict mapping table names to Polars DataFrames
            max_dimension_bytes: Max size for dimension tables (default 250 MB)
        """
        self.tables = tables
        self.relationships: List[TableRelationship] = []
        self.primary_keys: Dict[str, str] = {}
        self.classifications: Dict[str, TableClassification] = {}
        self.max_dimension_bytes = max_dimension_bytes

        # Normalize key column types across all tables
        self._normalize_key_columns()

        # Initialize DuckDB connection for SQL-based joins
        self.conn = duckdb.connect(':memory:')

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
        column_counts = {}
        for df in self.tables.values():
            for col in df.columns:
                column_counts[col] = column_counts.get(col, 0) + 1

        # Identify columns that appear in multiple tables (potential join keys)
        shared_columns = {col for col, count in column_counts.items() if count > 1}
        logger.debug(f"Found {len(shared_columns)} shared columns: {list(shared_columns)[:10]}")

        # Also include common key patterns
        key_patterns = [
            '_id', 'subject_id', 'hadm_id', 'stay_id', 'itemid',
            'icustay_id', 'transfer_id', 'caregiver_id', 'charttime'
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
                    cast_exprs = [pl.col(col).cast(pl.Utf8, strict=False).alias(col)
                                  for col in cols_to_cast]
                    self.tables[table_name] = df.with_columns(cast_exprs)
                    
                    # Verify cast succeeded
                    new_dtypes = {col: self.tables[table_name][col].dtype for col in cols_to_cast}
                    logger.debug(f"New dtypes for {table_name}: {new_dtypes}")
                    
                    # Check for any that didn't convert
                    failed_casts = [col for col in cols_to_cast if self.tables[table_name][col].dtype != pl.Utf8]
                    if failed_casts:
                        logger.warning(f"Failed to cast columns in {table_name}: {failed_casts}")
                        
                except Exception as e:
                    logger.error(f"Batch cast failed for {table_name}: {type(e).__name__}: {str(e)}")
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
                            logger.error(f"Failed to cast {table_name}.{col} from {df[col].dtype}: {type(col_e).__name__}: {str(col_e)}")
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

    def detect_relationships(self) -> List[TableRelationship]:
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
                        match_ratio = self._verify_referential_integrity(
                            parent_df, parent_pk, child_df, child_col
                        )

                        if match_ratio > 0.8:  # Threshold for FK confidence
                            # Calculate confidence based on name match + integrity
                            name_conf = 1.0 if parent_pk.lower() == child_col.lower() else 0.9
                            confidence = (name_conf + match_ratio) / 2

                            self.relationships.append(TableRelationship(
                                parent_table=parent_table,
                                child_table=child_table,
                                parent_key=parent_pk,
                                child_key=child_col,
                                relationship_type="one-to-many",
                                confidence=confidence,
                                match_ratio=match_ratio
                            ))

        # Sort by confidence (descending)
        self.relationships.sort(key=lambda r: r.confidence, reverse=True)

        return self.relationships

    def classify_tables(self, anchor_table: Optional[str] = None) -> Dict[str, TableClassification]:
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

            # Calculate cardinality metrics using sampled data
            total_rows = df.height

            # Use sampled uniqueness to prevent expensive full table scans
            sampled_unique, sampled_null_rate = self._compute_sampled_uniqueness(df, grain_key)

            # Estimate cardinality ratio from sample
            s = self._sample_df(df)
            non_null_sample_rows = s.height - int(sampled_null_rate * s.height)

            if sampled_unique > 0 and non_null_sample_rows > 0:
                # Estimate total unique values from sample ratio
                sample_uniq_ratio = sampled_unique / non_null_sample_rows
                estimated_total_unique = int(total_rows * sample_uniq_ratio)

                # Cardinality ratio: total_rows / unique_values
                # If ratio is ~1.0, table is unique on grain (dimension)
                # If ratio > 1.1, table has duplicates (fact/event)
                cardinality_ratio = total_rows / max(estimated_total_unique, 1)
                is_unique = (cardinality_ratio <= 1.05)  # Allow 5% tolerance for sampling error
            else:
                cardinality_ratio = float('inf')
                is_unique = False

            null_rate = sampled_null_rate

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
                is_unique=is_unique,
                estimated_bytes=estimated_bytes,
                has_time=has_time,
                is_bridge=is_bridge,
                is_n_side=is_n_side,
                time_col=time_col,
                df=df
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
                null_rate_in_grain=null_rate
            )

            logger.debug(
                f"Classified '{table_name}': {classification} "
                f"(grain={grain_level}, key={grain_key}, card_ratio={cardinality_ratio:.2f}, "
                f"bytes={estimated_bytes:,}, fks={fk_counts.get(table_name, 0)})"
            )

        return self.classifications

    def _detect_grain_key(self, df: pl.DataFrame) -> Optional[str]:
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
        patient_patterns = ['patient_id', 'subject_id', 'patientid', 'subjectid']
        for col in df.columns:
            if col.lower() in patient_patterns:
                return col

        # 2. Check explicit admission grain patterns
        admission_patterns = ['hadm_id', 'encounter_id', 'visit_id', 'admissionid', 'encounterid']
        for col in df.columns:
            if col.lower() in admission_patterns:
                return col

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
            score -= 1.0 * null_rate    # Penalize NULLs
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
            best_col = max(scores, key=scores.get)
            logger.debug(f"Selected grain key: '{best_col}' (score={scores[best_col]:.2f})")
            return best_col

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

        if any(p in grain_key_lower for p in ['patient', 'subject']):
            return "patient"
        elif any(p in grain_key_lower for p in ['hadm', 'encounter', 'visit', 'admission']):
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
                avg_str_len = sample[col].drop_nulls().str.len_chars().mean() or 0
                bytes_per_row += int(avg_str_len)
            elif dtype == pl.Boolean:
                bytes_per_row += 1
            else:
                # Default estimate for unknown types
                bytes_per_row += 8

        # Total estimate
        return bytes_per_row * df.height

    def _detect_time_column(self, df: pl.DataFrame) -> Tuple[Optional[str], bool]:
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
        time_patterns = ['time', 'date', 'timestamp', 'datetime']

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

    def _count_foreign_keys(self) -> Dict[str, int]:
        """
        Count number of foreign key relationships per table.

        Returns:
            Dict mapping table name to number of FKs
        """
        fk_counts: Dict[str, int] = {}

        for rel in self.relationships:
            # Child table has a foreign key
            fk_counts[rel.child_table] = fk_counts.get(rel.child_table, 0) + 1

        return fk_counts

    def _detect_bridge_table(
        self,
        table_name: str,
        df: pl.DataFrame,
        fk_count: int
    ) -> bool:
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
        fk_cols = [
            rel.child_key for rel in self.relationships
            if rel.child_table == table_name
        ]

        if len(fk_cols) < 2:
            return False

        # Sample for uniqueness checks
        s = self._sample_df(df)

        # Check if individual FKs are non-unique (on sample)
        fk_unique_flags = []
        for col in fk_cols:
            if col in s.columns:
                unique_count, _ = self._compute_sampled_uniqueness(df, col)
                is_unique = (unique_count == s.height)
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

    def _is_n_side_of_anchor(self, table_name: str, anchor_table: Optional[str]) -> bool:
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
        time_col: Optional[str],
        df: pl.DataFrame
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

    def _build_dimension_mart(
        self,
        anchor_table: Optional[str] = None,
        join_type: str = "left"
    ) -> pl.LazyFrame:
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

        # Verify anchor is unique on grain
        if not anchor_class.is_unique_on_grain:
            raise ValueError(
                f"Anchor table '{anchor_table}' is not unique on grain key "
                f"'{anchor_class.grain_key}' (cardinality_ratio={anchor_class.cardinality_ratio:.2f})"
            )

        # Start with anchor table as LazyFrame
        anchor_df = self.tables[anchor_table]
        mart = anchor_df.lazy()
        joined_tables = {anchor_table}

        logger.debug(
            f"Anchor '{anchor_table}': {anchor_df.height} rows, "
            f"grain_key={anchor_class.grain_key}"
        )

        # Build join graph using BFS from anchor
        queue = deque([anchor_table])
        join_count = 0

        while queue:
            current = queue.popleft()

            # Find relationships where current table is involved
            # Sort for deterministic traversal order
            for rel in sorted(
                self.relationships,
                key=lambda r: (r.parent_table, r.child_table, r.parent_key, r.child_key)
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

                    # CRITICAL: Verify RHS join key is unique (enforce 1:1 or many:1)
                    next_df = self.tables[next_table]

                    # Use sampled uniqueness to check if RHS key is unique
                    rhs_unique_count, rhs_null_rate = self._compute_sampled_uniqueness(
                        next_df, join_key_right
                    )
                    s = self._sample_df(next_df)
                    non_null_sample = s.height - int(rhs_null_rate * s.height)

                    if non_null_sample > 0:
                        rhs_uniq_ratio = rhs_unique_count / non_null_sample
                    else:
                        rhs_uniq_ratio = 0.0

                    # Allow 5% tolerance for sampling error
                    if rhs_uniq_ratio < 0.95:
                        logger.warning(
                            f"Skipping join to '{next_table}': RHS key '{join_key_right}' "
                            f"is not unique (uniq_ratio={rhs_uniq_ratio:.2%}, "
                            f"violates 1:1 or many:1 constraint)"
                        )
                        continue

                    # Join next dimension table
                    next_lazy = next_df.lazy()

                    mart = mart.join(
                        next_lazy,
                        left_on=join_key_left,
                        right_on=join_key_right,
                        how=join_type,
                        suffix=f"_{next_table}",
                        validate="m:1",  # fail fast if RHS key is not unique
                    )

                    joined_tables.add(next_table)
                    queue.append(next_table)
                    join_count += 1

                    logger.debug(
                        f"Joined dimension '{next_table}' on {join_key_left}={join_key_right} "
                        f"(rhs_uniq_ratio={rhs_uniq_ratio:.2%})"
                    )

        logger.info(
            f"Dimension mart built: anchor='{anchor_table}', "
            f"joined_dimensions={join_count}, total_tables={len(joined_tables)}"
        )

        return mart

    def _aggregate_fact_tables(
        self,
        grain_key: str,
        policy: AggregationPolicy = None
    ) -> Dict[str, pl.LazyFrame]:
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
            ValueError: If grain_key not found in fact table
        """
        import fnmatch

        # Use default policy if not specified
        if policy is None:
            policy = AggregationPolicy()

        logger.info(f"Aggregating fact/event tables by grain_key='{grain_key}'")

        # Ensure classifications exist
        if not self.classifications:
            self.classify_tables()

        # Filter to fact/event tables only
        fact_event_tables = {
            name: cls for name, cls in self.classifications.items()
            if cls.classification in ["fact", "event"]
        }

        if not fact_event_tables:
            logger.warning("No fact/event tables found for aggregation")
            return {}

        feature_tables = {}

        for table_name, cls in fact_event_tables.items():
            df = self.tables[table_name]

            # Verify grain_key exists
            if grain_key not in df.columns:
                logger.warning(
                    f"Skipping '{table_name}': grain_key '{grain_key}' not found in columns"
                )
                continue

            # Build aggregation expressions
            agg_exprs = []

            # Always include count
            agg_exprs.append(pl.len().alias("count"))

            for col in df.columns:
                if col == grain_key:
                    continue  # Skip grain key

                # Check if column matches code patterns
                is_code_col = any(
                    fnmatch.fnmatch(col.lower(), pattern.lower())
                    for pattern in policy.code_column_patterns
                )

                dtype = df[col].dtype

                # Count distinct for all columns
                agg_exprs.append(pl.col(col).n_unique().alias(f"{col}_count_distinct"))

                # Numeric aggregations
                if dtype in [pl.Int8, pl.Int16, pl.Int32, pl.Int64,
                            pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
                            pl.Float32, pl.Float64]:

                    # Safe aggregations: min/max (always allowed)
                    agg_exprs.append(pl.col(col).min().alias(f"{col}_min"))
                    agg_exprs.append(pl.col(col).max().alias(f"{col}_max"))

                    # Mean: only if allowed AND not a code column
                    if policy.allow_mean:
                        if is_code_col:
                            raise AggregationPolicyError(
                                f"Cannot compute mean on code column '{col}' in table '{table_name}'. "
                                f"Column matches pattern: {policy.code_column_patterns}. "
                                f"Code columns should not be averaged."
                            )
                        else:
                            agg_exprs.append(pl.col(col).mean().alias(f"{col}_mean"))

                    # Last: only if allowed
                    if policy.allow_last:
                        agg_exprs.append(pl.col(col).last().alias(f"{col}_last"))

                # Time aggregations
                elif dtype in [pl.Datetime, pl.Date]:
                    agg_exprs.append(pl.col(col).min().alias(f"{col}_min"))
                    agg_exprs.append(pl.col(col).max().alias(f"{col}_max"))

                    # Last: only if allowed
                    if policy.allow_last:
                        agg_exprs.append(pl.col(col).last().alias(f"{col}_last"))

                # String columns: only count distinct (no aggregations)
                # Already added count_distinct above

            # Build lazy aggregation
            lazy_df = df.lazy()
            aggregated = lazy_df.group_by(grain_key).agg(agg_exprs)

            # Store with _features suffix
            feature_table_name = f"{table_name}_features"
            feature_tables[feature_table_name] = aggregated

            logger.debug(
                f"Aggregated '{table_name}' → '{feature_table_name}': "
                f"{len(agg_exprs)} aggregations on {len(df.columns)} columns"
            )

        logger.info(f"Created {len(feature_tables)} feature tables from fact/event tables")

        return feature_tables

    def build_unified_cohort(
        self,
        anchor_table: Optional[str] = None,
        join_type: str = "left"
    ) -> pl.DataFrame:
        """
        Join all related tables into unified cohort view using DuckDB.

        Strategy:
        1. Find anchor table (root of join graph)
        2. Build join graph using BFS from anchor
        3. Execute joins in correct order
        4. Handle name collisions with suffixes

        Args:
            anchor_table: Root table for joins (auto-detected if None)
            join_type: Type of SQL join (left, inner, outer)

        Returns:
            Unified Polars DataFrame with all columns

        Example:
            >>> cohort = handler.build_unified_cohort(anchor_table='patients')
        """
        logger.info(f"Building unified cohort (anchor_table={anchor_table}, join_type={join_type})")
        
        # Auto-detect anchor table if not specified
        if anchor_table is None:
            anchor_table = self._find_anchor_table()
            logger.info(f"Auto-detected anchor table: {anchor_table}")

        if anchor_table not in self.tables:
            raise ValueError(f"Anchor table '{anchor_table}' not found in tables")
        
        logger.debug(f"Anchor table '{anchor_table}' has {self.tables[anchor_table].height} rows, {self.tables[anchor_table].width} columns")

        # Build join graph using BFS
        joined_tables = {anchor_table}
        join_clauses = []

        # Start with anchor table
        current_table_alias = anchor_table

        # BFS to find join order
        queue = [anchor_table]

        while queue:
            current = queue.pop(0)

            # Find relationships where current table is parent or child
            for rel in self.relationships:
                next_table = None
                join_on = None

                if rel.parent_table == current and rel.child_table not in joined_tables:
                    # Join child to parent
                    next_table = rel.child_table
                    join_on = f"{current}.{rel.parent_key} = {next_table}.{rel.child_key}"

                elif rel.child_table == current and rel.parent_table not in joined_tables:
                    # Join parent to child (reverse relationship)
                    next_table = rel.parent_table
                    join_on = f"{current}.{rel.child_key} = {next_table}.{rel.parent_key}"

                if next_table:
                    join_clauses.append((next_table, join_on, join_type))
                    joined_tables.add(next_table)
                    queue.append(next_table)

        # Build SQL query for joins
        if not join_clauses:
            # No joins possible, return anchor table
            logger.debug(f"No joins possible, returning anchor table: {anchor_table}")
            return self.tables[anchor_table]

        # Construct SQL query
        query = f"SELECT * FROM {anchor_table}"

        for table, on_clause, jtype in join_clauses:
            query += f" {jtype.upper()} JOIN {table} ON {on_clause}"

        logger.debug(f"Executing SQL query with {len(join_clauses)} joins")
        logger.debug(f"Query: {query[:500]}...")  # Log first 500 chars of query

        # Execute query using DuckDB
        try:
            result = self.conn.execute(query).pl()
            logger.debug(f"Query executed successfully: {result.height} rows, {result.width} columns")
            return result
        except Exception as e:
            logger.error(f"Error executing SQL query: {type(e).__name__}: {str(e)}")
            logger.error(f"Full query: {query}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise

    def _detect_primary_key(self, df: pl.DataFrame) -> Optional[str]:
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
        id_pattern_cols = [col for col in df.columns if 'id' in col.lower()]

        # Check ID pattern columns first
        for col in id_pattern_cols:
            if df[col].n_unique() == df.height and df[col].null_count() == 0:
                return col

        # Fallback: check all columns
        for col in df.columns:
            if df[col].n_unique() == df.height and df[col].null_count() == 0:
                return col

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
        parent_no_underscore = parent_lower.replace('_', '')
        child_no_underscore = child_lower.replace('_', '')

        if parent_no_underscore == child_no_underscore:
            return True

        return False

    def _verify_referential_integrity(
        self,
        parent_df: pl.DataFrame,
        parent_key: str,
        child_df: pl.DataFrame,
        child_col: str
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
        logger.debug(f"Child values dtype: {child_values.dtype}, len: {child_values.len()}, sample: {child_values.head(3).to_list()}")

        if child_values.len() == 0:
            logger.debug("Child values empty, returning 0.0")
            return 0.0

        # Get parent values
        parent_values = parent_df[parent_key].drop_nulls()
        logger.debug(f"Parent values dtype: {parent_values.dtype}, len: {parent_values.len()}, sample: {parent_values.head(3).to_list()}")

        # Cast both to string for type-safe comparison using pure Polars operations
        # This handles cases where one table has int64 and another has string
        try:
            # Cast both to Utf8 for consistent comparison
            logger.debug(f"Casting child from {child_values.dtype} to Utf8")
            child_str = child_values.cast(pl.Utf8, strict=False).drop_nulls().unique()
            logger.debug(f"Child after cast: dtype={child_str.dtype}, len={child_str.len()}, sample={child_str.head(3).to_list()}")
            
            logger.debug(f"Casting parent from {parent_values.dtype} to Utf8")
            parent_str = parent_values.cast(pl.Utf8, strict=False).drop_nulls().unique()
            logger.debug(f"Parent after cast: dtype={parent_str.dtype}, len={parent_str.len()}, sample={parent_str.head(3).to_list()}")

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
            logger.error(f"Error in referential integrity check ({parent_key} -> {child_col}): {type(e).__name__}: {str(e)}")
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
        logger.warning(
            "_find_anchor_table() is deprecated, use _find_anchor_by_centrality() instead"
        )

        # Count relationships for each table
        table_counts = {}

        for rel in self.relationships:
            table_counts[rel.parent_table] = table_counts.get(rel.parent_table, 0) + 1
            table_counts[rel.child_table] = table_counts.get(rel.child_table, 0) + 1

        # Prefer tables with "patient" or "subject" in name
        patient_tables = [
            t for t in self.tables.keys()
            if 'patient' in t.lower() or 'subject' in t.lower()
        ]

        if patient_tables:
            # Pick patient table with most relationships
            return max(patient_tables, key=lambda t: table_counts.get(t, 0))

        # Fallback: table with most relationships
        if table_counts:
            return max(table_counts, key=table_counts.get)

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
            name: cls for name, cls in self.classifications.items()
            if cls.classification == "dimension"
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

            if not cls.is_unique_on_grain:
                logger.debug(f"Excluding {name} from anchor candidates: not unique on grain")
                continue

            candidates[name] = cls

        if not candidates:
            raise ValueError(
                "No suitable anchor table found. All dimensions have >50% NULLs "
                "or non-unique grain keys."
            )

        # Count relationships per table
        relationship_counts = {}
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

            if any(p in col_lower for p in ['hadm_id', 'encounter_id']):
                score += 10

            if any(p in col_lower for p in ['patient_id', 'subject_id']):
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
        top_candidates = [
            name for name in top_candidates
            if candidates[name].null_rate_in_grain == min_null_rate
        ]

        if len(top_candidates) == 1:
            winner = top_candidates[0]
            logger.info(
                f"Selected anchor table '{winner}' (tie-breaker: null_rate={min_null_rate:.2%})"
            )
            return winner

        # Tie-breaker 2: Unique on grain (should all be True at this point, but check anyway)
        unique_candidates = [
            name for name in top_candidates
            if candidates[name].is_unique_on_grain
        ]

        if unique_candidates:
            top_candidates = unique_candidates

        if len(top_candidates) == 1:
            winner = top_candidates[0]
            logger.info(f"Selected anchor table '{winner}' (tie-breaker: unique on grain)")
            return winner

        # Tie-breaker 3: Smaller estimated bytes
        min_bytes = min(candidates[name].estimated_bytes for name in top_candidates)
        top_candidates = [
            name for name in top_candidates
            if candidates[name].estimated_bytes == min_bytes
        ]

        if len(top_candidates) == 1:
            winner = top_candidates[0]
            logger.info(
                f"Selected anchor table '{winner}' (tie-breaker: bytes={min_bytes:,})"
            )
            return winner

        # Tie-breaker 4: Patient grain over admission grain
        patient_grain_candidates = [
            name for name in top_candidates
            if candidates[name].grain == "patient"
        ]

        if patient_grain_candidates:
            top_candidates = patient_grain_candidates

        # Final tie-breaker: alphabetical order (deterministic)
        winner = sorted(top_candidates)[0]
        logger.info(
            f"Selected anchor table '{winner}' (tie-breaker: alphabetical, "
            f"grain={candidates[winner].grain})"
        )

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

    def close(self):
        """Close DuckDB connection."""
        if self.conn:
            self.conn.close()

    def __del__(self):
        """Cleanup DuckDB connection on deletion."""
        self.close()
