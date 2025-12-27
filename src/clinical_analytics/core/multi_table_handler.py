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

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Set, Literal
import polars as pl
import duckdb
import logging

logger = logging.getLogger(__name__)


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

            # Calculate cardinality metrics
            total_rows = df.height
            unique_grain_values = df[grain_key].n_unique()
            null_count = df[grain_key].null_count()
            null_rate = null_count / total_rows if total_rows > 0 else 0.0

            cardinality_ratio = total_rows / unique_grain_values if unique_grain_values > 0 else float('inf')
            is_unique = (unique_grain_values == total_rows - null_count)

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
        Detect grain key column using pattern matching and cardinality.

        Grain key patterns (in order of priority):
        1. patient_id, subject_id (patient grain)
        2. hadm_id, encounter_id, visit_id (admission grain)
        3. Any column with high uniqueness and *_id pattern

        Args:
            df: Polars DataFrame

        Returns:
            Grain key column name or None
        """
        # Patient grain patterns
        patient_patterns = ['patient_id', 'subject_id', 'patientid', 'subjectid']
        for col in df.columns:
            if col.lower() in patient_patterns:
                return col

        # Admission grain patterns
        admission_patterns = ['hadm_id', 'encounter_id', 'visit_id', 'admissionid', 'encounterid']
        for col in df.columns:
            if col.lower() in admission_patterns:
                return col

        # Fallback: highest cardinality column with _id suffix
        id_cols = [col for col in df.columns if col.lower().endswith('_id') or col.lower().endswith('id')]
        if id_cols:
            # Pick column with highest uniqueness
            best_col = max(id_cols, key=lambda c: df[c].n_unique())
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
        Detect time column in DataFrame.

        A valid time column must:
        1. Have datetime/date type OR name contains time/date patterns
        2. Not be constant (all same value)

        Args:
            df: Polars DataFrame

        Returns:
            Tuple of (column_name, has_valid_time_column)
        """
        time_patterns = ['time', 'date', 'timestamp', 'datetime', 'dt']

        for col in df.columns:
            # Check dtype
            is_time_type = df[col].dtype in [pl.Datetime, pl.Date]

            # Check name pattern
            col_lower = col.lower()
            has_time_pattern = any(p in col_lower for p in time_patterns)

            if is_time_type or has_time_pattern:
                # Verify not constant
                unique_count = df[col].n_unique()
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
        Detect if table is a bridge (many-to-many) table.

        Bridge table characteristics:
        1. Two or more foreign keys to different parent tables
        2. Neither FK is unique individually
        3. Composite key (fk1, fk2) is near-unique (>95% unique)
        4. High relationship degree but low column payload (<10 columns typically)

        Args:
            table_name: Name of table
            df: Polars DataFrame
            fk_count: Number of foreign keys

        Returns:
            True if bridge table detected
        """
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

        # Check if individual FKs are non-unique
        fk_unique_flags = [df[col].n_unique() == df.height for col in fk_cols if col in df.columns]

        if all(fk_unique_flags):
            # If all FKs are unique, not a bridge
            return False

        # Check if composite key is near-unique
        try:
            # Use first two FK columns for composite check
            fk1, fk2 = fk_cols[0], fk_cols[1]
            if fk1 in df.columns and fk2 in df.columns:
                composite_unique = df.select([fk1, fk2]).n_unique()
                composite_ratio = composite_unique / df.height if df.height > 0 else 0.0

                # Near-unique threshold: 95%
                is_composite_unique = composite_ratio > 0.95

                # Check column count (bridges typically have few columns)
                has_low_payload = df.width < 10

                return is_composite_unique and has_low_payload
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

        Strategy:
        - Table with most relationships (parent or child)
        - Prefer tables with "patient" or "subject" in name

        Returns:
            Anchor table name
        """
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
