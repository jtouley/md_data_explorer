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
from typing import Dict, List, Tuple, Optional, Set
import polars as pl
import duckdb
import logging

logger = logging.getLogger(__name__)


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

    def __init__(self, tables: Dict[str, pl.DataFrame]):
        """
        Initialize with dictionary of table_name -> Polars DataFrame.

        Args:
            tables: Dict mapping table names to Polars DataFrames
        """
        self.tables = tables
        self.relationships: List[TableRelationship] = []
        self.primary_keys: Dict[str, str] = {}

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
