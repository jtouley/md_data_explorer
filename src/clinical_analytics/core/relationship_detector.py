"""
Relationship Detector - Automatic Foreign Key Discovery

Extracted from MultiTableHandler for independent testability (Phase 0.3).

This module detects foreign key relationships between tables by:
- Detecting primary keys in each table
- Finding foreign key candidates based on naming patterns
- Verifying referential integrity
- Scoring confidence based on name similarity and integrity
"""

import logging
from dataclasses import dataclass

import polars as pl

logger = logging.getLogger(__name__)


@dataclass
class TableRelationship:
    """
    Detected relationship between two tables.

    Represents a foreign key relationship discovered through pattern matching
    and referential integrity analysis.
    """

    parent_table: str
    child_table: str
    parent_key: str
    child_key: str
    relationship_type: str  # e.g., "one-to-many"
    confidence: float  # 0-1 score based on name similarity + referential integrity
    match_ratio: float  # 0-1 ratio of child values matching parent values


class RelationshipDetector:
    """
    Detects foreign key relationships between tables.

    Uses naming patterns and referential integrity analysis to automatically
    discover relationships without requiring explicit schema definitions.
    """

    def detect_primary_key(self, df: pl.DataFrame) -> str | None:
        """
        Detect primary key column in a table.

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
                return col

        # Fallback: check all columns
        for col in df.columns:
            if df[col].n_unique() == df.height and df[col].null_count() == 0:
                return col

        return None

    def is_foreign_key_candidate(self, parent_key: str, child_col: str) -> bool:
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

        # Exact match or common FK patterns
        for pattern in [
            parent_lower,
            f"{parent_lower}_id",
            f"fk_{parent_lower}",
            f"{parent_lower}_fk",
        ]:
            if child_lower == pattern:
                return True

        # Handle cases like "patient_id" matching "patientid"
        parent_no_underscore = parent_lower.replace("_", "")
        child_no_underscore = child_lower.replace("_", "")

        if parent_no_underscore == child_no_underscore:
            return True

        return False

    def verify_referential_integrity(
        self, parent_df: pl.DataFrame, parent_key: str, child_df: pl.DataFrame, child_col: str
    ) -> float:
        """
        Verify referential integrity between parent and child tables.

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

    def detect_relationships(self, tables: dict[str, pl.DataFrame]) -> list[TableRelationship]:
        """
        Auto-detect foreign key relationships between tables.

        Strategy:
        1. Detect primary keys for each table
        2. For each table pair, check if child has column matching parent PK
        3. Verify referential integrity (% of child values in parent)
        4. Score confidence based on name similarity + integrity

        Args:
            tables: Dictionary mapping table names to Polars DataFrames

        Returns:
            List of detected relationships sorted by confidence (descending)
        """
        relationships = []
        primary_keys = {}

        # 1. Detect primary keys for each table
        for table_name, df in tables.items():
            pk = self.detect_primary_key(df)
            if pk:
                primary_keys[table_name] = pk
                logger.debug(f"Detected primary key for {table_name}: {pk}")

        # 2. Detect foreign key relationships
        for parent_table, parent_df in tables.items():
            parent_pk = primary_keys.get(parent_table)

            if not parent_pk:
                continue  # Skip tables without primary keys

            for child_table, child_df in tables.items():
                if parent_table == child_table:
                    continue  # Skip self-joins

                # Check if child has column matching parent's primary key
                for child_col in child_df.columns:
                    if self.is_foreign_key_candidate(parent_pk, child_col):
                        # Verify referential integrity
                        match_ratio = self.verify_referential_integrity(parent_df, parent_pk, child_df, child_col)

                        if match_ratio > 0.8:  # Threshold for FK confidence
                            # Calculate confidence based on name match + integrity
                            name_conf = 1.0 if parent_pk.lower() == child_col.lower() else 0.9
                            confidence = (name_conf + match_ratio) / 2

                            relationships.append(
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
                            logger.info(
                                f"Detected relationship: {parent_table}.{parent_pk} -> {child_table}.{child_col} "
                                f"(confidence={confidence:.2f}, match_ratio={match_ratio:.2f})"
                            )

        # Sort by confidence (descending)
        relationships.sort(key=lambda r: r.confidence, reverse=True)

        return relationships
