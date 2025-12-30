"""
Dataset Versioning Module

Implements content-based dataset versioning for idempotent query execution.
MVP scope: Basic content hashing with canonicalization.
Deferred to Phase 5+: Perfect deduplication, re-upload detection, storage reuse.
"""

import hashlib
import io
from typing import Any

import polars as pl


def compute_dataset_version(tables: list[pl.DataFrame]) -> str:
    """
    Compute content hash of canonicalized tables.

    MVP: Simple hash of sorted table data. Perfect dedup deferred to Phase 5+.

    Args:
        tables: List of Polars DataFrames (NOT LazyFrames - must be materialized)

    Returns:
        16-character hex hash (stable, order-independent)

    Example:
        >>> df1 = pl.DataFrame({"patient_id": [1, 2, 3], "age": [25, 30, 35]})
        >>> df2 = pl.DataFrame({"age": [25, 30, 35], "patient_id": [1, 2, 3]})  # Different column order
        >>> compute_dataset_version([df1]) == compute_dataset_version([df2])
        True
    """
    if not tables:
        raise ValueError("Cannot compute version of empty table list")

    # Compute hash for each table
    table_hashes = []
    for table in tables:
        if isinstance(table, pl.LazyFrame):
            raise TypeError("compute_dataset_version requires materialized DataFrame, not LazyFrame")

        # Canonicalize: Sort columns alphabetically, then sort rows by all columns
        canonical = _canonicalize_dataframe(table)

        # Compute table hash using Parquet serialization (handles all data types correctly)
        table_hash = _hash_dataframe(canonical)
        table_hashes.append(table_hash)

    # Aggregate table hashes into single dataset version
    combined = "|".join(table_hashes)
    dataset_hash = hashlib.sha256(combined.encode()).hexdigest()[:16]

    return dataset_hash


def _canonicalize_dataframe(df: pl.DataFrame) -> pl.DataFrame:
    """
    Canonicalize DataFrame for content-based hashing.

    Makes hash order-independent:
    - Sorts columns alphabetically
    - Sorts rows by all columns (lexicographic order)

    Args:
        df: Input DataFrame

    Returns:
        Canonicalized DataFrame (same data, deterministic order)
    """
    # Sort columns alphabetically
    sorted_cols = sorted(df.columns)
    df = df.select(sorted_cols)

    # Sort rows by all columns (stable, deterministic)
    # Use sorted_cols to ensure lexicographic ordering
    df = df.sort(by=sorted_cols)

    return df


def _hash_dataframe(df: pl.DataFrame) -> str:
    """
    Compute content hash of DataFrame using Parquet serialization.

    Parquet serialization correctly handles:
    - All Polars data types (including nested types)
    - Null values
    - Schema information (column names + types)

    Args:
        df: Canonicalized DataFrame

    Returns:
        16-character hex hash
    """
    # Serialize to Parquet (in-memory)
    buffer = io.BytesIO()
    df.write_parquet(buffer)
    parquet_bytes = buffer.getvalue()

    # Hash Parquet bytes
    return hashlib.sha256(parquet_bytes).hexdigest()[:16]


def compute_table_fingerprint(df: pl.DataFrame, table_name: str) -> dict[str, Any]:
    """
    Compute basic table fingerprint for provenance tracking.

    MVP: Row count and content hash only.
    Deferred to Phase 5+: Column stats, schema fingerprint, data quality metrics.

    Args:
        df: Table DataFrame
        table_name: Table name

    Returns:
        Dict with name, row_count, fingerprint
    """
    canonical = _canonicalize_dataframe(df)
    fingerprint = _hash_dataframe(canonical)

    return {
        "name": table_name,
        "row_count": df.height,
        "fingerprint": fingerprint,
    }
