# SPDX-License-Identifier: MIT
"""
Dataset Versioning Module

Implements content-based dataset versioning for idempotent query execution.

INVARIANT (Core Contract):
    A dataset is uniquely identified by (upload_id, dataset_version) where:
    - upload_id: User-scoped identifier (assigned at upload time)
    - dataset_version: Content hash (deterministic from data)

    This guarantees:
    - Identical data → Identical version → Storage reuse
    - Different data → Different version → Separate storage
    - Query results can be cached/memoized by (upload_id, dataset_version)

    Persistence key format: {upload_id}_{table_name}_{dataset_version}

    The version is a hash of canonical Arrow IPC bytes (canonical_table_bytes), so it depends
    on the data and its types only, not on the Polars version that computed it. Versions
    computed before PR #83 hashed Parquet bytes, whose footer names the Polars version; those
    stay as stored (they are part of persistence keys), and dedup uses the separately stored
    content_hash instead (ui/storage/user_datasets.py).

MVP scope: Basic content hashing with canonicalization.
Deferred to Phase 5+: Perfect deduplication, re-upload detection, cross-user storage reuse.
"""

import hashlib
import io
from typing import Any

import polars as pl
import polars.selectors as cs

# Names the byte form compute_dataset_version hashes. Stored next to each dataset's
# content_hash so a future change of the form is detected and backfilled, never compared
# across forms. Bump it whenever canonical_table_bytes changes its output.
CONTENT_HASH_SCHEME = "arrow-ipc-oldest-v1"


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

        # Canonical Arrow IPC bytes: independent of row/column order and of the Polars version
        table_hash = _hash_dataframe(table)
        table_hashes.append(table_hash)

    # Aggregate table hashes into single dataset version
    combined = "|".join(table_hashes)
    dataset_hash = hashlib.sha256(combined.encode()).hexdigest()[:16]

    return dataset_hash


def _canonicalize_dataframe(df: pl.DataFrame) -> pl.DataFrame:
    """
    Canonicalize DataFrame for content-based hashing.

    Makes hash order- and layout-independent:
    - Categorical/Enum columns become String: their physical codes depend on dictionary
      order (and on the string cache), and Polars tags them with its own field metadata
    - Sorts columns alphabetically
    - Sorts rows by all columns (lexicographic order)
    - Rechunks into one chunk, so how the frame was built does not change the bytes

    Args:
        df: Input DataFrame

    Returns:
        Canonicalized DataFrame (same data, deterministic order)
    """
    df = df.with_columns(cs.categorical().cast(pl.String), cs.enum().cast(pl.String))

    sorted_cols = sorted(df.columns)
    return df.select(sorted_cols).sort(by=sorted_cols).rechunk()


def canonical_table_bytes(df: pl.DataFrame) -> bytes:
    """
    Serialize a DataFrame to the byte form that content hashes are taken over.

    Arrow IPC stream at the oldest Polars compatibility level, uncompressed. Unlike Parquet,
    whose footer records ``created_by: Polars (python) <version>``, the stream holds only the
    schema (names and Arrow types) and the column buffers, which the Arrow spec fixes. The
    oldest compat level keeps Polars from switching a type's Arrow encoding (e.g. string views)
    in a later release. Verified equal under Polars 1.35.2 and 2.1.0 (PR #83);
    tests/storage/test_versioning.py pins the value for the synthetic cohort.

    Args:
        df: Any DataFrame; it is canonicalized first

    Returns:
        Bytes to hash. Do not persist them: they are a hashing input, not a storage format.
    """
    buffer = io.BytesIO()
    _canonicalize_dataframe(df).write_ipc_stream(
        buffer, compression="uncompressed", compat_level=pl.CompatLevel.oldest()
    )
    return buffer.getvalue()


def _hash_dataframe(df: pl.DataFrame) -> str:
    """
    Compute content hash of DataFrame from its canonical bytes.

    Args:
        df: DataFrame (canonicalized inside canonical_table_bytes)

    Returns:
        16-character hex hash
    """
    return hashlib.sha256(canonical_table_bytes(df)).hexdigest()[:16]


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
    fingerprint = _hash_dataframe(df)

    return {
        "name": table_name,
        "row_count": df.height,
        "fingerprint": fingerprint,
    }
