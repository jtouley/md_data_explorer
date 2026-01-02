"""
Test data caching infrastructure with content-based hashing.

Provides caching for expensive test fixtures:
- DataFrame caching (parquet files)
- Excel file caching
- Content-based cache keys (hash of data)

Cache location: tests/.test_cache/
"""

import hashlib
import os
import shutil
from io import BytesIO
from pathlib import Path

import polars as pl

# Cache directory location
CACHE_DIR = Path(__file__).parent.parent / ".test_cache"


def get_cache_dir() -> Path:
    """
    Get cache directory path.

    Can be overridden via TEST_CACHE_DIR environment variable.

    Returns:
        Path to cache directory
    """
    env_cache_dir = os.getenv("TEST_CACHE_DIR")
    if env_cache_dir:
        return Path(env_cache_dir)
    return CACHE_DIR


def hash_dataframe(df: pl.DataFrame) -> str:
    """
    Generate content-based hash for DataFrame.

    Uses parquet serialization to ensure consistent hashing
    regardless of DataFrame creation method.

    Args:
        df: Polars DataFrame to hash

    Returns:
        Hexadecimal hash string
    """
    # Convert to parquet bytes for consistent hashing
    buffer = BytesIO()
    df.write_parquet(buffer)
    parquet_bytes = buffer.getvalue()
    return hashlib.sha256(parquet_bytes).hexdigest()


def hash_file(file_path: Path) -> str:
    """
    Generate content-based hash for file.

    Args:
        file_path: Path to file

    Returns:
        Hexadecimal hash string
    """
    with open(file_path, "rb") as f:
        file_bytes = f.read()
    return hashlib.sha256(file_bytes).hexdigest()


def _get_cache_path(cache_dir: Path | None, cache_key: str, extension: str) -> Path:
    """Get cache file path for given key and extension."""
    if cache_dir is None:
        cache_dir = get_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{cache_key}{extension}"


def cache_dataframe(df: pl.DataFrame, cache_key: str, cache_dir: Path | None = None) -> Path:
    """
    Cache DataFrame as parquet file.

    Args:
        df: DataFrame to cache
        cache_dir: Directory for cache files
        cache_key: Unique key for this cache entry

    Returns:
        Path to cached parquet file
    """
    cache_path = _get_cache_path(cache_dir, cache_key, ".parquet")
    df.write_parquet(cache_path)
    return cache_path


def get_cached_dataframe(cache_key: str, cache_dir: Path | None = None) -> pl.DataFrame | None:
    """
    Retrieve cached DataFrame from parquet file.

    Args:
        cache_dir: Directory for cache files
        cache_key: Unique key for cache entry

    Returns:
        Cached DataFrame if exists, None otherwise
    """
    cache_path = _get_cache_path(cache_dir, cache_key, ".parquet")
    if not cache_path.exists():
        return None

    try:
        return pl.read_parquet(cache_path)
    except Exception:
        # If read fails, cache is corrupted, return None
        return None


def cache_excel_file(source_file: Path, cache_key: str, cache_dir: Path | None = None) -> Path:
    """
    Cache Excel file by copying to cache directory.

    Args:
        source_file: Source Excel file to cache
        cache_dir: Directory for cache files
        cache_key: Unique key for this cache entry

    Returns:
        Path to cached Excel file
    """
    cache_path = _get_cache_path(cache_dir, cache_key, ".xlsx")
    shutil.copy2(source_file, cache_path)
    return cache_path


def get_cached_excel_file(cache_key: str, cache_dir: Path | None = None) -> Path | None:
    """
    Retrieve cached Excel file path.

    Args:
        cache_dir: Directory for cache files
        cache_key: Unique key for cache entry

    Returns:
        Path to cached Excel file if exists, None otherwise
    """
    cache_path = _get_cache_path(cache_dir, cache_key, ".xlsx")
    if not cache_path.exists():
        return None

    return cache_path
