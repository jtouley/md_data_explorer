"""
Generic factory functions for creating test fixtures (DRY/SOLID).

Provides extensible factories for:
- Excel files (with configurable layout)
- Large CSV strings (with configurable columns)
- ZIP files (with configurable CSV files)

All factories support caching via tests.fixtures.cache.
"""

import io
import sys
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd
import polars as pl
import pytest

# Add tests to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from fixtures.cache import (
    cache_excel_file,
    get_cache_dir,
    get_cached_excel_file,
    hash_dataframe,
    hash_file,
)


def _create_synthetic_excel_file(
    tmp_path_factory: pytest.TempPathFactory,
    data: dict[str, list],
    filename: str,
    excel_config: dict[str, Any] | None = None,
) -> Path:
    """
    Generic factory for creating synthetic Excel files with caching.

    Args:
        tmp_path_factory: Pytest tmp_path_factory fixture
        data: Dictionary of column_name -> list of values
        filename: Output filename
        excel_config: Configuration for Excel layout:
            - header_row: int (default: 0) - Row index for headers
            - metadata_rows: list[dict] | None - Metadata rows before headers
                Each dict: {"row_index": int, "cells": list[str | None]}
            - use_dataframe_hash: bool (default: True) - Use DataFrame hash for cache key

    Returns:
        Path to Excel file (cached if available)
    """
    import pandas as pd

    config = excel_config or {}
    header_row = config.get("header_row", 0)
    metadata_rows = config.get("metadata_rows", None)
    use_dataframe_hash = config.get("use_dataframe_hash", True)

    # Create DataFrame from data
    df = pd.DataFrame(data)

    # Generate cache key
    cache_dir = get_cache_dir()
    if use_dataframe_hash:
        df_polars = pl.from_pandas(df)
        cache_key = hash_dataframe(df_polars)
    else:
        # For complex layouts, we need to hash the final file
        # Create temp file first to compute hash
        temp_path = tmp_path_factory.mktemp("excel_data") / f"temp_{filename}"
        _write_excel_with_layout(temp_path, df, header_row, metadata_rows)
        cache_key = hash_file(temp_path)
        temp_path.unlink()

    # Check cache first
    cached_file = get_cached_excel_file(cache_key, cache_dir)
    if cached_file is not None:
        return cached_file

    # Generate file if not cached
    excel_path = tmp_path_factory.mktemp("excel_data") / filename
    _write_excel_with_layout(excel_path, df, header_row, metadata_rows)

    # Cache the generated file
    cache_excel_file(excel_path, cache_key, cache_dir)

    return excel_path


def _write_excel_with_layout(
    excel_path: Path,
    df: pd.DataFrame,
    header_row: int,
    metadata_rows: list[dict[str, Any]] | None,
) -> None:
    """Write DataFrame to Excel with specified layout."""
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        current_row = 0

        # Write metadata rows if specified
        if metadata_rows:
            for metadata_row in metadata_rows:
                row_index = metadata_row.get("row_index", current_row)
                cells = metadata_row.get("cells", [])
                # Pad cells to match DataFrame width
                while len(cells) < len(df.columns):
                    cells.append(None)
                metadata_df = pd.DataFrame([cells[: len(df.columns)]])
                metadata_df.to_excel(writer, index=False, header=False, startrow=row_index)
                current_row = max(current_row, row_index + 1)

        # Write headers at specified row
        if header_row >= current_row:
            headers_df = pd.DataFrame([df.columns])
            headers_df.to_excel(writer, index=False, header=False, startrow=header_row)
            current_row = header_row + 1

        # Write data starting after headers
        df.to_excel(writer, index=False, header=False, startrow=current_row)


def make_large_csv(columns: dict[str, Callable[[int], str]], num_records: int = 1000000) -> str:
    """
    Generic factory for generating large CSV strings with caching.

    Args:
        columns: Dictionary of column_name -> function(i) that generates value for row i
        num_records: Number of records to generate (default: 1,000,000)

    Returns:
        CSV string with header and data rows

    Example:
        columns = {
            "patient_id": lambda i: f"P{i:06d}",
            "age": lambda i: str(20 + i % 100),
        }
        csv = make_large_csv(columns, num_records=1000)
    """
    # Generate header
    header = ",".join(columns.keys())

    # Generate rows
    rows = []
    for i in range(num_records):
        row_values = [str(fn(i)) for fn in columns.values()]
        rows.append(",".join(row_values))

    return header + "\n" + "\n".join(rows)


def make_large_zip(csv_files: dict[str, str]) -> bytes:
    """
    Generic factory for creating ZIP files with CSV content.

    Args:
        csv_files: Dictionary of filename -> CSV string content

    Returns:
        ZIP file bytes

    Example:
        csv_files = {
            "patients.csv": "patient_id,age\\nP001,45",
            "admissions.csv": "patient_id,date\\nP001,2020-01-01",
        }
        zip_bytes = make_large_zip(csv_files)
    """
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zip_file:
        for filename, csv_content in csv_files.items():
            zip_file.writestr(filename, csv_content)
    zip_buffer.seek(0)
    return zip_buffer.getvalue()
