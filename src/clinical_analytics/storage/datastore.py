"""
DataStore Class - Persistent DuckDB Storage

Manages persistent DuckDB storage for uploaded datasets.

INVARIANT (see versioning.py for full contract):
    Tables are persisted as: {upload_id}_{table_name}_{dataset_version}
    Guarantees: same (upload_id, dataset_version) → storage reuse

MVP scope: Basic save/load operations, lazy frame support.
Deferred to Phase 5+: Table deduplication, storage optimization, compression.
"""

import logging
import re
from pathlib import Path

import duckdb
import polars as pl

logger = logging.getLogger(__name__)


def _sanitize_table_name(name: str) -> str:
    """
    Sanitize table name to SQL-safe identifier.

    Replaces spaces, hyphens, and special characters with underscores.
    Ensures identifier starts with letter/underscore (not number).

    Args:
        name: Original table name (can contain spaces, hyphens, etc.)

    Returns:
        SQL-safe identifier (e.g., "statin_use_deidentified")
    """
    # Replace non-alphanumeric (except underscore) with underscore
    sanitized = re.sub(r"[^0-9a-zA-Z_]+", "_", name).strip("_").lower()

    # Ensure it starts with letter or underscore (not a number)
    if not sanitized or sanitized[0].isdigit():
        sanitized = f"t_{sanitized}" if sanitized else "t"

    return sanitized


class DataStore:
    """
    Manages persistent DuckDB storage.

    Boundary: IO is eager (DuckDB writes), transforms are lazy (return LazyFrame).

    MVP Features:
    - Save tables to persistent DuckDB
    - Load tables as Polars lazy frames
    - List all datasets and tables
    - ACID guarantees via DuckDB

    Deferred to Phase 5+:
    - Table deduplication (storage reuse)
    - Compression strategies
    - Archive old versions
    """

    def __init__(self, db_path: Path | str):
        """
        Initialize persistent DuckDB connection.

        Args:
            db_path: Path to DuckDB database file (will be created if not exists)
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Create persistent connection
        self.conn = duckdb.connect(str(self.db_path))
        logger.info(f"Initialized DataStore with persistent DuckDB at {self.db_path}")

    def save_table(
        self,
        table_name: str,
        data: pl.DataFrame,
        upload_id: str,
        dataset_version: str,
    ) -> None:
        """
        Save table to DuckDB with versioning.

        Enforces persistence invariant: same (upload_id, dataset_version) → same table.

        Table naming: {upload_id}_{table_name}_{dataset_version}

        Args:
            table_name: Original table name (e.g., "patients", "visits")
            data: Polars DataFrame (eager - IO boundary)
            upload_id: Upload identifier
            dataset_version: Content-based dataset version

        Note:
            IO Boundary: Accepts eager DataFrame for DuckDB write.
            Use collect() before calling if you have a LazyFrame.
        """
        # Build qualified table name (sanitize table_name for SQL safety)
        sanitized_table_name = _sanitize_table_name(table_name)
        qualified_name = f"{upload_id}_{sanitized_table_name}_{dataset_version}"

        # Create table in DuckDB (overwrite if exists for idempotency)
        # Use CREATE OR REPLACE for MVP (deferred: IF NOT EXISTS + dedup logic)
        self.conn.execute(f"CREATE OR REPLACE TABLE {qualified_name} AS SELECT * FROM data")

        logger.info(f"Saved table '{table_name}' ({data.height:,} rows) to DuckDB as '{qualified_name}'")

    def load_table(
        self,
        upload_id: str,
        table_name: str,
        dataset_version: str,
    ) -> pl.LazyFrame:
        """
        Load table as Polars lazy frame.

        Boundary: Returns LazyFrame (lazy transform), not eager DataFrame.

        Args:
            upload_id: Upload identifier
            table_name: Original table name
            dataset_version: Content-based dataset version

        Returns:
            Polars LazyFrame for lazy evaluation

        Raises:
            ValueError: If table not found
        """
        # Build qualified table name (sanitize table_name for SQL safety)
        sanitized_table_name = _sanitize_table_name(table_name)
        qualified_name = f"{upload_id}_{sanitized_table_name}_{dataset_version}"

        # Check if table exists
        tables = self.list_tables()
        if qualified_name not in tables:
            raise ValueError(f"Table '{qualified_name}' not found in DuckDB. Available tables: {', '.join(tables)}")

        # Load as lazy frame using pl.scan_sql() for deferred execution
        # Note: pl.scan_sql() was added in Polars 0.20+, fallback to read_database if needed
        lazy_df: pl.LazyFrame
        try:
            # Preferred: scan_sql for true lazy evaluation
            lazy_df = pl.scan_sql(  # type: ignore[attr-defined]
                query=f"SELECT * FROM {qualified_name}",
                connection_uri=str(self.db_path),
            )
        except AttributeError:
            # Fallback: read_database then convert to lazy (Polars < 0.20)
            eager_df = pl.read_database(
                query=f"SELECT * FROM {qualified_name}",
                connection=self.conn,
            )
            lazy_df = eager_df.lazy()

        logger.debug(f"Loaded table '{qualified_name}' as LazyFrame")
        return lazy_df

    def list_tables(self) -> list[str]:
        """
        List all tables in DuckDB.

        Returns:
            List of table names
        """
        result = self.conn.execute("SHOW TABLES").fetchall()
        return [row[0] for row in result]

    def list_datasets(self) -> list[dict]:
        """
        List all datasets (unique upload_ids) in DuckDB.

        Returns:
            List of dicts with upload_id and table_count

        Example:
            [
                {"upload_id": "upload_001", "table_count": 2},
                {"upload_id": "upload_002", "table_count": 1},
            ]
        """
        tables = self.list_tables()

        # Parse upload_id from table names
        # Format: {upload_id}_{sanitized_table_name}_{version}
        # Version can be:
        #   - 16 hex chars (new format: dataset_version from versioning.py)
        #   - Short string (old format: "v1", "v2", etc.)
        upload_ids = {}
        for table in tables:
            upload_id = None

            # Try new format first: version is always exactly 16 hex characters
            if len(table) >= 17:
                # Check if last segment looks like a 16-char hex version
                last_segment = table.rsplit("_", 1)[-1] if "_" in table else ""
                is_16_char_hex = len(last_segment) == 16 and all(c in "0123456789abcdef" for c in last_segment.lower())

                if is_16_char_hex:
                    # Remove version (last 16 chars) and trailing underscore (last 17 chars total)
                    without_version = table[:-17]

                    # Now we have: {upload_id}_{sanitized_table_name}
                    # Upload_id format: user_upload_YYYYMMDD_HHMMSS_8charhash
                    # Use regex to match the complete upload_id pattern
                    upload_id_pattern = r"^user_upload_\d{8}_\d{6}_[a-f0-9]{8}"
                    match = re.match(upload_id_pattern, without_version)
                    if match:
                        upload_id = match.group(0)

            # Fallback: Old format or non-standard upload_ids
            # Use rsplit to remove last two components (table_name + version)
            if upload_id is None:
                parts = table.rsplit("_", 2)  # Split from right, max 2 splits
                if len(parts) >= 1:
                    upload_id = parts[0]  # Everything before last two underscores

            if upload_id:
                if upload_id not in upload_ids:
                    upload_ids[upload_id] = 0
                upload_ids[upload_id] += 1

        return [{"upload_id": uid, "table_count": count} for uid, count in upload_ids.items()]

    def export_to_parquet(
        self,
        upload_id: str,
        table_name: str,
        dataset_version: str,
        parquet_dir: Path,
    ) -> Path:
        """
        Export DuckDB table to Parquet file.

        Phase 3: Columnar Parquet format enables lazy Polars scanning with predicate pushdown.

        Args:
            upload_id: Upload identifier
            table_name: Original table name
            dataset_version: Content-based dataset version
            parquet_dir: Directory to store Parquet files

        Returns:
            Path to created Parquet file

        Raises:
            ValueError: If table not found in DuckDB
        """
        # Build qualified table name (sanitize table_name for SQL safety)
        sanitized_table_name = _sanitize_table_name(table_name)
        qualified_name = f"{upload_id}_{sanitized_table_name}_{dataset_version}"

        # Check if table exists
        tables = self.list_tables()
        if qualified_name not in tables:
            raise ValueError(f"Table '{qualified_name}' not found in DuckDB. Available tables: {', '.join(tables)}")

        # Create parquet directory
        parquet_dir = Path(parquet_dir)
        parquet_dir.mkdir(parents=True, exist_ok=True)

        # Build Parquet file path (same naming as DuckDB table)
        parquet_path = parquet_dir / f"{qualified_name}.parquet"

        # Export using DuckDB's COPY TO (efficient, uses compression)
        self.conn.execute(f"COPY {qualified_name} TO '{parquet_path}' (FORMAT PARQUET, COMPRESSION SNAPPY)")

        logger.info(f"Exported table '{qualified_name}' to Parquet: {parquet_path}")
        return parquet_path

    @staticmethod
    def load_from_parquet(parquet_path: Path) -> pl.LazyFrame:
        """
        Load Parquet file as Polars lazy frame.

        Static method - no DataStore instance required for loading Parquet files.

        Args:
            parquet_path: Path to Parquet file

        Returns:
            Polars LazyFrame for lazy evaluation

        Raises:
            FileNotFoundError: If Parquet file doesn't exist
        """
        parquet_path = Path(parquet_path)
        if not parquet_path.exists():
            raise FileNotFoundError(f"Parquet file not found: {parquet_path}")

        # Use pl.scan_parquet() for true lazy IO (deferred read)
        lazy_df = pl.scan_parquet(parquet_path)
        logger.debug(f"Loaded Parquet file as LazyFrame: {parquet_path}")
        return lazy_df

    def close(self) -> None:
        """Close DuckDB connection."""
        if self.conn:
            self.conn.close()
            logger.debug(f"Closed DuckDB connection to {self.db_path}")
