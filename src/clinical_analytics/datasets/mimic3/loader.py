"""
MIMIC-III Dataset Loader - DuckDB-based SQL extraction.

This module handles loading MIMIC-III data from DuckDB or Postgres databases.
"""

from pathlib import Path
from typing import Any

import duckdb
import polars as pl


class MIMIC3Loader:
    """
    Loader for MIMIC-III clinical database using DuckDB.

    Supports both DuckDB files and PostgreSQL connections.
    """

    def __init__(self, db_path: str | Path | None = None, db_connection=None):
        """
        Initialize MIMIC-III loader.

        Args:
            db_path: Path to DuckDB file
            db_connection: Existing database connection (DuckDB or psycopg2)
        """
        self.db_path = Path(db_path) if db_path else None
        self.db_connection = db_connection
        self.conn: Any = None

    def connect(self) -> None:
        """Establish database connection."""
        if self.db_connection:
            self.conn = self.db_connection
        elif self.db_path:
            if not self.db_path.exists():
                raise FileNotFoundError(f"DuckDB file not found: {self.db_path}")
            self.conn = duckdb.connect(str(self.db_path))
        else:
            raise ValueError("Either db_path or db_connection must be provided")

    def disconnect(self) -> None:
        """Close database connection."""
        if self.conn and self.conn != self.db_connection:
            self.conn.close()
            self.conn = None

    def execute_query(self, query: str) -> pl.DataFrame:
        """
        Execute SQL query and return Polars DataFrame.

        Args:
            query: SQL query string

        Returns:
            Polars DataFrame with query results
        """
        if not self.conn:
            self.connect()

        try:
            # DuckDB supports direct Polars conversion
            result = self.conn.execute(query).pl()
            return pl.DataFrame(result) if result is not None else pl.DataFrame()
        except Exception as e:
            raise RuntimeError(f"Query execution failed: {e}")

    def load_cohort(self, query: str) -> pl.DataFrame:
        """
        Load cohort data using SQL query from config.

        Args:
            query: SQL query string (must be provided, no default)

        Returns:
            Polars DataFrame with cohort data

        Raises:
            ValueError: If query is None or empty
        """
        if not query or not query.strip():
            raise ValueError(
                "SQL query must be provided. Load query from config using datasets.yaml sql_queries.cohort_extraction"
            )

        return self.execute_query(query)

    def check_tables_exist(self) -> dict:
        """
        Check which MIMIC-III tables exist in the database.

        Returns:
            Dictionary mapping table names to existence status
        """
        if not self.conn:
            self.connect()

        if not self.conn:
            raise RuntimeError("Database connection not available")

        required_tables = ["patients", "admissions", "diagnoses_icd", "chartevents"]

        table_status = {}
        for table in required_tables:
            try:
                self.conn.execute(f"SELECT 1 FROM {table} LIMIT 1")
                table_status[table] = True
            except Exception:
                table_status[table] = False

        return table_status

    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()


def load_mimic3_from_duckdb(db_path: str | Path, query: str) -> pl.DataFrame:
    """
    Convenience function to load MIMIC-III data from DuckDB.

    Args:
        db_path: Path to DuckDB file
        query: SQL query string (required)

    Returns:
        Polars DataFrame with cohort data
    """
    with MIMIC3Loader(db_path=db_path) as loader:
        result = loader.load_cohort(query=query)
        return pl.DataFrame(result) if result is not None else pl.DataFrame()
