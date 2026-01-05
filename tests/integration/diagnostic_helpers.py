"""Diagnostic helper functions for integration tests.

These functions dump state at integration points to help diagnose failures.
"""

import json
from collections.abc import Callable
from typing import Any


def dump_storage_state(storage: Any, upload_id: str) -> dict[str, Any]:
    """Dump storage state for diagnostics.

    Args:
        storage: UserDatasetStorage instance
        upload_id: Upload identifier

    Returns:
        Dictionary with storage state information
    """
    state: dict[str, Any] = {
        "upload_id": upload_id,
        "metadata_dir": str(storage.metadata_dir),
        "raw_dir": str(storage.raw_dir),
    }

    # Get metadata if available
    metadata = storage.get_upload_metadata(upload_id)
    if metadata:
        state["metadata"] = {
            "dataset_name": metadata.get("dataset_name"),
            "dataset_version": metadata.get("dataset_version"),
            "table_count": metadata.get("table_count"),
            "row_count": metadata.get("row_count"),
            "column_count": metadata.get("column_count"),
            "columns": metadata.get("columns", [])[:10],  # First 10 columns
        }

    # Check file existence
    metadata_path = storage.metadata_dir / f"{upload_id}.json"
    state["metadata_file_exists"] = metadata_path.exists()
    if metadata_path.exists():
        state["metadata_file_size"] = metadata_path.stat().st_size

    # Check for unified cohort CSV
    unified_csv = storage.upload_dir / "unified" / f"{upload_id}_unified.csv"
    state["unified_csv_exists"] = unified_csv.exists()
    if unified_csv.exists():
        state["unified_csv_size"] = unified_csv.stat().st_size

    return state


def dump_duckdb_state(conn: Any) -> dict[str, Any]:
    """Dump DuckDB connection state for diagnostics.

    Args:
        conn: DuckDB connection (via SemanticLayer.con.con)

    Returns:
        Dictionary with DuckDB state information
    """
    state: dict[str, Any] = {}

    try:
        # Get list of tables
        tables_result = conn.execute("SHOW TABLES").fetchall()
        state["tables"] = [row[0] if isinstance(row, list | tuple) else str(row) for row in tables_result]

        # Get schema for each table
        schemas = {}
        for table_name in state["tables"]:
            try:
                schema_result = conn.execute(f"DESCRIBE {table_name}").fetchall()
                schemas[table_name] = [
                    {"column": row[0], "type": row[1]} if isinstance(row, list | tuple) else str(row)
                    for row in schema_result
                ]
            except Exception as e:
                schemas[table_name] = f"Error: {e}"

        state["schemas"] = schemas
    except Exception as e:
        state["error"] = str(e)

    return state


def dump_semantic_layer_state(semantic: Any) -> dict[str, Any]:
    """Dump semantic layer state for diagnostics.

    Args:
        semantic: SemanticLayer instance

    Returns:
        Dictionary with semantic layer state information
    """
    state: dict[str, Any] = {
        "dataset_name": semantic.dataset_name,
        "upload_id": semantic.upload_id,
        "dataset_version": semantic.dataset_version,
    }

    # Get dataset info
    try:
        dataset_info = semantic.get_dataset_info()
        if dataset_info:
            state["dataset_info"] = {
                "row_count": dataset_info.get("row_count"),
                "column_count": dataset_info.get("column_count"),
                "columns": dataset_info.get("columns", [])[:10],  # First 10 columns
            }
    except Exception as e:
        state["dataset_info_error"] = str(e)

    # Get config info
    if hasattr(semantic, "config"):
        config = semantic.config
        state["config"] = {
            "has_metrics": "metrics" in config if config else False,
            "has_dimensions": "dimensions" in config if config else False,
            "has_filters": "default_filters" in config if config else False,
        }

    # Get base view info
    try:
        base_view = semantic.get_base_view()
        if base_view:
            state["base_view"] = {
                "columns": list(base_view.columns)[:10],  # First 10 columns
            }
    except Exception as e:
        state["base_view_error"] = str(e)

    return state


def assert_with_diagnostics(condition: bool, diagnostic_fn: Callable[..., dict[str, Any]], **kwargs: Any) -> None:
    """Assert with automatic diagnostic dump on failure.

    Args:
        condition: Boolean condition to assert
        diagnostic_fn: Function that returns diagnostic dict
        **kwargs: Arguments to pass to diagnostic_fn

    Raises:
        AssertionError: If condition is False, after printing diagnostics
    """
    if not condition:
        print("\n" + "=" * 80)
        print("ASSERTION FAILED - DIAGNOSTIC DUMP:")
        print("=" * 80)
        try:
            diagnostics = diagnostic_fn(**kwargs)
            print(json.dumps(diagnostics, indent=2, default=str))
        except Exception as e:
            print(f"Error generating diagnostics: {e}")
        print("=" * 80 + "\n")
    assert condition
