"""
App Utilities - Session Recovery and Startup Functions

Provides utility functions for app initialization and session management.
"""

import logging
from pathlib import Path

from clinical_analytics.storage.datastore import DataStore
from clinical_analytics.ui.storage.user_datasets import UserDatasetStorage

logger = logging.getLogger(__name__)


def restore_datasets(storage: UserDatasetStorage, db_path: Path | str) -> list[dict]:
    """
    Detect existing datasets and return metadata for session recovery.

    Queries persistent DuckDB for available datasets and loads their metadata.
    Sorts by creation time (newest first) for better UX.

    Args:
        storage: UserDatasetStorage instance
        db_path: Path to persistent DuckDB database

    Returns:
        List of dataset metadata dicts, sorted by created_at (descending)

    Example:
        >>> restored = restore_datasets(storage, "data/analytics.duckdb")
        >>> if restored:
        ...     print(f"Found {len(restored)} previous datasets")
        ...     latest = restored[0]
        ...     print(f"Latest: {latest['dataset_name']}")
    """
    db_path = Path(db_path)

    # Return empty if DuckDB doesn't exist (first run)
    if not db_path.exists():
        logger.info("No persistent DuckDB found, starting fresh")
        return []

    try:
        # Connect to persistent DuckDB
        datastore = DataStore(db_path)

        # Get all unique upload_ids from DuckDB
        datasets_info = datastore.list_datasets()
        datastore.close()

        if not datasets_info:
            logger.info("No datasets found in persistent DuckDB")
            return []

        logger.info(f"Found {len(datasets_info)} datasets in persistent DuckDB")

        # Load metadata for each upload_id
        restored = []
        for dataset_info in datasets_info:
            upload_id = dataset_info["upload_id"]

            # Try to load metadata JSON
            metadata = storage.get_upload_metadata(upload_id)
            if not metadata:
                logger.warning(
                    f"Metadata missing for upload {upload_id}, skipping dataset. "
                    "DuckDB table exists but metadata JSON not found."
                )
                continue

            # Add to restored list
            restored.append(metadata)
            logger.debug(f"Restored dataset: {metadata.get('dataset_name', upload_id)} (upload_id: {upload_id})")

        # Sort by created_at (descending - newest first)
        # Handle missing created_at gracefully
        restored.sort(
            key=lambda d: d.get("created_at", ""),
            reverse=True,
        )

        logger.info(f"Successfully restored {len(restored)} datasets")
        return restored

    except Exception as e:
        logger.error(f"Error restoring datasets from DuckDB: {e}")
        import traceback

        logger.debug(traceback.format_exc())
        return []
