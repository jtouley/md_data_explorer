"""Headless CSV upload → JSON preview (issue #51).

UI-free entrypoint. Uses ``UserDatasetStorage.save_upload`` then reads
columns and row counts from the registered file.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import polars as pl

from clinical_analytics.datasets.demo import DEMO_DATASET_NAME, build_demo_metadata, get_demo_csv_path
from clinical_analytics.ui.storage.user_datasets import UserDatasetStorage


def preview_csv(
    csv_path: Path,
    *,
    upload_dir: Path | None = None,
    preview_rows: int = 5,
    overwrite: bool = True,
) -> dict[str, Any]:
    """
    Register ``csv_path`` via upload storage and return a JSON-serializable preview.

    Args:
        csv_path: Path to a CSV file.
        upload_dir: Optional override for ``UserDatasetStorage`` root.
            When omitted, uses an isolated temp directory (never ``data/uploads``).
        preview_rows: Number of sample rows to include.
        overwrite: Allow replacing an existing dataset with the same name.

    Returns:
        Dict with ``upload_id``, ``dataset_name``, ``columns``, ``row_count``, ``preview``.

    Raises:
        FileNotFoundError: If ``csv_path`` does not exist.
        RuntimeError: If ``save_upload`` fails.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")

    cleanup_dir: Path | None = None
    if upload_dir is None:
        cleanup_dir = Path(tempfile.mkdtemp(prefix="mdde-preview-"))
        upload_dir = cleanup_dir

    try:
        storage = UserDatasetStorage(upload_dir=upload_dir)

        # Use demo metadata when loading the bundled sample; otherwise a minimal mapping.
        # Only the bundled file gets demo metadata; a user's own CSV with the same name keeps its columns.
        if path.resolve() == get_demo_csv_path().resolve():
            metadata = build_demo_metadata()
        else:
            df_head = pl.read_csv(path, n_rows=50)
            cols = df_head.columns
            patient_id = "patient_id" if "patient_id" in cols else cols[0]
            outcome = "outcome" if "outcome" in cols else (cols[1] if len(cols) > 1 else cols[0])
            predictors = [c for c in cols if c not in {patient_id, outcome}]
            metadata = {
                "dataset_name": path.stem,
                "variable_mapping": {
                    "patient_id": patient_id,
                    "outcome": outcome,
                    "time_variables": {"time_zero": None},
                    "predictors": predictors,
                },
            }

        success, message, upload_id = storage.save_upload(
            file_bytes=path.read_bytes(),
            original_filename=path.name,
            metadata=metadata,
            overwrite=overwrite,
        )
        if not success or not upload_id:
            raise RuntimeError(f"Upload failed: {message}")

        stored = storage.raw_dir / f"{upload_id}.csv"
        lf = pl.scan_csv(stored)
        schema = lf.collect_schema()
        row_count = lf.select(pl.len()).collect().item()
        preview = lf.head(preview_rows).collect().to_dicts()

        return {
            "upload_id": upload_id,
            "dataset_name": metadata.get("dataset_name", DEMO_DATASET_NAME),
            "columns": list(schema.names()),
            "row_count": int(row_count),
            "preview": preview,
        }
    finally:
        if cleanup_dir is not None:
            shutil.rmtree(cleanup_dir, ignore_errors=True)


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint: print JSON preview to stdout."""
    parser = argparse.ArgumentParser(description="Load a CSV via upload storage and print a JSON preview.")
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Path to CSV (default: bundled data/demo/sample_cohort.csv)",
    )
    parser.add_argument(
        "--upload-dir",
        type=Path,
        default=None,
        help="Override UserDatasetStorage upload directory (tests use a temp dir).",
    )
    parser.add_argument("--preview-rows", type=int, default=5, help="Sample rows to include (default 5).")
    args = parser.parse_args(argv)

    csv_path = args.csv if args.csv is not None else get_demo_csv_path()
    try:
        payload = preview_csv(csv_path, upload_dir=args.upload_dir, preview_rows=args.preview_rows)
    except (FileNotFoundError, RuntimeError, OSError, pl.exceptions.PolarsError) as exc:
        print(json.dumps({"error": str(exc)}), file=sys.stderr)
        return 1

    print(json.dumps(payload, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
