"""Bundled sample clinical cohort for one-click demo onboarding."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from clinical_analytics.core.resources import demo_file

if TYPE_CHECKING:
    from clinical_analytics.ui.storage.user_datasets import UserDatasetStorage

DEMO_DATASET_NAME = "Sample Clinical Cohort"
DEMO_CSV_NAME = "sample_cohort.csv"

# Predictor columns shipped with the bundled CSV (excluding id/outcome).
DEMO_PREDICTORS = ("age", "sex", "diabetes", "hypertension")


def get_demo_csv_path() -> Path:
    """Absolute path to the bundled sample cohort CSV (shipped inside the package)."""
    return demo_file(DEMO_CSV_NAME)


DEMO_CLASH_SUFFIX = " (sample)"


def demo_display_name(storage: UserDatasetStorage) -> str:
    """The sample's display name: marked as the sample when a user dataset already has the plain name."""
    clash = any(
        meta.get("dataset_name") == DEMO_DATASET_NAME and meta.get("is_demo") is not True
        for meta in storage.list_uploads()
    )
    return f"{DEMO_DATASET_NAME}{DEMO_CLASH_SUFFIX}" if clash else DEMO_DATASET_NAME


def build_demo_metadata(display_name: str = DEMO_DATASET_NAME) -> dict[str, Any]:
    """
    Build upload metadata that skips the mapping wizard.

    Includes a complete ``variable_mapping`` so ``UploadedDataset.get_semantic_layer()``
    can initialize without clinician mapping steps.
    """
    predictors = list(DEMO_PREDICTORS)
    variable_types: dict[str, dict[str, Any]] = {
        "patient_id": {"type": "identifier", "metadata": {}, "missing_count": 0, "missing_pct": 0.0},
        "outcome": {"type": "binary", "metadata": {"unique_values": [0, 1]}, "missing_count": 0, "missing_pct": 0.0},
        "age": {"type": "continuous", "metadata": {}, "missing_count": 0, "missing_pct": 0.0},
        "sex": {
            "type": "categorical",
            "metadata": {"unique_values": ["F", "M"]},
            "missing_count": 0,
            "missing_pct": 0.0,
        },
        "diabetes": {
            "type": "binary",
            "metadata": {"unique_values": [0, 1]},
            "missing_count": 0,
            "missing_pct": 0.0,
        },
        "hypertension": {
            "type": "binary",
            "metadata": {"unique_values": [0, 1]},
            "missing_count": 0,
            "missing_pct": 0.0,
        },
    }
    return {
        "dataset_name": display_name,
        "description": "Bundled synthetic clinical cohort for first-run demos.",
        "is_demo": True,
        "variable_mapping": {
            "patient_id": "patient_id",
            "outcome": "outcome",
            "time_variables": {"time_zero": None},
            "predictors": predictors,
        },
        "variable_types": variable_types,
    }


def load_sample_dataset(
    storage: UserDatasetStorage,
    *,
    overwrite: bool = True,
    csv_path: Path | None = None,
) -> tuple[bool, str, str | None]:
    """
    Register the bundled sample cohort into ``UserDatasetStorage``.

    Args:
        storage: Upload storage target (production or temp for tests).
        overwrite: Allow replacing an existing dataset with the same name.
        csv_path: Optional override path (tests / alternate fixtures).

    Returns:
        ``(success, message, upload_id)`` from ``UserDatasetStorage.save_upload``.

    Raises:
        FileNotFoundError: If the sample cohort CSV is missing.
    """
    path = csv_path if csv_path is not None else get_demo_csv_path()
    if not path.exists():
        raise FileNotFoundError(
            f"Bundled sample cohort CSV not found at {path}. "
            "The installed package is incomplete (clinical_analytics/resources/demo)."
        )

    file_bytes = path.read_bytes()
    return storage.save_upload(
        file_bytes=file_bytes,
        original_filename=path.name,
        metadata=build_demo_metadata(demo_display_name(storage)),
        overwrite=overwrite,
    )
