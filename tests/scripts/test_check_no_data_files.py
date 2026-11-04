"""Tests for scripts/check_no_data_files.py (tracked-data guard)."""

import importlib.util
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "check_no_data_files.py"
_spec = importlib.util.spec_from_file_location("check_no_data_files", _SCRIPT)
check_no_data_files = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(check_no_data_files)


@pytest.mark.parametrize(
    ("path", "flagged"),
    [
        ("data/analytics.duckdb", True),
        ("data/clinical_analytics.db", True),
        ("data/uploads/raw/upl_1.csv", True),
        ("exports/cohort.parquet", True),
        ("notes/patients.xlsx", True),
        ("scratch/extract.csv", True),
        ("src/clinical_analytics/resources/demo/sample_cohort.csv", False),
        ("data/demo/sample_cohort.csv", True),
        ("src/clinical_analytics/resources/config/extract.csv", True),
        ("data/dictionaries/Codebook v1.0.pdf", True),
        ("tests/fixtures/synthetic/cohort.csv", False),
        ("src/clinical_analytics/core/semantic.py", False),
    ],
    ids=[
        "duckdb_store",
        "session_db",
        "uploaded_csv",
        "parquet_anywhere",
        "excel_anywhere",
        "csv_outside_allowlist",
        "synthetic_demo_csv",
        "old_demo_location_under_data",
        "csv_in_shipped_config",
        "dictionary_pdf_under_data",
        "synthetic_test_fixture",
        "source_code",
    ],
)
def test_violations_path_flaggedUnlessAllowlisted(path, flagged):
    # Act
    result = check_no_data_files.violations([path])

    # Assert
    assert (result == [path]) is flagged
