"""
Pytest configuration and fixtures for clinical analytics tests.
"""

import io
import sys
import tempfile
import zipfile
from pathlib import Path

import polars as pl
import pytest
import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.fixture(scope="session")
def project_root():
    """Return project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture(scope="session")
def test_data_dir(project_root):
    """Return test data directory."""
    return project_root / "data" / "raw"


@pytest.fixture
def temp_config_file():
    """Create a temporary config file for testing."""
    config_data = {
        "test_dataset": {
            "name": "test",
            "display_name": "Test Dataset",
            "status": "available",
            "init_params": {"source_path": "data/raw/test/test.csv"},
            "column_mapping": {"id": "patient_id", "date": "time_zero", "result": "outcome"},
            "analysis": {
                "default_outcome": "outcome",
                "default_predictors": ["age", "sex"],
                "categorical_variables": ["sex"],
            },
        }
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(config_data, f)
        temp_path = Path(f.name)

    yield temp_path

    # Cleanup
    temp_path.unlink()


@pytest.fixture
def sample_covid_ms_path(test_data_dir):
    """Return path to COVID-MS test data if available."""
    path = test_data_dir / "covid_ms" / "GDSI_OpenDataset_Final.csv"
    return path if path.exists() else None


@pytest.fixture
def sample_sepsis_path(test_data_dir):
    """Return path to Sepsis test data if available."""
    path = test_data_dir / "sepsis"
    return path if path.exists() else None


@pytest.fixture(scope="module")
def ask_questions_page():
    """
    Import the Ask Questions page module.

    Uses importlib because the filename contains an emoji.
    """
    import importlib.util
    import sys
    from pathlib import Path

    # Add src to path
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root / "src"))

    # Import from the page file (has emoji in name, so use importlib)
    page_path = project_root / "src" / "clinical_analytics" / "ui" / "pages" / "3_💬_Ask_Questions.py"
    spec = importlib.util.spec_from_file_location("ask_questions_page", page_path)
    ask_questions_page = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ask_questions_page)

    return ask_questions_page


# ============================================================================
# Test Data Generation Fixtures (DRY - Single Source of Truth)
# ============================================================================


@pytest.fixture
def large_test_data_csv(num_records: int = 1000000) -> str:
    """
    Generate large CSV data string for testing (meets 1KB minimum requirement).

    Args:
        num_records: Number of records to generate (default: 1,000,000)

    Returns:
        CSV string with patient_id and age columns
    """
    return "patient_id,age\n" + "\n".join([f"P{i:06d},{20 + i % 100}" for i in range(num_records)])


@pytest.fixture
def large_patients_csv(num_records: int = 1000000) -> str:
    """Generate large patients CSV with patient_id, age, sex columns."""
    return "patient_id,age,sex\n" + "\n".join(
        [f"P{i:06d},{20 + i % 100},{['M', 'F'][i % 2]}" for i in range(num_records)]
    )


@pytest.fixture
def large_admissions_csv(num_records: int = 1000000) -> str:
    """Generate large admissions CSV with patient_id and date columns."""
    return "patient_id,date\n" + "\n".join([f"P{i:06d},2020-01-{1 + i % 30:02d}" for i in range(num_records)])


@pytest.fixture
def large_admissions_with_admission_date_csv(num_records: int = 1000000) -> str:
    """Generate large admissions CSV with patient_id and admission_date columns."""
    return "patient_id,admission_date\n" + "\n".join([f"P{i:06d},2020-01-{1 + i % 30:02d}" for i in range(num_records)])


@pytest.fixture
def large_admissions_with_discharge_csv(num_records: int = 1000000) -> str:
    """Generate large admissions CSV with patient_id, admission_date, discharge_date columns."""
    return "patient_id,admission_date,discharge_date\n" + "\n".join(
        [f"P{i:06d},2020-01-{1 + i % 30:02d},2020-01-{5 + i % 30:02d}" for i in range(num_records)]
    )


@pytest.fixture
def large_diagnoses_csv(num_records: int = 1000000) -> str:
    """Generate large diagnoses CSV with patient_id, icd_code, diagnosis columns."""
    return "patient_id,icd_code,diagnosis\n" + "\n".join([f"P{i:06d},E11.9,Diabetes" for i in range(num_records)])


@pytest.fixture
def large_zip_with_csvs(large_patients_csv, large_admissions_csv) -> bytes:
    """
    Create a ZIP file containing large CSV files for testing.

    Returns:
        ZIP file bytes with patients.csv and admissions.csv
    """
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zip_file:
        zip_file.writestr("patients.csv", large_patients_csv)
        zip_file.writestr("admissions.csv", large_admissions_csv)
    zip_buffer.seek(0)
    return zip_buffer.getvalue()


@pytest.fixture
def large_zip_with_three_tables(large_patients_csv, large_admissions_csv, large_diagnoses_csv) -> bytes:
    """
    Create a ZIP file containing three large CSV files for testing.

    Returns:
        ZIP file bytes with patients.csv, admissions.csv, and diagnoses.csv
    """
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zip_file:
        zip_file.writestr("patients.csv", large_patients_csv)
        zip_file.writestr("admissions.csv", large_admissions_csv)
        zip_file.writestr("diagnoses.csv", large_diagnoses_csv)
    zip_buffer.seek(0)
    return zip_buffer.getvalue()


@pytest.fixture
def sample_patients_df() -> pl.DataFrame:
    """Create sample patients Polars DataFrame for testing."""
    return pl.DataFrame(
        {
            "patient_id": ["P001", "P002", "P003"],
            "age": [45, 62, 38],
            "sex": ["M", "F", "M"],
        }
    )


@pytest.fixture
def sample_admissions_df() -> pl.DataFrame:
    """Create sample admissions Polars DataFrame for testing."""
    return pl.DataFrame(
        {
            "patient_id": ["P001", "P002"],
            "admission_date": ["2020-01-01", "2020-02-01"],
            "discharge_date": ["2020-01-05", "2020-02-10"],
        }
    )
