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


@pytest.fixture
def sample_upload_df() -> pl.DataFrame:
    """Create sample upload Polars DataFrame for testing lazy frames."""
    return pl.DataFrame(
        {
            "patient_id": ["P001", "P002", "P003"],
            "outcome": [0, 1, 0],
            "age": [50, 60, 70],
        }
    )


@pytest.fixture
def sample_upload_metadata() -> dict:
    """Create sample upload metadata for testing."""
    return {
        "upload_timestamp": "2024-01-01T00:00:00",
        "original_filename": "test.csv",
    }


@pytest.fixture
def sample_variable_mapping() -> dict:
    """Create sample variable mapping for uploaded datasets."""
    return {
        "patient_id": "patient_id",
        "outcome": "outcome",
        "predictors": ["age"],
    }


# ============================================================================
# Excel Test Data Fixtures (DRY - Reusable across all Excel tests)
# ============================================================================


@pytest.fixture(scope="module")
def synthetic_dexa_excel_file(tmp_path_factory):
    """
    Create synthetic DEXA-like Excel file with headers in row 0 (standard format).

    Mimics: data/raw/LWTest/de-identified DEXA.xlsx
    - Headers in first row
    - 25-50 rows of clinical data
    - Mixed data types (strings, numbers, dates)

    Returns:
        Path to Excel file
    """
    import pandas as pd

    data = {
        "Race": ["Black or African-American"] * 30 + ["White"] * 20,
        "Gender": ["Male", "Female"] * 25,
        "Age": list(range(40, 90)),
        "Had DEXA Scan? Yes: 1 No: 2": [1] * 35 + [2] * 15,
        "Results of DEXA? 1: Normal 2: Osteopenia 3: Osteoporosis": [1, 2, 3] * 16 + [1, 2],
        "DEXA Score (T score)": [-2.5 + i * 0.1 for i in range(50)],
        "DEXA Score (Z score)": [-1.8 + i * 0.08 for i in range(50)],
        "CD4 Count": list(range(200, 1200, 20)),
        "Viral Load": ["<20"] * 30 + ["40", "120", "240"] * 6 + ["<20"] * 2,
        "Prior Tenofovir (TDF) use? 1: Yes 2: No 3: Unknown": [1, 2, 3] * 16 + [1, 2],
    }

    df = pd.DataFrame(data)
    excel_path = tmp_path_factory.mktemp("excel_data") / "synthetic_dexa.xlsx"
    df.to_excel(excel_path, index=False, engine="openpyxl")

    return excel_path


@pytest.fixture(scope="module")
def synthetic_statin_excel_file(tmp_path_factory):
    """
    Create synthetic Statin-like Excel file with empty row 1, headers in row 2.

    Mimics: data/raw/LWTest/Statin use - deidentified.xlsx
    - Empty first row
    - Headers in row 2 (index 1)
    - 25-50 rows of clinical data
    - Complex multi-column structure (27 columns)

    Returns:
        Path to Excel file
    """
    import pandas as pd

    # All columns must have exactly 50 rows
    n_rows = 50
    data = {
        "Race": (["Black or African-American"] * 30 + ["White", "Asian"] * 10)[:n_rows],
        "Gender": (["Male", "Female", "Transgender MtF"] * 17)[:n_rows],
        "Age": list(range(40, 90))[:n_rows],
        "Most Recent VL copies/mL": (["<20"] * 25 + ["40", "120", "240", "1740"] * 6 + ["<20"])[:n_rows],
        "Most Recent CD4 /uL": list(range(185, 185 + n_rows * 26, 26))[:n_rows],
        "Current Regimen 1: Biktarvy 2: Symtuza 3: Triumeq": ([1, 2, 3, 4, 5] * 10)[:n_rows],
        "Regimen (if 9 or other)": ([None] * 45 + ["Biktarvy, Darunavir/Ritonavir"] * 5)[:n_rows],
        "Total Cholesterol mg/dL": list(range(85, 85 + n_rows * 3, 3))[:n_rows],
        "LDL mg/dL": list(range(25, 25 + n_rows * 2, 2))[:n_rows],
        "HDL mg/dL": list(range(33, 33 + n_rows))[:n_rows],
        "Triglycerides mg/dL": list(range(5, 5 + n_rows * 5, 5))[:n_rows],
        "Systolic blood pressure": list(range(100, 100 + n_rows))[:n_rows],
        "Diastolic blood pressure": list(range(60, 60 + n_rows))[:n_rows],
        "Diabetes 1: Yes 2: No": ([1, 2] * 25)[:n_rows],
        "HTN 1: Yes 2: No": ([1, 2] * 25)[:n_rows],
    }

    df_data = pd.DataFrame(data)

    # Create Excel with empty first row, then headers, then data
    excel_path = tmp_path_factory.mktemp("excel_data") / "synthetic_statin.xlsx"
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        # Write empty first row
        empty_row = pd.DataFrame([[""] * len(df_data.columns)])
        empty_row.to_excel(writer, index=False, header=False, startrow=0)

        # Write headers in row 2 (index 1)
        headers_df = pd.DataFrame([df_data.columns])
        headers_df.to_excel(writer, index=False, header=False, startrow=1)

        # Write data starting from row 3 (index 2)
        df_data.to_excel(writer, index=False, header=False, startrow=2)

    return excel_path


@pytest.fixture(scope="module")
def synthetic_complex_excel_file(tmp_path_factory):
    """
    Create complex Excel file with metadata rows before headers.

    - Row 1: Metadata/notes (mostly empty, one cell with "Units")
    - Row 2: Actual headers
    - Row 3+: Data

    Returns:
        Path to Excel file
    """
    import pandas as pd

    # All columns must have exactly 45 rows
    n_rows = 45
    data = {
        "Race": (["Black or African-American"] * 25 + ["White"] * 20)[:n_rows],
        "Gender": (["Male", "Female"] * 23)[:n_rows],
        "Age": list(range(35, 80))[:n_rows],
        "Viral Load copies/mL": (["<20"] * 30 + ["40", "120", "240"] * 5)[:n_rows],
        "CD4 Count /uL": list(range(200, 200 + n_rows * 26, 26))[:n_rows],
        "Total Cholesterol mg/dL": list(range(100, 100 + n_rows * 3, 3))[:n_rows],
        "LDL mg/dL": list(range(30, 30 + n_rows * 2, 2))[:n_rows],
        "HDL mg/dL": list(range(35, 35 + n_rows))[:n_rows],
        "Triglycerides mg/dL": list(range(50, 50 + n_rows * 5, 5))[:n_rows],
        "Systolic BP mmHg": list(range(100, 100 + n_rows))[:n_rows],
        "Diastolic BP mmHg": list(range(60, 60 + n_rows))[:n_rows],
        "Diabetes Yes:1 No:2": ([1, 2] * 23)[:n_rows],
    }

    df_data = pd.DataFrame(data)

    # Create Excel with metadata row, then headers, then data
    excel_path = tmp_path_factory.mktemp("excel_data") / "synthetic_complex.xlsx"
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        # Write metadata row (row 1) - mostly empty, one cell with "Units"
        metadata_row = [None] * len(df_data.columns)
        metadata_row[7] = "Units"  # Put "Units" in column 8
        metadata_df = pd.DataFrame([metadata_row])
        metadata_df.to_excel(writer, index=False, header=False, startrow=0)

        # Write headers in row 2 (index 1)
        headers_df = pd.DataFrame([df_data.columns])
        headers_df.to_excel(writer, index=False, header=False, startrow=1)

        # Write data starting from row 3 (index 2)
        df_data.to_excel(writer, index=False, header=False, startrow=2)

    return excel_path


# ============================================================================
# Consolidated Test Fixtures (DRY - Single Source of Truth)
# ============================================================================


@pytest.fixture
def sample_cohort():
    """
    Standard Polars cohort fixture used across all tests.

    Returns Polars DataFrame with:
    - patient_id: Integer IDs [1, 2, 3, 4, 5]
    - outcome: Binary outcome [0, 1, 0, 1, 0]
    - age: Age values [45, 62, 38, 71, 55]
    - treatment: Treatment arm ["A", "A", "B", "B", "A"]
    """
    return pl.DataFrame(
        {
            "patient_id": [1, 2, 3, 4, 5],
            "outcome": [0, 1, 0, 1, 0],
            "age": [45, 62, 38, 71, 55],
            "treatment": ["A", "A", "B", "B", "A"],
        }
    )


@pytest.fixture
def mock_cohort():
    """
    Pandas cohort fixture for Streamlit UI tests.

    Returns Pandas DataFrame with UnifiedCohort schema:
    - patient_id: String IDs ["P0" - "P19"]
    - time_zero: Date range
    - outcome: Binary [0, 1] alternating
    - outcome_label: ["alive", "dead"] alternating
    - Predictors: age, score, group
    """
    import pandas as pd

    from clinical_analytics.core.schema import UnifiedCohort

    return pd.DataFrame(
        {
            UnifiedCohort.PATIENT_ID: [f"P{i}" for i in range(20)],
            UnifiedCohort.TIME_ZERO: pd.date_range("2023-01-01", periods=20),
            UnifiedCohort.OUTCOME: [0, 1] * 10,
            UnifiedCohort.OUTCOME_LABEL: ["alive", "dead"] * 10,
            # Predictors
            "age": [25, 30, 35, 40] * 5,
            "score": [1.5, 2.5, 3.5, 4.5] * 5,
            "group": ["A", "B"] * 10,
        }
    )


@pytest.fixture
def sample_context():
    """
    Direct AnalysisContext fixture for backward compatibility.

    Returns AnalysisContext with DESCRIBE intent and confidence=0.9.
    This is a simple, ready-to-use fixture.
    """
    from clinical_analytics.ui.components.question_engine import AnalysisContext, AnalysisIntent

    context = AnalysisContext(
        inferred_intent=AnalysisIntent.DESCRIBE,
        primary_variable="all",
    )
    context.confidence = 0.9
    return context


@pytest.fixture
def low_confidence_context():
    """
    AnalysisContext fixture with low confidence (0.4).

    Returns AnalysisContext configured for low-confidence feedback testing:
    - inferred_intent: COMPARE_GROUPS
    - primary_variable: "mortality"
    - grouping_variable: "treatment_arm"
    - confidence: 0.4 (below auto-execute threshold)
    - match_suggestions: Dictionary with collision suggestions
    """
    from clinical_analytics.ui.components.question_engine import AnalysisContext, AnalysisIntent

    context = AnalysisContext(
        inferred_intent=AnalysisIntent.COMPARE_GROUPS,
        primary_variable="mortality",
        grouping_variable="treatment_arm",
        research_question="compare mortality by treatment",
        match_suggestions={"mortality": ["mortality", "death", "outcome"]},
    )
    context.confidence = 0.4  # Low confidence
    return context


@pytest.fixture
def high_confidence_context():
    """
    AnalysisContext fixture with high confidence (0.9).

    Returns AnalysisContext configured for high-confidence auto-execute testing:
    - inferred_intent: COMPARE_GROUPS
    - primary_variable: "mortality"
    - grouping_variable: "treatment_arm"
    - confidence: 0.9 (above auto-execute threshold)
    """
    from clinical_analytics.ui.components.question_engine import AnalysisContext, AnalysisIntent

    context = AnalysisContext(
        inferred_intent=AnalysisIntent.COMPARE_GROUPS,
        primary_variable="mortality",
        grouping_variable="treatment_arm",
        research_question="compare mortality by treatment",
    )
    context.confidence = 0.9  # High confidence
    return context


@pytest.fixture
def mock_semantic_layer():
    """
    Factory fixture for creating mock SemanticLayer instances.

    Returns a function that creates a MagicMock with configurable column mappings.

    Usage:
        def test_example(mock_semantic_layer):
            mock = mock_semantic_layer(columns={
                "mortality": "mortality",
                "treatment": "treatment_arm"
            })
    """
    from unittest.mock import MagicMock

    def _make(columns=None, collision_suggestions=None):
        mock = MagicMock()
        default_columns = {
            "mortality": "mortality",
            "treatment": "treatment_arm",
            "age": "age",
        }
        mock.get_column_alias_index.return_value = columns or default_columns
        mock.get_collision_suggestions.return_value = collision_suggestions
        mock.get_collision_warnings.return_value = set()
        mock._normalize_alias = lambda x: x.lower().replace(" ", "_")
        return mock

    return _make


# ============================================================================
# Analysis Context Fixtures (for compute tests)
# ============================================================================


@pytest.fixture
def sample_numeric_df():
    """Create sample Polars DataFrame with numeric columns."""
    return pl.DataFrame(
        {
            "age": [25, 30, 35, 40, 45, 50, 55, 60],
            "score": [10, 20, 30, 40, 50, 60, 70, 80],
            "value": [1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5],
        }
    )


@pytest.fixture
def sample_categorical_df():
    """Create sample Polars DataFrame with categorical columns."""
    return pl.DataFrame(
        {
            "category": ["A", "B", "A", "B", "A", "B", "A", "B"],
            "status": ["active", "inactive", "active", "inactive", "active", "inactive", "active", "inactive"],
        }
    )


@pytest.fixture
def sample_mixed_df():
    """Create sample Polars DataFrame with mixed column types."""
    return pl.DataFrame(
        {
            "age": [25, 30, 35, 40, 45],
            "category": ["A", "B", "A", "B", "A"],
            "score": [10, 20, 30, 40, 50],
        }
    )


@pytest.fixture
def sample_context_describe():
    """Create AnalysisContext for descriptive analysis."""
    from clinical_analytics.ui.components.question_engine import AnalysisContext, AnalysisIntent

    context = AnalysisContext()
    context.inferred_intent = AnalysisIntent.DESCRIBE
    context.primary_variable = "all"
    return context


@pytest.fixture
def sample_context_compare():
    """Create AnalysisContext for comparison analysis."""
    from clinical_analytics.ui.components.question_engine import AnalysisContext, AnalysisIntent

    context = AnalysisContext()
    context.inferred_intent = AnalysisIntent.COMPARE_GROUPS
    context.primary_variable = "score"
    context.grouping_variable = "category"
    return context


@pytest.fixture
def sample_context_predictor():
    """Create AnalysisContext for predictor analysis."""
    from clinical_analytics.ui.components.question_engine import AnalysisContext, AnalysisIntent

    context = AnalysisContext()
    context.inferred_intent = AnalysisIntent.FIND_PREDICTORS
    context.primary_variable = "outcome"
    context.predictor_variables = ["age", "score"]
    return context


@pytest.fixture
def sample_context_survival():
    """Create AnalysisContext for survival analysis."""
    from clinical_analytics.ui.components.question_engine import AnalysisContext, AnalysisIntent

    context = AnalysisContext()
    context.inferred_intent = AnalysisIntent.EXAMINE_SURVIVAL
    context.time_variable = "time"
    context.event_variable = "event"
    return context


@pytest.fixture
def sample_context_relationship():
    """Create AnalysisContext for relationship analysis."""
    from clinical_analytics.ui.components.question_engine import AnalysisContext, AnalysisIntent

    context = AnalysisContext()
    context.inferred_intent = AnalysisIntent.EXPLORE_RELATIONSHIPS
    context.predictor_variables = ["age", "score", "value"]
    return context


# ============================================================================
# Mock Session State Fixture
# ============================================================================


@pytest.fixture
def mock_session_state():
    """Mock Streamlit session_state for UI tests."""
    return {}


@pytest.fixture
def sample_context_direct():
    """
    Direct AnalysisContext fixture (not a factory) for tests that expect it ready-to-use.

    Returns AnalysisContext with default DESCRIBE intent and confidence=0.9.
    Use this when you need a simple context object without customization.

    For customization, use the sample_context() factory fixture instead.
    """
    from clinical_analytics.ui.components.question_engine import AnalysisContext, AnalysisIntent

    context = AnalysisContext(
        inferred_intent=AnalysisIntent.DESCRIBE,
        primary_variable="all",
    )
    context.confidence = 0.9
    return context
