"""
Pytest configuration and fixtures for clinical analytics tests.
"""

import sys
import tempfile
from pathlib import Path

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
