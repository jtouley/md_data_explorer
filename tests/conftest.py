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
