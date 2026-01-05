"""
Pytest fixtures for loader module tests.
"""

import pytest

from clinical_analytics.ui.storage.user_datasets import UserDatasetStorage


@pytest.fixture
def upload_storage(tmp_path):
    """Create UserDatasetStorage with temp directory."""
    return UserDatasetStorage(upload_dir=tmp_path)
