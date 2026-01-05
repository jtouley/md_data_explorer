"""
Tests for pre-commit hook: check_test_fixtures.py

Tests ensure that:
1. Hook detects inline UserDatasetStorage creation
2. Hook detects duplicate DataFrame creation patterns
3. Hook allows fixtures and doesn't false-positive
"""

import importlib.util
import sys
from pathlib import Path

# Load the script module directly
script_path = Path(__file__).parent.parent.parent / "scripts" / "check_test_fixtures.py"
spec = importlib.util.spec_from_file_location("check_test_fixtures", script_path)
check_test_fixtures = importlib.util.module_from_spec(spec)
sys.modules["check_test_fixtures"] = check_test_fixtures
spec.loader.exec_module(check_test_fixtures)

check_file = check_test_fixtures.check_file
find_duplicate_dataframe_creation = check_test_fixtures.find_duplicate_dataframe_creation
find_inline_storage_creation = check_test_fixtures.find_inline_storage_creation


class TestCheckTestFixtures:
    """Test suite for pre-commit hook script."""

    def test_find_inline_storage_creation_detects_violation(self):
        """Test that inline UserDatasetStorage creation is detected."""
        # Arrange: Create test file with inline storage creation
        content = """
def test_something(tmp_path):
    storage = UserDatasetStorage(tmp_path / "uploads")
    # ... test code ...
"""
        filepath = Path("test_file.py")

        # Act
        violations = find_inline_storage_creation(content, filepath)

        # Assert: Violation detected
        assert len(violations) > 0, "Should detect inline UserDatasetStorage creation"
        assert any("UserDatasetStorage" in msg for _, msg in violations)

    def test_find_inline_storage_creation_allows_fixture(self):
        """Test that fixture definitions are allowed."""
        # Arrange: Create test file with fixture (should be allowed)
        content = """
@pytest.fixture
def upload_storage(tmp_path):
    return UserDatasetStorage(upload_dir=tmp_path)
"""
        filepath = Path("test_file.py")

        # Act
        violations = find_inline_storage_creation(content, filepath)

        # Assert: No violation (fixture is allowed)
        assert len(violations) == 0, "Fixture definitions should be allowed"

    def test_find_duplicate_dataframe_creation_detects_violation(self):
        """Test that duplicate DataFrame creation patterns are detected."""
        # Arrange: Create test file with duplicate DataFrame creation
        content = """
def test_one():
    df = pl.DataFrame({"id": [1, 2], "value": [10, 20]})
    # ... test code ...

def test_two():
    df = pl.DataFrame({"id": [1, 2], "value": [10, 20]})
    # ... test code ...
"""
        filepath = Path("test_file.py")

        # Act
        violations = find_duplicate_dataframe_creation(content, filepath)

        # Assert: Violation detected
        assert len(violations) > 0, "Should detect duplicate DataFrame creation"

    def test_check_file_returns_violations_for_bad_file(self, tmp_path):
        """Test that check_file returns violations for file with duplicate setup."""
        # Arrange: Create test file with violations
        test_file = tmp_path / "test_bad.py"
        test_file.write_text(
            """
import polars as pl
from clinical_analytics.ui.storage.user_datasets import UserDatasetStorage

def test_one(tmp_path):
    storage = UserDatasetStorage(tmp_path / "uploads")
    df = pl.DataFrame({"id": [1, 2], "value": [10, 20]})

def test_two(tmp_path):
    storage = UserDatasetStorage(tmp_path / "uploads")
    df = pl.DataFrame({"id": [1, 2], "value": [10, 20]})
"""
        )

        # Act
        violations = check_file(test_file)

        # Assert: Violations detected
        assert len(violations) > 0, "Should detect violations in test file"

    def test_check_file_allows_good_file(self, tmp_path):
        """Test that check_file allows file with fixtures."""
        # Arrange: Create test file with fixtures (should pass)
        test_file = tmp_path / "test_good.py"
        test_file.write_text(
            """
import polars as pl
import pytest
from clinical_analytics.ui.storage.user_datasets import UserDatasetStorage

@pytest.fixture
def upload_storage(tmp_path):
    return UserDatasetStorage(upload_dir=tmp_path / "uploads")

@pytest.fixture
def sample_df():
    return pl.DataFrame({"id": [1, 2], "value": [10, 20]})

def test_one(upload_storage, sample_df):
    # ... test code ...
    pass

def test_two(upload_storage, sample_df):
    # ... test code ...
    pass
"""
        )

        # Act
        violations = check_file(test_file)

        # Assert: No violations (fixtures are used)
        assert len(violations) == 0, "File with fixtures should pass"

    def test_check_file_skips_non_test_files(self, tmp_path):
        """Test that non-test files are skipped."""
        # Arrange: Create non-test file
        test_file = tmp_path / "regular_file.py"
        test_file.write_text(
            """
def some_function():
    storage = UserDatasetStorage(tmp_path / "uploads")
    df = pl.DataFrame({"id": [1, 2]})
"""
        )

        # Act
        violations = check_file(test_file)

        # Assert: No violations (not a test file)
        assert len(violations) == 0, "Non-test files should be skipped"
