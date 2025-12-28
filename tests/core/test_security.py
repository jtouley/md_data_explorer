"""
Tests for security functions.

Validates:
- SQL injection prevention via _validate_table_identifier()
- Path traversal prevention via _safe_extract_zip_member()
- UUID-based storage via _safe_store_upload()
"""

import zipfile
from pathlib import Path

import pytest


class TestValidateTableIdentifier:
    """Tests for SQL identifier validation."""

    def test_identifier_validation_valid_names_pass(self) -> None:
        """Test that valid SQL identifiers pass validation."""
        # Arrange
        from clinical_analytics.core.semantic import _validate_table_identifier

        valid_names = ["patients", "my_table", "Table123", "_private"]

        # Act & Assert
        for name in valid_names:
            assert _validate_table_identifier(name) == name

    def test_identifier_validation_sql_injection_raises_valueerror(self) -> None:
        """Test that SQL injection attempts are rejected."""
        # Arrange
        from clinical_analytics.core.semantic import _validate_table_identifier

        malicious_inputs = [
            '"; DROP TABLE patients; --',
            "table; SELECT * FROM secrets",
        ]

        # Act & Assert
        for malicious in malicious_inputs:
            with pytest.raises(ValueError, match="Invalid table identifier"):
                _validate_table_identifier(malicious)

    def test_identifier_validation_special_chars_raises_valueerror(self) -> None:
        """Test that special characters are rejected."""
        # Arrange
        from clinical_analytics.core.semantic import _validate_table_identifier

        invalid_names = ["my-table", "my.table", "my table"]

        # Act & Assert
        for name in invalid_names:
            with pytest.raises(ValueError, match="Invalid table identifier"):
                _validate_table_identifier(name)

    def test_identifier_validation_starts_with_number_raises_valueerror(self) -> None:
        """Test that identifiers starting with numbers are rejected."""
        # Arrange
        from clinical_analytics.core.semantic import _validate_table_identifier

        # Act & Assert
        with pytest.raises(ValueError, match="Invalid table identifier"):
            _validate_table_identifier("123table")

    def test_identifier_validation_empty_string_raises_valueerror(self) -> None:
        """Test that empty identifiers are rejected."""
        # Arrange
        from clinical_analytics.core.semantic import _validate_table_identifier

        # Act & Assert
        with pytest.raises(ValueError, match="cannot be empty"):
            _validate_table_identifier("")


class TestSafeExtractZipMember:
    """Tests for safe ZIP extraction."""

    def test_zip_extraction_valid_member_extracts_correctly(self, tmp_path: Path) -> None:
        """Test that valid ZIP members are extracted correctly."""
        # Arrange
        from clinical_analytics.ui.storage.user_datasets import _safe_extract_zip_member

        zip_path = tmp_path / "test.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("data.csv", "col1,col2\n1,2")

        extract_to = tmp_path / "extracted"
        extract_to.mkdir()

        # Act
        with zipfile.ZipFile(zip_path, "r") as zf:
            extracted = _safe_extract_zip_member(zf, "data.csv", extract_to)

        # Assert
        assert extracted.exists()
        assert extracted.read_text() == "col1,col2\n1,2"

    def test_zip_extraction_path_traversal_raises_securityerror(self, tmp_path: Path) -> None:
        """Test that path traversal attempts are blocked."""
        # Arrange
        from clinical_analytics.ui.storage.user_datasets import (
            SecurityError,
            _safe_extract_zip_member,
        )

        zip_path = tmp_path / "evil.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("../../../etc/passwd.csv", "malicious")

        extract_to = tmp_path / "extracted"
        extract_to.mkdir()

        # Act & Assert
        with zipfile.ZipFile(zip_path, "r") as zf:
            with pytest.raises(SecurityError, match="Path traversal"):
                _safe_extract_zip_member(zf, "../../../etc/passwd.csv", extract_to)

    def test_zip_extraction_absolute_path_raises_securityerror(self, tmp_path: Path) -> None:
        """Test that absolute paths in ZIP are blocked."""
        # Arrange
        from clinical_analytics.ui.storage.user_datasets import (
            SecurityError,
            _safe_extract_zip_member,
        )

        zip_path = tmp_path / "evil.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("/etc/passwd", "malicious")

        extract_to = tmp_path / "extracted"
        extract_to.mkdir()

        # Act & Assert
        with zipfile.ZipFile(zip_path, "r") as zf:
            with pytest.raises(SecurityError, match="Path traversal|Invalid"):
                _safe_extract_zip_member(zf, "/etc/passwd", extract_to)


class TestSafeStoreUpload:
    """Tests for UUID-based safe upload storage."""

    def test_upload_storage_uses_uuid_not_original_filename(self, tmp_path: Path) -> None:
        """Test that uploads are stored with UUID, not original filename."""
        # Arrange
        from clinical_analytics.ui.storage.user_datasets import _safe_store_upload

        file_bytes = b"test content"
        dangerous_filename = "dangerous;name.csv"

        # Act
        stored_path = _safe_store_upload(file_bytes, tmp_path, dangerous_filename)

        # Assert: Should NOT contain original filename
        assert "dangerous" not in str(stored_path)
        assert ";" not in str(stored_path)
        assert stored_path.is_relative_to(tmp_path)
        assert stored_path.read_bytes() == file_bytes
        assert stored_path.suffix == ".csv"

    def test_upload_storage_path_traversal_in_filename_ignored(self, tmp_path: Path) -> None:
        """Test that path traversal in original filename is safely ignored."""
        # Arrange
        from clinical_analytics.ui.storage.user_datasets import _safe_store_upload

        file_bytes = b"test content"
        malicious_filename = "../../../etc/passwd.csv"

        # Act
        stored_path = _safe_store_upload(file_bytes, tmp_path, malicious_filename)

        # Assert: Should still be within base_dir (UUID-based, ignores original filename)
        assert stored_path.is_relative_to(tmp_path)
        assert stored_path.exists()
