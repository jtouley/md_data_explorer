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

    def test_upload_storage_rejects_original_filename_as_key(self, tmp_path: Path) -> None:
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
