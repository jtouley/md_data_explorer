"""
Tests for upload security validation.

Tests file type validation, size limits, path traversal prevention, and other security measures.
"""

import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from clinical_analytics.ui.storage.user_datasets import UploadSecurityValidator


class TestFileTypeValidation:
    """Test file type validation."""

    def test_valid_csv(self):
        """CSV files should be accepted."""
        valid, error = UploadSecurityValidator.validate_file_type("data.csv")
        assert valid is True
        assert error == ""

    def test_valid_excel_xlsx(self):
        """Excel .xlsx files should be accepted."""
        valid, error = UploadSecurityValidator.validate_file_type("data.xlsx")
        assert valid is True
        assert error == ""

    def test_valid_excel_xls(self):
        """Excel .xls files should be accepted."""
        valid, error = UploadSecurityValidator.validate_file_type("data.xls")
        assert valid is True
        assert error == ""

    def test_valid_spss(self):
        """SPSS .sav files should be accepted."""
        valid, error = UploadSecurityValidator.validate_file_type("data.sav")
        assert valid is True
        assert error == ""

    def test_case_insensitive(self):
        """File extension check should be case-insensitive."""
        valid, error = UploadSecurityValidator.validate_file_type("DATA.CSV")
        assert valid is True

        valid, error = UploadSecurityValidator.validate_file_type("DATA.XLSX")
        assert valid is True

    def test_invalid_extension(self):
        """Files with invalid extensions should be rejected."""
        valid, error = UploadSecurityValidator.validate_file_type("data.exe")
        assert valid is False
        assert "not allowed" in error

    def test_no_extension(self):
        """Files without extension should be rejected."""
        valid, error = UploadSecurityValidator.validate_file_type("data")
        assert valid is False
        assert "no extension" in error

    def test_dangerous_extensions(self):
        """Dangerous file types should be rejected."""
        dangerous = ["data.exe", "data.bat", "data.sh", "data.py", "data.js"]
        for filename in dangerous:
            valid, error = UploadSecurityValidator.validate_file_type(filename)
            assert valid is False, f"Should reject {filename}"


class TestFileSizeValidation:
    """Test file size validation."""

    def test_valid_size(self):
        """Files within size limits should be accepted."""
        # 1MB file
        file_bytes = b"x" * (1024 * 1024)
        valid, error = UploadSecurityValidator.validate_file_size(file_bytes)
        assert valid is True
        assert error == ""

    def test_max_size(self):
        """Files at exactly max size should be accepted."""
        # Exactly 100MB
        file_bytes = b"x" * (100 * 1024 * 1024)
        valid, error = UploadSecurityValidator.validate_file_size(file_bytes)
        assert valid is True

    def test_too_large(self):
        """Files over 100MB should be rejected."""
        # 101MB
        file_bytes = b"x" * (101 * 1024 * 1024)
        valid, error = UploadSecurityValidator.validate_file_size(file_bytes)
        assert valid is False
        assert "too large" in error.lower()

    def test_too_small(self):
        """Files under 1KB should be rejected."""
        # 500 bytes
        file_bytes = b"x" * 500
        valid, error = UploadSecurityValidator.validate_file_size(file_bytes)
        assert valid is False
        assert "too small" in error.lower()

    def test_empty_file(self):
        """Empty files should be rejected."""
        file_bytes = b""
        valid, error = UploadSecurityValidator.validate_file_size(file_bytes)
        assert valid is False


class TestFilenameSecure:
    """Test filename sanitization."""

    def test_sanitize_normal(self):
        """Normal filenames should pass through."""
        result = UploadSecurityValidator.sanitize_filename("patient_data_2024.csv")
        assert result == "patient_data_2024.csv"

    def test_sanitize_path_traversal(self):
        """Path traversal attempts should be neutralized."""
        result = UploadSecurityValidator.sanitize_filename("../../etc/passwd")
        assert "/" not in result
        assert ".." not in result
        # Should only keep filename
        assert result == "passwd"

    def test_sanitize_absolute_path(self):
        """Absolute paths should be reduced to filename only."""
        result = UploadSecurityValidator.sanitize_filename("/var/www/data.csv")
        assert result == "data.csv"

    def test_sanitize_windows_path(self):
        """Windows paths should be handled."""
        result = UploadSecurityValidator.sanitize_filename("C:\\Users\\data.csv")
        assert "\\" not in result
        assert ":" not in result

    def test_sanitize_special_chars(self):
        """Special characters should be replaced with underscores."""
        result = UploadSecurityValidator.sanitize_filename("data <test> (file).csv")
        assert "<" not in result
        assert ">" not in result
        assert "(" not in result
        assert ")" not in result

    def test_sanitize_spaces(self):
        """Spaces should be preserved or handled."""
        result = UploadSecurityValidator.sanitize_filename("patient data 2024.csv")
        # Spaces might be replaced with underscores or preserved
        assert "patient" in result
        assert "data" in result

    def test_sanitize_unicode(self):
        """Unicode characters should be handled."""
        result = UploadSecurityValidator.sanitize_filename("données_français.csv")
        # Should handle or remove unicode
        assert ".csv" in result


class TestCompleteValidation:
    """Test complete validation workflow."""

    def test_valid_upload(self):
        """Valid files should pass all checks."""
        filename = "patient_data.csv"
        file_bytes = b"patient_id,age,sex\n001,45,M\n002,62,F\n" * 100  # Valid CSV

        valid, error = UploadSecurityValidator.validate(filename, file_bytes)
        assert valid is True
        assert error == ""

    def test_invalid_type(self):
        """Invalid file types should fail validation."""
        filename = "malware.exe"
        file_bytes = b"x" * 5000

        valid, error = UploadSecurityValidator.validate(filename, file_bytes)
        assert valid is False
        assert error != ""

    def test_too_large_file(self):
        """Oversized files should fail validation."""
        filename = "huge.csv"
        file_bytes = b"x" * (101 * 1024 * 1024)  # 101MB

        valid, error = UploadSecurityValidator.validate(filename, file_bytes)
        assert valid is False

    def test_multiple_issues(self):
        """Files with multiple issues should report appropriately."""
        filename = "bad.exe"  # Wrong type
        file_bytes = b"x" * 100  # Too small

        valid, error = UploadSecurityValidator.validate(filename, file_bytes)
        assert valid is False
        # Should report the first issue encountered


class TestSecurityConstants:
    """Test security configuration constants."""

    def test_allowed_extensions_defined(self):
        """Allowed extensions should be defined."""
        assert hasattr(UploadSecurityValidator, 'ALLOWED_EXTENSIONS')
        assert len(UploadSecurityValidator.ALLOWED_EXTENSIONS) > 0

    def test_size_limits_defined(self):
        """Size limits should be defined."""
        assert hasattr(UploadSecurityValidator, 'MAX_FILE_SIZE_BYTES')
        assert hasattr(UploadSecurityValidator, 'MIN_FILE_SIZE_BYTES')

    def test_size_limits_reasonable(self):
        """Size limits should be reasonable."""
        assert UploadSecurityValidator.MAX_FILE_SIZE_BYTES == 100 * 1024 * 1024  # 100MB
        assert UploadSecurityValidator.MIN_FILE_SIZE_BYTES == 1024  # 1KB

    def test_allowed_extensions_secure(self):
        """Only safe file types should be allowed."""
        dangerous_extensions = {'.exe', '.bat', '.sh', '.py', '.js', '.php'}
        allowed = UploadSecurityValidator.ALLOWED_EXTENSIONS

        # None of the dangerous types should be allowed
        for dangerous in dangerous_extensions:
            assert dangerous not in allowed, f"Dangerous extension {dangerous} should not be allowed"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
