"""
Manual tests for upload security validation (no pytest required).
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from clinical_analytics.ui.storage.user_datasets import UploadSecurityValidator


def test_file_type_validation():
    """Test file type validation."""
    print("Testing file type validation...")

    # Valid types
    assert UploadSecurityValidator.validate_file_type("data.csv")[0], "CSV should be valid"
    assert UploadSecurityValidator.validate_file_type("data.xlsx")[0], "XLSX should be valid"
    assert UploadSecurityValidator.validate_file_type("data.xls")[0], "XLS should be valid"
    assert UploadSecurityValidator.validate_file_type("data.sav")[0], "SAV should be valid"

    # Case insensitive
    assert UploadSecurityValidator.validate_file_type("DATA.CSV")[0], "Should be case-insensitive"

    # Invalid types
    assert not UploadSecurityValidator.validate_file_type("data.exe")[0], "EXE should be invalid"
    assert not UploadSecurityValidator.validate_file_type("data")[0], "No extension should be invalid"

    print("✅ File type validation tests passed")


def test_file_size_validation():
    """Test file size validation."""
    print("Testing file size validation...")

    # Valid size (1MB)
    file_1mb = b"x" * (1024 * 1024)
    assert UploadSecurityValidator.validate_file_size(file_1mb)[0], "1MB file should be valid"

    # Max size (100MB)
    file_100mb = b"x" * (100 * 1024 * 1024)
    assert UploadSecurityValidator.validate_file_size(file_100mb)[0], "100MB file should be valid"

    # Too large (101MB)
    file_101mb = b"x" * (101 * 1024 * 1024)
    assert not UploadSecurityValidator.validate_file_size(file_101mb)[0], "101MB file should be invalid"

    # Too small (500 bytes)
    file_500b = b"x" * 500
    assert not UploadSecurityValidator.validate_file_size(file_500b)[0], "500 byte file should be invalid"

    # Empty
    empty_file = b""
    assert not UploadSecurityValidator.validate_file_size(empty_file)[0], "Empty file should be invalid"

    print("✅ File size validation tests passed")


def test_filename_sanitization():
    """Test filename sanitization."""
    print("Testing filename sanitization...")

    # Normal filename
    result = UploadSecurityValidator.sanitize_filename("patient_data_2024.csv")
    assert result == "patient_data_2024.csv", f"Normal filename failed: {result}"

    # Path traversal
    result = UploadSecurityValidator.sanitize_filename("../../etc/passwd")
    assert "/" not in result, "Path traversal not prevented"
    assert result == "passwd", f"Should extract filename only: {result}"

    # Absolute path
    result = UploadSecurityValidator.sanitize_filename("/var/www/data.csv")
    assert result == "data.csv", f"Should extract filename from absolute path: {result}"

    # Special characters
    result = UploadSecurityValidator.sanitize_filename("data <test>.csv")
    assert "<" not in result and ">" not in result, "Special characters not sanitized"

    print("✅ Filename sanitization tests passed")


def test_complete_validation():
    """Test complete validation workflow."""
    print("Testing complete validation...")

    # Valid upload
    filename = "patient_data.csv"
    file_bytes = b"patient_id,age,sex\n001,45,M\n002,62,F\n" * 100
    valid, error = UploadSecurityValidator.validate(filename, file_bytes)
    assert valid, f"Valid file should pass: {error}"

    # Invalid type
    filename = "malware.exe"
    file_bytes = b"x" * 5000
    valid, error = UploadSecurityValidator.validate(filename, file_bytes)
    assert not valid, "Executable should fail validation"

    # Too large
    filename = "huge.csv"
    file_bytes = b"x" * (101 * 1024 * 1024)
    valid, error = UploadSecurityValidator.validate(filename, file_bytes)
    assert not valid, "Oversized file should fail validation"

    print("✅ Complete validation tests passed")


def test_security_constants():
    """Test security constants."""
    print("Testing security constants...")

    # Check constants exist
    assert hasattr(UploadSecurityValidator, 'ALLOWED_EXTENSIONS')
    assert hasattr(UploadSecurityValidator, 'MAX_FILE_SIZE_BYTES')
    assert hasattr(UploadSecurityValidator, 'MIN_FILE_SIZE_BYTES')

    # Check values
    assert UploadSecurityValidator.MAX_FILE_SIZE_BYTES == 100 * 1024 * 1024
    assert UploadSecurityValidator.MIN_FILE_SIZE_BYTES == 1024

    # Check no dangerous extensions
    dangerous = {'.exe', '.bat', '.sh', '.py', '.js', '.php'}
    allowed = UploadSecurityValidator.ALLOWED_EXTENSIONS
    for ext in dangerous:
        assert ext not in allowed, f"Dangerous extension {ext} should not be allowed"

    print("✅ Security constants tests passed")


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("Running Upload Security Tests")
    print("=" * 60)

    try:
        test_file_type_validation()
        test_file_size_validation()
        test_filename_sanitization()
        test_complete_validation()
        test_security_constants()

        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("=" * 60)
        return 0

    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(run_all_tests())
