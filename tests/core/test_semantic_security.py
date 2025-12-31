"""
Tests for semantic layer SQL injection mitigation (Phase 0.2).

Tests the security functions that prevent SQL injection attacks
by validating table identifiers against an allowlist pattern.

Test name follows: test_unit_scenario_expectedBehavior
"""

import pytest

from clinical_analytics.core.semantic import _safe_identifier, _validate_table_identifier


class TestSQLInjectionMitigation:
    """Test suite for Phase 0.2: SQL injection mitigation."""

    def test_validate_table_identifier_accepts_valid_identifier(self):
        """Valid SQL identifier should pass validation (Phase 0.2)."""
        # Arrange: Valid identifiers
        valid_identifiers = [
            "patients",
            "patient_data",
            "PatientData",
            "_private_table",
            "table123",
            "t1",
        ]

        # Act & Assert: All should be accepted
        for identifier in valid_identifiers:
            result = _validate_table_identifier(identifier)
            assert result == identifier

    def test_validate_table_identifier_rejects_sql_injection_attempts(self):
        """SQL injection attempts should be rejected (Phase 0.2)."""
        # Arrange: SQL injection attempts
        injection_attempts = [
            "users; DROP TABLE users--",
            "users' OR '1'='1",
            "users/*comment*/",
            "users;--",
            "users UNION SELECT * FROM passwords",
            "users' UNION SELECT",
            "../etc/passwd",
            "..\\windows\\system32",
        ]

        # Act & Assert: All should raise ValueError
        for malicious_input in injection_attempts:
            with pytest.raises(ValueError, match="Invalid table identifier"):
                _validate_table_identifier(malicious_input)

    def test_validate_table_identifier_rejects_special_characters(self):
        """Special characters should be rejected (Phase 0.2)."""
        # Arrange: Identifiers with special chars
        special_char_inputs = [
            "user-table",
            "user.table",
            "user table",
            "user@table",
            "user#table",
            "user$table",
            "user%table",
            "user^table",
            "user&table",
            "user*table",
            "user(table",
            "user)table",
        ]

        # Act & Assert: All should raise ValueError
        for invalid_input in special_char_inputs:
            with pytest.raises(ValueError, match="Invalid table identifier"):
                _validate_table_identifier(invalid_input)

    def test_validate_table_identifier_rejects_empty_string(self):
        """Empty string should be rejected (Phase 0.2)."""
        # Arrange: Empty identifier
        empty_input = ""

        # Act & Assert: Should raise ValueError
        with pytest.raises(ValueError, match="cannot be empty"):
            _validate_table_identifier(empty_input)

    def test_validate_table_identifier_rejects_too_long_identifier(self):
        """Identifier exceeding 255 chars should be rejected (Phase 0.2)."""
        # Arrange: Identifier > 255 characters
        too_long = "a" * 256

        # Act & Assert: Should raise ValueError
        with pytest.raises(ValueError, match="too long"):
            _validate_table_identifier(too_long)

    def test_validate_table_identifier_accepts_max_length_identifier(self):
        """Identifier at exactly 255 chars should be accepted (Phase 0.2)."""
        # Arrange: Identifier exactly 255 characters
        max_length = "a" * 255

        # Act: Validate
        result = _validate_table_identifier(max_length)

        # Assert: Should pass
        assert result == max_length

    def test_validate_table_identifier_rejects_starting_with_digit(self):
        """Identifier starting with digit should be rejected (Phase 0.2)."""
        # Arrange: Identifiers starting with digit
        digit_start_inputs = ["1users", "9table", "0data"]

        # Act & Assert: All should raise ValueError
        for invalid_input in digit_start_inputs:
            with pytest.raises(ValueError, match="Invalid table identifier"):
                _validate_table_identifier(invalid_input)

    def test_safe_identifier_sanitizes_user_input(self):
        """User input should be sanitized to SQL-safe identifier (Phase 0.2)."""
        # Arrange: User-provided names with special chars
        user_inputs = [
            "My Dataset!",
            "dataset-with-hyphens",
            "dataset.with.dots",
            "dataset with spaces",
            "émoji😀data",
        ]

        # Act: Generate safe identifiers
        results = [_safe_identifier(name) for name in user_inputs]

        # Assert: All results should be valid SQL identifiers
        for result in results:
            # Should pass validation
            assert _validate_table_identifier(result) == result
            # Should only contain safe chars
            assert all(c.isalnum() or c == "_" for c in result)

    def test_safe_identifier_handles_collision_prevention(self):
        """Different inputs should produce different identifiers (Phase 0.2)."""
        # Arrange: Similar names that normalize to same base
        similar_names = [
            "my-dataset",
            "my_dataset",
            "my.dataset",
            "my dataset",
        ]

        # Act: Generate safe identifiers
        results = [_safe_identifier(name) for name in similar_names]

        # Assert: All should be unique (hash suffix prevents collision)
        assert len(set(results)) == len(similar_names)

    def test_safe_identifier_starts_with_letter_or_underscore(self):
        """Generated identifiers should always start with letter/underscore (Phase 0.2)."""
        # Arrange: Names that start with digits or special chars
        problematic_names = [
            "123data",
            "9dataset",
            "-dataset",
            ".dataset",
            " dataset",
        ]

        # Act: Generate safe identifiers
        results = [_safe_identifier(name) for name in problematic_names]

        # Assert: All should start with letter or underscore
        for result in results:
            assert result[0].isalpha() or result[0] == "_"

    def test_safe_identifier_enforces_max_length(self):
        """Generated identifiers should respect max length (Phase 0.2)."""
        # Arrange: Very long name
        long_name = "a" * 200

        # Act: Generate safe identifier with custom max_len
        result = _safe_identifier(long_name, max_len=30)

        # Assert: Result should be <= max_len + hash length + 1 (underscore)
        # max_len (30) + underscore (1) + hash (8) = 39
        assert len(result) <= 39

    def test_safe_identifier_preserves_readability(self):
        """Generated identifiers should preserve original name readability (Phase 0.2)."""
        # Arrange: Readable dataset name
        name = "MIMIC-IV-Clinical-Data"

        # Act: Generate safe identifier
        result = _safe_identifier(name)

        # Assert: Should contain "mimic" and "iv" and "clinical" and "data"
        base_part = result.rsplit("_", 1)[0]  # Remove hash suffix
        assert "mimic" in base_part.lower()
        assert "clinical" in base_part.lower()
        assert "data" in base_part.lower()
