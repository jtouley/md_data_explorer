"""
Tests for NL Query validation functionality.

Following AGENTS.md guidelines:
- AAA pattern (Arrange-Act-Assert)
- Descriptive test names: test_unit_scenario_expectedBehavior
- Test isolation (no shared mutable state)
"""


class TestValidationResult:
    """Test suite for ValidationResult dataclass."""

    def test_validation_result_exists(self):
        """Test that ValidationResult dataclass exists."""
        # Arrange & Act & Assert
        from clinical_analytics.core.nl_query_engine import ValidationResult

        assert ValidationResult is not None

    def test_validation_result_is_dataclass(self):
        """Test that ValidationResult is a dataclass."""
        # Arrange
        from dataclasses import is_dataclass

        from clinical_analytics.core.nl_query_engine import ValidationResult

        # Act & Assert
        assert is_dataclass(ValidationResult)

    def test_validation_result_has_required_fields(self):
        """Test that ValidationResult has is_valid, errors, warnings fields."""
        # Arrange
        from clinical_analytics.core.nl_query_engine import ValidationResult

        # Act
        result = ValidationResult(
            is_valid=True,
            errors=[],
            warnings=["Some warning"],
        )

        # Assert
        assert result.is_valid is True
        assert result.errors == []
        assert result.warnings == ["Some warning"]

    def test_validation_result_invalid_with_errors(self):
        """Test ValidationResult with is_valid=False and errors."""
        # Arrange
        from clinical_analytics.core.nl_query_engine import ValidationResult

        # Act
        result = ValidationResult(
            is_valid=False,
            errors=["Type mismatch: column 'age' expects float, got str"],
            warnings=[],
        )

        # Assert
        assert result.is_valid is False
        assert len(result.errors) == 1
        assert "age" in result.errors[0]

    def test_validation_result_defaults(self):
        """Test ValidationResult with default values."""
        # Arrange
        from clinical_analytics.core.nl_query_engine import ValidationResult

        # Act: Create with only is_valid (if defaults exist)
        result = ValidationResult(is_valid=True)

        # Assert: Defaults should be empty lists
        assert result.errors == []
        assert result.warnings == []
