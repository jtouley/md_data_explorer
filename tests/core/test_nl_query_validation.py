"""
Tests for NL Query validation functionality.

Following AGENTS.md guidelines:
- AAA pattern (Arrange-Act-Assert)
- Descriptive test names: test_unit_scenario_expectedBehavior
- Test isolation (no shared mutable state)
"""


class TestInferColumnTypeFromView:
    """Test suite for _infer_column_type_from_view method."""

    def test_infer_column_type_numeric_column_returns_numeric_type(self, make_semantic_layer):
        """Test that numeric columns return numeric type info."""
        # Arrange
        from clinical_analytics.core.nl_query_engine import NLQueryEngine

        semantic = make_semantic_layer(
            dataset_name="test_infer",
            data={"patient_id": ["P1", "P2"], "age": [45.0, 52.0], "score": [100, 200]},
        )
        engine = NLQueryEngine(semantic)

        # Act
        type_info = engine._infer_column_type_from_view("age")

        # Assert
        assert type_info is not None
        assert type_info.get("numeric") is True
        assert "dtype" in type_info

    def test_infer_column_type_string_column_returns_string_type(self, make_semantic_layer):
        """Test that string columns return string type info."""
        # Arrange
        from clinical_analytics.core.nl_query_engine import NLQueryEngine

        semantic = make_semantic_layer(
            dataset_name="test_infer_str",
            data={"patient_id": ["P1", "P2"], "name": ["Alice", "Bob"]},
        )
        engine = NLQueryEngine(semantic)

        # Act
        type_info = engine._infer_column_type_from_view("name")

        # Assert
        assert type_info is not None
        assert type_info.get("numeric") is False

    def test_infer_column_type_missing_column_returns_none(self, make_semantic_layer):
        """Test that missing columns return None."""
        # Arrange
        from clinical_analytics.core.nl_query_engine import NLQueryEngine

        semantic = make_semantic_layer(
            dataset_name="test_infer_missing",
            data={"patient_id": ["P1", "P2"], "age": [45.0, 52.0]},
        )
        engine = NLQueryEngine(semantic)

        # Act
        type_info = engine._infer_column_type_from_view("nonexistent_column")

        # Assert
        assert type_info is None


class TestBuildRagContextWithTypes:
    """Test suite for _build_rag_context with type information."""

    def test_build_rag_context_includes_column_types(self, make_semantic_layer):
        """Test that _build_rag_context includes column_types dict."""
        # Arrange
        from clinical_analytics.core.nl_query_engine import NLQueryEngine

        semantic = make_semantic_layer(
            dataset_name="test_rag_types",
            data={
                "patient_id": ["P1", "P2"],
                "age": [45.0, 52.0],
                "name": ["Alice", "Bob"],
            },
        )
        engine = NLQueryEngine(semantic)

        # Act
        context = engine._build_rag_context("describe patients")

        # Assert
        assert "column_types" in context
        assert isinstance(context["column_types"], dict)
        assert "age" in context["column_types"]
        assert context["column_types"]["age"]["numeric"] is True

    def test_build_rag_context_column_types_has_all_columns(self, make_semantic_layer):
        """Test that column_types includes all columns from base view."""
        # Arrange
        from clinical_analytics.core.nl_query_engine import NLQueryEngine

        semantic = make_semantic_layer(
            dataset_name="test_rag_all_cols",
            data={
                "patient_id": ["P1", "P2"],
                "age": [45.0, 52.0],
                "score": [100, 200],
            },
        )
        engine = NLQueryEngine(semantic)

        # Act
        context = engine._build_rag_context("describe patients")

        # Assert
        column_types = context["column_types"]
        assert "patient_id" in column_types
        assert "age" in column_types
        assert "score" in column_types


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
