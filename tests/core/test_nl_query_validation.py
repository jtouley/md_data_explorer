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


class TestDBAValidation:
    """Test suite for _dba_validate_llm method."""

    def test_dba_validate_llm_method_exists(self, make_semantic_layer):
        """Test that _dba_validate_llm method exists."""
        # Arrange
        from clinical_analytics.core.nl_query_engine import NLQueryEngine

        semantic = make_semantic_layer(
            dataset_name="test_dba_exists",
            data={"patient_id": ["P1", "P2"], "age": [45.0, 52.0]},
        )
        engine = NLQueryEngine(semantic)

        # Act & Assert
        assert hasattr(engine, "_dba_validate_llm")
        assert callable(engine._dba_validate_llm)

    def test_dba_validate_llm_returns_validation_result(self, make_semantic_layer):
        """Test that _dba_validate_llm returns a ValidationResult."""
        # Arrange
        from unittest.mock import MagicMock, patch

        from clinical_analytics.core.nl_query_engine import NLQueryEngine, QueryIntent, ValidationResult

        semantic = make_semantic_layer(
            dataset_name="test_dba_returns",
            data={"patient_id": ["P1", "P2"], "age": [45.0, 52.0]},
        )
        engine = NLQueryEngine(semantic)

        intent = QueryIntent(
            intent_type="DESCRIBE",
            primary_variable="age",
            confidence=0.9,
        )

        # Mock the LLM client to return a valid response
        mock_client = MagicMock()
        mock_client.is_available.return_value = True
        mock_client.generate.return_value = '{"is_valid": true, "errors": [], "warnings": []}'

        with patch.object(engine, "_get_ollama_client", return_value=mock_client):
            # Act
            result = engine._dba_validate_llm(intent, "describe age")

        # Assert
        assert isinstance(result, ValidationResult)
        assert result.is_valid is True

    def test_dba_validate_llm_invalid_returns_errors(self, make_semantic_layer):
        """Test that _dba_validate_llm returns errors for invalid intent."""
        # Arrange
        from unittest.mock import MagicMock, patch

        from clinical_analytics.core.nl_query_engine import NLQueryEngine, QueryIntent, ValidationResult
        from clinical_analytics.core.query_plan import FilterSpec

        semantic = make_semantic_layer(
            dataset_name="test_dba_invalid",
            data={"patient_id": ["P1", "P2"], "age": [45.0, 52.0]},
        )
        engine = NLQueryEngine(semantic)

        intent = QueryIntent(
            intent_type="DESCRIBE",
            primary_variable="age",
            filters=[FilterSpec(column="age", operator="!=", value="n/a")],
            confidence=0.8,
        )

        # Mock the LLM client to return invalid response
        mock_client = MagicMock()
        mock_client.is_available.return_value = True
        mock_client.generate.return_value = (
            '{"is_valid": false, "errors": ["Type mismatch: age expects numeric"], "warnings": []}'
        )

        with patch.object(engine, "_get_ollama_client", return_value=mock_client):
            # Act
            result = engine._dba_validate_llm(intent, "remove the n/a")

        # Assert
        assert isinstance(result, ValidationResult)
        assert result.is_valid is False
        assert len(result.errors) >= 1


class TestAnalystValidation:
    """Test suite for _analyst_validate_llm method."""

    def test_analyst_validate_llm_method_exists(self, make_semantic_layer):
        """Test that _analyst_validate_llm method exists."""
        # Arrange
        from clinical_analytics.core.nl_query_engine import NLQueryEngine

        semantic = make_semantic_layer(
            dataset_name="test_analyst_exists",
            data={"patient_id": ["P1", "P2"], "age": [45.0, 52.0]},
        )
        engine = NLQueryEngine(semantic)

        # Act & Assert
        assert hasattr(engine, "_analyst_validate_llm")
        assert callable(engine._analyst_validate_llm)


class TestManagerApproval:
    """Test suite for _manager_approve_llm method."""

    def test_manager_approve_llm_method_exists(self, make_semantic_layer):
        """Test that _manager_approve_llm method exists."""
        # Arrange
        from clinical_analytics.core.nl_query_engine import NLQueryEngine

        semantic = make_semantic_layer(
            dataset_name="test_manager_exists",
            data={"patient_id": ["P1", "P2"], "age": [45.0, 52.0]},
        )
        engine = NLQueryEngine(semantic)

        # Act & Assert
        assert hasattr(engine, "_manager_approve_llm")
        assert callable(engine._manager_approve_llm)


class TestRetryWithFeedback:
    """Test suite for _retry_with_dba_feedback_llm method."""

    def test_retry_with_dba_feedback_llm_method_exists(self, make_semantic_layer):
        """Test that _retry_with_dba_feedback_llm method exists."""
        # Arrange
        from clinical_analytics.core.nl_query_engine import NLQueryEngine

        semantic = make_semantic_layer(
            dataset_name="test_retry_exists",
            data={"patient_id": ["P1", "P2"], "age": [45.0, 52.0]},
        )
        engine = NLQueryEngine(semantic)

        # Act & Assert
        assert hasattr(engine, "_retry_with_dba_feedback_llm")
        assert callable(engine._retry_with_dba_feedback_llm)


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
