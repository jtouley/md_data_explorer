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


class TestConfigCaching:
    """Test suite for validation config caching in NLQueryEngine."""

    def test_nlquery_engine_caches_validation_config(self, make_semantic_layer):
        """Test that NLQueryEngine caches validation config to avoid repeated loading."""
        # Arrange
        from unittest.mock import MagicMock, patch

        from clinical_analytics.core.config_loader import load_validation_config
        from clinical_analytics.core.nl_query_engine import NLQueryEngine, QueryIntent

        semantic = make_semantic_layer(
            dataset_name="test_cache",
            data={"patient_id": ["P1", "P2"], "age": [45.0, 52.0]},
        )
        engine = NLQueryEngine(semantic)

        # Verify engine has a cached config attribute (None initially)
        assert hasattr(engine, "_validation_config"), "Engine should have _validation_config attribute"

        intent = QueryIntent(
            intent_type="DESCRIBE",
            primary_variable="age",
            confidence=0.9,
        )

        # Mock the LLM client
        mock_client = MagicMock()
        mock_client.is_available.return_value = True
        mock_client.generate.return_value = '{"is_valid": true, "errors": [], "warnings": []}'

        # Track config loading calls
        load_count = [0]
        real_config = load_validation_config()

        def mock_load_validation_config():
            load_count[0] += 1
            return real_config

        with patch.object(engine, "_get_ollama_client", return_value=mock_client):
            with patch(
                "clinical_analytics.core.config_loader.load_validation_config",
                mock_load_validation_config,
            ):
                # Act - call validation twice
                engine._dba_validate_llm(intent, "describe age")
                engine._dba_validate_llm(intent, "describe age again")

        # Assert - config should only be loaded once (cached)
        assert load_count[0] == 1, f"Config should be loaded only once, was loaded {load_count[0]} times"


class TestValidationErrorHandling:
    """Test suite for validation error handling - fail closed, not open."""

    def test_dba_validate_llm_exception_returns_invalid(self, make_semantic_layer):
        """Test that _dba_validate_llm returns is_valid=False on exception."""
        # Arrange
        from unittest.mock import MagicMock, patch

        from clinical_analytics.core.nl_query_engine import NLQueryEngine, QueryIntent

        semantic = make_semantic_layer(
            dataset_name="test_exception_handling",
            data={"patient_id": ["P1", "P2"], "age": [45.0, 52.0]},
        )
        engine = NLQueryEngine(semantic)

        intent = QueryIntent(
            intent_type="DESCRIBE",
            primary_variable="age",
            confidence=0.9,
        )

        # Mock the LLM client to raise an exception
        mock_client = MagicMock()
        mock_client.is_available.return_value = True
        mock_client.generate.side_effect = Exception("LLM connection error")

        with patch.object(engine, "_get_ollama_client", return_value=mock_client):
            # Act
            result = engine._dba_validate_llm(intent, "describe age")

        # Assert - should return is_valid=False on exception (fail closed)
        assert result.is_valid is False, "Validation should return is_valid=False on exception"
        assert len(result.errors) >= 1, "Should include error message"
        assert "error" in result.errors[0].lower() or "connection" in result.errors[0].lower()


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


class TestMultiLayerValidationIntegration:
    """Test suite for multi-layer validation in _llm_parse."""

    def test_llm_parse_calls_dba_validation(self, make_semantic_layer):
        """Test that _llm_parse calls DBA validation when feature flag is enabled."""
        # Arrange
        from unittest.mock import MagicMock, patch

        from clinical_analytics.core.config_loader import load_nl_query_config
        from clinical_analytics.core.nl_query_engine import NLQueryEngine, ValidationResult

        semantic = make_semantic_layer(
            dataset_name="test_integration",
            data={"patient_id": ["P1", "P2"], "age": [45.0, 52.0]},
        )
        engine = NLQueryEngine(semantic)

        # Mock OllamaClient to return valid intent
        mock_client = MagicMock()
        mock_client.is_available.return_value = True
        mock_client.generate.return_value = '{"intent_type": "DESCRIBE", "primary_variable": "age", "confidence": 0.9}'

        # Track DBA validation calls
        dba_called = []

        def mock_dba_validate(intent, query):
            dba_called.append({"intent": intent.intent_type, "query": query})
            return ValidationResult(is_valid=True, errors=(), warnings=())

        # Enable feature flag for this test
        real_config = load_nl_query_config()
        mock_config = {**real_config, "enable_multi_layer_validation": True}

        with patch.object(engine, "_get_ollama_client", return_value=mock_client):
            with patch.object(engine, "_dba_validate_llm", side_effect=mock_dba_validate):
                with patch(
                    "clinical_analytics.core.config_loader.load_nl_query_config",
                    return_value=mock_config,
                ):
                    # Act
                    result = engine._llm_parse("describe age")

        # Assert - DBA validation must be called when flag is enabled
        assert len(dba_called) == 1, f"DBA validation should be called once, was called {len(dba_called)} times"
        assert dba_called[0]["query"] == "describe age"
        assert result.intent_type == "DESCRIBE"

    def test_llm_parse_retries_on_dba_validation_failure(self, make_semantic_layer):
        """Test that _llm_parse retries when DBA validation fails (with feature flag enabled)."""
        # Arrange
        from unittest.mock import MagicMock, patch

        from clinical_analytics.core.config_loader import load_nl_query_config
        from clinical_analytics.core.nl_query_engine import NLQueryEngine, QueryIntent, ValidationResult

        semantic = make_semantic_layer(
            dataset_name="test_retry",
            data={"patient_id": ["P1", "P2"], "age": [45.0, 52.0]},
        )
        engine = NLQueryEngine(semantic)

        mock_client = MagicMock()
        mock_client.is_available.return_value = True
        mock_client.generate.return_value = '{"intent_type": "DESCRIBE", "primary_variable": "age", "confidence": 0.9}'

        # DBA returns invalid on first call
        retry_called = []

        def mock_dba_validate(intent, query):
            return ValidationResult(is_valid=False, errors=("Type mismatch",), warnings=())

        def mock_retry(query, errors, history):
            retry_called.append({"query": query, "errors": errors})
            return QueryIntent(intent_type="DESCRIBE", confidence=0.7, parsing_tier="retry")

        # Enable feature flag for this test
        real_config = load_nl_query_config()
        mock_config = {**real_config, "enable_multi_layer_validation": True}

        with patch.object(engine, "_get_ollama_client", return_value=mock_client):
            with patch.object(engine, "_dba_validate_llm", side_effect=mock_dba_validate):
                with patch.object(engine, "_retry_with_dba_feedback_llm", side_effect=mock_retry):
                    with patch(
                        "clinical_analytics.core.config_loader.load_nl_query_config",
                        return_value=mock_config,
                    ):
                        # Act
                        result = engine._llm_parse("describe age")

        # Assert - retry should be called and intent should be from retry
        assert len(retry_called) == 1, "Retry should be called once"
        # Errors are converted to list when passed to retry function
        assert retry_called[0]["errors"] == ["Type mismatch"]
        assert result.confidence == 0.7  # Confidence from retry mock


class TestBuildTypeRulesSection:
    """Test suite for _build_type_rules_section method."""

    def test_build_type_rules_section_exists(self, make_semantic_layer):
        """Test that _build_type_rules_section method exists."""
        # Arrange
        from clinical_analytics.core.nl_query_engine import NLQueryEngine

        semantic = make_semantic_layer(
            dataset_name="test_type_rules",
            data={"patient_id": ["P1", "P2"], "age": [45.0, 52.0]},
        )
        engine = NLQueryEngine(semantic)

        # Act & Assert
        assert hasattr(engine, "_build_type_rules_section")
        assert callable(engine._build_type_rules_section)

    def test_build_type_rules_section_formats_numeric_columns(self, make_semantic_layer):
        """Test that numeric columns are formatted correctly."""
        # Arrange
        from clinical_analytics.core.nl_query_engine import NLQueryEngine

        semantic = make_semantic_layer(
            dataset_name="test_numeric_rules",
            data={"patient_id": ["P1"], "age": [45.0]},
        )
        engine = NLQueryEngine(semantic)

        column_types = {
            "age": {"type": "numeric", "numeric": True, "dtype": "float64"},
        }

        # Act
        result = engine._build_type_rules_section(column_types)

        # Assert
        assert "age" in result
        assert "NUMERIC" in result
        assert "numbers only" in result.lower()


class TestFeatureFlag:
    """Test suite for enable_multi_layer_validation feature flag."""

    def test_llm_parse_only_calls_dba_validation_when_enabled(self, make_semantic_layer):
        """Test that _llm_parse only calls DBA validation (not analyst/manager) when enabled."""
        # Arrange
        from unittest.mock import MagicMock, patch

        from clinical_analytics.core.config_loader import load_nl_query_config
        from clinical_analytics.core.nl_query_engine import NLQueryEngine, ValidationResult

        semantic = make_semantic_layer(
            dataset_name="test_dba_only",
            data={"patient_id": ["P1", "P2"], "age": [45.0, 52.0]},
        )
        engine = NLQueryEngine(semantic)

        # Mock OllamaClient to return valid intent
        mock_client = MagicMock()
        mock_client.is_available.return_value = True
        mock_client.generate.return_value = '{"intent_type": "DESCRIBE", "primary_variable": "age", "confidence": 0.9}'

        # Track validation calls
        dba_called = []

        def mock_dba_validate(intent, query):
            dba_called.append(True)
            return ValidationResult(is_valid=True, errors=(), warnings=())

        # Enable feature flag
        real_config = load_nl_query_config()
        mock_config = {**real_config, "enable_multi_layer_validation": True}

        with patch.object(engine, "_get_ollama_client", return_value=mock_client):
            with patch.object(engine, "_dba_validate_llm", side_effect=mock_dba_validate):
                with patch(
                    "clinical_analytics.core.config_loader.load_nl_query_config",
                    return_value=mock_config,
                ):
                    # Act
                    result = engine._llm_parse("describe age")

        # Assert - only DBA validation should be called
        assert len(dba_called) == 1, "DBA validation should be called once"
        assert result.intent_type == "DESCRIBE"

        # Analyst and Manager should NOT exist (removed per PR38)
        assert not hasattr(engine, "_analyst_validate_llm"), "_analyst_validate_llm should be removed"
        assert not hasattr(engine, "_manager_approve_llm"), "_manager_approve_llm should be removed"

    def test_config_loader_enable_multi_layer_validation_defaults_false(self):
        """Test that enable_multi_layer_validation defaults to false."""
        # Arrange
        from clinical_analytics.core.config_loader import load_nl_query_config

        # Act
        config = load_nl_query_config()

        # Assert
        assert "enable_multi_layer_validation" in config
        assert config["enable_multi_layer_validation"] is False

    def test_llm_parse_skips_validation_when_flag_disabled(self, make_semantic_layer):
        """Test that _llm_parse skips validation when feature flag is disabled."""
        # Arrange
        from unittest.mock import MagicMock, patch

        from clinical_analytics.core.config_loader import load_nl_query_config
        from clinical_analytics.core.nl_query_engine import NLQueryEngine

        semantic = make_semantic_layer(
            dataset_name="test_flag_disabled",
            data={"patient_id": ["P1", "P2"], "age": [45.0, 52.0]},
        )
        engine = NLQueryEngine(semantic)

        # Mock OllamaClient to return valid intent
        mock_client = MagicMock()
        mock_client.is_available.return_value = True
        mock_client.generate.return_value = '{"intent_type": "DESCRIBE", "primary_variable": "age", "confidence": 0.9}'

        # Track validation calls
        dba_called = []

        def mock_dba_validate(intent, query):
            dba_called.append(True)
            from clinical_analytics.core.nl_query_engine import ValidationResult

            return ValidationResult(is_valid=True, errors=(), warnings=())

        # Get real config and override just the validation flag
        real_config = load_nl_query_config()
        mock_config = {**real_config, "enable_multi_layer_validation": False}

        with patch.object(engine, "_get_ollama_client", return_value=mock_client):
            with patch.object(engine, "_dba_validate_llm", side_effect=mock_dba_validate):
                with patch(
                    "clinical_analytics.core.config_loader.load_nl_query_config",
                    return_value=mock_config,
                ):
                    # Act
                    result = engine._llm_parse("describe age")

        # Assert - DBA validation should NOT be called when flag is disabled
        assert len(dba_called) == 0, "DBA validation should not be called when flag is disabled"
        assert result.intent_type == "DESCRIBE"


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
            errors=(),
            warnings=("Some warning",),
        )

        # Assert
        assert result.is_valid is True
        assert result.errors == ()
        assert result.warnings == ("Some warning",)

    def test_validation_result_invalid_with_errors(self):
        """Test ValidationResult with is_valid=False and errors."""
        # Arrange
        from clinical_analytics.core.nl_query_engine import ValidationResult

        # Act
        result = ValidationResult(
            is_valid=False,
            errors=("Type mismatch: column 'age' expects float, got str",),
            warnings=(),
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

        # Assert: Defaults should be empty tuples (frozen dataclass)
        assert result.errors == ()
        assert result.warnings == ()
