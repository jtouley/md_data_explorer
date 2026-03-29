"""Tests ensuring core layer has zero UI dependencies.

Phase 0 of Electron UI Migration: Extract Streamlit + UI imports from core layer.

These tests verify architectural boundaries:
1. Core layer has no Streamlit imports
2. Core layer has no UI layer imports
3. clarifying_questions returns data structures (not renders UI)
4. query_service uses core.analysis_types (not ui.components)
"""

import ast
import subprocess
from pathlib import Path


class TestCoreLayerNoStreamlitImports:
    """Verify core layer has zero Streamlit imports."""

    def test_core_layer_has_no_streamlit_imports(self) -> None:
        """Scan core layer for any Streamlit imports - should find none."""
        # Arrange
        core_path = Path("src/clinical_analytics/core")

        # Act: Use grep to find Streamlit imports
        result = subprocess.run(
            ["grep", "-r", "-E", "^import streamlit|^from streamlit", str(core_path)],
            capture_output=True,
            text=True,
        )

        # Assert: No matches should be found (return code 1 = no matches)
        assert result.returncode == 1, f"Found Streamlit imports in core layer:\n{result.stdout}"

    def test_core_layer_has_no_ui_imports(self) -> None:
        """Scan core layer for any UI layer imports - should find none."""
        # Arrange
        core_path = Path("src/clinical_analytics/core")

        # Act: Use grep to find UI layer imports
        result = subprocess.run(
            ["grep", "-r", "-E", "from clinical_analytics\\.ui", str(core_path)],
            capture_output=True,
            text=True,
        )

        # Assert: No matches should be found (return code 1 = no matches)
        assert result.returncode == 1, f"Found UI layer imports in core layer:\n{result.stdout}"


class TestClarifyingQuestionsReturnsData:
    """Verify clarifying_questions returns data structures, not renders UI."""

    def test_clarifying_questions_module_has_no_streamlit(self) -> None:
        """clarifying_questions.py should have zero Streamlit imports after refactor."""
        # Arrange
        clarifying_path = Path("src/clinical_analytics/core/clarifying_questions.py")

        # Act: Parse AST to find imports
        source = clarifying_path.read_text()
        tree = ast.parse(source)

        streamlit_imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "streamlit" or alias.name.startswith("streamlit."):
                        streamlit_imports.append(f"import {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                if node.module and (node.module == "streamlit" or node.module.startswith("streamlit.")):
                    streamlit_imports.append(f"from {node.module} import ...")

        # Assert: No Streamlit imports
        assert len(streamlit_imports) == 0, f"clarifying_questions.py still has Streamlit imports: {streamlit_imports}"

    def test_clarification_request_is_dataclass(self) -> None:
        """ClarificationRequest should be a dataclass for API serialization."""
        # Arrange & Act
        # Assert: Is a dataclass with expected fields
        import dataclasses

        from clinical_analytics.core.clarifying_questions import ClarificationRequest

        assert dataclasses.is_dataclass(ClarificationRequest), "ClarificationRequest should be a dataclass"

        # Check expected fields exist
        field_names = {f.name for f in dataclasses.fields(ClarificationRequest)}
        assert "question_type" in field_names
        assert "prompt" in field_names
        assert "options" in field_names

    def test_generate_clarifications_returns_list(self) -> None:
        """generate_clarifications() should return list of ClarificationRequest."""
        # Arrange
        from unittest.mock import MagicMock

        from clinical_analytics.core.clarifying_questions import (
            ClarificationRequest,
            generate_clarifications,
        )
        from clinical_analytics.core.nl_query_engine import QueryIntent

        mock_semantic_layer = MagicMock()
        mock_semantic_layer.get_available_dimensions.return_value = {"age_group": "Age Group"}
        mock_semantic_layer.get_collision_suggestions.return_value = []
        mock_semantic_layer.get_data_quality_warnings.return_value = []

        intent = QueryIntent(
            intent_type="DESCRIBE",
            confidence=0.2,  # Low confidence triggers clarification
            primary_variable=None,
        )

        # Act
        result = generate_clarifications(intent, mock_semantic_layer, ["age", "sex", "outcome"])

        # Assert: Returns list of ClarificationRequest (may be empty if no clarifications needed)
        assert isinstance(result, list)
        for item in result:
            assert isinstance(item, ClarificationRequest)


class TestQueryServiceUsesCorrectTypes:
    """Verify query_service imports from core.analysis_types."""

    def test_query_service_imports_from_core(self) -> None:
        """query_service.py should import AnalysisContext/AnalysisIntent from core."""
        # Arrange
        query_service_path = Path("src/clinical_analytics/core/query_service.py")

        # Act: Parse AST to find imports
        source = query_service_path.read_text()
        tree = ast.parse(source)

        ui_imports = []
        core_analysis_types_imports = []

        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.module and "clinical_analytics.ui" in node.module:
                    ui_imports.append(f"from {node.module} import ...")
                if node.module and "clinical_analytics.core.analysis_types" in node.module:
                    core_analysis_types_imports.append(f"from {node.module}")

        # Assert: No UI imports, has core.analysis_types import
        assert len(ui_imports) == 0, f"query_service.py imports from UI layer: {ui_imports}"
        assert (
            len(core_analysis_types_imports) > 0
        ), "query_service.py should import from clinical_analytics.core.analysis_types"

    def test_analysis_types_module_exists(self) -> None:
        """core/analysis_types.py should exist with AnalysisIntent and AnalysisContext."""
        # Arrange & Act
        from clinical_analytics.core.analysis_types import AnalysisContext, AnalysisIntent

        # Assert: Classes exist and are correct types
        assert AnalysisIntent is not None
        assert AnalysisContext is not None

        # AnalysisIntent should be an Enum
        from enum import Enum

        assert issubclass(AnalysisIntent, Enum), "AnalysisIntent should be an Enum"

        # AnalysisContext should be a dataclass
        import dataclasses

        assert dataclasses.is_dataclass(AnalysisContext), "AnalysisContext should be a dataclass"

    def test_analysis_context_has_expected_fields(self) -> None:
        """AnalysisContext should have all required fields for query execution."""
        # Arrange & Act
        import dataclasses

        from clinical_analytics.core.analysis_types import AnalysisContext

        field_names = {f.name for f in dataclasses.fields(AnalysisContext)}

        # Assert: Expected fields exist
        expected_fields = {
            "inferred_intent",
            "primary_variable",
            "grouping_variable",
            "predictor_variables",
            "time_variable",
            "event_variable",
            "filters",
            "query_plan",
        }
        missing = expected_fields - field_names
        assert len(missing) == 0, f"AnalysisContext missing fields: {missing}"
