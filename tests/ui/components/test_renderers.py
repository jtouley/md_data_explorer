"""
Tests for Renderer Registry - Strategy pattern for analysis result rendering.

Tests verify registry operations and individual renderer behavior.
"""

from unittest.mock import MagicMock, patch

import pytest
from clinical_analytics.core.analysis_result import AnalysisResult
from clinical_analytics.ui.components.renderers import (
    RendererRegistry,
)


class TestRendererRegistry:
    """Test suite for Renderer registry."""

    def test_renderer_registry_known_type_returns_renderer(self):
        """Test that registry returns renderer for known type."""
        # Arrange
        registry = RendererRegistry()

        @registry.register("test_type")
        class TestRenderer:
            def render(self, result: AnalysisResult, *, query_text: str | None = None) -> None:
                pass

        # Act
        renderer = registry.get("test_type")

        # Assert
        assert renderer is not None
        assert hasattr(renderer, "render")

    def test_renderer_registry_unknown_type_raises_valueerror(self):
        """Test that registry raises ValueError for unknown type."""
        # Arrange
        registry = RendererRegistry()

        # Act & Assert
        with pytest.raises(ValueError, match="Unknown result type"):
            registry.get("nonexistent_type")

    def test_renderer_register_decorator_adds_to_registry(self):
        """Test that register decorator adds renderer to registry."""
        # Arrange
        registry = RendererRegistry()

        # Act
        @registry.register("decorated_type")
        class DecoratedRenderer:
            def render(self, result: AnalysisResult, *, query_text: str | None = None) -> None:
                pass

        # Assert
        assert "decorated_type" in registry._renderers
        renderer = registry.get("decorated_type")
        assert isinstance(renderer, DecoratedRenderer)

    def test_render_result_dispatches_to_correct_renderer(self):
        """Test that render_result calls correct renderer based on type."""
        # Arrange
        registry = RendererRegistry()
        mock_render = MagicMock()

        @registry.register("mock_type")
        class MockRenderer:
            def render(self, result: AnalysisResult, *, query_text: str | None = None) -> None:
                mock_render(result, query_text)

        result = AnalysisResult(type="mock_type", payload={"data": 1})

        # Act
        registry.render(result, query_text="test query")

        # Assert
        mock_render.assert_called_once()
        call_args = mock_render.call_args
        assert call_args[0][0] == result
        assert call_args[0][1] == "test query"

    def test_descriptive_renderer_renders_stats(self):
        """Test that descriptive renderer handles stats payload."""
        # Arrange
        from clinical_analytics.ui.components.renderers import DescriptiveRenderer

        renderer = DescriptiveRenderer()
        result = AnalysisResult(
            type="descriptive",
            payload={"mean": 45.5, "std": 12.3, "count": 100},
        )

        # Act - should not raise
        with patch("streamlit.write"), patch("streamlit.dataframe"):
            renderer.render(result, query_text="What is the average?")

        # Assert - renderer executed without error
        # Note: Actual UI assertions would require Streamlit testing framework

    def test_comparison_renderer_renders_groups(self):
        """Test that comparison renderer handles group comparison payload."""
        # Arrange
        from clinical_analytics.ui.components.renderers import ComparisonRenderer

        renderer = ComparisonRenderer()
        result = AnalysisResult(
            type="comparison",
            payload={"group_a": 45.5, "group_b": 52.1, "p_value": 0.03},
        )

        # Act - should not raise
        with patch("streamlit.write"):
            with patch("streamlit.metric"):
                renderer.render(result, query_text="Compare groups")

    def test_count_renderer_renders_count(self):
        """Test that count renderer handles count payload."""
        # Arrange
        from clinical_analytics.ui.components.renderers import CountRenderer

        renderer = CountRenderer()
        result = AnalysisResult(
            type="count",
            payload={"count": 42, "label": "patients"},
        )

        # Act - should not raise
        with patch("streamlit.metric"):
            renderer.render(result, query_text="How many patients?")

    def test_renderer_with_error_shows_friendly_message(self):
        """Test that renderer displays friendly error message when present."""
        # Arrange
        from clinical_analytics.ui.components.renderers import DescriptiveRenderer

        renderer = DescriptiveRenderer()
        result = AnalysisResult(
            type="descriptive",
            payload={"error": "Division by zero"},
            friendly_error_message="Unable to calculate due to missing data.",
        )

        # Act - should not raise
        with patch("streamlit.error") as mock_error:
            renderer.render(result, query_text="Calculate average")

        # Assert
        mock_error.assert_called_once_with("Unable to calculate due to missing data.")
