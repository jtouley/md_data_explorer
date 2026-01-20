"""
Renderer Registry - Strategy pattern for analysis result rendering.

Replaces if/elif ladder with extensible registry of renderers.
Each analysis type has its own renderer class.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import streamlit as st

from clinical_analytics.core.analysis_result import AnalysisResult


@runtime_checkable
class Renderer(Protocol):
    """Protocol for result renderers."""

    def render(self, result: AnalysisResult, *, query_text: str | None = None) -> None:
        """Render analysis result to UI."""
        ...


# Global registry for module-level decorator
RENDERERS: dict[str, Renderer] = {}


def register(result_type: str):
    """Decorator to register a renderer for a result type in global registry."""

    def decorator(renderer_class: type) -> type:
        RENDERERS[result_type] = renderer_class()
        return renderer_class

    return decorator


def render_result(result: AnalysisResult, *, query_text: str | None = None) -> None:
    """Render result using global registry."""
    renderer = RENDERERS.get(result.type)
    if not renderer:
        raise ValueError(f"Unknown result type: {result.type}")
    renderer.render(result, query_text=query_text)


class RendererRegistry:
    """
    Registry for result renderers with Strategy pattern.

    Allows dynamic registration and lookup of renderers by result type.
    """

    def __init__(self) -> None:
        """Initialize empty registry."""
        self._renderers: dict[str, Renderer] = {}

    def register(self, result_type: str):
        """Decorator to register a renderer for a result type."""

        def decorator(renderer_class: type) -> type:
            self._renderers[result_type] = renderer_class()
            return renderer_class

        return decorator

    def get(self, result_type: str) -> Renderer:
        """Get renderer for result type, raising ValueError if not found."""
        renderer = self._renderers.get(result_type)
        if renderer is None:
            raise ValueError(f"Unknown result type: {result_type}")
        return renderer

    def render(self, result: AnalysisResult, *, query_text: str | None = None) -> None:
        """Render result using registered renderer."""
        renderer = self.get(result.type)
        renderer.render(result, query_text=query_text)


# =============================================================================
# Concrete Renderer Implementations
# =============================================================================


class BaseRenderer:
    """Base class with common rendering utilities."""

    def _render_error(self, result: AnalysisResult) -> bool:
        """Render error message if present. Returns True if error was rendered."""
        if result.friendly_error_message:
            st.error(result.friendly_error_message)
            return True
        if "error" in result.payload:
            st.error(f"Error: {result.payload['error']}")
            return True
        return False

    def _render_interpretation(self, result: AnalysisResult) -> None:
        """Render LLM interpretation if present."""
        if result.llm_interpretation:
            st.info(result.llm_interpretation)


@register("descriptive")
class DescriptiveRenderer(BaseRenderer):
    """Renderer for descriptive statistics results."""

    def render(self, result: AnalysisResult, *, query_text: str | None = None) -> None:
        """Render descriptive statistics."""
        if self._render_error(result):
            return

        self._render_interpretation(result)

        payload = result.payload
        if "mean" in payload:
            st.metric("Mean", f"{payload['mean']:.2f}")
        if "std" in payload:
            st.metric("Std Dev", f"{payload['std']:.2f}")
        if "count" in payload:
            st.metric("Count", payload["count"])


@register("comparison")
class ComparisonRenderer(BaseRenderer):
    """Renderer for group comparison results."""

    def render(self, result: AnalysisResult, *, query_text: str | None = None) -> None:
        """Render group comparison."""
        if self._render_error(result):
            return

        self._render_interpretation(result)

        payload = result.payload
        cols = st.columns(3)
        if "group_a" in payload:
            cols[0].metric("Group A", f"{payload['group_a']:.2f}")
        if "group_b" in payload:
            cols[1].metric("Group B", f"{payload['group_b']:.2f}")
        if "p_value" in payload:
            cols[2].metric("P-value", f"{payload['p_value']:.4f}")


@register("predictor")
class PredictorRenderer(BaseRenderer):
    """Renderer for predictor/risk factor results."""

    def render(self, result: AnalysisResult, *, query_text: str | None = None) -> None:
        """Render predictor analysis."""
        if self._render_error(result):
            return

        self._render_interpretation(result)
        st.write("Predictor analysis results:")
        st.json(result.payload)


@register("survival")
class SurvivalRenderer(BaseRenderer):
    """Renderer for survival analysis results."""

    def render(self, result: AnalysisResult, *, query_text: str | None = None) -> None:
        """Render survival analysis."""
        if self._render_error(result):
            return

        self._render_interpretation(result)
        st.write("Survival analysis results:")
        st.json(result.payload)


@register("relationship")
class RelationshipRenderer(BaseRenderer):
    """Renderer for relationship/correlation results."""

    def render(self, result: AnalysisResult, *, query_text: str | None = None) -> None:
        """Render relationship analysis."""
        if self._render_error(result):
            return

        self._render_interpretation(result)
        st.write("Relationship analysis results:")
        st.json(result.payload)


@register("count")
class CountRenderer(BaseRenderer):
    """Renderer for count/aggregation results."""

    def render(self, result: AnalysisResult, *, query_text: str | None = None) -> None:
        """Render count result."""
        if self._render_error(result):
            return

        self._render_interpretation(result)

        payload = result.payload
        label = payload.get("label", "Count")
        count = payload.get("count", 0)
        st.metric(label, count)
