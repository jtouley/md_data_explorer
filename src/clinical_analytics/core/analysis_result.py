"""
AnalysisResult - Typed domain model for analysis execution results.

This dataclass provides a consistent, immutable structure for all analysis
results flowing through the system. It replaces ad-hoc dict structures with
typed fields for better IDE support and runtime safety.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class AnalysisResult:
    """
    Immutable result of an analysis execution.

    Attributes:
        type: Analysis type identifier (e.g., "descriptive", "comparison", "predictor")
        payload: Raw analysis data as dict (stats, p-values, etc.)
        friendly_error_message: Human-readable error message for failures
        llm_interpretation: LLM-generated interpretation of results
        run_key: Cache key for result deduplication
    """

    type: str
    payload: dict[str, Any]
    friendly_error_message: str | None = None
    llm_interpretation: str | None = None
    run_key: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AnalysisResult:
        """
        Construct an AnalysisResult from a raw dictionary.

        Args:
            data: Dictionary with keys matching AnalysisResult fields.
                  Required: 'type', 'payload'
                  Optional: 'friendly_error_message', 'llm_interpretation', 'run_key'

        Returns:
            Frozen AnalysisResult instance.
        """
        return cls(
            type=data["type"],
            payload=data["payload"],
            friendly_error_message=data.get("friendly_error_message"),
            llm_interpretation=data.get("llm_interpretation"),
            run_key=data.get("run_key"),
        )
