"""
AnalysisExecutor - Orchestration for analysis execution pipeline.

Single point of control for cache check -> execute -> enrich -> store.
Decouples orchestration logic from UI rendering.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from typing import TYPE_CHECKING

import structlog

from clinical_analytics.core.analysis_result import AnalysisResult
from clinical_analytics.core.error_translation import translate_error_with_llm
from clinical_analytics.core.result_cache import CachedResult, ResultCache
from clinical_analytics.core.result_interpretation import interpret_result_with_llm
from clinical_analytics.core.state_store import StateStore

if TYPE_CHECKING:
    from clinical_analytics.core.conversation_manager import ConversationManager

logger = structlog.get_logger(__name__)


class AnalysisExecutor:
    """
    Orchestrates analysis execution with caching and enrichment.

    Responsibilities:
    - Check cache before execution
    - Store results in cache after execution
    - Enrich results with error translation
    - Enrich results with LLM interpretation
    - Update conversation history
    """

    def __init__(self, state_store: StateStore, result_cache: ResultCache) -> None:
        """
        Initialize executor with dependencies.

        Args:
            state_store: Session state abstraction
            result_cache: Result caching with LRU eviction
        """
        self._state = state_store
        self._cache = result_cache

    def get_cached(self, run_key: str, dataset_version: str) -> AnalysisResult | None:
        """
        Get cached result if available.

        Args:
            run_key: Unique identifier for the analysis run
            dataset_version: Version of the dataset

        Returns:
            Cached AnalysisResult if found, None otherwise
        """
        cached = self._cache.get(run_key, dataset_version)
        if cached is not None:
            logger.info("cache_hit", run_key=run_key, dataset_version=dataset_version)
            return AnalysisResult(
                type=cached.result.get("type", "unknown"),
                payload=cached.result.get("payload", {}),
                friendly_error_message=cached.result.get("friendly_error_message"),
                llm_interpretation=cached.result.get("llm_interpretation"),
                run_key=run_key,
            )
        logger.debug("cache_miss", run_key=run_key, dataset_version=dataset_version)
        return None

    def cache_result(
        self,
        result: AnalysisResult,
        run_key: str,
        query_text: str,
        dataset_version: str,
    ) -> None:
        """
        Store result in cache.

        Args:
            result: Analysis result to cache
            run_key: Unique identifier for the analysis run
            query_text: Original query text
            dataset_version: Version of the dataset
        """
        cached = CachedResult(
            run_key=run_key,
            query=query_text,
            result={
                "type": result.type,
                "payload": result.payload,
                "friendly_error_message": result.friendly_error_message,
                "llm_interpretation": result.llm_interpretation,
            },
            timestamp=datetime.now(),
            dataset_version=dataset_version,
        )
        self._cache.put(cached)
        logger.info("result_cached", run_key=run_key, dataset_version=dataset_version)

    def enrich_with_error_translation(self, result: AnalysisResult) -> AnalysisResult:
        """
        Enrich error result with friendly message.

        Args:
            result: Analysis result potentially containing error

        Returns:
            AnalysisResult with friendly_error_message if error detected
        """
        if "error" not in result.payload:
            return result

        error_msg = result.payload.get("error", "")
        logger.debug("translating_error", error=error_msg)

        friendly = translate_error_with_llm(error_msg)
        return replace(result, friendly_error_message=friendly)

    def enrich_with_interpretation(self, result: AnalysisResult, query_text: str) -> AnalysisResult:
        """
        Enrich successful result with LLM interpretation.

        Args:
            result: Successful analysis result
            query_text: Original query for context

        Returns:
            AnalysisResult with llm_interpretation
        """
        if "error" in result.payload:
            return result

        logger.debug("generating_interpretation", query=query_text)
        interpretation = interpret_result_with_llm(result.payload, query_text)
        return replace(result, llm_interpretation=interpretation)

    def update_conversation_history(
        self,
        result: AnalysisResult,
        query_text: str,
        run_key: str,
    ) -> None:
        """
        Update conversation manager with result.

        Args:
            result: Analysis result to record
            query_text: Original query
            run_key: Run key for result association
        """
        conversation: ConversationManager | None = self._state.get("conversation_manager")
        if conversation is None:
            logger.debug("no_conversation_manager")
            return

        # Build response content from result
        if result.friendly_error_message:
            content = result.friendly_error_message
        elif result.llm_interpretation:
            content = result.llm_interpretation
        else:
            content = f"Analysis complete: {result.type}"

        conversation.add_message(
            role="assistant",
            content=content,
            run_key=run_key,
        )
        logger.info("conversation_updated", run_key=run_key)
