# SPDX-License-Identifier: MIT
"""
QueryService - Pure Python query execution service.

Extracted from Streamlit UI to enable UI-agnostic execution.
Manages query parsing, validation, and execution without Streamlit dependencies.
"""

import threading
from dataclasses import dataclass
from typing import Any

from clinical_analytics.core.analysis_types import AnalysisContext, AnalysisIntent
from clinical_analytics.core.conversation_manager import ConversationManager
from clinical_analytics.core.execution_gate import PLAN_CHANGED_REASON, format_plan_line
from clinical_analytics.core.nl_query_config import AUTO_EXECUTE_CONFIDENCE_THRESHOLD
from clinical_analytics.core.nl_query_engine import NLQueryEngine
from clinical_analytics.core.query_plan import QueryPlan
from clinical_analytics.core.result_presentation import present_analysis
from clinical_analytics.core.semantic import SemanticLayer


@dataclass
class QueryResult:
    """Result of query execution."""

    plan: QueryPlan | None  # QueryPlan from NLQueryEngine (None if parse failed)
    issues: list[dict[str, Any]]  # Validation issues (list of dicts with 'message', 'severity')
    result: dict[str, Any] | None  # Analysis result if executed
    confidence: float  # Parsing confidence (0.0-1.0)
    run_key: str | None  # Deterministic run key (None if parse failed)
    context: AnalysisContext | None  # Analysis context (None if parse failed)
    # Statistical result from SemanticLayer.format_execution_result (None if not executed)
    analysis: dict[str, Any] | None = None
    claim: str | None = None  # Plain-language claim (issue #52)
    requires_confirmation: bool = False  # Low-confidence hold
    failure_reason: str | None = None  # Hold or refuse reason
    plan_line: str | None = None  # One-line plan summary for confirm UI


class QueryService:
    """
    Pure Python query execution service.

    Extracted from QuestionEngine and Ask_Questions.py to enable UI-agnostic execution.
    Zero Streamlit dependencies.
    """

    def __init__(self, semantic_layer: SemanticLayer) -> None:
        """
        Initialize query service with semantic layer.

        Args:
            semantic_layer: SemanticLayer instance for query parsing and execution
        """
        self.semantic_layer = semantic_layer
        self.nl_engine = NLQueryEngine(semantic_layer)
        self.conversation_manager = ConversationManager()

    def ask(
        self,
        question: str,
        dataset_id: str | None = None,
        upload_id: str | None = None,
        dataset_version: str | None = None,
        conversation_history: list[dict[str, Any]] | None = None,
        confirmed: bool = False,
        confirm_plan_line: str | None = None,
    ) -> QueryResult:
        """Parse and execute a query on this layer's single DuckDB connection."""
        lock = getattr(self.semantic_layer, "_con_lock", None)
        if lock is None or not hasattr(lock, "acquire"):
            lock = threading.Lock()
            try:
                self.semantic_layer._con_lock = lock
            except (AttributeError, TypeError):
                pass
        with lock:
            return self._ask_unlocked(
                question,
                dataset_id=dataset_id,
                upload_id=upload_id,
                dataset_version=dataset_version,
                conversation_history=conversation_history,
                confirmed=confirmed,
                confirm_plan_line=confirm_plan_line,
            )

    def _ask_unlocked(
        self,
        question: str,
        dataset_id: str | None = None,
        upload_id: str | None = None,
        dataset_version: str | None = None,
        conversation_history: list[dict[str, Any]] | None = None,
        confirmed: bool = False,
        confirm_plan_line: str | None = None,
    ) -> QueryResult:
        """
        Parse and execute query, return structured result.

        Args:
            question: Natural language query text
            dataset_id: Optional dataset identifier for logging
            upload_id: Optional upload identifier for logging
            dataset_version: Optional dataset version identifier (defaults to upload_id or dataset_id)
            conversation_history: Optional conversation history for context-aware parsing

        Returns:
            QueryResult with plan, issues, result, confidence, run_key, context
        """
        # Normalize query (extracted from Ask_Questions.py normalize_query)
        normalized_query = self.conversation_manager.normalize_query(question)

        # Reject empty queries
        if not normalized_query:
            return QueryResult(
                plan=None,
                issues=[{"message": "Query cannot be empty", "severity": "error"}],
                result=None,
                confidence=0.0,
                run_key=None,
                context=None,
            )

        # Parse query with NLQueryEngine
        query_intent = self.nl_engine.parse_query(
            normalized_query,
            dataset_id=dataset_id,
            upload_id=upload_id,
            conversation_history=conversation_history,
        )

        # Handle parse failure
        if query_intent is None:
            return QueryResult(
                plan=None,
                issues=[{"message": "Failed to parse query", "severity": "error"}],
                result=None,
                confidence=0.0,
                run_key=None,
                context=None,
            )

        # Convert QueryIntent to QueryPlan
        dataset_version = dataset_version or upload_id or dataset_id or "unknown"
        query_plan: QueryPlan = self.nl_engine._intent_to_plan(query_intent, dataset_version)

        # Generate run_key using semantic layer (canonical implementation)
        run_key = self._generate_run_key(query_plan, normalized_query)

        # Validate query plan. The semantic layer returns {valid, error}, not issues.
        validation_result = self.semantic_layer._validate_query_plan(query_plan)
        issues = list(validation_result.get("issues") or [])
        if not validation_result.get("valid", False) and not any(item.get("severity") == "error" for item in issues):
            issues.append(
                {
                    "message": validation_result.get("error") or "Query plan is invalid",
                    "severity": "error",
                }
            )

        # Convert AnalysisContext for compatibility (if needed)
        context = self._intent_to_context(query_intent, query_plan)

        # Execute via semantic gate (completeness / confidence hold / confirm)
        result: dict[str, Any] | None = None
        analysis: dict[str, Any] | None = None
        claim: str | None = None
        requires_confirmation = False
        failure_reason: str | None = None
        plan_line: str | None = None
        execution_run_key: str | None = run_key

        # A confirmation covers the plan the user saw. Re-parsing (e.g. via the LLM tier) can
        # yield a different plan; that one must be shown and confirmed on its own.
        # A confirmation without the plan line it covers releases nothing.
        shown_plan_line = format_plan_line(query_plan, context)
        plan_changed = False
        if confirmed and (not confirm_plan_line or shown_plan_line != confirm_plan_line):
            confirmed = False
            plan_changed = bool(confirm_plan_line)

        if not validation_result.get("valid", False):
            # Fail closed: an invalid plan is refused here and never reaches execution.
            failure_reason = next(
                (str(item["message"]) for item in issues if item.get("severity") == "error"),
                "Query plan is invalid",
            )
            return QueryResult(
                plan=query_plan,
                issues=issues,
                result=None,
                confidence=query_intent.confidence,
                run_key=None,
                context=context,
                failure_reason=failure_reason,
            )

        if plan_changed:
            # Hold before execution: the gate only sees confidence, so a confident reparse
            # to a different plan would otherwise run without the user ever seeing it.
            return QueryResult(
                plan=query_plan,
                issues=issues,
                result=None,
                confidence=query_intent.confidence,
                run_key=None,
                context=context,
                requires_confirmation=True,
                failure_reason=PLAN_CHANGED_REASON,
                plan_line=shown_plan_line,
            )

        try:
            execution_result = self.semantic_layer.execute_query_plan(
                query_plan,
                confidence_threshold=AUTO_EXECUTE_CONFIDENCE_THRESHOLD,
                query_text=normalized_query,
                confirmed=confirmed,
            )
            result = execution_result if isinstance(execution_result, dict) else {"value": execution_result}

            if isinstance(execution_result, dict):
                requires_confirmation = bool(execution_result.get("requires_confirmation"))
                failure_reason = execution_result.get("failure_reason")
                next_run_key = execution_result.get("run_key")
                if isinstance(next_run_key, str) and next_run_key:
                    execution_run_key = next_run_key
                elif requires_confirmation or execution_result.get("success") is False:
                    execution_run_key = None

                if requires_confirmation:
                    plan_line = shown_plan_line
                elif execution_result.get("success") is False:
                    reason = failure_reason or ((execution_result.get("warnings") or ["Execution failed"])[0])
                    issues.append({"message": str(reason), "severity": "error"})
                elif execution_result.get("success") is True:
                    # Run the statistics for the plan's intent (t-test, regression, ...).
                    analysis = self.semantic_layer.format_execution_result(execution_result, context)
                    if analysis.get("error"):
                        issues.append({"message": str(analysis["error"]), "severity": "error"})
                    else:
                        claim = present_analysis(analysis).get("claim")
        except Exception as e:
            issues.append({"message": f"Execution failed: {str(e)}", "severity": "error"})

        return QueryResult(
            plan=query_plan,
            issues=issues,
            result=result,
            confidence=query_intent.confidence,
            run_key=execution_run_key,
            context=context,
            analysis=analysis,
            claim=claim,
            requires_confirmation=requires_confirmation,
            failure_reason=failure_reason,
            plan_line=plan_line,
        )

    def _generate_run_key(self, plan: QueryPlan, query_text: str) -> str:
        """
        Generate deterministic run key from plan and query text.

        Delegates to semantic_layer._generate_run_key() (canonical implementation).

        Args:
            plan: QueryPlan to generate key for
            query_text: Normalized query text

        Returns:
            Deterministic hash key for caching
        """
        return self.semantic_layer._generate_run_key(plan, query_text)

    def _intent_to_context(self, intent, plan: QueryPlan) -> AnalysisContext:
        """
        Convert QueryIntent to AnalysisContext for compatibility.

        Args:
            intent: QueryIntent from NLQueryEngine
            plan: QueryPlan from NLQueryEngine

        Returns:
            AnalysisContext instance
        """
        context = AnalysisContext()

        # Map intent type
        intent_map = {
            "DESCRIBE": AnalysisIntent.DESCRIBE,
            "COMPARE_GROUPS": AnalysisIntent.COMPARE_GROUPS,
            "FIND_PREDICTORS": AnalysisIntent.FIND_PREDICTORS,
            "SURVIVAL": AnalysisIntent.EXAMINE_SURVIVAL,
            "CORRELATIONS": AnalysisIntent.EXPLORE_RELATIONSHIPS,
            "COUNT": AnalysisIntent.COUNT,
        }
        context.inferred_intent = intent_map.get(intent.intent_type, AnalysisIntent.DESCRIBE)

        # Set variables from intent
        context.primary_variable = intent.primary_variable
        context.grouping_variable = intent.grouping_variable
        context.predictor_variables = intent.predictor_variables
        context.time_variable = intent.time_variable
        context.event_variable = intent.event_variable
        context.filters = intent.filters

        # Set query plan (type: ignore for AnalysisContext.query_plan which is typed as None)
        context.query_plan = plan

        return context
