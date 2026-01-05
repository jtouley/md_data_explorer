"""
QueryService - Pure Python query execution service.

Extracted from Streamlit UI to enable UI-agnostic execution.
Manages query parsing, validation, and execution without Streamlit dependencies.
"""

from dataclasses import dataclass
from typing import Any

from clinical_analytics.core.conversation_manager import ConversationManager
from clinical_analytics.core.nl_query_engine import NLQueryEngine
from clinical_analytics.core.query_plan import QueryPlan
from clinical_analytics.core.semantic import SemanticLayer
from clinical_analytics.ui.components.question_engine import AnalysisContext, AnalysisIntent


@dataclass
class QueryResult:
    """Result of query execution."""

    plan: QueryPlan | None  # QueryPlan from NLQueryEngine (None if parse failed)
    issues: list[dict[str, Any]]  # Validation issues (list of dicts with 'message', 'severity')
    result: dict[str, Any] | None  # Analysis result if executed
    confidence: float  # Parsing confidence (0.0-1.0)
    run_key: str | None  # Deterministic run key (None if parse failed)
    context: AnalysisContext | None  # Analysis context (None if parse failed)


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

        # Validate query plan
        validation_result = self.semantic_layer._validate_query_plan(query_plan)
        issues = validation_result.get("issues", [])

        # Convert AnalysisContext for compatibility (if needed)
        context = self._intent_to_context(query_intent, query_plan)

        # Execute if valid
        result = None
        if validation_result.get("valid", False):
            try:
                execution_result = self.semantic_layer.execute_query_plan(query_plan)
                result = execution_result
            except Exception as e:
                issues.append({"message": f"Execution failed: {str(e)}", "severity": "error"})

        return QueryResult(
            plan=query_plan,
            issues=issues,
            result=result,
            confidence=query_intent.confidence,
            run_key=run_key,
            context=context,
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
