# SPDX-License-Identifier: MIT
"""
QueryService - Pure Python query execution service.

Extracted from Streamlit UI to enable UI-agnostic execution.
Manages query parsing, validation, and execution without Streamlit dependencies.
"""

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl
import structlog

from clinical_analytics.analysis.grouping import grouping_problem
from clinical_analytics.analysis.variable_types import short_name
from clinical_analytics.contracts.declarations import Declarations, DeclarationStore, MemoryDeclarationStore
from clinical_analytics.contracts.errors import ContractError, ContractStoreError
from clinical_analytics.contracts.quality import QualityReport
from clinical_analytics.contracts.store import latest_report
from clinical_analytics.core import answer_checks
from clinical_analytics.core.analysis_types import AnalysisContext, AnalysisIntent
from clinical_analytics.core.conversation_manager import ConversationManager
from clinical_analytics.core.execution_gate import PLAN_CHANGED_REASON, format_plan_line, plan_fingerprint
from clinical_analytics.core.log_redaction import error_fields
from clinical_analytics.core.nl_query_config import AUTO_EXECUTE_CONFIDENCE_THRESHOLD
from clinical_analytics.core.nl_query_engine import NLQueryEngine
from clinical_analytics.core.plan_values import absent_value_reason
from clinical_analytics.core.query_plan import CohortStepError, QueryPlan
from clinical_analytics.core.result_presentation import present_analysis
from clinical_analytics.core.semantic import SemanticLayer
from clinical_analytics.core.word_accounting import hold_reason, unjustified_elements, unjustified_reason, unused_words

_GROUP_SIZE_COLUMNS = frozenset({"n", "count", "len", "n_rows", "n_pairs", "events"})
logger = structlog.get_logger()
NO_MATCH_REASON = "No patients matched the question's limits, so there is no value to report."


def all_null_statistics_reason(result: Any, plan: QueryPlan) -> str | None:
    """Plain-language refusal when every requested statistic came back empty, with the real cause.

    A frame of null statistics is not an answer. The reason says why: no patient matched the
    limits, the patients who matched have no recorded value, or (correlation) too few pairs or
    a column that does not vary. Counts are exempt: zero patients is a real answer.
    """
    if result is None or (plan.intent == "COUNT" and not (plan.proportion_column or plan.rate_column)):
        return None
    frame = result if isinstance(result, pl.DataFrame) else _to_polars(result)
    if frame is None:
        return None
    if frame.height == 0:
        return NO_MATCH_REASON
    stats = [
        name for name in frame.columns if name not in {plan.group_by, plan.metric} and name not in _GROUP_SIZE_COLUMNS
    ]
    if not stats or not frame.select(pl.all_horizontal(pl.col(stats).is_null().all())).item():
        return None
    matched = int(frame.get_column("n_rows").sum()) if "n_rows" in frame.columns else None
    if matched == 0:
        return NO_MATCH_REASON
    if plan.proportion_column or plan.rate_column:
        column = plan.proportion_column or plan.rate_column
        return f"No patient who matched has '{column}' recorded, so there is no share to report."
    if "n_pairs" in frame.columns:
        if int(frame.get_column("n_pairs").sum()) < 2:
            return "Cannot compute the correlation: fewer than 2 patients who matched have both values recorded."
        return (
            "Cannot compute the correlation: one of the two columns has the same value for every patient who matched."
        )
    if plan.metric:
        return f"Every patient who matched is missing '{plan.metric}', so there is no value to report."
    return "The patients who matched have no recorded values to report."


def _to_polars(result: Any) -> pl.DataFrame | None:
    try:
        return pl.from_pandas(result)
    except (TypeError, ValueError, ImportError):
        return None


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
    plan_hash: str | None = None  # Canonical hash of the executable plan; a confirmation must return it


class QueryService:
    """
    Pure Python query execution service.

    Extracted from QuestionEngine and Ask_Questions.py to enable UI-agnostic execution.
    Zero Streamlit dependencies.
    """

    def __init__(
        self,
        semantic_layer: SemanticLayer,
        declarations: DeclarationStore | None = None,
        data_dir: Path | None = None,
    ) -> None:
        """
        Initialize query service with semantic layer.

        Args:
            semantic_layer: SemanticLayer instance for query parsing and execution
            declarations: Where confirmed grain and category mappings are stored (default: this
                process only; the API passes the data directory's file store)
            data_dir: Data directory holding quality reports; None: answers carry no report notes
        """
        self.semantic_layer = semantic_layer
        self.declarations: DeclarationStore = declarations if declarations is not None else MemoryDeclarationStore()
        self.data_dir = data_dir
        self._applied_levels: set[tuple[str, tuple[str, ...]]] = set()
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
        confirm_plan_hash: str | None = None,
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
                confirm_plan_hash=confirm_plan_hash,
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
        confirm_plan_hash: str | None = None,
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

        # A confirmation covers exactly the executable plan the user reviewed: it is bound to
        # the canonical plan hash, never to the display line (two plans can share a line).
        # Re-parsing can yield a different plan; that one is held and shown on its own.
        shown_plan_line = format_plan_line(query_plan, context)
        plan_hash = plan_fingerprint(query_plan, context)
        plan_changed = False
        if confirmed and confirm_plan_hash != plan_hash:
            confirmed = False
            plan_changed = True

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
                plan_hash=plan_hash,
            )

        # Refused whatever the confirmation: a grouping by a many-level column (an exact lab
        # value per group) would label each group with a few patients' values.
        grouping_refusal = self._grouping_refusal(query_plan)
        if grouping_refusal:
            return QueryResult(
                plan=query_plan,
                issues=[*issues, {"message": grouping_refusal, "severity": "error"}],
                result=None,
                confidence=query_intent.confidence,
                run_key=None,
                context=context,
                failure_reason=grouping_refusal,
            )

        # Data checks on the columns the plan reads: refuse what would be wrong whatever the
        # doctor says; hold what only they can settle (grain, category spellings, implausible
        # values) until they confirm this exact plan; otherwise note it on the answer.
        declaration_key = dataset_id or upload_id or "dataset"
        findings, unit = self._data_findings(query_plan, declaration_key, dataset_version)
        if findings.refusal:
            return QueryResult(
                plan=query_plan,
                issues=[*issues, {"message": findings.refusal, "severity": "error"}],
                result=None,
                confidence=query_intent.confidence,
                run_key=None,
                context=context,
                failure_reason=findings.refusal,
            )
        if confirmed:
            self._declare(declaration_key, findings)

        # Word accounting (all tiers): a plan that leaves a word of the question unused answers
        # a different question. Hold it for review regardless of confidence; only the user's
        # confirmation of this exact plan (by hash) releases it.
        # Reverse accounting: every element of the plan must also be asked for by the question.
        # The data-check holds join the same hold, so one confirmation covers what was shown.
        if not confirmed:
            accounting = self._accounting_hold_reason(normalized_query, query_plan, context)
            accounting_reason = " ".join([r for r in [accounting, *findings.holds] if r]) or None
            if accounting_reason:
                return QueryResult(
                    plan=query_plan,
                    issues=issues,
                    result=None,
                    confidence=query_intent.confidence,
                    run_key=None,
                    context=context,
                    requires_confirmation=True,
                    failure_reason=accounting_reason,
                    plan_line=shown_plan_line,
                    plan_hash=plan_hash,
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
                    empty_reason = all_null_statistics_reason(execution_result.get("result"), query_plan)
                    if empty_reason:
                        failure_reason = empty_reason
                        issues.append({"message": empty_reason, "severity": "error"})
                    else:
                        # Run the statistics for the plan's intent (t-test, regression, ...).
                        analysis = self.semantic_layer.format_execution_result(execution_result, context)
                        if analysis.get("error"):
                            issues.append({"message": str(analysis["error"]), "severity": "error"})
                        else:
                            analysis = self._with_findings(analysis, findings, unit)
                            claim = present_analysis(analysis).get("claim")
        except CohortStepError as e:
            # Already plain language: say why the limit could not be applied.
            failure_reason = str(e)
            issues.append({"message": failure_reason, "severity": "error"})
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
            plan_hash=plan_hash,
        )

    def _data_findings(
        self, plan: QueryPlan, dataset_key: str, dataset_version: str | None
    ) -> tuple[answer_checks.Findings, str | None]:
        """Run the data checks for ``plan``; returns the findings and what n counts.

        Stored declarations are applied first (count patients once rows are visits; merge
        confirmed spellings), so a confirmed question is not held again.
        """
        findings = answer_checks.Findings()
        try:
            declarations = self.declarations.load(dataset_key)
        except ContractStoreError as e:
            logger.error("declarations_unreadable", dataset_key=dataset_key, **error_fields(e))
            declarations = Declarations()
        self._apply_levels(declarations)
        layer = self.semantic_layer
        grain = answer_checks.grain_of(layer)
        visits = grain.repeated and answer_checks.declared_visits(grain, declarations)
        layer.count_entity_key = grain.key if visits else None  # type: ignore[attr-defined]
        # A COUNT counts patients (distinct key once rows are visits); every other answer,
        # shares included, is over rows: visits at visit grain.
        counts_patients = plan.intent == "COUNT" and not (plan.proportion_column or plan.rate_column)
        unit = None if grain.key is None else ("visits" if visits and not counts_patients else "patients")

        columns = answer_checks.plan_columns(plan)
        known = [c for c in [*columns, *([grain.key] if grain.key else [])] if c in layer.get_base_view().columns]
        frame = layer.get_base_view().select(*dict.fromkeys(known)).to_polars() if known else pl.DataFrame()

        findings.refusal = answer_checks.metric_refusal(plan, frame)
        if findings.refusal:
            return findings, unit
        if grain.repeated and grain.key is not None and not visits:
            findings.holds.append(answer_checks.grain_hold_reason(grain.key))
            findings.grain_to_declare = grain.declaration()
            unit = "patients" if counts_patients else "visits"
        if grain.repeated and grain.key and plan.intent == "COUNT" and plan.group_by:
            findings.refusal = answer_checks.grouped_count_refusal(frame, grain.key, plan.group_by)
            if findings.refusal:
                return findings, unit
        for _column, mapping, sentence in answer_checks.variant_findings(plan, frame, declarations):
            findings.holds.append(sentence)
            findings.levels_to_declare.append(mapping)
        for _column, hold, note in answer_checks.implausible_sentences(plan, frame):
            findings.holds.append(hold)
            findings.notes.append(note)
        findings.notes.extend(answer_checks.report_notes(self._latest_report(dataset_key, dataset_version), columns))
        return findings, unit

    def _latest_report(self, dataset_key: str, dataset_version: str | None) -> QualityReport | None:
        if self.data_dir is None or not dataset_version:
            return None
        try:
            return latest_report(self.data_dir, dataset_key, dataset_version)
        except (ContractError, ValueError) as e:
            logger.error("quality_report_unreadable", dataset_key=dataset_key, **error_fields(e))
            return None

    def _declare(self, dataset_key: str, findings: answer_checks.Findings) -> None:
        """Store what the doctor confirmed by confirming this plan, and apply it now."""
        if findings.grain_to_declare is None and not findings.levels_to_declare:
            return
        declarations = self.declarations.load(dataset_key)
        if findings.grain_to_declare is not None:
            declarations = declarations.with_grain(findings.grain_to_declare)
            self.semantic_layer.count_entity_key = findings.grain_to_declare.key  # type: ignore[attr-defined]
        for mapping in findings.levels_to_declare:
            declarations = declarations.with_levels(mapping)
        self.declarations.save(dataset_key, declarations)
        self._apply_levels(declarations)
        logger.info(
            "data_declarations_confirmed",
            dataset_key=dataset_key,
            grain=findings.grain_to_declare is not None,
            level_mappings=len(findings.levels_to_declare),
        )

    def _apply_levels(self, declarations: Declarations) -> None:
        """Merge each confirmed set of spellings into one level, once per layer."""
        columns = set(self.semantic_layer.get_base_view().columns)
        for mapping in declarations.levels:
            marker = (mapping.column, mapping.spellings)
            if marker in self._applied_levels or mapping.column not in columns:
                continue
            self.semantic_layer.remap_levels(mapping.column, {s: mapping.canonical for s in mapping.spellings})
            self._applied_levels.add(marker)

    @staticmethod
    def _with_findings(analysis: dict[str, Any], findings: answer_checks.Findings, unit: str | None) -> dict[str, Any]:
        """The analysis with what n counts and the data-check notes the answer must carry.

        A column whose implausible values the analysis already discloses gets no second note.
        """
        disclosed = (analysis.get("implausible") or {}).get("column")
        notes = [n for n in findings.notes if not (disclosed and n.startswith(f"{short_name(disclosed)} has "))]
        extra: dict[str, Any] = {}
        if unit is not None:
            extra["unit"] = unit
        if notes:
            extra["data_notes"] = notes
        return {**analysis, **extra}

    def _grouping_refusal(self, plan: QueryPlan) -> str | None:
        """Why the plan's grouping column cannot split patients into groups, or None.

        A correlation's second column is a measurement, not a grouping, so it is exempt.
        """
        if not plan.group_by or plan.intent == "CORRELATIONS":
            return None
        view = self.semantic_layer.get_base_view()
        if plan.group_by not in view.columns:
            return None  # validation reports the missing column
        levels = int(view[plan.group_by].nunique().execute())
        return grouping_problem(plan.group_by, levels)

    def _accounting_hold_reason(self, question: str, plan: QueryPlan, context: AnalysisContext) -> str | None:
        """Why this plan must be reviewed before it runs, or None when question and plan agree.

        Forward: a word of the question the plan does not use. Reverse: an element of the plan
        the question does not ask for. Then: a limit on a value its column does not hold.
        """
        columns = list(self.semantic_layer.get_base_view().columns)
        raw_index = self.semantic_layer.get_column_alias_index()
        alias_index = raw_index if isinstance(raw_index, dict) else None
        values_for = self.nl_engine._distinct_values
        unused = unused_words(question, plan, columns, values_for, alias_index=alias_index, context=context)
        if unused:
            return hold_reason(unused)
        unasked = unjustified_elements(question, plan, columns, values_for, alias_index=alias_index, context=context)
        if unasked:
            return unjustified_reason(unasked)
        # Value presence: a limit on a value the column does not hold would report "0 patients".
        return absent_value_reason(plan, columns, values_for)

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
