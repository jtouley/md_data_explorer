# SPDX-License-Identifier: MIT
"""
QueryService - Pure Python query execution service.

Extracted from Streamlit UI to enable UI-agnostic execution.
Manages query parsing, validation, and execution without Streamlit dependencies.
"""

import dataclasses
import hashlib
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl
import structlog

from clinical_analytics.analysis.grouping import grouping_problem
from clinical_analytics.analysis.variable_types import short_name
from clinical_analytics.contracts.declarations import (
    Declarations,
    DeclarationStore,
    LevelMapping,
    MemoryDeclarationStore,
    UnitRule,
)
from clinical_analytics.contracts.errors import ContractError, ContractStoreError, InvalidDeclarationKeyError
from clinical_analytics.contracts.quality import QualityReport
from clinical_analytics.contracts.store import latest_report
from clinical_analytics.core import answer_checks, unit_checks
from clinical_analytics.core.analysis_types import AnalysisContext, AnalysisIntent
from clinical_analytics.core.conversation_manager import ConversationManager
from clinical_analytics.core.execution_gate import PLAN_CHANGED_REASON, format_plan_line, plan_fingerprint
from clinical_analytics.core.log_redaction import error_fields
from clinical_analytics.core.nl_query_config import AUTO_EXECUTE_CONFIDENCE_THRESHOLD
from clinical_analytics.core.nl_query_engine import NLQueryEngine
from clinical_analytics.core.plan_values import absent_value_reason, plan_values
from clinical_analytics.core.query_plan import CohortStepError, QueryPlan
from clinical_analytics.core.result_presentation import present_analysis
from clinical_analytics.core.semantic import SemanticLayer
from clinical_analytics.core.word_accounting import (
    coded_grouping_reason,
    explain_hold,
    indicator_statistic_reason,
    measurement_grouping_reason,
    unjustified_elements,
    unjustified_reason,
)

_GROUP_SIZE_COLUMNS = frozenset({"n", "count", "len", "n_rows", "n_pairs", "events"})
logger = structlog.get_logger()
NO_MATCH_REASON = "No patients matched the question's limits, so there is no value to report."
CONFIRMATION_NOT_SAVED_REASON = (
    "I couldn't save your confirmation, so the question was not run. Try again; if it keeps "
    "happening, the dataset's saved settings may be damaged."
)
INVALID_DATASET_KEY_REASON = (
    "This dataset's ID is not valid, so the question was not run. Open the dataset again from the list."
)
_ENTITY_COUNTED = frozenset({"proportion", "rate"})


def _unit_of(analysis: dict[str, Any], plan: QueryPlan, findings: answer_checks.Findings) -> str | None:
    """What n counts in ``analysis``: "patients" only when the counted entity is the patient
    key (or rows are one per patient), rows as "visits" or "rows" by the grain, and no unit for
    any other entity (a COUNT of distinct wards is not a count of patients)."""
    kind = analysis.get("type")
    frequency = kind == "descriptive" and analysis.get("value_counts") is not None and not analysis.get("is_numeric")
    visit_key = findings.key if findings.visits else None
    if kind == "count":
        entity = plan.entity_key or visit_key
    elif kind in _ENTITY_COUNTED or frequency:
        entity = visit_key  # shares and frequency tables count distinct patients once rows are visits
    else:
        entity = None  # a statistic over rows
    if entity is not None:
        return "patients" if entity == findings.key else None
    if findings.visits:
        return "visits"
    return "patients" if findings.rows_are_patients else "rows"


def _confirmed_value(value: Any, lookup: dict[str, str]) -> Any:
    """``value`` with each confirmed spelling replaced by its merged level (lists item by item)."""
    if isinstance(value, list):
        return [_confirmed_value(item, lookup) for item in value]
    return lookup.get(value, value) if isinstance(value, str) else value


def remapped_plan(plan: QueryPlan, mappings: tuple[LevelMapping, ...]) -> QueryPlan:
    """``plan`` with every limit, share and rate value that is a confirmed spelling replaced by its
    merged level (after ``remap_levels`` the base view holds only the merged level). For
    execution only: word accounting, the hash and the plan line read the plan as parsed."""
    lookups: dict[str, dict[str, str]] = {}
    for mapping in mappings:
        lookups.setdefault(mapping.column, {}).update({s: mapping.canonical for s in mapping.spellings})
    if not lookups:
        return plan

    def mapped(column: str | None, value: Any) -> Any:
        return _confirmed_value(value, lookups.get(str(column), {}))

    return dataclasses.replace(
        plan,
        filters=[dataclasses.replace(f, value=mapped(f.column, f.value)) for f in plan.filters],
        cohort_steps=[
            dataclasses.replace(step, value=mapped(step.column, step.value)) if step.op == "eq" else step
            for step in plan.cohort_steps
        ],
        proportion_value=mapped(plan.proportion_column, plan.proportion_value),
        rate_value=mapped(plan.rate_column, plan.rate_value),
    )


def merged_words(plan: QueryPlan, mappings: tuple[LevelMapping, ...]) -> dict[tuple[str, str], tuple[str, ...]]:
    """(column, merged level) -> the words a plan line shows beside it: the spelling the question
    used, then the other spellings held by at least ``MIN_CELL_SIZE`` rows (``LevelMapping.named``),
    so "sex = M" reads "sex = Male (includes 'M', 'male')" and a rare spelling is never named."""
    asked: dict[tuple[str, str], list[str]] = {}
    by_spelling = {(m.column, s): m for m in mappings for s in m.spellings}
    for column, value, _compared in plan_values(plan):
        for item in value if isinstance(value, list) else [value]:
            mapping = by_spelling.get((str(column), item)) if isinstance(item, str) else None
            if mapping is None:
                continue
            words = asked.setdefault((mapping.column, mapping.canonical), [])
            if item != mapping.canonical and item not in words:
                words.append(item)
            for named in sorted(mapping.named):
                if named != mapping.canonical and named not in words:
                    words.append(named)
    return {key: tuple(words) for key, words in asked.items() if words}


def option_hash(plan_hash: str, choice: unit_checks.UnitChoice) -> str:
    """The hash that confirms ``plan_hash`` with ``choice`` (how a unit hold is settled): each
    option of a hold has its own, so the doctor's pick is bound to the plan they saw."""
    return hashlib.sha256(f"{plan_hash}|{choice.key}".encode()).hexdigest()


def confirmation_hash(parsed: QueryPlan, executed: QueryPlan, context: AnalysisContext) -> str:
    """The hash a confirmation must return: the plan as parsed, plus the plan that runs when a
    confirmed merge changes it, so a merge confirmed after the hold was shown changes the hash."""
    parsed_hash = plan_fingerprint(parsed, context)
    if executed == parsed:
        return parsed_hash
    executed_hash = plan_fingerprint(executed, context)
    return hashlib.sha256(f"{parsed_hash}:{executed_hash}".encode()).hexdigest()


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


@dataclass(frozen=True)
class ConfirmOption:
    """One way to settle a hold (a unit hold offers two or more): send ``plan_hash`` back as
    ``confirm_plan_hash`` to run the plan with this choice."""

    label: str
    plan_hash: str


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
    # One-line plan summary: the line held for confirmation; on an answer, the line it ran on
    # when the plan alone would not say so (merged spellings, unit conversions), else None and
    # format_plan_line(plan, context) gives the same line.
    plan_line: str | None = None
    plan_hash: str | None = None  # Canonical hash of the executable plan; a confirmation must return it
    # A unit hold is settled by picking one of these; a plain plan_hash confirmation does not pick.
    confirm_options: list[ConfirmOption] = dataclasses.field(default_factory=list)


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
        # A question with no dataset id never persists a declaration: what the doctor confirms
        # holds for this service only, never under a key another dataset could share.
        self._unsaved = MemoryDeclarationStore()
        self.data_dir = data_dir
        self._applied_levels: set[tuple[str, tuple[str, ...], str | None]] = set()
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
            lock = threading.RLock()
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

        # Confirmed declarations, read once per question. Merged spellings are applied before the
        # parse, so the parser reads the merged levels ("male" names Male, not a tie of spellings).
        dataset_version = dataset_version or upload_id or dataset_id or "unknown"
        declaration_key = dataset_id or upload_id
        declarations = self._load_declarations(declaration_key)
        if declarations is None:
            return QueryResult(
                plan=None,
                issues=[{"message": INVALID_DATASET_KEY_REASON, "severity": "error"}],
                result=None,
                confidence=0.0,
                run_key=None,
                context=None,
                failure_reason=INVALID_DATASET_KEY_REASON,
            )
        mappings = declarations.levels_for(dataset_version)
        self._apply_levels(mappings)
        self._apply_units(declarations.units_for(dataset_version))

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
        # The plan as parsed is what the question is checked against; a limit on a confirmed
        # spelling runs on the merged level ("M" -> "Male"), and the line shown says so.
        executed_plan = remapped_plan(query_plan, mappings)
        predictors = [str(p) for p in getattr(context, "predictor_variables", None) or []]
        # The line names each unit conversion already applied to a column the plan reads.
        converted = unit_checks.applied_unit_rules(query_plan, declarations.units_for(dataset_version), predictors)
        used = {*answer_checks.plan_columns(query_plan), *predictors}
        one_unit = [
            unit_checks.one_unit_line(column, unit)
            for column, unit in declarations.one_unit_for(dataset_version).items()
            if column in used
        ]
        merged = merged_words(query_plan, mappings)
        shown_plan_line = unit_checks.with_unit_rules(
            format_plan_line(executed_plan, context, merged=merged), converted, one_unit
        )
        plan_hash = confirmation_hash(query_plan, executed_plan, context)
        plan_changed = False
        # A unit hold is settled by one of its options, each with its own hash (option_hash).
        choice: unit_checks.UnitChoice | None = None
        if confirmed and confirm_plan_hash != plan_hash:
            choice = next((c for c in unit_checks.UNIT_CHOICES if option_hash(plan_hash, c) == confirm_plan_hash), None)
            if choice is None:
                confirmed = False
                plan_changed = True

        if not validation_result.get("valid", False):
            # Fail closed: an invalid plan never reaches execution, and it is refused, never held:
            # confirming it could not make it run. When the parser added filters and the question
            # has words the plan does not use, that misreading is the cause the reader can fix,
            # so it is the reason given.
            misread = (
                self._unused_words_reason(normalized_query, query_plan, context, mappings)
                if query_plan.filters and self._plan_columns_exist(query_plan)
                else None
            )
            failure_reason = misread or next(
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

        # A grouping by a measurement makes one group per exact value, which identifies
        # patients. Refused whatever the words say, and a confirmation cannot release it.
        measured = measurement_grouping_reason(
            query_plan,
            list(self.semantic_layer.get_base_view().columns),
            self.nl_engine._distinct_values,
            row_count=self._row_count(),
            declared_type=self._declared_type,
        )
        if measured:
            return QueryResult(
                plan=query_plan,
                issues=[*issues, {"message": measured, "severity": "error"}],
                result=None,
                confidence=query_intent.confidence,
                run_key=None,
                context=context,
                failure_reason=measured,
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
        findings = self._data_findings(
            query_plan,
            declaration_key,
            dataset_version,
            declarations,
            predictors,
            question=normalized_query,
            choice=choice if confirmed else None,
        )
        options = [
            ConfirmOption(label=o.label, plan_hash=option_hash(plan_hash, o.choice)) for o in findings.unit_options
        ]
        if options and confirmed and not findings.refusal:
            # Confirmed without picking how to read the units: nothing is stored or run.
            return QueryResult(
                plan=query_plan,
                issues=issues,
                result=None,
                confidence=query_intent.confidence,
                run_key=None,
                context=context,
                requires_confirmation=True,
                failure_reason=" ".join(findings.holds),
                plan_line=shown_plan_line,
                plan_hash=plan_hash,
                confirm_options=options,
            )
        if confirmed and not findings.refusal:
            findings.refusal, declarations = self._declare(declaration_key, findings, declarations)
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
        # Word accounting (all tiers): a plan that leaves a word of the question unused answers
        # a different question. Hold it for review regardless of confidence; only the user's
        # confirmation of this exact plan (by hash) releases it.
        # Reverse accounting: every element of the plan must also be asked for by the question.
        # The data-check holds join the same hold, so one confirmation covers what was shown.
        if not confirmed:
            accounting = self._accounting_hold_reason(normalized_query, query_plan, context, mappings)
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
                    # With a limit the question gives in the other unit, the line says how it runs.
                    plan_line=unit_checks.with_unit_rules(
                        format_plan_line(executed_plan, context, merged=merged),
                        converted,
                        [*one_unit, *findings.plan_line_notes],
                    ),
                    plan_hash=plan_hash,
                    confirm_options=options,
                )

        # A limit on a spelling the doctor confirmed as one category reads the merged level
        # ("M" -> "Male"), so it is not a limit on a value the merged view no longer holds. Remapped
        # again here: this confirmation may have just merged the spellings.
        parsed_plan, levels = query_plan, declarations.levels_for(dataset_version)
        query_plan = remapped_plan(parsed_plan, levels)
        # A number in the other unit is compared in the column's unit (7 mmol/L -> 126.1 mg/dL);
        # the question's constraints are still checked against the numbers it states.
        stated_plan = query_plan
        query_plan = unit_checks.converted_plan(query_plan, findings.limit_conversions)
        # The line the answer ran on: the plan as confirmed, with what this confirmation merged
        # and converted, so it reads as the line a later question on the same plan shows.
        answer_line = unit_checks.with_unit_rules(
            format_plan_line(stated_plan, context, merged=merged_words(parsed_plan, levels)),
            converted,
            [*one_unit, *findings.plan_line_notes],
        )
        if getattr(context, "query_plan", None) is not None:
            context.query_plan = query_plan  # the answer formats the plan that ran

        try:
            stated = {"stated_plan": stated_plan} if stated_plan is not query_plan else {}
            execution_result = self.semantic_layer.execute_query_plan(
                query_plan,
                confidence_threshold=AUTO_EXECUTE_CONFIDENCE_THRESHOLD,
                query_text=normalized_query,
                confirmed=confirmed,
                **stated,
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
                            analysis = self._with_findings(analysis, findings, query_plan)
                            claim = present_analysis(analysis).get("claim")
                            # Every answer comes with the line it ran on; None only where the
                            # plan's own line is that line (no merge, conversion or converted limit).
                            if answer_line != format_plan_line(query_plan, context):
                                plan_line = answer_line
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

    def _store(self, dataset_key: str | None) -> tuple[DeclarationStore, str]:
        """The store and key for a dataset's declarations: in memory for this service when the
        question names no dataset."""
        if dataset_key is None:
            return self._unsaved, "unsaved"
        return self.declarations, dataset_key

    def _load_declarations(self, dataset_key: str | None) -> Declarations | None:
        """The dataset's declarations; None when its id is not a valid key (the question is then
        refused). Unreadable declarations count as none, so a confirmed question is held again."""
        store, key = self._store(dataset_key)
        try:
            return store.load(key)
        except InvalidDeclarationKeyError as e:
            logger.error("declarations_key_invalid", **error_fields(e))
            return None
        except ContractStoreError as e:
            # Unreadable: nothing confirmed applies, so the question is held again.
            logger.error("declarations_unreadable", dataset_key=dataset_key, **error_fields(e))
            return Declarations()

    def _data_findings(
        self,
        plan: QueryPlan,
        dataset_key: str | None,
        dataset_version: str | None,
        declarations: Declarations,
        predictors: list[str] | None = None,
        question: str = "",
        choice: unit_checks.UnitChoice | None = None,
    ) -> answer_checks.Findings:
        """Run the data checks for ``plan``. ``findings.visits``: rows are declared visits on this
        dataset version.

        ``declarations`` (loaded once per question, merges already applied) decide whether rows
        are visits, so a confirmed question is not held again. Every check is a DuckDB aggregate
        (answer_checks), and value counts are shared per column: no column is copied per question.
        """
        findings = answer_checks.Findings()
        layer = self.semantic_layer
        grain = answer_checks.grain_of(layer)
        visits = grain.repeated and answer_checks.declared_visits(grain, declarations, dataset_version)
        layer.count_entity_key = grain.key if visits else None
        findings.keyed, findings.visits = grain.key is not None, visits
        findings.key, findings.rows_are_patients = grain.key, grain.rows_are_patients

        findings.refusal = answer_checks.metric_refusal(plan, layer)
        if findings.refusal:
            return findings
        mixed = self._unit_findings(findings, plan, declarations, dataset_version, predictors or [], question, choice)
        if findings.refusal:
            return findings
        if grain.repeated and grain.key is not None and not visits:
            findings.holds.append(answer_checks.grain_hold_reason(grain.key))
            findings.grain_to_declare = grain.declaration(dataset_version)
        if grain.repeated and grain.key is not None:
            # Checked before and after the confirmation: patients are counted once per group, and
            # a share counts patients, only when each patient has one value.
            if plan.intent == "COUNT" and plan.group_by and not (plan.proportion_column or plan.rate_column):
                findings.refusal = answer_checks.grouped_count_refusal(layer, grain.key, plan.group_by)
            elif plan.proportion_column or plan.rate_column:
                findings.refusal = answer_checks.share_refusal(layer, grain.key, plan)
            if findings.refusal:
                return findings
        for _column, mapping, sentence in answer_checks.variant_findings(plan, layer, declarations, dataset_version):
            findings.holds.append(sentence)
            findings.levels_to_declare.append(mapping)
        for column, hold, note in answer_checks.implausible_sentences(plan, layer):
            if column in mixed:
                continue  # one unit's values look implausible in the other: the unit hold covers it
            findings.holds.append(hold)
            findings.notes.append(note)
        null_keys = answer_checks.missing_key_note(grain, visits or grain.repeated)
        if null_keys:
            findings.notes.append(null_keys)
        columns = answer_checks.plan_columns(plan)
        report = self._latest_report(dataset_key, dataset_version)
        findings.notes.extend(answer_checks.report_notes(report, columns))
        findings.notes.extend(answer_checks.missingness_notes(layer, columns, answer_checks.flagged_columns(report)))
        return findings

    def _unit_findings(
        self,
        findings: answer_checks.Findings,
        plan: QueryPlan,
        declarations: Declarations,
        dataset_version: str | None,
        predictors: list[str],
        question: str,
        choice: unit_checks.UnitChoice | None,
    ) -> set[str]:
        """Mixed units (DQ M3) and limits on converted columns: hold until the doctor picks an
        option (``findings.unit_options``), refuse when no option exists. With ``choice`` (a
        confirmed option) record what it stores and which limits it converts. Returns the
        columns read as mixed."""
        declared = declarations.one_unit_for(dataset_version)
        mixed = unit_checks.mixed_columns(plan, self.semantic_layer, dataset_version, predictors, declared)
        unsettled = next((found for found in mixed if not unit_checks.settleable(found)), None)
        if unsettled is not None:
            findings.refusal = unit_checks.mix_sentence(unsettled)
            return {found.mix.column for found in mixed}
        used = {*answer_checks.plan_columns(plan), *predictors}
        findings.notes.extend(unit_checks.one_unit_note(c, unit) for c, unit in declared.items() if c in used)
        stored_rules = unit_checks.applied_unit_rules(plan, declarations.units_for(dataset_version), predictors)
        findings.notes.extend(f"Data check: {unit_checks.unit_rule_text(rule)}." for rule in stored_rules)
        stored = unit_checks.unit_limits(plan, question, unit_checks.last_rules(stored_rules))
        pending = unit_checks.unit_limits(
            plan, question, {found.mix.column: found.rule for found in mixed if found.rule is not None}
        )
        offered = unit_checks.unit_options(mixed, unit_checks.unstated(stored), unit_checks.unstated(pending))
        picked = next((option.choice for option in offered if option.choice == choice), None)
        if offered and picked is None:
            findings.holds.extend(unit_checks.mix_sentence(found) for found in mixed)
            if unit_checks.unstated(stored):
                findings.holds.append(unit_checks.limit_sentence(unit_checks.unstated(stored)))
            if any(o.choice.units == "convert" for o in offered) and unit_checks.unstated(pending):
                findings.holds.append(unit_checks.limit_sentence(unit_checks.unstated(pending)))
            findings.unit_options = offered
            return {found.mix.column for found in mixed}
        units = picked.units if picked is not None else None
        if units == "convert":
            findings.units_to_declare.extend(found.rule for found in mixed if found.rule is not None)
        elif units is not None:
            chosen = [found.one_unit(units) for found in mixed]
            findings.one_unit_to_declare.extend(d for d in chosen if d is not None)
        # The line the answer comes with names what the choice did, as a later question's does.
        findings.plan_line_notes = [
            *(unit_checks.unit_rule_text(rule) for rule in findings.units_to_declare),
            *(unit_checks.one_unit_line(d.column, d.unit) for d in findings.one_unit_to_declare),
        ]
        limits = [*stored, *(pending if units == "convert" else [])]
        in_other = picked is not None and picked.limits == "source"
        findings.limit_conversions = [
            limit
            for limit in limits
            if limit.stated == limit.rule.source_unit or (in_other and limit in unit_checks.unstated(limits))
        ]
        findings.plan_line_notes += [unit_checks.limit_line_note(limit) for limit in findings.limit_conversions]
        return {found.mix.column for found in mixed}

    def _latest_report(self, dataset_key: str | None, dataset_version: str | None) -> QualityReport | None:
        if self.data_dir is None or not dataset_version or dataset_key is None:
            return None
        try:
            return latest_report(self.data_dir, dataset_key, dataset_version)
        except (ContractError, ValueError) as e:
            logger.error("quality_report_unreadable", dataset_key=dataset_key, **error_fields(e))
            return None

    def _declare(
        self, dataset_key: str | None, findings: answer_checks.Findings, declarations: Declarations
    ) -> tuple[str | None, Declarations]:
        """Store what the doctor confirmed by confirming this plan, and apply it now.

        Returns (refusal, declarations now stored). The refusal is plain when the confirmation
        cannot be stored: the answer would otherwise rest on a confirmation the next question
        does not know about.
        """
        grain, levels, units = findings.grain_to_declare, findings.levels_to_declare, findings.units_to_declare
        one_unit = findings.one_unit_to_declare
        if grain is None and not levels and not units and not one_unit:
            return None, declarations

        def change(declarations: Declarations) -> Declarations:
            if grain is not None:
                declarations = declarations.with_grain(grain)
            for mapping in levels:
                declarations = declarations.with_levels(mapping)
            for rule in units:
                declarations = declarations.with_unit(rule)
            for declaration in one_unit:
                declarations = declarations.with_one_unit(declaration)
            return declarations

        store, key = self._store(dataset_key)
        try:
            declarations = store.update(key, change)
        except ContractStoreError as e:
            logger.error("declarations_not_saved", dataset_key=dataset_key, **error_fields(e))
            return CONFIRMATION_NOT_SAVED_REASON, declarations
        if grain is not None:
            self.semantic_layer.count_entity_key = grain.key
            findings.visits = True
        if levels:
            self._apply_levels(declarations.levels_for(levels[0].dataset_version))
        if units:
            self._apply_units(declarations.units_for(units[0].dataset_version))
            findings.notes.extend(f"Data check: {unit_checks.unit_rule_text(rule)}." for rule in units)
        findings.notes.extend(unit_checks.one_unit_note(d.column, d.unit) for d in one_unit)
        logger.info(
            "data_declarations_confirmed",
            dataset_key=dataset_key,
            grain=grain is not None,
            level_mappings=len(levels),
            unit_rules=len(units),
            one_unit=len(one_unit),
        )
        return None, declarations

    def _apply_levels(self, mappings: tuple[LevelMapping, ...]) -> None:
        """Merge each set of confirmed spellings into one level, once per layer.

        The NL engine then reads the merged levels only, so "male" names Male instead of tying
        Male and male. Word accounting still knows the spellings (``_values_with_spellings``), so
        a plan that names "M" is checked as asked and runs on Male.
        """
        if not mappings:
            return
        columns = set(self.semantic_layer.get_base_view().columns)
        for mapping in mappings:
            marker = (mapping.column, mapping.spellings, mapping.dataset_version)
            if marker in self._applied_levels or mapping.column not in columns:
                continue
            self.semantic_layer.remap_levels(mapping.column, {s: mapping.canonical for s in mapping.spellings})
            self._applied_levels.add(marker)
            cache: dict[str, list[Any]] | None = getattr(self.nl_engine, "_distinct_value_cache", None)
            if cache is not None:
                cache.pop(mapping.column, None)

    def _apply_units(self, rules: tuple[UnitRule, ...]) -> None:
        """Convert each confirmed unit rule's values, in the order confirmed, so every later
        question reads the column in one unit. The layer applies each rule once (keyed by the
        rule and its dataset version, under its lock), whichever service on it asks first."""
        if not rules:
            return
        columns = set(self.semantic_layer.get_base_view().columns)
        for rule in rules:
            if rule.column not in columns:
                continue
            applied = self.semantic_layer.convert_units(
                rule.model_dump_json(),
                rule.column,
                rule.cutoff,
                below=rule.side == "below",
                scale=rule.scale,
                offset=rule.offset,
            )
            if not applied:
                continue
            cache: dict[str, list[Any]] | None = getattr(self.nl_engine, "_distinct_value_cache", None)
            if cache is not None:
                cache.pop(rule.column, None)
            logger.info("unit_rule_applied", column=rule.column, source_unit=rule.source_unit, target=rule.target_unit)

    def _values_with_spellings(self, mappings: tuple[LevelMapping, ...]) -> Callable[[str], list[Any]]:
        """The column values word accounting checks a plan against: the merged levels, plus the
        confirmed spellings each one includes (the parse may name "M")."""
        base = self.nl_engine._distinct_values
        extra: dict[str, set[str]] = {}
        for mapping in mappings:
            extra.setdefault(mapping.column, set()).update(mapping.spellings)
        if not extra:
            return base

        def values_for(column: str) -> list[Any]:
            values = base(column)
            if column not in extra:
                return values
            merged = set(values) | extra[column]
            return sorted(merged, key=lambda value: (value is None, str(value)))

        return values_for

    @staticmethod
    def _with_findings(analysis: dict[str, Any], findings: answer_checks.Findings, plan: QueryPlan) -> dict[str, Any]:
        """The analysis with what n counts and the data-check notes the answer must carry.

        Every n says what it counts: patients for counts, shares and frequency tables (distinct
        patients once rows are visits), visits for the statistics computed over visit rows. A
        column whose implausible values the analysis already discloses gets no second note.
        """
        disclosed = (analysis.get("implausible") or {}).get("column")
        notes = [n for n in findings.notes if not (disclosed and n.startswith(f"{short_name(disclosed)} has "))]
        unit = _unit_of(analysis, plan, findings) if findings.keyed else None
        extra: dict[str, Any] = {"unit": unit} if unit else {}
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

    def _plan_columns_exist(self, plan: QueryPlan) -> bool:
        """Every column the plan names is in the dataset (an invalid plan may name one that is not)."""
        columns = set(self.semantic_layer.get_base_view().columns)
        named = {plan.metric, plan.group_by, plan.proportion_column, plan.rate_column}
        named |= {spec.column for spec in plan.filters} | {step.column for step in plan.cohort_steps}
        return {column for column in named if column} <= columns

    def _row_count(self) -> int | None:
        count = self.semantic_layer.get_base_row_count()
        return count if isinstance(count, int) and not isinstance(count, bool) else None

    def _declared_type(self, column: str) -> str | None:
        """The column's type as the dataset's own metadata declares it ("continuous", "categorical")."""
        metadata = self.semantic_layer.get_column_metadata(column)
        declared = metadata.get("type") if isinstance(metadata, dict) else None
        return declared if isinstance(declared, str) else None

    def _unused_words_reason(
        self, question: str, plan: QueryPlan, context: AnalysisContext, mappings: tuple[LevelMapping, ...] = ()
    ) -> str | None:
        """The sentence naming the question words this plan leaves out, or None."""
        columns = list(self.semantic_layer.get_base_view().columns)
        raw_index = self.semantic_layer.get_column_alias_index()
        alias_index = raw_index if isinstance(raw_index, dict) else None
        values_for = self._values_with_spellings(mappings)
        return explain_hold(question, plan, columns, values_for, alias_index=alias_index, context=context)

    def _accounting_hold_reason(
        self, question: str, plan: QueryPlan, context: AnalysisContext, mappings: tuple[LevelMapping, ...] = ()
    ) -> str | None:
        """Why this plan must be reviewed before it runs, or None when question and plan agree.

        Forward: a word of the question the plan does not use. Reverse: an element of the plan
        the question does not ask for. Then: a grouping by many integer codes, and a limit on a
        value its column does not hold. Run on the plan as parsed; a confirmed spelling
        (``mappings``) is a value its column holds.
        """
        columns = list(self.semantic_layer.get_base_view().columns)
        raw_index = self.semantic_layer.get_column_alias_index()
        alias_index = raw_index if isinstance(raw_index, dict) else None
        values_for = self._values_with_spellings(mappings)
        explained = self._unused_words_reason(question, plan, context, mappings)
        if explained:
            return explained
        unasked = unjustified_elements(question, plan, columns, values_for, alias_index=alias_index, context=context)
        if unasked:
            return unjustified_reason(unasked)
        coded = coded_grouping_reason(
            plan, columns, values_for, row_count=self._row_count(), declared_type=self._declared_type
        )
        if coded:
            return coded
        indicator = indicator_statistic_reason(question, plan, values_for)
        if indicator:
            return indicator
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
