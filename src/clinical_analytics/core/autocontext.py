# SPDX-License-Identifier: MIT
"""
AutoContext Packager for Tier 3 LLM Fallback

Builds bounded, privacy-safe context packs to enable grounded query parsing. For an uploaded
dataset with a data contract, the contract is the only source (``contracts.context``); other
datasets (built-in YAML configs, uploads from before contracts) are described from schema
inference, documentation and alias mappings. Neither path carries observed values: no minimum,
maximum or value counts reach the model.
"""

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, TypeAlias

import structlog
import tiktoken
from ibis.common.exceptions import IbisError

if TYPE_CHECKING:
    from clinical_analytics.contracts.context import ContractColumn, ContractContext

logger = structlog.get_logger()

_MIN_DESCRIPTION_TERM = 4

ColumnDtype: TypeAlias = Literal["numeric", "categorical", "datetime", "id", "coded"]

# What SemanticLayer.get_base_view() raises for a config the data cannot satisfy: an outcome
# without "source_column" (KeyError) or a column the table lacks (IbisTypeError).
_BASE_VIEW_ERRORS: tuple[type[Exception], ...] = (IbisError, KeyError)


@dataclass
class ColumnContext:
    """Column metadata for AutoContext."""

    name: str
    normalized_name: str
    system_aliases: list[str]
    user_aliases: list[str]
    dtype: ColumnDtype
    units: str | None = None
    codebook: dict[str, str] | None = None
    description: str | None = None
    is_phi: bool = False
    exclusion_patterns: list[Any] | None = None
    semantic_type: str | None = None


@dataclass
class AutoContext:
    """Bounded, privacy-safe context pack for Tier 3 LLM."""

    dataset: dict[str, str]  # {upload_id, dataset_version, display_name}
    entity_keys: list[str]  # Candidate columns ranked
    columns: list[ColumnContext]
    glossary: dict[str, Any]  # Extracted terms, abbreviations, notes
    constraints: dict[str, Any] = field(default_factory=lambda: {"no_row_level_data": True, "max_tokens": 4000})
    relationships: list[Any] | None = None
    grain: str | None = None  # what one row is, from the contract ("One row per secret_name")


def _extract_entity_keys(inferred_schema: Any) -> list[str]:
    """
    Extract entity keys from inferred schema with deterministic ranking.

    Ranking priority:
    1. patient_id_column (highest priority)
    2. encounter_id (if present in schema)
    3. Other ID-like columns

    Args:
        inferred_schema: InferredSchema instance

    Returns:
        List of entity key column names (ranked by priority)
    """
    entity_keys = []

    # Priority 1: patient_id_column
    if inferred_schema.patient_id_column:
        entity_keys.append(inferred_schema.patient_id_column)

    # Priority 2: Look for encounter_id or similar in all columns
    # Get all columns from schema
    all_columns = []
    if inferred_schema.patient_id_column:
        all_columns.append(inferred_schema.patient_id_column)
    all_columns.extend(inferred_schema.outcome_columns)
    all_columns.extend(inferred_schema.time_columns)
    all_columns.extend(inferred_schema.event_columns)
    all_columns.extend(inferred_schema.categorical_columns)
    all_columns.extend(inferred_schema.continuous_columns)

    # Look for encounter_id or similar patterns
    for col in all_columns:
        col_lower = col.lower()
        if (
            "encounter" in col_lower
            or "visit" in col_lower
            or (col_lower.endswith("_id") and col != inferred_schema.patient_id_column)
        ):
            if col not in entity_keys:
                entity_keys.append(col)

    return entity_keys


def _normalize_column_name(name: str) -> str:
    """
    Normalize column name for AutoContext (lowercase, underscore-separated).

    Args:
        name: Original column name

    Returns:
        Normalized name (e.g., "Current Regimen" -> "current_regimen")
    """
    # Lowercase
    normalized = name.lower()

    # Replace spaces and special chars with underscores
    normalized = re.sub(r"[^\w]+", "_", normalized)

    # Remove multiple underscores
    normalized = re.sub(r"_+", "_", normalized)

    # Strip leading/trailing underscores
    normalized = normalized.strip("_")

    return normalized


def _extract_glossary_from_docs(doc_context: str | None, max_terms: int = 50) -> dict[str, str]:
    """
    Extract glossary terms from documentation context.

    Looks for patterns like:
    - "BMI: Body Mass Index"
    - "LDL: Low-density lipoprotein"
    - Sections with headers "Abbreviations", "Definitions", "Glossary"

    Args:
        doc_context: Documentation text
        max_terms: Maximum number of terms to extract (default: 50)

    Returns:
        Dictionary mapping term -> definition
    """
    if not doc_context:
        return {}

    glossary = {}

    # Pattern 1: "Term: Definition" or "Term - Definition"
    pattern1 = re.compile(r"^[\s*-]*([A-Z][A-Za-z0-9\s]+?)\s*[:-]\s*(.+?)(?:\n|$)", re.MULTILINE)
    matches = pattern1.findall(doc_context)
    for term, definition in matches:
        term = term.strip()
        definition = definition.strip()
        if len(term) > 1 and len(definition) > 3:  # Filter out noise
            glossary[term] = definition
            if len(glossary) >= max_terms:
                break

    # Pattern 2: Look for sections with headers
    section_patterns = [
        r"(?i)abbreviations?\s*:?\s*\n(.*?)(?=\n\n|\n[A-Z]|$)",
        r"(?i)definitions?\s*:?\s*\n(.*?)(?=\n\n|\n[A-Z]|$)",
        r"(?i)glossary\s*:?\s*\n(.*?)(?=\n\n|\n[A-Z]|$)",
    ]

    for pattern in section_patterns:
        matches = re.findall(pattern, doc_context, re.DOTALL)
        for section_text in matches:
            # Extract term: definition pairs from section
            term_defs = re.findall(r"^[\s*-]*([A-Z][A-Za-z0-9\s]+?)\s*[:-]\s*(.+?)(?:\n|$)", section_text, re.MULTILINE)
            for term, definition in term_defs:
                term = term.strip()
                definition = definition.strip()
                if len(term) > 1 and len(definition) > 3:
                    glossary[term] = definition
                    if len(glossary) >= max_terms:
                        break
            if len(glossary) >= max_terms:
                break
        if len(glossary) >= max_terms:
            break

    # Limit to top max_terms
    if len(glossary) > max_terms:
        # Keep first max_terms (deterministic)
        glossary = dict(list(glossary.items())[:max_terms])

    return glossary


def _estimate_tokens(text: str) -> int:
    """
    Estimate token count using tiktoken (accurate for OpenAI models).

    Uses cl100k_base encoding (standard for GPT-4).
    Falls back to character approximation if encoding fails.

    Args:
        text: Text to estimate

    Returns:
        Token count
    """
    try:
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    except Exception:
        # Fallback to approximation if encoding fails
        return len(text) // 4


def _build_column_contexts(
    semantic_layer: Any,
    inferred_schema: Any,
    query_terms: list[str] | None = None,
    resolved_metadata: Any = None,
) -> list[ColumnContext]:
    """
    Build column contexts from semantic layer and schema.

    Args:
        semantic_layer: SemanticLayer instance
        inferred_schema: InferredSchema instance
        query_terms: Optional query terms for relevance filtering
        resolved_metadata: Optional ResolvedDatasetMetadata with enriched column metadata

    Returns:
        List of ColumnContext objects (enriched if resolved_metadata provided)
    """
    # Get all columns from semantic layer
    alias_index = semantic_layer.get_column_alias_index()
    if not alias_index:
        return []

    # Get unique canonical columns
    canonical_columns = list(set(alias_index.values()))

    column_contexts = []

    for col_name in canonical_columns:
        # Get aliases for this column
        system_aliases = [alias for alias, canonical in alias_index.items() if canonical == col_name]

        # Get column metadata
        col_metadata = semantic_layer.get_column_metadata(col_name)

        # Determine dtype
        dtype: Literal["numeric", "categorical", "datetime", "id", "coded"] = "categorical"  # default

        # Check schema first (most reliable)
        if col_name == inferred_schema.patient_id_column or "id" in col_name.lower():
            dtype = "id"
        elif col_name in inferred_schema.continuous_columns:
            dtype = "numeric"
        elif col_name in inferred_schema.time_columns:
            dtype = "datetime"
        elif col_name in inferred_schema.categorical_columns:
            # Check if it's coded (numeric categorical)
            if col_metadata:
                var_type = col_metadata.get("type", "categorical")
                metadata_info = col_metadata.get("metadata", {})
                is_numeric = metadata_info.get("numeric", False)
                if var_type in ("categorical", "binary") and is_numeric:
                    dtype = "coded"
                else:
                    dtype = "categorical"
            else:
                dtype = "categorical"
        elif col_metadata:
            # Fallback to metadata
            var_type = col_metadata.get("type", "categorical")
            metadata_info = col_metadata.get("metadata", {})
            is_numeric = metadata_info.get("numeric", False)

            if var_type == "numeric" or var_type == "continuous":
                dtype = "numeric"
            elif var_type == "datetime":
                dtype = "datetime"
            elif var_type in ("categorical", "binary") and is_numeric:
                dtype = "coded"
            else:
                dtype = "categorical"

        # Get codebook from metadata
        codebook = None
        if col_metadata and "codebook" in col_metadata:
            codebook = col_metadata["codebook"]
        elif inferred_schema.dictionary_metadata:
            codebook = inferred_schema.dictionary_metadata.codebooks.get(col_name)

        # Get units from metadata or column name
        units = None
        if col_metadata and "units" in col_metadata:
            units = col_metadata["units"]
        else:
            # Try to extract from column name using column_parser
            from clinical_analytics.core.column_parser import parse_column_name

            parsed = parse_column_name(col_name)
            if parsed.unit:
                units = parsed.unit

        # No stats: min, max and value counts are observed patient values (a level of 1-10
        # patients among them), and the model needs none of them to read a question.

        # Enrich with resolved metadata if available
        user_aliases: list[str] = []
        description: str | None = None
        is_phi: bool = False
        exclusion_patterns: list[Any] | None = None
        semantic_type: str | None = None

        if resolved_metadata and col_name in resolved_metadata.columns:
            resolved_col = resolved_metadata.columns[col_name]

            if resolved_col.aliases:
                user_aliases = list(resolved_col.aliases)
            if resolved_col.description:
                description = resolved_col.description
            is_phi = resolved_col.is_phi
            if resolved_col.exclusion_patterns:
                exclusion_patterns = list(resolved_col.exclusion_patterns)
            if resolved_col.semantic_type:
                semantic_type = resolved_col.semantic_type.value
            if resolved_col.unit:
                units = resolved_col.unit
            if resolved_col.codebook:
                codebook = {**codebook, **resolved_col.codebook} if codebook else resolved_col.codebook

        # Create ColumnContext with enriched fields
        column_context = ColumnContext(
            name=col_name,
            normalized_name=_normalize_column_name(col_name),
            system_aliases=system_aliases,
            user_aliases=user_aliases,
            dtype=dtype,
            units=units,
            codebook=codebook,
            description=description,
            is_phi=is_phi,
            exclusion_patterns=exclusion_patterns,
            semantic_type=semantic_type,
        )

        column_contexts.append(column_context)

    return column_contexts


def _filter_columns_by_relevance(
    columns: list[ColumnContext],
    query_terms: list[str] | None,
) -> list[ColumnContext]:
    """
    Filter columns by relevance to query terms (deterministic scoring).

    Args:
        columns: List of ColumnContext objects
        query_terms: Optional query terms for filtering

    Returns:
        Filtered and ranked list of ColumnContext objects
    """
    if not query_terms:
        return columns

    # Score each column by relevance
    scored_columns = []
    query_terms_lower = [term.lower() for term in query_terms]

    for col in columns:
        score = 0.0

        # Check column name
        col_name_lower = col.name.lower()
        for term in query_terms_lower:
            if term in col_name_lower:
                score += 1.0
            # Whole word match gets higher score
            if re.search(rf"\b{re.escape(term)}\b", col_name_lower):
                score += 2.0

        # Check aliases
        for alias in col.system_aliases:
            alias_lower = alias.lower()
            for term in query_terms_lower:
                if term in alias_lower:
                    score += 0.5
                if re.search(rf"\b{re.escape(term)}\b", alias_lower):
                    score += 1.0

        # Documented names and descriptions (from the contract): whole words of 4+ letters only,
        # so "the" or "of" do not rank a column.
        documented = " ".join([*col.user_aliases, col.description or ""]).lower()
        for term in query_terms_lower:
            if len(term) >= _MIN_DESCRIPTION_TERM and re.search(rf"\b{re.escape(term)}\b", documented):
                score += 0.5

        scored_columns.append((score, col))

    # Sort by score (descending)
    scored_columns.sort(key=lambda x: x[0], reverse=True)

    return [col for _, col in scored_columns]


def _enforce_token_budget(
    autocontext: AutoContext,
    max_tokens: int,
) -> AutoContext:
    """
    Enforce token budget by truncating columns if needed.

    Args:
        autocontext: AutoContext to truncate
        max_tokens: Maximum token budget

    Returns:
        AutoContext with columns truncated to fit budget
    """
    # Estimate current token count
    current_tokens = _estimate_tokens(str(autocontext.dataset))
    current_tokens += _estimate_tokens(str(autocontext.entity_keys))
    current_tokens += _estimate_tokens(str(autocontext.glossary))
    current_tokens += _estimate_tokens(str(autocontext.constraints))

    # Estimate tokens per column (approximate)
    tokens_per_column = 50  # Conservative estimate

    # Calculate how many columns we can fit
    available_tokens = max_tokens - current_tokens
    max_columns = max(1, available_tokens // tokens_per_column)

    # Truncate columns if needed
    if len(autocontext.columns) > max_columns:
        autocontext.columns = autocontext.columns[:max_columns]

    # Update constraints
    autocontext.constraints["max_tokens"] = max_tokens

    return autocontext


def _reconstruct_inferred_schema_from_semantic_layer(semantic_layer: Any) -> Any:
    """
    Reconstruct minimal InferredSchema from semantic layer config and base view.

    Used when full InferredSchema is not available (e.g., in _llm_parse()).

    Args:
        semantic_layer: SemanticLayer instance

    Returns:
        Minimal InferredSchema object
    """
    from clinical_analytics.core.schema_inference import InferredSchema

    schema = InferredSchema()

    # Extract patient_id from column_mapping (uploaded configs are built from metadata normalized
    # by storage.upload_metadata, so these fields are never None)
    column_mapping = semantic_layer.config.get("column_mapping", {})
    for col, mapped in column_mapping.items():
        if mapped == "patient_id":
            schema.patient_id_column = col
            break

    # Extract outcome columns
    outcomes = semantic_layer.config.get("outcomes", {})
    schema.outcome_columns = list(outcomes.keys())

    # Get all columns from base view
    # A view the config cannot build (an outcome without source_column, a mapped column the data
    # lacks) leaves the context without columns, logged; any other error is a bug and propagates.
    try:
        base_view = semantic_layer.get_base_view()
        all_columns = list(base_view.columns)
    except _BASE_VIEW_ERRORS as e:
        logger.warning(
            "autocontext_base_view_unavailable",
            dataset=getattr(semantic_layer, "dataset_name", None),
            error_type=type(e).__name__,
        )
        all_columns = []

    # Classify columns from variable_types if available
    variable_types = semantic_layer.config.get("variable_types", {})
    for col in all_columns:
        if col == schema.patient_id_column or col in schema.outcome_columns:
            continue

        # A config not built from normalized upload metadata (a YAML dataset) can hold a
        # non-dict entry, e.g. {"age": "numeric"}: classify that column with the default type.
        col_meta = variable_types.get(col)
        if col_meta is not None and not isinstance(col_meta, dict):
            logger.warning("autocontext_variable_type_not_a_dict", column=col, found_type=type(col_meta).__name__)
        var_type = col_meta.get("type", "categorical") if isinstance(col_meta, dict) else "categorical"
        if var_type in ("numeric", "continuous"):
            schema.continuous_columns.append(col)
        elif var_type == "datetime":
            schema.time_columns.append(col)
        else:
            schema.categorical_columns.append(col)

    return schema


def build_autocontext(
    semantic_layer: Any,
    inferred_schema: Any = None,
    doc_context: str | None = None,
    query_terms: list[str] | None = None,
    max_tokens: int = 4000,
    resolved_metadata: Any = None,
    contract: "ContractContext | None" = None,
) -> AutoContext:
    """
    Build AutoContext pack from schema inference, documentation, and aliases.

    **Deterministic construction** - no LLM inference. Same inputs always produce same output.
    Think: compiler metadata, not chat memory.

    Args:
        semantic_layer: SemanticLayer instance
        inferred_schema: Optional InferredSchema from Phase 2 (if None, reconstructed from semantic layer)
        doc_context: Extracted documentation text from Phase 1
        query_terms: Optional query terms for relevance filtering
        max_tokens: Maximum token budget (default: 4000)
        resolved_metadata: Optional ResolvedDatasetMetadata with enriched column metadata
        contract: The upload version's data contract; when given it is the only source (the
            schema, documentation and metadata arguments are not read)

    Returns:
        AutoContext pack with schema context (enriched if resolved_metadata provided)
    """
    if contract is not None:
        return _autocontext_from_contract(semantic_layer, contract, query_terms, max_tokens)

    # Reconstruct inferred_schema if not provided
    if inferred_schema is None:
        inferred_schema = _reconstruct_inferred_schema_from_semantic_layer(semantic_layer)

    # Step 1: Extract entity keys (deterministic ranking)
    entity_keys = _extract_entity_keys(inferred_schema)

    # Step 2: Build column catalog (with resolved metadata enrichment if available)
    column_contexts = _build_column_contexts(semantic_layer, inferred_schema, query_terms, resolved_metadata)

    # Step 3: Filter columns by relevance if query_terms provided
    if query_terms:
        column_contexts = _filter_columns_by_relevance(column_contexts, query_terms)

    # Step 4: Extract glossary from doc_context
    glossary = _extract_glossary_from_docs(doc_context, max_terms=50)

    # Step 5: Build dataset info
    dataset_info = {
        "upload_id": getattr(semantic_layer, "upload_id", None) or "unknown",
        "dataset_version": getattr(semantic_layer, "dataset_version", None) or "unknown",
        "display_name": semantic_layer.config.get("display_name", semantic_layer.dataset_name),
    }

    # Step 5b: Extract relationships from resolved metadata
    relationships: list[Any] | None = None
    if resolved_metadata and resolved_metadata.relationships:
        relationships = list(resolved_metadata.relationships)

    # Step 6: Create AutoContext
    autocontext = AutoContext(
        dataset=dataset_info,
        entity_keys=entity_keys,
        columns=column_contexts,
        glossary=glossary,
        constraints={"no_row_level_data": True, "max_tokens": max_tokens},
        relationships=relationships,
    )

    # Step 7: Enforce token budget
    autocontext = _enforce_token_budget(autocontext, max_tokens)

    return autocontext


def _contract_dtype(column: "ContractColumn") -> ColumnDtype:
    if column.is_key:
        return "id"
    if column.codebook:
        return "coded"
    if column.logical_type in ("number", "integer"):
        return "numeric"
    if column.logical_type in ("date", "timestamp", "time"):
        return "datetime"
    return "categorical"


def _autocontext_from_contract(
    semantic_layer: Any, contract: "ContractContext", query_terms: list[str] | None, max_tokens: int
) -> AutoContext:
    """AutoContext read only from the version's contract: keys first, then by relevance."""
    columns = [
        ColumnContext(
            name=c.name,
            normalized_name=_normalize_column_name(c.name),
            system_aliases=[],
            user_aliases=list(c.synonyms),
            dtype=_contract_dtype(c),
            units=c.unit,
            codebook=dict(c.codebook) or None,
            description=c.description,
            is_phi=c.is_phi,
            semantic_type=c.semantic_type,
        )
        for c in contract.columns
    ]
    if query_terms:
        columns = _filter_columns_by_relevance(columns, query_terms)
    columns.sort(key=lambda c: c.dtype != "id")  # stable: the identifier first, the rest by relevance
    autocontext = AutoContext(
        dataset={
            "upload_id": contract.upload_id,
            "dataset_version": contract.dataset_version,
            "display_name": semantic_layer.config.get("display_name", semantic_layer.dataset_name),
        },
        entity_keys=[contract.key] if contract.key else [],
        columns=columns,
        glossary={},
        constraints={"no_row_level_data": True, "max_tokens": max_tokens},
        grain=contract.grain,
    )
    return _enforce_token_budget(autocontext, max_tokens)
