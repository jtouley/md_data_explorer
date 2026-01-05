"""
AutoContext Packager for Tier 3 LLM Fallback

Builds bounded, privacy-safe context packs from schema inference,
documentation, and alias mappings to enable grounded query parsing.
"""

import re
from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class ColumnContext:
    """Column metadata for AutoContext."""

    name: str
    normalized_name: str
    system_aliases: list[str]
    user_aliases: list[str]
    dtype: Literal["numeric", "categorical", "datetime", "id", "coded"]
    units: str | None = None
    codebook: dict[str, str] | None = None
    stats: dict[str, Any] | None = None


@dataclass
class AutoContext:
    """Bounded, privacy-safe context pack for Tier 3 LLM."""

    dataset: dict[str, str]  # {upload_id, dataset_version, display_name}
    entity_keys: list[str]  # Candidate columns ranked
    columns: list[ColumnContext]
    glossary: dict[str, Any]  # Extracted terms, abbreviations, notes
    constraints: dict[str, Any] = field(default_factory=lambda: {"no_row_level_data": True, "max_tokens": 4000})


def _extract_entity_keys(inferred_schema) -> list[str]:
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
    Estimate token count from text (approximation: 1 token ≈ 4 characters).

    Args:
        text: Text to estimate

    Returns:
        Estimated token count
    """
    return len(text) // 4


def _build_column_contexts(
    semantic_layer,
    inferred_schema,
    query_terms: list[str] | None = None,
) -> list[ColumnContext]:
    """
    Build column contexts from semantic layer and schema.

    Args:
        semantic_layer: SemanticLayer instance
        inferred_schema: InferredSchema instance
        query_terms: Optional query terms for relevance filtering

    Returns:
        List of ColumnContext objects
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

        # Build lightweight stats (aggregated only, no row-level data)
        stats: dict[str, Any] | None = None
        if col_metadata and "metadata" in col_metadata:
            metadata_info = col_metadata["metadata"]
            stats_dict: dict[str, Any] = {}
            # Only include aggregated stats
            for key in ["min", "max", "mean", "median", "std", "count", "unique_count"]:
                if key in metadata_info:
                    stats_dict[key] = metadata_info[key]
            # Top values should be counts, not raw values
            if "top_values" in metadata_info:
                top_vals = metadata_info["top_values"]
                if isinstance(top_vals, dict):
                    stats_dict["top_values"] = top_vals  # Already in value -> count format
            if stats_dict:
                stats = stats_dict

        # Create ColumnContext
        column_context = ColumnContext(
            name=col_name,
            normalized_name=_normalize_column_name(col_name),
            system_aliases=system_aliases,
            user_aliases=[],  # TODO: Get from ADR003 persisted mappings
            dtype=dtype,
            units=units,
            codebook=codebook,
            stats=stats,
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


def _reconstruct_inferred_schema_from_semantic_layer(semantic_layer):
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

    # Extract patient_id from column_mapping
    column_mapping = semantic_layer.config.get("column_mapping", {})
    for col, mapped in column_mapping.items():
        if mapped == "patient_id":
            schema.patient_id_column = col
            break

    # Extract outcome columns
    outcomes = semantic_layer.config.get("outcomes", {})
    schema.outcome_columns = list(outcomes.keys())

    # Get all columns from base view
    try:
        base_view = semantic_layer.get_base_view()
        all_columns = list(base_view.columns)
    except Exception:
        all_columns = []

    # Classify columns from variable_types if available
    variable_types = semantic_layer.config.get("variable_types", {})
    for col in all_columns:
        if col == schema.patient_id_column or col in schema.outcome_columns:
            continue

        col_meta = variable_types.get(col, {})
        var_type = col_meta.get("type", "categorical")
        if var_type in ("numeric", "continuous"):
            schema.continuous_columns.append(col)
        elif var_type == "datetime":
            schema.time_columns.append(col)
        else:
            schema.categorical_columns.append(col)

    return schema


def build_autocontext(
    semantic_layer,
    inferred_schema=None,
    doc_context: str | None = None,
    query_terms: list[str] | None = None,
    max_tokens: int = 4000,
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

    Returns:
        AutoContext pack with schema context
    """
    # Reconstruct inferred_schema if not provided
    if inferred_schema is None:
        inferred_schema = _reconstruct_inferred_schema_from_semantic_layer(semantic_layer)

    # Step 1: Extract entity keys (deterministic ranking)
    entity_keys = _extract_entity_keys(inferred_schema)

    # Step 2: Build column catalog
    column_contexts = _build_column_contexts(semantic_layer, inferred_schema, query_terms)

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

    # Step 6: Create AutoContext
    autocontext = AutoContext(
        dataset=dataset_info,
        entity_keys=entity_keys,
        columns=column_contexts,
        glossary=glossary,
        constraints={"no_row_level_data": True, "max_tokens": max_tokens},
    )

    # Step 7: Enforce token budget
    autocontext = _enforce_token_budget(autocontext, max_tokens)

    # Step 8: Privacy validation (assertion check)
    # Verify no row-level data in stats
    for col in autocontext.columns:
        if col.stats:
            # Stats should only contain aggregated data
            allowed_keys = {"min", "max", "mean", "median", "std", "count", "unique_count", "top_values"}
            assert all(key in allowed_keys for key in col.stats.keys()), (
                f"Column {col.name} stats contains non-aggregated data: {col.stats}"
            )
            # top_values should be value -> count mapping, not raw row data
            if "top_values" in col.stats:
                top_vals = col.stats["top_values"]
                if isinstance(top_vals, dict):
                    assert all(isinstance(v, int | float) for v in top_vals.values()), (
                        f"Column {col.name} top_values contains non-count data: {top_vals}"
                    )

    return autocontext
