# SPDX-License-Identifier: MIT
"""The data contract as the model's context: a typed read view of an upload version's contract.

The model (``NLQueryEngine._llm_parse``) is told about an uploaded dataset only what its
contract says: column names and types, descriptions, units and codebooks from the data
dictionary or accepted metadata, the glossary, the identifier and the grain. The contract holds
no observed values (no levels, minimum or maximum, no counts; ``contracts.builder``), so neither
does the context.

A contract written before the builder read data dictionaries (it lacks
``config.DICTIONARY_READ_PROPERTY``) is immutable like any other, so ``with_dictionary`` reads the
upload's dictionary into its view by the same rules: the same parser, and codes only when the
data holds no value outside them (``builder.documented_enum``).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, TypeAlias

import polars as pl
from open_data_contract_standard.model import OpenDataContractStandard, SchemaObject, SchemaProperty

from clinical_analytics.contracts import config
from clinical_analytics.contracts.dictionary import ColumnDoc
from clinical_analytics.contracts.errors import ContractDamagedError
from clinical_analytics.contracts.quality import custom_value

# How the model is told a column's type: one mapping for the column catalog, the type rules and
# the validation prompts (``core.nl_query_engine``), so no two of them disagree.
TypeCategory: TypeAlias = Literal["numeric", "datetime", "boolean", "string"]


def type_category(logical_type: str) -> TypeCategory:
    """The prompt's name for an ODCS ``logicalType`` ("time" is a datetime, like "date")."""
    if logical_type in config.NUMERIC_LOGICAL_TYPES:
        return "numeric"
    if logical_type in config.DATETIME_LOGICAL_TYPES:
        return "datetime"
    if logical_type == "boolean":
        return "boolean"
    return "string"


@dataclass(frozen=True)
class ContractColumn:
    """One column as the contract documents it."""

    name: str  # the physical (stored) column name: what a plan must use
    logical_type: str
    label: str | None = None
    description: str | None = None
    unit: str | None = None
    codebook: dict[str, str] = field(default_factory=dict)
    synonyms: tuple[str, ...] = ()
    is_key: bool = False
    is_phi: bool = False
    semantic_type: str | None = None

    @property
    def type_category(self) -> TypeCategory:
        return type_category(self.logical_type)


@dataclass(frozen=True)
class ContractContext:
    """What the model is told about one upload version."""

    upload_id: str
    dataset_version: str
    grain: str | None
    key: str | None
    columns: tuple[ContractColumn, ...]
    glossary: dict[str, str] = field(default_factory=dict)
    # False for a contract written before the builder read dictionaries (``with_dictionary``).
    dictionary_read: bool = True


def _column(prop: SchemaProperty, dictionary_read: bool) -> ContractColumn:
    """One property; codes a dictionary added before its codes were checked against the data (a
    contract without ``config.DICTIONARY_READ_PROPERTY``) are left out."""
    name = prop.physicalName or prop.name
    if not name:
        raise ContractDamagedError("A contract property has no name")
    semantic = custom_value(prop, "clinicalSemanticType")
    unit = custom_value(prop, "unit")
    added = custom_value(prop, config.DICTIONARY_CODES_PROPERTY)
    unchecked = {str(v) for v in added} if isinstance(added, list) and not dictionary_read else set()
    return ContractColumn(
        name=name,
        logical_type=str(prop.logicalType or "string"),
        label=prop.businessName,
        description=prop.description,
        unit=str(unit) if unit else None,
        codebook={
            str(e.value): str(e.label)
            for e in prop.enum or []
            if e.value is not None and e.label and str(e.value) not in unchecked
        },
        synonyms=tuple(s.synonym for s in prop.synonyms or [] if s.synonym),
        is_key=bool(prop.criticalDataElement),
        is_phi="phi" in (prop.tags or []),
        semantic_type=str(semantic) if semantic else None,
    )


def _unified(contract: OpenDataContractStandard) -> SchemaObject:
    obj = next((o for o in contract.schema_ or [] if o.name == config.UNIFIED_OBJECT_NAME), None)
    if obj is None:
        raise ContractDamagedError(f"The contract has no '{config.UNIFIED_OBJECT_NAME}' object")
    return obj


def context_from_contract(contract: OpenDataContractStandard) -> ContractContext:
    """The model's view of ``contract`` (its unified object).

    Raises:
        ContractDamagedError: no unified object, or a property without a name.
    """
    obj = _unified(contract)
    dictionary_read = custom_value(contract, config.DICTIONARY_READ_PROPERTY) is True
    columns = tuple(_column(p, dictionary_read) for p in obj.properties or [])
    glossary = custom_value(contract, config.GLOSSARY_PROPERTY)
    return ContractContext(
        upload_id=str(custom_value(contract, "uploadId") or contract.id),
        dataset_version=str(custom_value(contract, "datasetVersion") or ""),
        grain=obj.dataGranularityDescription,
        key=next((c.name for c in columns if c.is_key), None),
        columns=columns,
        glossary={str(k): str(v) for k, v in glossary.items()} if isinstance(glossary, dict) else {},
        dictionary_read=dictionary_read,
    )


def load_contract_context(path: Path) -> ContractContext | None:
    """The context for the contract at ``path``, or None when the version has none (an upload from
    before contracts).

    Raises:
        ContractDamagedError: the saved contract is damaged.
        ContractStoreError: it cannot be read.
    """
    from clinical_analytics.contracts.service import read_saved_contract

    if not path.exists():
        return None
    return context_from_contract(read_saved_contract(path))


def _documented(column: ContractColumn, doc: ColumnDoc | None, values: Callable[[str], pl.Series]) -> ContractColumn:
    """``column`` with what ``doc`` adds, as ``builder._property`` would have written it: accepted
    metadata (already in the contract) wins, and codes join only when the data allows them."""
    if doc is None or column.is_key:
        return column
    codebook = dict(column.codebook)
    if doc.codebook and column.logical_type != "boolean":
        from clinical_analytics.contracts.builder import documented_enum, typed_code

        series = values(column.name)
        enum: list[dict[str, Any]] = []
        for code, label in column.codebook.items():
            value = typed_code(code, series.dtype)
            if value is not None:
                enum.append({"value": value, "label": label})
        codebook.update({str(e["value"]): str(e["label"]) for e in documented_enum(doc.codebook, series, enum)})
    return dataclasses.replace(
        column,
        description=column.description or doc.description,
        unit=column.unit or doc.unit,
        codebook=codebook,
    )


def with_dictionary(
    context: ContractContext,
    docs: Mapping[str, ColumnDoc],
    glossary: Mapping[str, str],
    values: Callable[[str], pl.Series],
) -> ContractContext:
    """``context`` of a contract written before the builder read dictionaries, with what the
    upload's dictionary documents (``service.documented_columns``) read in by the builder's rules.

    ``values(column)`` gives the column's distinct stored values; only columns the dictionary
    gives codes for are read. A context whose contract read its dictionary is returned as is.
    """
    if context.dictionary_read:
        return context
    columns = tuple(_documented(c, docs.get(c.name), values) for c in context.columns)
    return dataclasses.replace(
        context, columns=columns, glossary={**dict(glossary), **context.glossary}, dictionary_read=True
    )
