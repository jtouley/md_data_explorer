# SPDX-License-Identifier: MIT
"""The data contract as the model's context: a typed read view of an upload version's contract.

The model (``NLQueryEngine._llm_parse``) is told about an uploaded dataset only what its
contract says: column names and types, descriptions, units and codebooks from the data
dictionary or accepted metadata, the identifier and the grain. The contract holds no observed
values (no levels, minimum or maximum, no counts; ``contracts.builder``), so neither does the
context.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from open_data_contract_standard.model import OpenDataContractStandard, SchemaObject, SchemaProperty

from clinical_analytics.contracts import config
from clinical_analytics.contracts.errors import ContractDamagedError


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


@dataclass(frozen=True)
class ContractContext:
    """What the model is told about one upload version."""

    upload_id: str
    dataset_version: str
    grain: str | None
    key: str | None
    columns: tuple[ContractColumn, ...]


def _custom(prop: SchemaProperty | OpenDataContractStandard, name: str) -> Any:
    return next((c.value for c in prop.customProperties or [] if c.property == name), None)


def _column(prop: SchemaProperty) -> ContractColumn:
    name = prop.physicalName or prop.name
    if not name:
        raise ContractDamagedError("A contract property has no name")
    semantic = _custom(prop, "clinicalSemanticType")
    unit = _custom(prop, "unit")
    return ContractColumn(
        name=name,
        logical_type=str(prop.logicalType or "string"),
        label=prop.businessName,
        description=prop.description,
        unit=str(unit) if unit else None,
        codebook={str(e.value): str(e.label) for e in prop.enum or [] if e.value is not None and e.label},
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
    columns = tuple(_column(p) for p in obj.properties or [])
    return ContractContext(
        upload_id=str(_custom(contract, "uploadId") or contract.id),
        dataset_version=str(_custom(contract, "datasetVersion") or ""),
        grain=obj.dataGranularityDescription,
        key=next((c.name for c in columns if c.is_key), None),
        columns=columns,
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
