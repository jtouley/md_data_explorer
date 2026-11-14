# SPDX-License-Identifier: MIT
"""What a governed query may read: the columns of the dataset's contract, and where they are.

The contract (``contracts/<upload_id>/<version>.odcs.yaml``) is the only schema: a column that is
not one of its properties does not exist for SQL, and its identifier and PHI tags decide where a
column may appear. The rows are the stored Parquet of the same version (written by ``storage``),
with the dataset's confirmed level merges applied, so SQL sees the same levels as the intent path.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeAlias

from open_data_contract_standard.model import OpenDataContractStandard, SchemaProperty
from sqlglot import exp

from clinical_analytics.contracts.config import UNIFIED_OBJECT_NAME
from clinical_analytics.contracts.declarations import LevelMapping
from clinical_analytics.sql.errors import SqlRefusedError
from clinical_analytics.sql.render import render

LogicalType: TypeAlias = Literal[
    "string", "integer", "number", "boolean", "date", "timestamp", "time", "array", "object"
]
_NUMERIC: frozenset[str] = frozenset({"integer", "number", "boolean"})


@dataclass(frozen=True)
class ColumnSpec:
    name: str  # the stored (physical) column name
    logical_type: str
    is_id: bool  # the patient identifier: only ever counted
    phi: bool  # tagged PHI in the contract: only coarsened (year, month) or used in a filter

    @property
    def numeric(self) -> bool:
        return self.logical_type in _NUMERIC


@dataclass(frozen=True)
class SqlCatalog:
    """The single table ``unified`` a query reads, described by the contract."""

    columns: tuple[ColumnSpec, ...]
    parquet_path: Path
    levels: tuple[LevelMapping, ...] = ()

    def column(self, name: str) -> ColumnSpec | None:
        """A column by name, case-insensitive as DuckDB identifiers are."""
        folded = name.casefold()
        return next((c for c in self.columns if c.name.casefold() == folded), None)

    @property
    def id_column(self) -> ColumnSpec | None:
        return next((c for c in self.columns if c.is_id), None)

    def view_sql(self) -> str:
        """The view over the stored Parquet: the contract's columns only, level merges applied."""
        merges = {m.column: m for m in self.levels}
        select = []
        for col in self.columns:
            ref = exp.column(col.name, quoted=True)
            mapping = merges.get(col.name)
            value: exp.Expr = ref
            if mapping is not None and col.logical_type == "string" and mapping.spellings:
                value = (
                    exp.case()
                    .when(
                        exp.In(this=ref, expressions=[exp.Literal.string(s) for s in mapping.spellings]),
                        exp.Literal.string(mapping.canonical),
                    )
                    .else_(ref)
                )
            select.append(exp.alias_(value, col.name, quoted=True))
        reader = exp.func("read_parquet", exp.Literal.string(str(self.parquet_path)))
        query = exp.select(*select).from_(reader)
        return f'CREATE VIEW "{UNIFIED_OBJECT_NAME}" AS {render(query)}'

    @classmethod
    def from_contract(
        cls, contract: OpenDataContractStandard, parquet_path: Path, levels: Iterable[LevelMapping] = ()
    ) -> SqlCatalog:
        """The catalog of the contract's unified object.

        Raises:
            SqlRefusedError: the contract has no unified object or no columns.
        """
        unified = next((o for o in contract.schema_ or [] if o.name == UNIFIED_OBJECT_NAME), None)
        if unified is None or not unified.properties:
            raise SqlRefusedError("This dataset's data contract lists no columns, so SQL cannot be checked against it.")
        columns = tuple(_spec(p) for p in unified.properties)
        names = {c.name for c in columns}
        return cls(columns=columns, parquet_path=parquet_path, levels=tuple(m for m in levels if m.column in names))


def _spec(prop: SchemaProperty) -> ColumnSpec:
    name = prop.physicalName or prop.name
    if not name:
        raise SqlRefusedError("This dataset's data contract has a column without a name.")
    tags = set(prop.tags or [])
    return ColumnSpec(
        name=name,
        logical_type=str(prop.logicalType or "string"),
        is_id=bool(prop.criticalDataElement),
        phi="phi" in tags,
    )
