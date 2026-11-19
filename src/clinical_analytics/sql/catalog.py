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
from clinical_analytics.contracts.quality import is_id_property, is_restricted_property
from clinical_analytics.core.shown_counts import FINGERPRINT_SQL, ROW_ORDER_SQL
from clinical_analytics.sql.errors import SqlRefusedError
from clinical_analytics.sql.render import render

LogicalType: TypeAlias = Literal[
    "string", "integer", "number", "boolean", "date", "timestamp", "time", "array", "object"
]
_NUMERIC: frozenset[str] = frozenset({"integer", "number", "boolean"})
# The view with row positions the log of shown counts reads, and its position column.
POSITIONED_VIEW = "__mdde_positioned"
POSITIONED_COLUMN = "__mdde_row_position"


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
        """The view over the stored Parquet: the contract's columns only, level merges applied.

        A column can have several confirmed merges (one per canonical level: f/female to F, m/male
        to M); each becomes one WHEN of the column's CASE, so every merge applies.
        """
        reader = exp.func("read_parquet", exp.Literal.string(str(self.parquet_path)))
        return f'CREATE VIEW "{UNIFIED_OBJECT_NAME}" AS {render(exp.select(*self._columns()).from_(reader))}'

    def positioned_view_sql(self) -> str:
        """The same view with each row's number (``shown_counts.ROW_ORDER_SQL``, as the question
        box numbers them) as ``POSITIONED_COLUMN``, named ``POSITIONED_VIEW``: read only by the
        log of shown counts (``sql.privacy``); a query can't name it (the validator reads one table)."""
        reader = render(exp.func("read_parquet", exp.Literal.string(str(self.parquet_path))))
        numbered = f'SELECT row_number() OVER (ORDER BY {ROW_ORDER_SQL}) - 1 AS "{POSITIONED_COLUMN}", * FROM {reader}'
        columns = ", ".join(render(c) for c in self._columns())
        return f'CREATE VIEW "{POSITIONED_VIEW}" AS SELECT "{POSITIONED_COLUMN}", {columns} FROM ({numbered})'

    def fingerprint_sql(self) -> str:
        """Row count and hash sum of the stored rows: ``SemanticLayer.row_fingerprint``'s value."""
        reader = render(exp.func("read_parquet", exp.Literal.string(str(self.parquet_path))))
        return f"SELECT {FINGERPRINT_SQL} FROM {reader}"

    def _columns(self) -> list[exp.Expr]:
        merges: dict[str, list[LevelMapping]] = {}
        for mapping in self.levels:
            if mapping.spellings:
                merges.setdefault(mapping.column, []).append(mapping)
        select: list[exp.Expr] = []
        for col in self.columns:
            ref = exp.column(col.name, quoted=True)
            value: exp.Expr = ref
            if col.logical_type == "string" and merges.get(col.name):
                case = exp.case()
                for mapping in merges[col.name]:
                    spellings = [exp.Literal.string(s) for s in mapping.spellings]
                    case = case.when(
                        exp.In(this=ref.copy(), expressions=spellings), exp.Literal.string(mapping.canonical)
                    )
                value = case.else_(ref.copy())
            select.append(exp.alias_(value, col.name, quoted=True))
        return select

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
        folded = [c.name.casefold() for c in columns]
        if len(set(folded)) < len(folded):
            # DuckDB names are case-insensitive: it reads the second of "Age" and "age" as
            # "age_1", so one contract column would silently show the other's values.
            raise SqlRefusedError(
                "This dataset has columns whose names differ only in capitalisation, so SQL cannot tell them apart."
            )
        names = {c.name for c in columns}
        return cls(columns=columns, parquet_path=parquet_path, levels=tuple(m for m in levels if m.column in names))


def _spec(prop: SchemaProperty) -> ColumnSpec:
    name = prop.physicalName or prop.name
    if not name:
        raise SqlRefusedError("This dataset's data contract has a column without a name.")
    # The same rules as the quality report's, so the two never disagree on what is an ID or PHI.
    return ColumnSpec(
        name=name,
        logical_type=str(prop.logicalType or "string"),
        is_id=is_id_property(prop),
        phi=is_restricted_property(prop),
    )
