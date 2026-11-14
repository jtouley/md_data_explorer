# SPDX-License-Identifier: MIT
"""DuckDB SQL text from a sqlglot tree: the one place governed SQL is turned into text."""

from __future__ import annotations

from sqlglot import exp
from sqlglot.dialects.dialect import Dialect

_DUCKDB = Dialect.get_or_raise("duckdb")


def render(node: exp.Expr, *, pretty: bool = False) -> str:
    return _DUCKDB.generate(node, pretty=pretty)
