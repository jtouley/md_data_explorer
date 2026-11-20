# SPDX-License-Identifier: MIT
"""Limits of the governed SQL path, in one place."""

from __future__ import annotations

from typing import Final

from clinical_analytics.contracts.config import UNIFIED_OBJECT_NAME

# The one table a query may read: the contract's object for the analysed (unified) cohort.
TABLE_NAME: Final = UNIFIED_OBJECT_NAME
MAX_SQL_CHARS: Final = 10_000
# The API's hard cap on the request body's query: above MAX_SQL_CHARS, so a long query gets the
# validator's plain-language refusal; far beyond it, the request is rejected before parsing.
MAX_SQL_REQUEST_CHARS: Final = 4 * MAX_SQL_CHARS
MAX_AST_NODES: Final = 400
# Groups over the whole table (rows the WHERE removes included): more is not shown as a table.
MAX_GROUPS: Final = 200
# Quantiles a query may ask for: a fine grid of quantiles would trace one patient's position.
ALLOWED_QUANTILES: Final = frozenset({0.1, 0.25, 0.5, 0.75, 0.9})
TIMEOUT_SECONDS: Final = 10.0
