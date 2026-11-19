# SPDX-License-Identifier: MIT
"""Limits of the governed SQL path, in one place."""

from __future__ import annotations

from typing import Final

from clinical_analytics.contracts.config import UNIFIED_OBJECT_NAME
from clinical_analytics.core.nl_query_config import SMALL_CELL_THRESHOLD

# The one table a query may read: the contract's object for the analysed (unified) cohort.
TABLE_NAME: Final = UNIFIED_OBJECT_NAME
MAX_SQL_CHARS: Final = 10_000
# The API's hard cap on the request body's query: above MAX_SQL_CHARS, so a long query gets the
# validator's plain-language refusal; far beyond it, the request is rejected before parsing.
MAX_SQL_REQUEST_CHARS: Final = 4 * MAX_SQL_CHARS
MAX_AST_NODES: Final = 400
# Groups over the whole table (rows the WHERE removes included): every one is checked.
MAX_GROUPS: Final = 200
# A value aggregate (avg, sum, ...) over a column with at most this many distinct values is
# treated as counting patients per value: each value's count is a checked cell, so moments such
# as avg and stddev together cannot reveal a small group of patients with one value.
MAX_DISCRETE_LEVELS: Final = 20
# Quantiles a query may ask for: a fine grid of quantiles would trace one patient's position.
ALLOWED_QUANTILES: Final = frozenset({0.1, 0.25, 0.5, 0.75, 0.9})
# Conditions one answer may have: the AND terms of its WHERE, FILTERs and count_ifs, plus an
# IS NOT NULL for each column a count(column) or value aggregate reads. Every combination is a
# cell per group (2^4 = 16; 3,200 at MAX_GROUPS), so more are refused, never checked in part.
MAX_CONDITIONS: Final = 4
TIMEOUT_SECONDS: Final = 10.0
SMALL_CELL: Final = SMALL_CELL_THRESHOLD
