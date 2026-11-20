# SPDX-License-Identifier: MIT
"""The one place a count of 1-10 is masked: when an answer leaves the machine (ADR015).

On screen every answer shows its true numbers. Anything written out of the app as a file or a
packet (an export, a results packet for a model, a results file of the acceptance harness, a
published log) goes through ``export_answer`` or ``export_text`` first, and every count from 1 to
``SMALL_CELL_THRESHOLD - 1`` in it becomes ``EXPORT_MARKER`` ("<11"), with a visible note that
the rule was applied. This is plain per-cell masking: each count is judged on its own, and
nothing else is hidden to stop a masked count being worked out from the rest.

What counts as a count:

* in a table, every whole number (``int`` or a numpy integer, never ``bool``) in any cell: the
  app's tables hold counts as integers (n, count, events, censored, crosstab cells) and every
  other statistic as a float, and labels as text. A whole-number float is masked too where it is
  a count: in a column named like one (``COUNT_COLUMNS``) or in a row whose ``statistic`` is
  "count" (Polars ``describe()``);
* in a sentence, every whole number written on its own, since a sentence cannot say which of its
  numbers are counts ("n=7", "on 3 rows", "Type 2"). A label digit is masked too: over-masking a
  label is the safe direction at this boundary. Decimals ("0.71"), percents ("5%"), numbers
  inside words or codes ("A1c", "2024-01-05") and numbers of two or more digits besides 10 are
  left as written. Text is NFKC-normalised first, so a full-width "５" is read as 5.

Zero is not masked: it is no one. The threshold must lie in ``THRESHOLD_RANGE``; anything else
fails at import, since a threshold of 1 would mask nothing and leave the boundary open.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping
from typing import Any, Final, TypedDict

import numpy as np

from clinical_analytics.core.nl_query_config import SMALL_CELL_THRESHOLD
from clinical_analytics.core.result_presentation import Presentation, TablePayload

THRESHOLD_RANGE: Final = range(2, 101)
if isinstance(SMALL_CELL_THRESHOLD, bool) or SMALL_CELL_THRESHOLD not in THRESHOLD_RANGE:
    raise ValueError(
        f"small_cell_threshold must be a whole number from {THRESHOLD_RANGE.start} to {THRESHOLD_RANGE.stop - 1}, "
        f"got {SMALL_CELL_THRESHOLD!r}: the export boundary would not mask small counts"
    )
SMALL_MIN: Final = 1
SMALL_MAX: Final = SMALL_CELL_THRESHOLD - 1
# Columns whose whole-number floats are counts (a count column read back as Float64).
COUNT_COLUMNS: Final = frozenset({"n", "count", "events", "censored", "n_rows", "len"})
EXPORT_MARKER: Final = f"<{SMALL_CELL_THRESHOLD}"
EXPORT_NOTE: Final = (
    f'Counts from {SMALL_MIN} to {SMALL_MAX} are written as "{EXPORT_MARKER}" because this answer left the '
    f"machine; the app shows the exact numbers."
)

# A whole number from SMALL_MIN to SMALL_MAX standing on its own (also written 5.0): not part of a
# word, a code, a decimal, a percent or a thousands-grouped number. A sentence's full stop after it
# does not protect it.
_ALTERNATIVES: Final = "|".join(str(n) for n in range(SMALL_MAX, SMALL_MIN - 1, -1))
_SMALL_NUMBER: Final = re.compile(rf"(?<![\w.,])(?:{_ALTERNATIVES})(?:\.0+)?(?![\w%]|[.,]\d)")


class ExportedAnswer(TypedDict):
    """An answer as it may leave the machine: the claim and table with small counts masked, how
    many were masked, and the visible note saying so (None when nothing was masked)."""

    claim: str | None
    table: TablePayload | None
    masked: int
    note: str | None


def is_small_count(value: object) -> bool:
    """A whole number (``int`` or a numpy integer, never a bool) from 1 to ``SMALL_MAX``."""
    if isinstance(value, bool | np.bool_) or not isinstance(value, int | np.integer):
        return False
    return SMALL_MIN <= int(value) <= SMALL_MAX


def _small_float_count(value: object) -> bool:
    """A whole-number float from 1 to ``SMALL_MAX`` (judged a count only where its column says so)."""
    return (
        isinstance(value, float | np.floating) and float(value).is_integer() and SMALL_MIN <= float(value) <= SMALL_MAX
    )


def _mask_text(text: str) -> tuple[str, int]:
    return _SMALL_NUMBER.subn(EXPORT_MARKER, unicodedata.normalize("NFKC", text))


def export_text(text: str) -> str:
    """``text`` with every whole number from 1 to 10 written as "<11" (see the module docstring).
    For a line of a log or report that is published; an answer goes through ``export_answer``."""
    return _mask_text(text)[0]


def _mask_cell(value: Any) -> tuple[Any, int]:
    if is_small_count(value):
        return EXPORT_MARKER, 1
    if isinstance(value, str):
        return _mask_text(value)
    if isinstance(value, Mapping):
        total, out = 0, {}
        for key, item in value.items():
            masked_key, n_key = _mask_text(str(key))
            masked_item, n_item = _mask_cell(item)
            out[masked_key] = masked_item
            total += n_key + n_item
        return out, total
    if isinstance(value, list | tuple):
        total, items = 0, []
        for item in value:
            masked_item, n_item = _mask_cell(item)
            items.append(masked_item)
            total += n_item
        return items, total
    return value, 0


def export_answer(presentation: Mapping[str, Any] | Presentation | None) -> ExportedAnswer:
    """The presented answer (``result_presentation.present_analysis``) as it may leave the machine.

    Never changes its input, and returns only the claim and table: any other key of the input is
    left behind. A table that is not a mapping of ``columns`` and ``rows`` is not passed through
    as it is: it is dropped, and the note says so (an unknown shape could hold counts this
    function cannot find).

    Raises:
        TypeError: ``presentation`` is neither None nor a mapping.
    """
    if presentation is not None and not isinstance(presentation, Mapping):
        raise TypeError(f"export_answer takes a presentation mapping, not {type(presentation).__name__}")
    if presentation is None:
        return {"claim": None, "table": None, "masked": 0, "note": None}
    masked = 0
    notes: list[str] = []
    claim = presentation.get("claim")
    out_claim: str | None = None
    if isinstance(claim, str):
        out_claim, masked = _mask_text(claim)
    elif claim is not None:
        notes.append("The answer's sentence was left out because it was not text.")
    out_table, in_table, dropped = _export_table(presentation.get("table"))
    masked += in_table
    if dropped:
        notes.append("The answer's table was left out because its shape was not recognised.")
    if masked:
        notes.insert(0, EXPORT_NOTE)
    return {"claim": out_claim, "table": out_table, "masked": masked, "note": " ".join(notes) or None}


def _export_table(table: Any) -> tuple[TablePayload | None, int, bool]:
    """(the table with small counts masked, how many, whether it was dropped as unrecognised)."""
    if table is None:
        return None, 0, False
    if not isinstance(table, Mapping):
        return None, 0, True
    columns, rows = table.get("columns"), table.get("rows")
    if not isinstance(columns, list) or not isinstance(rows, list) or not all(isinstance(r, Mapping) for r in rows):
        return None, 0, True
    masked = 0
    out_columns: list[str] = []
    for column in columns:
        text, n = _mask_cell(str(column))
        out_columns.append(text)
        masked += n
    out_rows: list[dict[str, Any]] = []
    for row in rows:
        out_row: dict[str, Any] = {}
        count_row = str(row.get("statistic", "")).strip().lower() == "count"
        for key, value in row.items():
            masked_key, n_key = _mask_cell(str(key))
            if (count_row or str(key).strip().lower() in COUNT_COLUMNS) and _small_float_count(value):
                masked_value, n_value = EXPORT_MARKER, 1
            else:
                masked_value, n_value = _mask_cell(value)
            out_row[masked_key] = masked_value
            masked += n_key + n_value
        out_rows.append(out_row)
    return {"columns": out_columns, "rows": out_rows}, masked, False
