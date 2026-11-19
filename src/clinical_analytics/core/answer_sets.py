# SPDX-License-Identifier: MIT
"""The rows and patients behind the counts a question-box answer shows (for ``core.shown_counts``).

The presenters list what they show as ``ShownCount`` (a value and its conditions within the
answer's cohort). Here each one becomes its exact set: the plan's cohort is read again from the
layer, with each row's stored position, through the same filter code the answer ran
(``SemanticLayer.shown_rows``), and each count's conditions are applied to it in Polars. Every set
must have exactly the size the answer shows; any other size means this module and the answer
disagree on what was counted, and the answer is withheld (``ShownCountMismatchError``).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl

from clinical_analytics.analysis.numeric_text import is_numeric_text, missing_token_expr
from clinical_analytics.core.query_plan import QueryPlan
from clinical_analytics.core.semantic import ROW_POSITION
from clinical_analytics.core.shown_counts import ROWS_UNIT, Condition, ShownCount, ShownSet, Universe


@dataclass(frozen=True)
class Refine:
    """Filters an answer applied again after reading the plan's cohort (``analysis.compute``):
    the columns they read and the function that applies them to the cohort's rows."""

    columns: tuple[str, ...]
    apply: Callable[[pl.DataFrame], pl.DataFrame]


class ShownCountMismatchError(RuntimeError):
    """A shown count's set does not have the size the answer shows (never names the sizes)."""


def layer_universe(layer: Any) -> Universe:
    """The layer's rows as the log knows them; patients mapped to their first row once the
    question service counts patients over visit rows (``count_entity_key``)."""
    rows = int(layer.get_base_row_count())
    fingerprint = str(layer.row_fingerprint())
    key = getattr(layer, "count_entity_key", None)
    if isinstance(key, str) and key:
        return Universe(rows=rows, fingerprint=fingerprint, patient_key=key, patient_of_row=layer.patient_of_row(key))
    return Universe(rows=rows, fingerprint=fingerprint)


def _numeric_text(layer: Any, frame: pl.DataFrame, column: str) -> bool:
    """Numbers stored as text: their missing tokens ("N/A", ".") are missing values."""
    return frame.schema[column] == pl.String and is_numeric_text(*layer.base_value_counts(column))


def _expression(layer: Any, frame: pl.DataFrame, condition: Condition) -> pl.Expr:
    column = condition.column
    if column not in frame.columns:
        raise ShownCountMismatchError("a shown count names a column the cohort does not have")
    ref = pl.col(column)
    if condition.kind == "eq":
        return ref.is_null() if condition.value is None else (ref == pl.lit(condition.value)).fill_null(False)
    if condition.kind == "eq_text":
        values = [v for v in frame.get_column(column).unique().to_list() if v is not None and str(v) == condition.value]
        return ref.is_in(values) if values else pl.lit(False)
    if condition.kind == "event":
        typed = layer._equality_value(layer.get_base_view(), column, condition.value)
        return (ref == pl.lit(typed)).fill_null(False)
    present = ref.is_not_null()
    if condition.kind == "notnull":
        return present
    if condition.kind in ("recorded", "complete"):
        if _numeric_text(layer, frame, column):
            present = present & ~missing_token_expr(column).fill_null(False)
        if condition.kind == "complete" and frame.schema[column].is_float():
            present = present & ref.is_finite().fill_null(False)
        return present
    raise ShownCountMismatchError(f"unknown condition kind {condition.kind!r}")


def answer_sets(
    layer: Any, plan: QueryPlan, counts: Sequence[ShownCount], universe: Universe, refine: Refine | None = None
) -> list[ShownSet]:
    """The set behind each count, in the log's units. ``refine``: filters the answer applied
    again to the cohort (an answer from ``analysis.compute``).

    Raises:
        ShownCountMismatchError: a set's size differs from the count shown, or a condition can't
            be read on the cohort.
        CohortStepError: the cohort can't be read again.
    """
    if not counts:
        return []
    columns = [c.column for count in counts for c in count.conditions]
    if universe.patient_key is not None:
        columns.append(universe.patient_key)
    if refine is not None:
        columns.extend(refine.columns)
    frame = layer.shown_rows(plan, columns)
    if refine is not None:
        frame = refine.apply(frame)
    positions = frame.get_column(ROW_POSITION).to_numpy()
    sets: list[ShownSet] = []
    for count in counts:
        mask = pl.lit(True)
        for condition in count.conditions:
            mask = mask & _expression(layer, frame, condition)
        flags = frame.with_columns(mask.alias("__shown")).get_column("__shown").to_numpy().astype(bool)
        chosen = positions[flags]
        if count.unit == "counted" and universe.patient_of_row is not None and universe.patients_unit is not None:
            reps = universe.patient_of_row[chosen]
            shown = ShownSet(unit=universe.patients_unit, positions=np.unique(reps[reps >= 0]))
        else:
            shown = ShownSet(unit=ROWS_UNIT, positions=np.unique(chosen))
        if shown.size != count.value:
            raise ShownCountMismatchError("a shown count's set has another size than the answer shows")
        sets.append(shown)
    return sets
