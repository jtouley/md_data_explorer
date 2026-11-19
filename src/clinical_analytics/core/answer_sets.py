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

import dataclasses
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


def _same_patients(entity_of_row: np.ndarray, patient_of_row: np.ndarray) -> bool:
    """Whether an ID's values and the patients pair one to one: every row has both or neither,
    each value is on one patient's rows only, and each patient has one value."""
    keyed, patients = entity_of_row >= 0, patient_of_row >= 0
    if not np.array_equal(keyed, patients):
        return False
    pairs = np.unique(np.stack([entity_of_row[keyed], patient_of_row[keyed]]), axis=1)
    return bool(np.unique(pairs[0]).size == pairs.shape[1] == np.unique(pairs[1]).size)


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
    if condition.kind == "isnull":
        return ref.is_null()
    if condition.kind == "outside":
        return _outside(frame, column, condition.value)
    if condition.kind == "upper_value":
        levels: pl.Series = frame.get_column(column).drop_nulls().unique()
        if levels.len() != 2:
            raise _NotACountError
        return (ref == pl.lit(levels.max())).fill_null(False)
    raise ShownCountMismatchError(f"unknown condition kind {condition.kind!r}")


class _NotACountError(Exception):
    """An ``upper_value`` count on a column without exactly two values: no count is shown."""


def _outside(frame: pl.DataFrame, column: str, bounds: Any) -> pl.Expr:
    """Values outside ``(minimum, maximum)`` read as the plausibility check reads them: numbers as
    they are, text through ``parse_numeric_text`` (missing tokens and blanks are no value)."""
    from clinical_analytics.analysis.numeric_text import parse_numeric_text

    minimum, maximum = bounds
    series = frame.get_column(column)
    if series.dtype == pl.String:
        parsed = parse_numeric_text(series)
        if parsed is None:
            raise ShownCountMismatchError("values outside a range on text that is not numbers")
        series = parsed.values
    values = series.cast(pl.Float64, strict=False)
    below = (values < minimum).fill_null(False)
    above = (values > maximum).fill_null(False) if maximum is not None else pl.Series([False] * values.len())
    outside = values.is_finite().fill_null(False) & (below | above)
    return pl.lit(outside.alias("__outside"))


def answer_sets(
    layer: Any,
    plan: QueryPlan,
    counts: Sequence[ShownCount],
    universe: Universe,
    refine: Refine | None = None,
    entity_key: str | None = None,
) -> list[ShownSet]:
    """The set behind each count, in the log's units. ``refine``: filters the answer applied
    again to the cohort (an answer from ``analysis.compute``). ``entity_key``: the ID a COUNT
    counts distinct values of (``QueryPlan.entity_key``); each value is its first row, a value
    missing is no unit, and the set is in rows, where extra rows over patients are checked
    (admissions minus patients).

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
    if entity_key is not None:
        columns.append(entity_key)
    if refine is not None:
        columns.extend(refine.columns)
    frames: dict[str, pl.DataFrame] = {}

    def frame_for(scope: str) -> pl.DataFrame:
        if scope not in frames:
            if scope == "dataset":  # every row: a data note counts the whole dataset
                every = dataclasses.replace(plan, filters=[], cohort_steps=[], group_by=None)
                frames[scope] = layer.shown_rows(every, columns)
            else:
                cohort = layer.shown_rows(plan, columns)
                frames[scope] = refine.apply(cohort) if refine is not None else cohort
        return frames[scope]

    sets: list[ShownSet] = []
    for count in counts:
        frame = frame_for(count.scope)
        positions = frame.get_column(ROW_POSITION).to_numpy()
        try:
            flags = np.ones(frame.height, dtype=bool)
            for condition in count.conditions:
                column = frame.with_columns(_expression(layer, frame, condition).alias("__shown")).get_column("__shown")
                flags &= column.fill_null(False).to_numpy().astype(bool)
        except _NotACountError:
            continue
        chosen = positions[flags]
        if count.unit == "counted" and entity_key is not None and entity_key != universe.patient_key:
            entity_of_row = layer.patient_of_row(entity_key)
            if universe.patient_of_row is not None and _same_patients(entity_of_row, universe.patient_of_row):
                # An ID that names patients one to one (an mrn beside patient_id): its distinct
                # count is a patient count, so it joins the patient sets it can be compared with.
                reps = universe.patient_of_row[chosen[entity_of_row[chosen] >= 0]]
                shown = ShownSet(unit=str(universe.patients_unit), positions=np.unique(reps[reps >= 0]))
            else:
                firsts = entity_of_row[chosen]
                shown = ShownSet(unit=ROWS_UNIT, positions=np.unique(firsts[firsts >= 0]))
        elif count.unit == "counted" and universe.patient_of_row is not None and universe.patients_unit is not None:
            reps = universe.patient_of_row[chosen]
            shown = ShownSet(unit=universe.patients_unit, positions=np.unique(reps[reps >= 0]))
        else:
            shown = ShownSet(unit=ROWS_UNIT, positions=np.unique(chosen))
        if count.value is not None and shown.size != count.value:
            raise ShownCountMismatchError("a shown count's set has another size than the answer shows")
        sets.append(shown)
    return sets
