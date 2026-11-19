# SPDX-License-Identifier: MIT
"""The log of shown counts: for each dataset's rows, the patients behind every count shown.

Per-answer suppression (``result_presentation``, ``sql.privacy``) sees one answer. It cannot see
that "patients with a 2019 row" (60), "patients with a 2020 row" (40) and "patients with a row
before 2021" (95) give the 5 patients with rows in both years, or that the total minus "over 65"
minus "women" plus "women over 65" gives the men of 65 or under. This log can (ADR013 §12).

What is stored. Each count an answer shows is a set: the rows or patients of its cohort that meet
the answer's conditions (a filter, a group cell, a value). The set is stored exactly, as a bitmap
over the dataset's rows numbered by content (a patient, or another counted ID, is its first row),
never as IDs or values. A count of visit rows and a count of patients are different units.

The guarantee. Per unit, the shown sets (every set shown on these rows, the new answer's sets and
the totals the app shows) split the units into atoms: the units each set either holds or not.
A count a reader can work out by adding and subtracting shown counts is a combination of atoms
in the row space of the sets x atoms matrix. An answer is withheld, and nothing of it logged,
when any group of 1 to ``SMALL_CELL_THRESHOLD - 1`` units (any 0/1 combination of atoms of that
total size, so any union of atoms, not only one atom or one pair region) lies in that row space.
For visit rows: a set of visit rows with 1-10 more rows than patients is withheld once the count
of its patients lies in the patient sets' row space. When the check would exceed its bounds it
is not run and the answer is withheld: nothing is ever shown unchecked.

Not covered: bounds a reader can draw from inequalities (a value pinned between two derivable
limits) rather than equalities.

Keyed by the rows' fingerprint, not by upload id or version: the same file uploaded twice, or
relabelled, shares one log. Fail closed: a log that can't be read or saved, or that is full,
withholds every answer on those rows until it is fixed.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import json
import os
import secrets
import threading
import zlib
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Literal

import numpy as np
import pydantic
import structlog
from pydantic import BaseModel, ConfigDict, Field

from clinical_analytics.contracts.config import SHOWN_COUNTS_DIRNAME
from clinical_analytics.core.nl_query_config import SMALL_CELL_THRESHOLD
from clinical_analytics.storage.atomic import atomic_write, atomic_write_bytes
from clinical_analytics.storage.metadata_lock import FileLockTimeoutError, file_lock

logger = structlog.get_logger()

LOG_FORMAT: Final = 3
# Distinct sets one dataset's rows may hold. A grouped answer adds one set per group shown, so this
# is about 300 grouped answers of 30 groups with n and events; the exact check's own bounds
# (``MAX_SPAN_WORK``, ``MAX_MATRIX_FLOATS``) usually stop first on large data.
MAX_LOGGED_SETS: Final = 20_000
# Sets one answer may add (a table of 30 groups with n and events per group is 62).
MAX_NEW_SETS: Final = 256
# Bounds of the exact check: operations of the SVD (sets x atoms x min(sets, atoms)), floats of
# the sets x atoms matrix (and of the small-atom matrix), and seconds of the integer program.
# Above any of them the answer is withheld ("could not be checked"), never shown unchecked.
MAX_SPAN_WORK: Final = 20_000_000_000
MAX_MATRIX_FLOATS: Final = 25_000_000
MILP_TIME_LIMIT_S: Final = 10.0
# Residual below which a 0/1 region counts as a combination of shown sets.
SPAN_TOLERANCE: Final = 1e-6
ROWS_UNIT: Final = "rows"
# The session time zone of every DuckDB connection that reads rows as text for a digest (the
# layer's, the SQL sandbox's): a timestamp with a time zone is cast to text in it.
DIGEST_TIME_ZONE: Final = "UTC"


def _quoted(column: str) -> str:
    return '"' + column.replace('"', '""') + '"'


def row_digest_sql(columns: Sequence[str]) -> str:
    """A row's digest from its values alone: SHA-256 of its values as text (missing as NUL), each
    length-prefixed and sorted, so column order and names do not change it. Shared by both paths
    (the question box's DuckDB table and the SQL sandbox's ``read_parquet``) and pinned to SHA-256,
    never DuckDB's ``hash()``. ``columns``: from ``content_columns``."""
    if not columns:
        return "sha256('')"
    return f"sha256(array_to_string(list_sort([{', '.join(_value_text(c) for c in columns)}]), chr(31)))"


def _value_text(column: str) -> str:
    """A column's value as length-prefixed text (missing as NUL), so values can't run together."""
    value = f"coalesce(CAST({_quoted(column)} AS VARCHAR), chr(0))"
    return f"length({value})::VARCHAR || ':' || {value}"


def row_tiebreak_sql(columns: Sequence[str]) -> str:
    """Orders rows whose values are the same multiset in different columns (a=1, b=2 and a=2,
    b=1): their values in ``columns``' order, which ``content_columns`` sets by the columns'
    values, not names. Rows equal in every column still tie; they are interchangeable, since
    every condition holds for both or neither."""
    values = [_value_text(c) for c in columns]
    return f"sha256(concat_ws(chr(31), {', '.join(values)}))" if values else "''"


def row_number_sql(columns: Sequence[str]) -> str:
    """Each row's number on both paths: by digest, ties broken by ``row_tiebreak_sql``, from 0."""
    return f"row_number() OVER (ORDER BY {row_digest_sql(columns)}, {row_tiebreak_sql(columns)}) - 1"


def fingerprint_sql(columns: Sequence[str], source: str) -> str:
    """The rows' count and fingerprint (SHA-256 of their sorted digests) over ``source``: the same
    for the same rows whatever the file, row order, column order or names (and on both paths)."""
    return (
        f"SELECT count(*) AS n, sha256(coalesce(string_agg(d, '' ORDER BY d), '')) AS h "
        f"FROM (SELECT {row_digest_sql(columns)} AS d FROM {source})"
    )


def synthesized_columns(config: Any) -> tuple[str, ...]:
    """The columns an upload made up (``identifier_rules.SYNTHESIZED_COLUMNS_CONFIG_KEY``: a
    patient_id built from other columns or copied, a row number), from a layer's config. Left out
    of a row's digest, so the same rows uploaded with and without one, or in another row order,
    match."""
    from clinical_analytics.core.identifier_rules import SYNTHESIZED_COLUMNS_CONFIG_KEY

    found = (config or {}).get(SYNTHESIZED_COLUMNS_CONFIG_KEY) if isinstance(config, dict) else None
    return tuple(str(c) for c in found) if isinstance(found, list | tuple) else ()


def column_profile_sql(columns: Sequence[str], source: str) -> str:
    """One row: per column, how many distinct values it holds (as text, missing as one value),
    then a digest of its values (SHA-256 of them sorted): a name-free key for the column."""
    parts = []
    for c in columns:
        value = _value_text(c)
        parts.append(f"count(DISTINCT {value})")
        parts.append(f"sha256(coalesce(string_agg({value}, chr(31) ORDER BY {value}), ''))")
    return f"SELECT {', '.join(parts) or '0'} FROM {source}"


def content_columns(columns: Sequence[str], profile: Sequence[Any], synthesized: Sequence[str]) -> list[str]:
    """The columns a row's digest reads, in a name-free order. ``profile``: ``column_profile_sql``'s
    row. Kept: columns that tell rows apart (more than one value: a column holding one value
    everywhere, added or dropped, does not change which rows are which) and that the upload did
    not make up. Ordered by their values' digest, then (two columns with the same values) by
    their place in the file, never by name, so the row-order tie-break (``row_tiebreak_sql``)
    does not move when a column is renamed."""
    made_up = set(synthesized)
    kept = [
        (str(profile[2 * i + 1]), i, c) for i, c in enumerate(columns) if int(profile[2 * i]) > 1 and c not in made_up
    ]
    return [c for _, _, c in sorted(kept)]


WITHHELD_REASON: Final = (
    "This answer is not shown: together with answers already shown for this dataset, it would let "
    f"a reader work out a count of 1 to {SMALL_CELL_THRESHOLD - 1} patients. Ask with different limits or groups."
)
UNAVAILABLE_REASON: Final = (
    "This answer is not shown: the record of counts already shown for this dataset can't be read or "
    "saved, so it can't be checked against them."
)
NOT_CHECKED_REASON: Final = (
    "This answer is not shown: its counts could not be checked against the counts already shown for this dataset."
)
TOO_MANY_COUNTS_REASON: Final = (
    "This answer is not shown: it holds too many counts to check against the counts already shown. "
    "Ask with a coarser grouping."
)
FULL_REASON: Final = (
    "This answer is not shown: so many different counts have already been shown for this dataset "
    "that new ones can't be checked against them. Ask questions that reuse limits and groups already shown."
)

ConditionKind = Literal["eq", "eq_text", "event", "notnull", "complete", "recorded", "isnull", "outside", "upper_value"]
CountUnit = Literal["counted", "rows"]
CountScope = Literal["cohort", "dataset"]


@dataclass(frozen=True)
class Condition:
    """One condition a shown count's rows meet, within the answer's cohort (its filters).

    ``eq``: the column equals ``value`` (None: is missing), as the query computed it.
    ``eq_text``: ``str()`` of the column's value is ``value`` (compute.py names levels as text).
    ``event``: the column equals the plan's share or rate value, read as the share SQL reads it.
    ``notnull``: has a value (NaN counts, as SQL ``count(column)`` counts it).
    ``complete``: has a finite value, a missing token in numbers stored as text excluded.
    ``recorded``: has a value as a share counts it (``SemanticLayer._share_sql``).
    ``isnull``: has no value (NaN is a value, as SQL ``count(column)`` reads it).
    ``outside``: a number (text read as one, as the plausibility check reads it) below
    ``value[0]`` or above ``value[1]`` (None: no upper bound).
    ``upper_value``: the larger of a column's two values in the cohort (a mean of a two-valued
    column gives back how many have it); a count on a column with more values is not one shown.
    """

    kind: ConditionKind
    column: str
    value: Any = None


@dataclass(frozen=True)
class ShownCount:
    """A count an answer shows: ``value`` units of its cohort meeting every condition. ``value``
    None: a count the reader works out (a mean of a 0/1 column times n), not printed, so its set's
    size is not checked against it.

    ``unit``: "counted" is what the answer counts (patients once rows are declared visits, else
    rows); "rows" is rows whatever the grain (a statistic over visit rows reports its n in rows).
    """

    value: int | None
    conditions: tuple[Condition, ...] = ()
    unit: CountUnit = "counted"
    # "dataset": within every row (a data note's count), not the answer's cohort.
    scope: CountScope = "cohort"


@dataclass(frozen=True)
class Universe:
    """The rows of one dataset: their count, a fingerprint of their content (the log's key), and
    (when patients are counted over visit rows) each row's patient as the first row with its ID
    (-1 for a row with no ID). Rows are numbered by content, so the same rows uploaded twice, or
    with columns renamed, are numbered alike (``SemanticLayer.row_fingerprint``)."""

    rows: int
    fingerprint: str
    patient_key: str | None = None
    patient_of_row: np.ndarray | None = field(default=None, compare=False, repr=False)

    @property
    def patients_unit(self) -> str | None:
        """The patient unit, named by which rows belong to which patient (not by the ID column's
        name), so renaming the ID column can't start a parallel unit."""
        if self.patient_of_row is None:
            return None
        digest = hashlib.sha256(np.ascontiguousarray(self.patient_of_row, dtype=np.int64).tobytes()).hexdigest()
        return f"patients:{digest[:32]}"


@dataclass(frozen=True)
class ShownSet:
    """A shown count as its exact set: ``positions`` are rows (unit "rows") or each patient's
    first row (unit "patients:<key>")."""

    unit: str
    positions: np.ndarray = field(compare=False, repr=False)

    @property
    def size(self) -> int:
        return int(self.positions.size)


class ShownLogUnavailableError(RuntimeError):
    """The log can't be read or saved, or belongs to other rows: answers are withheld."""


@dataclass(frozen=True)
class Verdict:
    """Whether an answer may be shown, and why not (a plain sentence for the reader)."""

    shown: bool
    reason: str | None = None
    added: int = 0


# --- bitmaps -------------------------------------------------------------------------------


def _words(rows: int) -> int:
    return max(1, (rows + 63) // 64)


def to_bitmap(positions: np.ndarray, rows: int) -> np.ndarray:
    """Positions 0..rows-1 as a little-endian bitmap of uint64 words."""
    flags = np.zeros(_words(rows) * 64, dtype=bool)
    flags[positions] = True
    return np.packbits(flags, bitorder="little").view(np.uint64)


def from_bitmap(words: np.ndarray, rows: int) -> np.ndarray:
    flags = np.unpackbits(words.view(np.uint8), bitorder="little")[:rows]
    return np.flatnonzero(flags)


def _popcount(words: np.ndarray) -> np.ndarray:
    """Set bits per row of a 2-D array of words (per array for 1-D)."""
    counts: np.ndarray = np.bitwise_count(words).sum(axis=-1, dtype=np.int64)
    return counts


def _small(values: np.ndarray) -> np.ndarray:
    return (values >= 1) & (values <= SMALL_CELL_THRESHOLD - 1)


# --- state --------------------------------------------------------------------------------


@dataclass
class _Entry:
    unit: str
    words: np.ndarray
    size: int


@dataclass
class _State:
    rows: int
    fingerprint: str
    entries: list[_Entry]
    index: set[tuple[str, bytes]]

    @classmethod
    def empty(cls, universe: Universe) -> _State:
        return cls(rows=universe.rows, fingerprint=universe.fingerprint, entries=[], index=set())

    def add(self, entry: _Entry) -> None:
        self.entries.append(entry)
        self.index.add(_identity(entry.unit, entry.words))


def _identity(unit: str, words: np.ndarray) -> tuple[str, bytes]:
    return unit, hashlib.sha256(words.tobytes()).digest()


class _StoredSet(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    unit: str = Field(min_length=1, max_length=300)
    size: int = Field(ge=0)
    bits: str  # zlib-compressed bitmap, base64


class _StoredLog(BaseModel):
    """The file: validated whole on every read (an external boundary)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    format: Literal[3]
    rows: int = Field(ge=0)
    fingerprint: str = Field(min_length=1)
    # Which install key the file was written under (a digest of it, never the key): a log under
    # another key is found and withholds, never silently orphaned.
    key_id: str = Field(min_length=1)
    sets: list[_StoredSet] = Field(max_length=MAX_LOGGED_SETS)


def _encode(state: _State, fingerprint: str, key_id: str) -> bytes:
    """The file for ``state``; ``fingerprint`` is what the file records for its rows (the file
    log stores a keyed digest of the rows' fingerprint, never the fingerprint itself)."""
    stored = _StoredLog(
        format=LOG_FORMAT,
        rows=state.rows,
        fingerprint=fingerprint,
        key_id=key_id,
        sets=[
            _StoredSet(
                unit=e.unit,
                size=e.size,
                bits=base64.b64encode(zlib.compress(e.words.tobytes(), 6)).decode("ascii"),
            )
            for e in state.entries
        ],
    )
    return json.dumps(stored.model_dump(mode="json"), separators=(",", ":")).encode("utf-8")


def _decode(data: bytes, universe: Universe, fingerprint: str, key_id: str) -> _State:
    """The stored log, checked: these rows (count and the recorded ``fingerprint``), each bitmap
    the size stored.

    Raises:
        ShownLogUnavailableError: the file is not a valid log of these rows.
    """
    try:
        stored = _StoredLog.model_validate_json(data)
    except pydantic.ValidationError as e:
        raise ShownLogUnavailableError(f"shown-count log is invalid ({e.error_count()} errors)") from e
    if stored.key_id != key_id:
        raise ShownLogUnavailableError("shown-count log was written under another install key")
    if stored.rows != universe.rows or not hmac.compare_digest(stored.fingerprint, fingerprint):
        raise ShownLogUnavailableError("shown-count log belongs to other rows")
    state = _State(rows=stored.rows, fingerprint=universe.fingerprint, entries=[], index=set())
    expected = _words(stored.rows) * 8
    for item in stored.sets:
        try:
            raw = zlib.decompress(base64.b64decode(item.bits, validate=True))
        except (binascii.Error, zlib.error, ValueError) as e:
            raise ShownLogUnavailableError("shown-count log holds a set that can't be decoded") from e
        if len(raw) != expected:
            raise ShownLogUnavailableError("shown-count log holds a set of the wrong length")
        words = np.frombuffer(raw, dtype=np.uint64).copy()
        if int(_popcount(words)) != item.size:
            raise ShownLogUnavailableError("shown-count log holds a set whose size does not match")
        state.add(_Entry(unit=item.unit, words=words, size=item.size))
    return state


# --- the check ----------------------------------------------------------------------------


class _TooLargeError(Exception):
    """The exact check would exceed one of its bounds: the answer is withheld, not checked."""


def _patients_of(positions: np.ndarray, universe: Universe) -> np.ndarray:
    """The first row of each patient with a row in ``positions`` (rows with no ID dropped)."""
    assert universe.patient_of_row is not None
    reps = universe.patient_of_row[positions]
    return np.unique(reps[reps >= 0])


def _universe_sets(universe: Universe) -> dict[str, np.ndarray]:
    """The totals the app shows: every row, and every patient when patients are counted."""
    sets = {ROWS_UNIT: to_bitmap(np.arange(universe.rows), universe.rows)}
    if universe.patients_unit is not None and universe.patient_of_row is not None:
        sets[universe.patients_unit] = to_bitmap(
            np.unique(universe.patient_of_row[universe.patient_of_row >= 0]), universe.rows
        )
    return sets


def find_conflict(state: _State, new: Sequence[_Entry], universe: Universe) -> str | None:
    """Which check the answer fails, or None when it may be shown.

    "derivable": a group of 1 to threshold-1 units is a combination of shown counts (module
    docstring); "visits": extra visit rows of 1-10 follow from shown counts; "unchecked": the
    check would exceed its bounds. Only the units the answer adds sets to are checked: the other
    units' atoms do not change.

    Raises:
        ShownLogUnavailableError: the answer adds patient sets but the rows have no patient map.
    """
    totals = _universe_sets(universe)
    try:
        for unit in sorted({e.unit for e in new}):
            member = totals.get(unit)
            if member is None:
                raise ShownLogUnavailableError("patient sets without a patient map for these rows")
            entries = [e for e in state.entries if e.unit == unit] + [e for e in new if e.unit == unit]
            atoms = _Atoms.of([*(e.words for e in entries), member], from_bitmap(member, universe.rows), universe.rows)
            if atoms.small_group_derivable():
                return "derivable"
        return _visits_conflict(state, new, universe)
    except _TooLargeError:
        logger.info("shown_count_check_too_large", logged=len(state.entries), new=len(new))
        return "unchecked"


@dataclass(frozen=True)
class _Atoms:
    """The atoms of one unit's shown sets: ``matrix`` (sets x atoms, 0/1) and each atom's size.

    Raises (``of``):
        _TooLargeError: the matrix would exceed ``MAX_MATRIX_FLOATS`` or the SVD ``MAX_SPAN_WORK``.
    """

    matrix: np.ndarray
    sizes: np.ndarray

    @classmethod
    def of(cls, words: Sequence[np.ndarray], members: np.ndarray, rows: int) -> _Atoms:
        # Each unit's signature (which sets hold it), packed 8 sets to a byte, 8 sets at a time:
        # never a sets x rows array of bytes.
        chunks = []
        for start in range(0, len(words), 8):
            bits = np.stack(
                [np.unpackbits(w.view(np.uint8), bitorder="little")[:rows] for w in words[start : start + 8]]
            )
            chunks.append(np.packbits(bits[:, members], axis=0))
        signatures = np.vstack(chunks).T
        unique, sizes = np.unique(signatures, axis=0, return_counts=True)
        sets, atoms = len(words), int(unique.shape[0])
        if sets * atoms > MAX_MATRIX_FLOATS or sets * atoms * min(sets, atoms) > MAX_SPAN_WORK:
            raise _TooLargeError
        matrix = np.unpackbits(unique, axis=1, count=sets).T.astype(np.float64)
        return cls(matrix=matrix, sizes=sizes.astype(np.int64))

    def small_group_derivable(self) -> bool:
        """Whether some group of 1 to threshold-1 units is a combination of the sets.

        Such a group holds only atoms of 1 to threshold-1 units ("small"). First, cheaply: a
        small atom alone (its leverage on the row space is 1), and whether any direction on the
        small atoms lies in the row space at all (the null space of ``(I - P)`` restricted to
        them, P the projection onto the row space; empty for most answers). Otherwise an integer
        program over the atoms that null space touches decides it (``_group_in_row_space``).
        """
        small = np.flatnonzero(_small(self.sizes))
        if small.size == 0:
            return False
        if self.count * small.size > MAX_MATRIX_FLOATS:
            raise _TooLargeError
        basis = _row_space(self.matrix)  # r x atoms, orthonormal rows
        leverage = (basis[:, small] ** 2).sum(axis=0)
        if np.any(1.0 - leverage < SPAN_TOLERANCE):
            return True
        if self.count * small.size * min(self.count, small.size) > MAX_SPAN_WORK:
            raise _TooLargeError  # the second SVD, over atoms x small atoms
        residual = -(basis.T @ basis[:, small])  # (I - P) restricted to the small atoms' columns
        residual[small, np.arange(small.size)] += 1.0
        _, singular, right = np.linalg.svd(residual, full_matrices=False)
        null = right[int(np.sum(singular > SPAN_TOLERANCE)) :].T  # small x d
        if null.shape[1] == 0:
            return False
        touched = small[np.abs(null).max(axis=1) > SPAN_TOLERANCE]
        return _group_in_row_space(self.matrix, self.sizes, touched, basis)

    @property
    def count(self) -> int:
        return int(self.sizes.size)


def _group_in_row_space(sets: np.ndarray, sizes: np.ndarray, candidates: np.ndarray, basis: np.ndarray) -> bool:
    """Whether a union of ``candidates`` atoms totalling 1 to threshold-1 units is a combination
    of the rows of ``sets``: an integer program (HiGHS) with a free weight per set and a 0/1 choice
    per candidate atom, the weighted sets equal to the chosen atoms and to 0 on every other atom.

    A group it finds is checked against the row space again; one that does not check out is still
    treated as derivable (withheld, never shown on a rounding).

    Raises:
        _TooLargeError: the program hit its time limit or did not finish.
    """
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import csr_matrix, hstack

    n_sets, n_atoms = sets.shape
    choose = np.zeros((n_atoms, candidates.size))
    choose[candidates, np.arange(candidates.size)] = 1.0
    equal = hstack([csr_matrix(sets.T), csr_matrix(-choose)])
    size_row = np.concatenate([np.zeros(n_sets), sizes[candidates].astype(np.float64)])
    constraints = [
        LinearConstraint(equal, 0.0, 0.0),
        LinearConstraint(size_row[None, :], 1.0, float(SMALL_CELL_THRESHOLD - 1)),
    ]
    integrality = np.concatenate([np.zeros(n_sets), np.ones(candidates.size)])
    bounds = Bounds(
        np.concatenate([np.full(n_sets, -np.inf), np.zeros(candidates.size)]),
        np.concatenate([np.full(n_sets, np.inf), np.ones(candidates.size)]),
    )
    result = milp(
        np.zeros(n_sets + candidates.size),
        constraints=constraints,
        integrality=integrality,
        bounds=bounds,
        options={"time_limit": MILP_TIME_LIMIT_S, "presolve": True},
    )
    if result.status == 2:  # infeasible: no such group
        return False
    if result.status != 0 or result.x is None:
        raise _TooLargeError
    group = np.zeros(n_atoms)
    group[candidates] = np.round(result.x[n_sets:])
    if not _in_basis(group[None, :], basis):
        logger.warning("shown_count_group_not_confirmed")
    return True


def _row_space(sets: np.ndarray) -> np.ndarray:
    """An orthonormal basis (rows) of the row space of ``sets``.

    The rank cut is the numerical noise floor (largest singular value x size x machine epsilon):
    a direction kept by mistake only makes more groups derivable (more withheld); a true direction
    below it would need a singular value under about 1e-13 of the largest."""
    _, singular, basis = np.linalg.svd(sets, full_matrices=False)
    tolerance = singular.max(initial=0.0) * max(sets.shape) * np.finfo(np.float64).eps
    kept: np.ndarray = basis[singular > tolerance]
    return kept


def _in_basis(regions: np.ndarray, basis: np.ndarray) -> bool:
    """Whether any row of ``regions`` lies in the row space spanned by ``basis``."""
    if regions.shape[0] == 0:
        return False
    residual = regions - (regions @ basis.T) @ basis
    return bool(np.any(np.abs(residual).max(axis=1) < SPAN_TOLERANCE))


def _visits_conflict(state: _State, new: Sequence[_Entry], universe: Universe) -> str | None:
    """Visit rows next to their patients: a set of visit rows with 1 to threshold-1 more rows than
    patients is withheld once the count of its patients is a combination of shown patient counts
    (the extra visits then follow). Checked whenever the answer adds a set of either kind.

    Raises:
        _TooLargeError: the patient sets exceed the check's bounds.
    """
    unit = universe.patients_unit
    if unit is None or universe.patient_of_row is None:
        return None
    if not any(e.unit in (unit, ROWS_UNIT) for e in new):
        return None
    entries = [*state.entries, *new]
    # Extra rows over patients, from the patient map these rows have now: a set of rows logged
    # before the rows were declared visits is checked with the declaration in force.
    visits = [
        e
        for e in entries
        if e.unit == ROWS_UNIT
        and 1
        <= e.size - int(_patients_of(from_bitmap(e.words, universe.rows), universe).size)
        <= SMALL_CELL_THRESHOLD - 1
    ]
    if not visits:
        return None
    total = _universe_sets(universe)[unit]
    patients = [*(e.words for e in entries if e.unit == unit), total]
    targets = [to_bitmap(_patients_of(from_bitmap(e.words, universe.rows), universe), universe.rows) for e in visits]
    atoms = _Atoms.of([*patients, *targets], from_bitmap(total, universe.rows), universe.rows)
    known = atoms.matrix[: len(patients)]
    if _in_basis(atoms.matrix[len(patients) :], _row_space(known)):
        return "visits"
    return None


def _entries(sets: Sequence[ShownSet], universe: Universe) -> list[_Entry]:
    """The answer's sets as entries, each distinct set once."""
    out: list[_Entry] = []
    seen: set[tuple[str, bytes]] = set()
    for shown in sets:
        words = to_bitmap(shown.positions, universe.rows)
        identity = _identity(shown.unit, words)
        if identity in seen:
            continue
        seen.add(identity)
        out.append(_Entry(unit=shown.unit, words=words, size=shown.size))
    return out


# --- the stores ---------------------------------------------------------------------------


class ShownCountLog:
    """Admit an answer's sets for one dataset's rows, or say why it is withheld.

    Subclasses store the state, keyed by the rows' fingerprint; this class owns the rule. ``admit``
    is the only entry point: the check and the write happen under one lock, so two answers checked
    at once can't both pass against a log that holds neither.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()

    def admit(self, dataset: str, version: str, universe: Universe, sets: Sequence[ShownSet]) -> Verdict:
        """Check ``sets`` against the log of these rows; log them when the answer may be shown.
        ``dataset`` and ``version`` only label the log events: the log belongs to the rows.

        Never raises for the log itself: an unreadable or unsaved log withholds the answer.
        """
        log = logger.bind(dataset_id=dataset)
        for shown in sets:
            if shown.positions.size and (shown.positions.min() < 0 or shown.positions.max() >= universe.rows):
                log.error("shown_count_out_of_range")
                return Verdict(shown=False, reason=UNAVAILABLE_REASON)
        try:
            with self._lock, self._locked(universe):
                state = self._load(universe) or _State.empty(universe)
                new = [e for e in _entries(sets, universe) if _identity(e.unit, e.words) not in state.index]
                if not new:
                    return Verdict(shown=True)
                if len(new) > MAX_NEW_SETS:
                    log.warning("shown_count_answer_too_large", new=len(new))
                    return Verdict(shown=False, reason=TOO_MANY_COUNTS_REASON)
                if len(state.entries) + len(new) > MAX_LOGGED_SETS:
                    log.warning("shown_count_log_full", logged=len(state.entries), new=len(new))
                    return Verdict(shown=False, reason=FULL_REASON)
                try:
                    conflict = find_conflict(state, new, universe)
                except (MemoryError, np.linalg.LinAlgError, ValueError) as e:
                    log.error("shown_count_check_failed", error_type=type(e).__name__)
                    return Verdict(shown=False, reason=NOT_CHECKED_REASON)
                if conflict == "unchecked":
                    log.warning("shown_count_not_checked", logged=len(state.entries), new=len(new))
                    return Verdict(shown=False, reason=NOT_CHECKED_REASON)
                if conflict is not None:
                    # Never the counts: the group found is a count of 1 to 10 patients.
                    log.info("shown_count_withheld", check=conflict, logged=len(state.entries), new=len(new))
                    return Verdict(shown=False, reason=WITHHELD_REASON)
                for entry in new:
                    state.add(entry)
                self._save(universe, state)
                log.info("shown_count_logged", logged=len(state.entries), new=len(new))
                return Verdict(shown=True, added=len(new))
        except ShownLogUnavailableError as e:
            log.error("shown_count_log_unavailable", error=str(e))
            return Verdict(shown=False, reason=UNAVAILABLE_REASON)

    @contextmanager
    def _locked(self, universe: Universe) -> Iterator[None]:
        yield

    def _load(self, universe: Universe) -> _State | None:
        raise NotImplementedError

    def _save(self, universe: Universe, state: _State) -> None:
        raise NotImplementedError


class _PublishedEstimates(ShownCountLog):
    """The log for a dataset declared public population estimates (ADR014): it admits every
    answer and records nothing. Such a table's rows are published estimates, not people, shown as
    published by the owner's policy (ADR014 section 7) with no small-cell rule, so a cross-question
    check would only withhold published values. Used only where the declaration is in force
    (``Declarations.population_estimates_for``)."""

    def admit(self, dataset: str, version: str, universe: Universe, sets: Sequence[ShownSet]) -> Verdict:
        return Verdict(shown=True)


PUBLISHED_ESTIMATES_LOG: Final = _PublishedEstimates()


class MemoryShownCountLog(ShownCountLog):
    """The log in this process only (a service with no data directory: tests, tools)."""

    def __init__(self) -> None:
        super().__init__()
        self._states: dict[tuple[int, str], _State] = {}

    def _load(self, universe: Universe) -> _State | None:
        return self._states.get((universe.rows, universe.fingerprint))

    def _save(self, universe: Universe, state: _State) -> None:
        self._states[(universe.rows, universe.fingerprint)] = state


INSTALL_KEY_NAME: Final = ".install_key"


def _key_id(key: bytes) -> str:
    """Which install key a log was written under: a digest of the key, never the key."""
    return hashlib.sha256(b"shown-count-log-key-id:" + key).hexdigest()[:32]


class FileShownCountLog(ShownCountLog):
    """``<data_dir>/shown_counts/<keyed digest of the rows' fingerprint>.json``, one file per set
    of rows.

    The name and the file hold an HMAC of the fingerprint under a per-install key
    (``shown_counts/.install_key``, made once), never the fingerprint: a file copied elsewhere does
    not say which rows it is about. Losing the key while logs exist withholds every answer (the
    logs could no longer be found, which would reset them). Kept when a dataset is deleted: a
    re-upload of the same rows must find what was shown on them. Holds bitmaps of row numbers
    only. Written with ``atomic_write`` under a sidecar lock file shared with every process on the
    data directory.
    """

    def __init__(self, data_dir: Path) -> None:
        super().__init__()
        self.data_dir = data_dir
        self._checked_key: bytes | None = None

    @property
    def _root(self) -> Path:
        return self.data_dir / SHOWN_COUNTS_DIRNAME

    def _key(self) -> bytes:
        """The install key, made once under a lock and read back (no two processes make two),
        the directory 0700 and the key 0600. Checked once per log object: every log under the
        directory was written under this key.

        Raises:
            ShownLogUnavailableError: it can't be read or made, logs exist without it, or a log was
                written under another key (every answer is withheld until that is resolved).
        """
        if self._checked_key is not None:
            return self._checked_key
        path = self._root / INSTALL_KEY_NAME
        try:
            self._root.mkdir(mode=0o700, parents=True, exist_ok=True)
            os.chmod(self._root, 0o700)
            with file_lock(self._root / f"{INSTALL_KEY_NAME}.lock"):
                if not path.exists():
                    if any(self._root.glob("*.json")):
                        raise ShownLogUnavailableError("shown-count logs exist but their install key is gone")

                    def write(part: Path) -> None:
                        fd = os.open(part, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                        with os.fdopen(fd, "wb") as handle:
                            handle.write(secrets.token_bytes(32))

                    atomic_write(path, write, exclusive=True)
                key = path.read_bytes()
        except (OSError, FileLockTimeoutError) as e:
            raise ShownLogUnavailableError(f"shown-count install key unavailable: {type(e).__name__}") from e
        if len(key) != 32:
            raise ShownLogUnavailableError("shown-count install key is invalid")
        key_id = _key_id(key)
        for log_file in self._root.glob("*.json"):
            try:
                written_under = json.loads(log_file.read_bytes()).get("key_id")
            except (OSError, ValueError, AttributeError) as e:
                raise ShownLogUnavailableError("a shown-count log can't be read") from e
            if written_under != key_id:
                raise ShownLogUnavailableError("a shown-count log was written under another install key")
        self._checked_key = key
        return key

    def _keyed(self, universe: Universe) -> str:
        message = f"{universe.rows}:{universe.fingerprint}".encode()
        return hmac.new(self._key(), message, hashlib.sha256).hexdigest()

    def _path(self, universe: Universe) -> Path:
        return self._root / f"{self._keyed(universe)[:40]}.json"

    @contextmanager
    def _locked(self, universe: Universe) -> Iterator[None]:
        path = self._path(universe)
        try:
            with file_lock(path.with_name(f".{path.name}.lock")):
                yield
        except (FileLockTimeoutError, OSError) as e:
            raise ShownLogUnavailableError(f"shown-count log lock failed: {type(e).__name__}") from e

    def _load(self, universe: Universe) -> _State | None:
        path = self._path(universe)
        try:
            data = path.read_bytes()
        except FileNotFoundError:
            return None
        except OSError as e:
            raise ShownLogUnavailableError(f"shown-count log unreadable: {type(e).__name__}") from e
        return _decode(data, universe, self._keyed(universe), _key_id(self._key()))

    def _save(self, universe: Universe, state: _State) -> None:
        try:
            atomic_write_bytes(self._path(universe), _encode(state, self._keyed(universe), _key_id(self._key())))
        except OSError as e:
            raise ShownLogUnavailableError(f"shown-count log not saved: {type(e).__name__}") from e
