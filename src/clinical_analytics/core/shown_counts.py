# SPDX-License-Identifier: MIT
"""The log of shown counts: for each dataset version, the patients behind every count shown.

Per-answer suppression (``result_presentation``, ``sql.privacy``) sees one answer. It cannot see
that "patients with a 2019 row" (60), "patients with a 2020 row" (40) and "patients with a row
before 2021" (95) give the 5 patients with rows in both years, or that "ever yes" + "ever no" -
the total gives the patients whose value changed. This log can (ADR013 section 12).

What is stored. Each count an answer shows is a set: the rows or patients of its cohort that meet
the answer's conditions (a filter, a group cell, a value). The set is stored exactly, as a bitmap
over the dataset version's rows in stored order (a patient is the first row with their ID), never
as IDs or values. A count of visit rows and a count of patients are different units.

The check. Before an answer is shown, each new set N is compared with every set L already shown
for this version, with the dataset itself (its row and patient totals, which the app shows) and
with the answer's other sets. The answer is withheld, and nothing logged, when for any pair
``|N - L|``, ``|L - N|`` or ``|N & L|`` is 1 to ``SMALL_CELL_THRESHOLD - 1``. Every Venn region of
every pair of shown sets is then 0 or at least the threshold, so no count a reader works out from
two shown counts (a difference, an overlap, a union through the total) is 1 to 10. For visit
rows: a count of visit rows next to the count of their patients gives the extra visits, so a set
of visit rows whose extra visits are 1-10 is withheld when a shown patient set is within 10
patients of its patients.

Not covered: a region of three or more shown sets that no pair of them isolates (stated in
ADR013 section 12 and ``docs/security/THREAT_MODEL.md``).

Fail closed. A log that can't be read, or that was written for other rows (its row count and
fingerprint differ), withholds every answer of that version until it is fixed; so does a log
that can't be saved, or one that already holds ``MAX_LOGGED_SETS`` sets.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
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
from clinical_analytics.storage.atomic import atomic_write_bytes
from clinical_analytics.storage.metadata_lock import FileLockTimeoutError, file_lock

logger = structlog.get_logger()

LOG_FORMAT: Final = 1
# Distinct sets one dataset version may hold. Each answer is checked against all of them; at the
# cap a 100,000-row dataset holds 2,000 x 12.5 KB = 25 MB of bitmaps and a check of a new set reads
# them once (numpy, about 10 ms). At the cap, answers that would add a set are withheld.
MAX_LOGGED_SETS: Final = 2000
# Sets one answer may add (a table of 30 groups with n and events per group is 62).
MAX_NEW_SETS: Final = 256
ROWS_UNIT: Final = "rows"

WITHHELD_REASON: Final = (
    "This answer is not shown: together with answers already shown for this dataset, it would let "
    f"a reader work out a count of 1 to {SMALL_CELL_THRESHOLD - 1} patients. Ask with different limits or groups."
)
UNAVAILABLE_REASON: Final = (
    "This answer is not shown: the record of counts already shown for this dataset can't be read or "
    "saved, so it can't be checked against them."
)
FULL_REASON: Final = (
    "This answer is not shown: so many different counts have already been shown for this dataset "
    "that new ones can't be checked against them. Ask questions that reuse limits and groups already shown."
)

ConditionKind = Literal["eq", "eq_text", "event", "notnull", "complete", "recorded"]
CountUnit = Literal["counted", "rows"]


@dataclass(frozen=True)
class Condition:
    """One condition a shown count's rows meet, within the answer's cohort (its filters).

    ``eq``: the column equals ``value`` (None: is missing), as the query computed it.
    ``eq_text``: ``str()`` of the column's value is ``value`` (compute.py names levels as text).
    ``event``: the column equals the plan's share or rate value, read as the share SQL reads it.
    ``notnull``: has a value (NaN counts, as SQL ``count(column)`` counts it).
    ``complete``: has a finite value, a missing token in numbers stored as text excluded.
    ``recorded``: has a value as a share counts it (``SemanticLayer._share_sql``).
    """

    kind: ConditionKind
    column: str
    value: Any = None


@dataclass(frozen=True)
class ShownCount:
    """A count an answer shows: ``value`` units of its cohort meeting every condition.

    ``unit``: "counted" is what the answer counts (patients once rows are declared visits, else
    rows); "rows" is rows whatever the grain (a statistic over visit rows reports its n in rows).
    """

    value: int
    conditions: tuple[Condition, ...] = ()
    unit: CountUnit = "counted"


@dataclass(frozen=True)
class Universe:
    """The rows of one dataset version: their count, a fingerprint of their stored order and
    content, and (when patients are counted over visit rows) each row's patient as the first
    row with its ID (-1 for a row with no ID)."""

    rows: int
    fingerprint: str
    patient_key: str | None = None
    patient_of_row: np.ndarray | None = field(default=None, compare=False, repr=False)

    @property
    def patients_unit(self) -> str | None:
        return f"patients:{self.patient_key}" if self.patient_key is not None else None


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
    return np.bitwise_count(words).sum(axis=-1, dtype=np.int64)


def _small(values: np.ndarray) -> np.ndarray:
    return (values >= 1) & (values <= SMALL_CELL_THRESHOLD - 1)


# --- state --------------------------------------------------------------------------------


@dataclass
class _Entry:
    unit: str
    words: np.ndarray
    size: int
    excess: int | None  # visit-row sets: rows minus their patients; None otherwise


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
    excess: int | None = Field(default=None, ge=0)
    bits: str  # zlib-compressed bitmap, base64


class _StoredLog(BaseModel):
    """The file: validated whole on every read (an external boundary)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    format: Literal[1]
    dataset: str
    version: str
    rows: int = Field(ge=0)
    fingerprint: str = Field(min_length=1)
    sets: list[_StoredSet] = Field(max_length=MAX_LOGGED_SETS)


def _encode(state: _State, dataset: str, version: str) -> bytes:
    stored = _StoredLog(
        format=LOG_FORMAT,
        dataset=dataset,
        version=version,
        rows=state.rows,
        fingerprint=state.fingerprint,
        sets=[
            _StoredSet(
                unit=e.unit,
                size=e.size,
                excess=e.excess,
                bits=base64.b64encode(zlib.compress(e.words.tobytes(), 6)).decode("ascii"),
            )
            for e in state.entries
        ],
    )
    return json.dumps(stored.model_dump(mode="json"), separators=(",", ":")).encode("utf-8")


def _decode(data: bytes, dataset: str, version: str) -> _State:
    """The stored log, checked: the right dataset and version, and each bitmap the size stored.

    Raises:
        ShownLogUnavailableError: the file is not a valid log of this dataset version.
    """
    try:
        stored = _StoredLog.model_validate_json(data)
    except pydantic.ValidationError as e:
        raise ShownLogUnavailableError(f"shown-count log is invalid ({e.error_count()} errors)") from e
    if stored.dataset != dataset or stored.version != version:
        raise ShownLogUnavailableError("shown-count log belongs to another dataset version")
    state = _State(rows=stored.rows, fingerprint=stored.fingerprint, entries=[], index=set())
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
        state.add(_Entry(unit=item.unit, words=words, size=item.size, excess=item.excess))
    return state


# --- the check ----------------------------------------------------------------------------


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


def _pair_conflict(words: np.ndarray, size: int, others: np.ndarray, sizes: np.ndarray) -> bool:
    """Whether ``words`` and any row of ``others`` have a Venn region of 1 to threshold-1."""
    if others.shape[0] == 0:
        return False
    both = _popcount(others & words)
    return bool(np.any(_small(both) | _small(size - both) | _small(sizes - both)))


def find_conflict(state: _State, new: Sequence[_Entry], universe: Universe) -> str | None:
    """Which check the answer fails ("derivable", "overlap", "visits"), or None when it may be shown.

    Per unit, the shown sets (logged, new and the totals) split the units into atoms: the units
    each shown set either holds or not. A count a reader can work out by adding and subtracting
    shown counts is a combination of atoms in the row space of the sets' atom matrix. The answer
    is withheld when a region of 1 to threshold-1 units is in that row space, for each atom and
    each region of a pair of shown sets (intersection, either difference). Beyond
    ``MAX_SPAN_WORK`` the stricter pairwise rule is used instead: every pair region of 1 to
    threshold-1 withholds, derivable or not.
    """
    totals = _universe_sets(universe)
    units = {e.unit for e in new} | {e.unit for e in state.entries}
    for unit in sorted(units):
        entries = [e for e in state.entries if e.unit == unit] + [e for e in new if e.unit == unit]
        words = [e.words for e in entries] + ([totals[unit]] if unit in totals else [])
        member = totals.get(unit)
        if member is None:  # a patients unit logged before, with no patient map now: fail closed
            return "derivable"
        atoms = _Atoms.of(words, from_bitmap(member, universe.rows), universe.rows)
        if atoms.work > MAX_SPAN_WORK:
            logger.info("shown_count_span_too_large", sets=len(words), atoms=atoms.count)
            if _pairwise_conflict(entries, new, totals.get(unit)):
                return "overlap"
        elif atoms.small_region_derivable():
            return "derivable"
    return _visits_conflict(state, new, universe)


# Above this many operations (sets^2 x atoms) the exact check is replaced by the pairwise rule.
MAX_SPAN_WORK: Final = 4_000_000_000


@dataclass(frozen=True)
class _Atoms:
    """The atoms of one unit's shown sets: ``matrix`` (sets x atoms, 0/1) and each atom's size."""

    matrix: np.ndarray
    sizes: np.ndarray

    @classmethod
    def of(cls, words: Sequence[np.ndarray], members: np.ndarray, rows: int) -> _Atoms:
        # Each unit's signature (which sets hold it), packed 8 sets to a byte, 8 sets at a time:
        # never a sets x rows array of bytes (200 MB at the cap on 100,000 rows).
        chunks = []
        for start in range(0, len(words), 8):
            bits = np.stack(
                [np.unpackbits(w.view(np.uint8), bitorder="little")[:rows] for w in words[start : start + 8]]
            )
            chunks.append(np.packbits(bits[:, members], axis=0))
        signatures = np.vstack(chunks).T
        unique, sizes = np.unique(signatures, axis=0, return_counts=True)
        matrix = np.unpackbits(unique, axis=1, count=len(words)).T.astype(np.float64)
        return cls(matrix=matrix, sizes=sizes.astype(np.int64))

    @property
    def count(self) -> int:
        return int(self.sizes.size)

    @property
    def work(self) -> int:
        return int(self.matrix.shape[0]) ** 2 * max(self.count, 1)

    def small_region_derivable(self) -> bool:
        """Whether an atom or a pair region of 1 to threshold-1 units is a combination of sets."""
        sets = self.matrix
        weighted = sets * self.sizes
        both = sets @ weighted.T  # units in set i and set j
        only = weighted.sum(axis=1)[:, None] - both  # units in set i, not in set j
        other = ~np.eye(sets.shape[0], dtype=bool)
        first, second = np.nonzero(_small(both) & other)
        left, right = np.nonzero(_small(only) & other)
        regions = np.vstack(
            [
                np.eye(self.count, dtype=np.float64)[_small(self.sizes)],
                sets[first] * sets[second],
                sets[left] * (1.0 - sets[right]),
            ]
        )
        return bool(regions.shape[0]) and _in_span(regions, sets)


def _in_span(regions: np.ndarray, sets: np.ndarray) -> bool:
    """Whether any row of ``regions`` is a linear combination of the rows of ``sets``."""
    _, singular, basis = np.linalg.svd(sets, full_matrices=False)
    tolerance = singular.max(initial=0.0) * max(sets.shape) * np.finfo(np.float64).eps
    basis = basis[singular > tolerance]
    residual = regions - (regions @ basis.T) @ basis
    return bool(np.any(np.abs(residual).max(axis=1) < SPAN_TOLERANCE))


# Residual below which a 0/1 region counts as a combination of shown sets.
SPAN_TOLERANCE: Final = 1e-6


def _pairwise_conflict(entries: Sequence[_Entry], new: Sequence[_Entry], total: np.ndarray) -> bool:
    """Every pair with a new set: any region of 1 to threshold-1, derivable or not."""
    others = [e for e in entries if not any(e is n for n in new)]
    words = [e.words for e in others] + [total]
    sizes = [e.size for e in others] + [int(_popcount(total))]
    for entry in new:
        if _pair_conflict(entry.words, entry.size, np.vstack(words), np.asarray(sizes, dtype=np.int64)):
            return True
        words.append(entry.words)
        sizes.append(entry.size)
    return False


def _visits_conflict(state: _State, new: Sequence[_Entry], universe: Universe) -> str | None:
    """Visit rows next to their patients: a set of visit rows with 1 to threshold-1 more rows than
    patients is withheld once the count of its patients can be worked out from shown patient
    counts (it is a combination of them). Checked whenever a set of either kind is new."""
    unit = universe.patients_unit
    if unit is None or universe.patient_of_row is None:
        return None
    if not any(e.unit in (unit, ROWS_UNIT) for e in new):
        return None
    patients = [e.words for e in [*state.entries, *new] if e.unit == unit]
    patients.append(_universe_sets(universe)[unit])
    members = from_bitmap(patients[-1], universe.rows)
    for entry in [*state.entries, *new]:
        if entry.unit != ROWS_UNIT or entry.excess is None or not (1 <= entry.excess <= SMALL_CELL_THRESHOLD - 1):
            continue
        reps = to_bitmap(_patients_of(from_bitmap(entry.words, universe.rows), universe), universe.rows)
        atoms = _Atoms.of([*patients, reps], members, universe.rows)
        if _in_span(atoms.matrix[-1:], atoms.matrix[:-1]):
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
        excess = None
        if shown.unit == ROWS_UNIT and universe.patient_of_row is not None:
            excess = shown.size - int(_patients_of(shown.positions, universe).size)
        out.append(_Entry(unit=shown.unit, words=words, size=shown.size, excess=excess))
    return out


# --- the stores ---------------------------------------------------------------------------


class ShownCountLog:
    """Admit an answer's sets for one dataset version, or say why it is withheld.

    Subclasses store the state; this class owns the rule. ``admit`` is the only entry point: the
    check and the write happen under one lock, so two answers checked at once can't both pass
    against a log that holds neither.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()

    def admit(self, dataset: str, version: str, universe: Universe, sets: Sequence[ShownSet]) -> Verdict:
        """Check ``sets`` against the version's log; log them when the answer may be shown.

        Never raises for the log itself: an unreadable or unsaved log withholds the answer.
        """
        log = logger.bind(dataset_id=dataset)
        for shown in sets:
            if shown.positions.size and (shown.positions.min() < 0 or shown.positions.max() >= universe.rows):
                log.error("shown_count_out_of_range")
                return Verdict(shown=False, reason=UNAVAILABLE_REASON)
        try:
            with self._lock, self._locked(dataset, version):
                state = self._load(dataset, version) or _State.empty(universe)
                if state.rows != universe.rows or state.fingerprint != universe.fingerprint:
                    log.error("shown_count_log_other_rows")
                    return Verdict(shown=False, reason=UNAVAILABLE_REASON)
                new = [e for e in _entries(sets, universe) if _identity(e.unit, e.words) not in state.index]
                if not new:
                    return Verdict(shown=True)
                if len(new) > MAX_NEW_SETS or len(state.entries) + len(new) > MAX_LOGGED_SETS:
                    log.warning("shown_count_log_full", logged=len(state.entries), new=len(new))
                    return Verdict(shown=False, reason=FULL_REASON)
                conflict = find_conflict(state, new, universe)
                if conflict is not None:
                    # Never the counts: the region found is a count of 1 to 10 patients.
                    log.info("shown_count_withheld", check=conflict, logged=len(state.entries), new=len(new))
                    return Verdict(shown=False, reason=WITHHELD_REASON)
                for entry in new:
                    state.add(entry)
                self._save(dataset, version, state)
                log.info("shown_count_logged", logged=len(state.entries), new=len(new))
                return Verdict(shown=True, added=len(new))
        except ShownLogUnavailableError as e:
            log.error("shown_count_log_unavailable", error=str(e))
            return Verdict(shown=False, reason=UNAVAILABLE_REASON)

    @contextmanager
    def _locked(self, dataset: str, version: str) -> Iterator[None]:
        yield

    def _load(self, dataset: str, version: str) -> _State | None:
        raise NotImplementedError

    def _save(self, dataset: str, version: str, state: _State) -> None:
        raise NotImplementedError


class MemoryShownCountLog(ShownCountLog):
    """The log in this process only (a service with no data directory: tests, tools)."""

    def __init__(self) -> None:
        super().__init__()
        self._states: dict[tuple[str, str], _State] = {}

    def _load(self, dataset: str, version: str) -> _State | None:
        return self._states.get((dataset, version))

    def _save(self, dataset: str, version: str, state: _State) -> None:
        self._states[(dataset, version)] = state


class FileShownCountLog(ShownCountLog):
    """``<data_dir>/shown_counts/<dataset id>/<hash of version>.json``, one file per version.

    A new version starts with an empty log (its rows are different); an old version's file stays
    until the dataset is deleted (``storage.dataset_deletion``). Written with ``atomic_write`` under
    a sidecar lock file shared with every process on the data directory.
    """

    def __init__(self, data_dir: Path) -> None:
        super().__init__()
        self.data_dir = data_dir

    def _path(self, dataset: str, version: str) -> Path:
        from clinical_analytics.storage.dataset_deletion import InvalidDatasetIdError, validate_dataset_id

        try:
            checked = validate_dataset_id(dataset)
        except InvalidDatasetIdError as e:
            raise ShownLogUnavailableError("invalid dataset id for the shown-count log") from e
        name = hashlib.sha256(version.encode("utf-8")).hexdigest()[:32]
        return self.data_dir / SHOWN_COUNTS_DIRNAME / checked / f"{name}.json"

    @contextmanager
    def _locked(self, dataset: str, version: str) -> Iterator[None]:
        path = self._path(dataset, version)
        try:
            with file_lock(path.with_name(f".{path.name}.lock")):
                yield
        except (FileLockTimeoutError, OSError) as e:
            raise ShownLogUnavailableError(f"shown-count log lock failed: {type(e).__name__}") from e

    def _load(self, dataset: str, version: str) -> _State | None:
        path = self._path(dataset, version)
        try:
            data = path.read_bytes()
        except FileNotFoundError:
            return None
        except OSError as e:
            raise ShownLogUnavailableError(f"shown-count log unreadable: {type(e).__name__}") from e
        return _decode(data, dataset, version)

    def _save(self, dataset: str, version: str, state: _State) -> None:
        try:
            atomic_write_bytes(self._path(dataset, version), _encode(state, dataset, version))
        except OSError as e:
            raise ShownLogUnavailableError(f"shown-count log not saved: {type(e).__name__}") from e
