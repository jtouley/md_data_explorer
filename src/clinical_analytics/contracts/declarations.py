# SPDX-License-Identifier: MIT
"""What the doctor confirmed about a dataset: its grain, and which category spellings are one.

A declaration is made by confirming a held question and is stored so the next question does not
ask again. Each one carries the dataset version it was made on and the facts it was made on (row
and patient counts, the spellings), and is ignored when either no longer matches: a re-upload
(a new version, even of the same shape) asks again instead of applying an old answer to new
data. Running a confirmation twice stores the same thing.

Declarations are stored only for a dataset with an id; there is no shared fallback key.
``update`` reads, changes and writes under one lock, so two confirmations do not drop each other.

Stored as ``contracts/<dataset id>/declarations.json`` in the data directory, next to the
dataset's contracts, so deleting the dataset removes it too. Holds no patient values: column
names, counts and category labels only.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal, Protocol

import pydantic
import structlog
from pydantic import BaseModel, ConfigDict, Field, model_validator

from clinical_analytics.contracts.errors import ContractStoreError, InvalidDeclarationKeyError
from clinical_analytics.contracts.store import contracts_root
from clinical_analytics.storage.dataset_deletion import InvalidDatasetIdError, validate_dataset_id

logger = structlog.get_logger()

DECLARATIONS_FILENAME = "declarations.json"


class GrainDeclaration(BaseModel):
    """What a row is, as the doctor confirmed it on ``rows``/``patients``.

    * ``visit``: rows are visits of patients identified by ``key`` (the question box's hold).
    * ``patient``: each row is one patient, confirmed where IDs can't show it (no ID column, so
      ``key`` is None, or rows without a ``key``); ``rows`` equals ``patients``. Governed SQL
      needs it to count patients, not rows, before hiding small counts.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    grain: Literal["visit", "patient"] = "visit"
    key: str | None = None
    rows: int = Field(..., ge=0)
    patients: int = Field(..., ge=0)
    dataset_version: str | None = None

    @model_validator(mode="after")
    def _grain_is_consistent(self) -> GrainDeclaration:
        if self.grain == "visit" and not self.key:
            raise ValueError("a visit grain names the patient key")
        if self.key is not None and not self.key.strip():
            raise ValueError("a grain's key is a column name or None, never blank")
        if self.grain == "patient" and self.rows != self.patients:
            raise ValueError("a patient grain has one patient per row (rows == patients)")
        return self


class LevelMapping(BaseModel):
    """Spellings of one category that the doctor confirmed are the same level."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    column: str
    spellings: tuple[str, ...]
    canonical: str
    # Spellings held by at least MIN_CELL_SIZE rows when confirmed: the only ones a plan line
    # may name beside the merged level (a label of 1 to 10 rows is never shown).
    named: tuple[str, ...] = ()
    dataset_version: str | None = None


class Declarations(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    # Keyed by dataset version, like ``levels``: a confirmation that finishes on a retired layer
    # after a re-upload stores its own version's grain and never replaces the current version's.
    grains: tuple[GrainDeclaration, ...] = ()
    levels: tuple[LevelMapping, ...] = ()

    @model_validator(mode="before")
    @classmethod
    def _read_single_grain(cls, data: Any) -> Any:
        """Read a file written before grains were kept per version (one ``grain`` slot)."""
        if isinstance(data, dict) and "grain" in data:
            if "grains" in data:
                raise ValueError("declarations hold both 'grain' and 'grains'")
            data = dict(data)
            legacy = data.pop("grain")
            data["grains"] = () if legacy is None else (legacy,)
        return data

    def with_grain(self, grain: GrainDeclaration) -> Declarations:
        """These declarations with ``grain`` as its version's grain (other versions' are kept)."""
        kept = tuple(g for g in self.grains if g.dataset_version != grain.dataset_version)
        return self.model_copy(update={"grains": (*kept, grain)})

    def grain_for(self, dataset_version: str | None) -> GrainDeclaration | None:
        """The grain confirmed on ``dataset_version``, or None (others belong to other data)."""
        return next((g for g in self.grains if g.dataset_version == dataset_version), None)

    def with_levels(self, mapping: LevelMapping) -> Declarations:
        kept = tuple(
            m
            for m in self.levels
            if not (
                m.column == mapping.column
                and m.canonical == mapping.canonical
                and m.dataset_version == mapping.dataset_version
            )
        )
        return self.model_copy(update={"levels": (*kept, mapping)})

    def levels_for(self, dataset_version: str | None) -> tuple[LevelMapping, ...]:
        """The level mappings confirmed on ``dataset_version`` (others belong to other data)."""
        return tuple(m for m in self.levels if m.dataset_version == dataset_version)

    def for_version(self, dataset_version: str | None) -> Declarations:
        """Only what was confirmed on ``dataset_version``: all a question on that version reads."""
        grain = self.grain_for(dataset_version)
        return Declarations(grains=() if grain is None else (grain,), levels=self.levels_for(dataset_version))


class DeclarationStore(Protocol):
    def load(self, dataset_id: str) -> Declarations: ...

    def save(self, dataset_id: str, declarations: Declarations) -> None: ...

    def update(self, dataset_id: str, change: Callable[[Declarations], Declarations]) -> Declarations: ...


class MemoryDeclarationStore:
    """Declarations for this process only (tests, the headless evaluator, a dataset with no id)."""

    def __init__(self) -> None:
        self._items: dict[str, Declarations] = {}
        self._lock = threading.Lock()

    def load(self, dataset_id: str) -> Declarations:
        return self._items.get(dataset_id, Declarations())

    def save(self, dataset_id: str, declarations: Declarations) -> None:
        self._items[dataset_id] = declarations

    def update(self, dataset_id: str, change: Callable[[Declarations], Declarations]) -> Declarations:
        with self._lock:
            updated = change(self.load(dataset_id))
            self.save(dataset_id, updated)
            return updated


# One lock for every file store in the process: a read-modify-write on two QueryService
# instances (two layers of one dataset) must not interleave.
_store_lock = threading.RLock()


class FileDeclarationStore:
    """Declarations under ``<data_dir>/contracts/<dataset id>/declarations.json``."""

    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir

    def _path(self, dataset_id: str) -> Path:
        """The dataset's declarations file.

        Raises:
            InvalidDeclarationKeyError: ``dataset_id`` is not a plain identifier (``../``).
        """
        try:
            checked = validate_dataset_id(dataset_id)
        except InvalidDatasetIdError as e:
            raise InvalidDeclarationKeyError("Invalid dataset id for declarations") from e
        return contracts_root(self.data_dir) / checked / DECLARATIONS_FILENAME

    def load(self, dataset_id: str) -> Declarations:
        """The dataset's declarations, or none.

        Raises:
            ContractStoreError: the file exists but cannot be read or is not valid.
        """
        path = self._path(dataset_id)
        if not path.is_file():
            return Declarations()
        try:
            return Declarations.model_validate_json(path.read_text(encoding="utf-8"))
        except OSError as e:
            raise ContractStoreError(f"Could not read declarations: {e.strerror or type(e).__name__}") from e
        except pydantic.ValidationError as e:
            raise ContractStoreError(f"Declarations file is invalid ({e.error_count()} errors)") from e

    def save(self, dataset_id: str, declarations: Declarations) -> None:
        """Replace the dataset's declarations atomically (write, fsync, rename).

        Raises:
            ContractStoreError: the file could not be written.
        """
        path = self._path(dataset_id)
        text = json.dumps(declarations.model_dump(mode="json"), indent=1, sort_keys=True)
        with _store_lock:
            tmp: str | None = None
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp-", suffix=".part")
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    f.write(text)
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(tmp, path)
                tmp = None  # renamed into place: nothing to clean up
            except OSError as e:
                raise ContractStoreError(f"Could not write declarations: {e.strerror or type(e).__name__}") from e
            finally:
                if tmp is not None:
                    Path(tmp).unlink(missing_ok=True)
        logger.info("declarations_saved", dataset_id=dataset_id, grains=len(declarations.grains))

    def update(self, dataset_id: str, change: Callable[[Declarations], Declarations]) -> Declarations:
        """Load, apply ``change`` and save as one step under the store lock.

        Raises:
            ContractStoreError: the file could not be read, is invalid, or could not be written.
        """
        with _store_lock:
            updated = change(self.load(dataset_id))
            self.save(dataset_id, updated)
            return updated
