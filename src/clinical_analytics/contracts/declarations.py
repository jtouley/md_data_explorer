# SPDX-License-Identifier: MIT
"""What the doctor confirmed about a dataset: its grain, and which category spellings are one.

A declaration is made by confirming a held question and is stored so the next question does not
ask again. Each one carries the facts it was made on (row and patient counts, the spellings),
and is ignored when the data no longer matches them: a re-upload asks again instead of
applying an old answer to new data. Running a confirmation twice stores the same thing.

Stored as ``contracts/<dataset id>/declarations.json`` in the data directory, next to the
dataset's contracts, so deleting the dataset removes it too. Holds no patient values: column
names, counts and category labels only.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
from pathlib import Path
from typing import Literal, Protocol

import pydantic
import structlog
from pydantic import BaseModel, ConfigDict, Field

from clinical_analytics.contracts.errors import ContractStoreError
from clinical_analytics.contracts.store import contracts_root
from clinical_analytics.storage.dataset_deletion import validate_dataset_id

logger = structlog.get_logger()

DECLARATIONS_FILENAME = "declarations.json"


class GrainDeclaration(BaseModel):
    """Rows are visits of patients identified by ``key`` (declared on ``rows``/``patients``)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    grain: Literal["visit"] = "visit"
    key: str
    rows: int = Field(..., ge=0)
    patients: int = Field(..., ge=0)


class LevelMapping(BaseModel):
    """Spellings of one category that the doctor confirmed are the same level."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    column: str
    spellings: tuple[str, ...]
    canonical: str


class Declarations(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    grain: GrainDeclaration | None = None
    levels: tuple[LevelMapping, ...] = ()

    def with_grain(self, grain: GrainDeclaration) -> Declarations:
        return self.model_copy(update={"grain": grain})

    def with_levels(self, mapping: LevelMapping) -> Declarations:
        kept = tuple(m for m in self.levels if not (m.column == mapping.column and m.canonical == mapping.canonical))
        return self.model_copy(update={"levels": (*kept, mapping)})


class DeclarationStore(Protocol):
    def load(self, dataset_id: str) -> Declarations: ...

    def save(self, dataset_id: str, declarations: Declarations) -> None: ...


class MemoryDeclarationStore:
    """Declarations for this process only (tests, the headless evaluator)."""

    def __init__(self) -> None:
        self._items: dict[str, Declarations] = {}

    def load(self, dataset_id: str) -> Declarations:
        return self._items.get(dataset_id, Declarations())

    def save(self, dataset_id: str, declarations: Declarations) -> None:
        self._items[dataset_id] = declarations


_write_lock = threading.Lock()


class FileDeclarationStore:
    """Declarations under ``<data_dir>/contracts/<dataset id>/declarations.json``."""

    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir

    def _path(self, dataset_id: str) -> Path:
        return contracts_root(self.data_dir) / validate_dataset_id(dataset_id) / DECLARATIONS_FILENAME

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
        with _write_lock:
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp-", suffix=".part")
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    f.write(text)
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(tmp, path)
            except OSError as e:
                raise ContractStoreError(f"Could not write declarations: {e.strerror or type(e).__name__}") from e
        logger.info("declarations_saved", dataset_id=dataset_id, grain=bool(declarations.grain))
