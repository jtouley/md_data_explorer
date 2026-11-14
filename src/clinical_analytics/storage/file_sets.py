# SPDX-License-Identifier: MIT
"""One table from many files: a ZIP that holds one text file per patient.

PhysioNet's Sepsis Challenge 2019 ships 40,336 files (``training_setA/p000001.psv``, ...), one per
ICU stay, all with the same columns; the patient ID is only in the file name. Read as 40,336
tables, nothing could be asked of them. A *file set* is read as one table instead:

* Members are grouped when their file names are numbered with one prefix (``p000001``,
  ``p100001``), they have the same extension and their header lines are identical. A group needs
  at least ``MIN_SET_FILES`` files; anything else stays a table of its own.
* The file name becomes a column. It is ``patient_id`` when the files hold no patient ID of their
  own and the prefix names a patient (``PATIENT_FILE_PREFIXES``: ``p``, ``patient``, ...);
  otherwise ``source_file`` (``visit1``, ``visit2`` are batches, not patients). When the files
  sit in more than one folder, ``source_folder`` says which (Sepsis 2019: hospital system A or B).
* The table is staged like any text table (``storage.staging``), and a note says how many files
  were read into it.

Same-header files whose names are not numbered (``primary_cohort.csv``, ``study_cohort.csv``)
stay separate tables: one can be a subset of another, and joining them would count rows twice.
"""

from __future__ import annotations

import io
import re
from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import PurePosixPath
from typing import Final

import polars as pl
import structlog

from clinical_analytics.core.schema_inference import patient_key_by_name
from clinical_analytics.storage.staging import (
    ColumnTypeNote,
    StagedTable,
    StagingError,
    is_id_column_name,
    stage_text_frame,
)

logger = structlog.get_logger()

MIN_SET_FILES: Final = 2

# Numbered file names with one of these prefixes (case-insensitive, without a trailing "_" or "-")
# are read as one patient per file.
PATIENT_FILE_PREFIXES: Final = frozenset({"p", "pt", "pat", "patient", "subject", "subj", "participant", "case"})

PATIENT_ID_COLUMN: Final = "patient_id"
SOURCE_FILE_COLUMN: Final = "source_file"
SOURCE_FOLDER_COLUMN: Final = "source_folder"

_NUMBERED_STEM: Final = re.compile(r"^(?P<prefix>[A-Za-z]+[_-]?)(?P<number>\d+)$")
_DEFAULT_SET_NAME: Final = "records"


@dataclass(frozen=True)
class TextMember:
    """A delimited text file from an archive: its path there, its (decompressed) bytes, separator."""

    path: str
    content: bytes
    separator: str

    @property
    def stem(self) -> str:
        name = PurePosixPath(self.path).name
        return name.split(".", 1)[0]

    @property
    def folder(self) -> str:
        return str(PurePosixPath(self.path).parent)


@dataclass(frozen=True)
class FileSet:
    """Files read as one table (``name``), in archive order."""

    name: str
    prefix: str
    members: tuple[TextMember, ...]


def split_file_sets(members: list[TextMember]) -> tuple[list[FileSet], list[TextMember]]:
    """Group the members that form file sets; return the sets and the other members, in order."""
    groups: dict[tuple[str, str, str, bytes], list[TextMember]] = defaultdict(list)
    for member in members:
        match = _NUMBERED_STEM.match(member.stem)
        if match:
            prefix = match["prefix"].rstrip("_-").lower()
            suffix = PurePosixPath(member.path).name.split(".", 1)[1].lower()
            groups[(prefix, suffix, member.separator, _header_line(member.content))].append(member)
    in_set: set[str] = set()
    sets: list[FileSet] = []
    for (prefix, _suffix, _sep, _header), group in groups.items():
        if len(group) >= MIN_SET_FILES:
            sets.append(FileSet(name=_set_name(group), prefix=prefix, members=tuple(group)))
            in_set.update(m.path for m in group)
    singles = [m for m in members if m.path not in in_set]
    if sets:
        logger.info("zip_file_sets_found", sets=len(sets), files=len(in_set))
    return sets, singles


def read_file_set(file_set: FileSet) -> StagedTable:
    """Read a file set as one staged table, with the file name (and folder) as columns.

    Raises:
        StagingError: a file cannot be parsed, or a row has no ID and no values.
    """
    first = file_set.members[0]
    separator = first.separator
    try:
        columns = pl.read_csv(io.BytesIO(first.content), separator=separator, n_rows=0).columns
    except (pl.exceptions.ComputeError, pl.exceptions.NoDataError) as e:
        raise StagingError(_unreadable(file_set.name, e)) from e
    id_column = _id_column(file_set.prefix, columns)
    added = [c for c in (id_column, SOURCE_FOLDER_COLUMN) if c in columns]
    if added:
        raise StagingError(
            f"Files in '{file_set.name}' already have a column named '{added[0]}', a name the upload adds "
            "from the file and folder names. Rename that column in the files and upload again."
        )

    text, rows_per_file = _read_members(file_set.members, columns, separator, file_set.name)
    stems = [m.stem for m in file_set.members]
    folders = [m.folder for m in file_set.members]
    if len(set(stems)) < len(stems):  # the same name in two folders: keep both apart
        stems = [f"{folder}/{stem}" for folder, stem in zip(folders, stems, strict=True)]
    extra = [pl.Series(id_column, stems).repeat_by(pl.Series(rows_per_file)).explode()]
    if len(set(folders)) > 1:
        extra.append(pl.Series(SOURCE_FOLDER_COLUMN, folders).repeat_by(pl.Series(rows_per_file)).explode())
    with_names = pl.concat([pl.DataFrame(extra), text], how="horizontal")

    staged = stage_text_frame(with_names)
    files = len(file_set.members)
    note = ColumnTypeNote(id_column, "String", "from_file_names", files, files)
    logger.info("zip_file_set_read", files=len(file_set.members), rows=with_names.height, columns=with_names.width)
    return replace(staged, notes=(note, *staged.notes))


def _read_members(
    members: tuple[TextMember, ...], columns: list[str], separator: str, name: str
) -> tuple[pl.DataFrame, list[int]]:
    """All members' rows as text, and the row count of each.

    The bodies are joined and parsed once (one parse instead of one per file). A body's rows are
    its lines; when the parsed height disagrees (a quoted value spanning lines, a blank line),
    each file is parsed on its own instead, which is slower but exact.
    """
    bodies: list[bytes] = []
    for member in members:
        body = member.content.split(b"\n", 1)[1] if b"\n" in member.content else b""
        if body and not body.endswith(b"\n"):
            body += b"\n"
        bodies.append(body)
    counts = [body.count(b"\n") for body in bodies]
    try:
        joined = pl.read_csv(
            io.BytesIO(b"".join(bodies)),
            separator=separator,
            has_header=False,
            new_columns=columns,
            infer_schema=False,
        )
        if joined.height == sum(counts) and joined.width == len(columns):
            return joined, counts
        logger.info("zip_file_set_line_count_mismatch", files=len(members))
        frames = [
            pl.read_csv(io.BytesIO(m.content), separator=separator, infer_schema=False).select(columns) for m in members
        ]
    except (pl.exceptions.ComputeError, pl.exceptions.NoDataError, pl.exceptions.ColumnNotFoundError) as e:
        raise StagingError(_unreadable(name, e)) from e
    return pl.concat(frames, how="vertical"), [f.height for f in frames]


def _id_column(prefix: str, columns: list[str]) -> str:
    """``patient_id`` for one file per patient; ``source_file`` when the files hold patient IDs."""
    holds_ids = patient_key_by_name(columns) is not None or any(is_id_column_name(c) for c in columns)
    if prefix in PATIENT_FILE_PREFIXES and not holds_ids:
        return PATIENT_ID_COLUMN
    return SOURCE_FILE_COLUMN


def _set_name(group: list[TextMember]) -> str:
    """The folders' shared name (``training_setA``, ``training_setB`` -> ``training_set``), else
    the one folder, else ``records``."""
    folders = sorted({PurePosixPath(m.folder).name for m in group if m.folder not in ("", ".")})
    if not folders:
        return _DEFAULT_SET_NAME
    if len(folders) == 1:
        return folders[0]
    shared = folders[0]
    for folder in folders[1:]:
        while not folder.startswith(shared):
            shared = shared[:-1]
    return shared.rstrip("_- ") or _DEFAULT_SET_NAME


def _unreadable(name: str, error: Exception) -> str:
    return (
        f"Files in '{name}' could not be read ({type(error).__name__}). "
        "Check that every file has the same header row, then upload the ZIP again."
    )


def _header_line(content: bytes) -> bytes:
    line = content.split(b"\n", 1)[0]
    return line.rstrip(b"\r").removeprefix(b"\xef\xbb\xbf")
