# SPDX-License-Identifier: MIT
"""File sets: a ZIP of numbered per-patient files with the same columns becomes one table."""

from __future__ import annotations

import polars as pl
import pytest

from clinical_analytics.storage.file_sets import (
    PATIENT_ID_COLUMN,
    SOURCE_FILE_COLUMN,
    SOURCE_FOLDER_COLUMN,
    TextMember,
    read_file_set,
    split_file_sets,
)
from clinical_analytics.storage.staging import StagingError

_HEADER = "HR|Lactate|Gender|SepsisLabel"


def _psv(path: str, rows: list[str], header: str = _HEADER) -> TextMember:
    return TextMember(path, ("\n".join([header, *rows]) + "\n").encode(), "|")


def _patient_files() -> list[TextMember]:
    """Hospital A and B, PhysioNet-style: one file per ICU stay, hourly rows, NaN markers."""
    return [
        _psv("training_setA/p000001.psv", ["80|NaN|0|0", "82|2.1|0|0", "85|NaN|0|1"]),
        _psv("training_setA/p000002.psv", ["70|NaN|1|0"]),
        _psv("training_setB/p100001.psv", ["90|1.4|1|0", "NaN|NaN|1|0"]),
    ]


def test_splitFileSets_numberedSameHeader_isOneSet_othersStayTables() -> None:
    # Arrange: three patient files plus an unrelated table and same-header unnumbered files
    other = _psv("dictionary.psv", ["a|b|c|d"], header="w|x|y|z")
    primary = _psv("primary_cohort.psv", ["1|2|0|0"])
    study = _psv("study_cohort.psv", ["1|2|0|0"])
    members = [other, *_patient_files(), primary, study]

    # Act
    sets, singles = split_file_sets(members)

    # Assert
    assert [(s.name, s.prefix, len(s.members)) for s in sets] == [("training_set", "p", 3)]
    assert singles == [other, primary, study]


@pytest.mark.parametrize(
    "members",
    [
        pytest.param([], id="empty"),
        pytest.param([_psv("p1.psv", ["1|2|0|0"])], id="one_file"),
        pytest.param([_psv("p1.psv", ["1|2|0|0"]), _psv("p2.psv", ["1|0"], header="HR|Gender")], id="headers_differ"),
        pytest.param(
            [_psv("p1.psv", ["1|2|0|0"]), TextMember("p2.csv", b"HR|Lactate|Gender|SepsisLabel\n1|2|0|0\n", "|")],
            id="extensions_differ",
        ),
    ],
)
def test_splitFileSets_noGroupOfTwo_leavesEveryFileATable(members: list[TextMember]) -> None:
    # Arrange: members above

    # Act
    sets, singles = split_file_sets(members)

    # Assert
    assert sets == []
    assert singles == members


def test_readFileSet_patientFiles_oneTable_withPatientIdAndFolder() -> None:
    # Arrange
    (file_set,), _ = split_file_sets(_patient_files())

    # Act
    staged = read_file_set(file_set)

    # Assert
    data = staged.data
    assert data.columns == [PATIENT_ID_COLUMN, SOURCE_FOLDER_COLUMN, "HR", "Lactate", "Gender", "SepsisLabel"]
    assert data.height == 6
    assert data[PATIENT_ID_COLUMN].to_list() == ["p000001"] * 3 + ["p000002"] + ["p100001"] * 2
    assert data[SOURCE_FOLDER_COLUMN].to_list() == ["training_setA"] * 4 + ["training_setB"] * 2
    assert data.schema["Lactate"] == pl.Float64
    assert data.schema["Gender"] == pl.Int64
    assert data["Lactate"].null_count() == 4
    note = staged.notes[0]
    assert (note.column, note.reason, note.failed_values) == (PATIENT_ID_COLUMN, "from_file_names", 3)


def test_readFileSet_oneFolder_noFolderColumn_tableNamedForFolder() -> None:
    # Arrange
    files = [_psv("records/pt_01.psv", ["80|1|0|0"]), _psv("records/pt_02.psv", ["81|1|1|0"])]
    (file_set,), _ = split_file_sets(files)

    # Act
    staged = read_file_set(file_set)

    # Assert
    assert file_set.name == "records"
    assert SOURCE_FOLDER_COLUMN not in staged.data.columns
    assert staged.data[PATIENT_ID_COLUMN].to_list() == ["pt_01", "pt_02"]


@pytest.mark.parametrize(
    ("paths", "header", "id_column"),
    [
        pytest.param(["visit1.csv", "visit2.csv"], "HR,Gender", SOURCE_FILE_COLUMN, id="prefix_not_a_patient"),
        pytest.param(["p1.csv", "p2.csv"], "patient_id,HR", SOURCE_FILE_COLUMN, id="files_hold_patient_ids"),
        pytest.param(["site1.csv", "site2.csv"], "mrn,HR", SOURCE_FILE_COLUMN, id="files_hold_mrn"),
        pytest.param(["subject_7.csv", "subject_8.csv"], "HR,Gender", PATIENT_ID_COLUMN, id="subject_prefix"),
    ],
)
def test_readFileSet_idColumn_patientOnlyWhenFilesAreThePatients(paths: list[str], header: str, id_column: str) -> None:
    # Arrange
    files = [TextMember(p, f"{header}\n{i},80\n".encode(), ",") for i, p in enumerate(paths)]
    (file_set,), _ = split_file_sets(files)

    # Act
    staged = read_file_set(file_set)

    # Assert
    assert staged.data.columns[0] == id_column
    assert staged.data[id_column].to_list() == [p.split(".")[0] for p in paths]


def test_readFileSet_sameNameInTwoFolders_idKeepsFolder() -> None:
    # Arrange
    files = [_psv("A/p1.psv", ["80|1|0|0"]), _psv("B/p1.psv", ["81|1|1|0"])]
    (file_set,), _ = split_file_sets(files)

    # Act
    staged = read_file_set(file_set)

    # Assert
    assert staged.data[PATIENT_ID_COLUMN].to_list() == ["A/p1", "B/p1"]


def test_readFileSet_quotedValueSpanningLines_rowCountsStayExact() -> None:
    # Arrange: a quoted note with a line break, and a file with no trailing newline or rows
    files = [
        TextMember("p1.csv", b'note,HR\n"line one\nline two",80\nok,81\n', ","),
        TextMember("p2.csv", b"note,HR\nfine,82", ","),
        TextMember("p3.csv", b"note,HR\n", ","),
    ]
    (file_set,), _ = split_file_sets(files)

    # Act
    staged = read_file_set(file_set)

    # Assert
    assert staged.data[PATIENT_ID_COLUMN].to_list() == ["p1", "p1", "p2"]
    assert staged.data["HR"].to_list() == [80, 81, 82]


def test_readFileSet_headerAlreadyHasAddedName_raisesStagingError() -> None:
    # Arrange
    files = [_psv(f"{d}/p{i}.psv", ["1|x"], header="HR|source_folder") for i, d in enumerate("AB")]
    (file_set,), _ = split_file_sets(files)

    # Act / Assert
    with pytest.raises(StagingError, match="already have a column named 'source_folder'"):
        read_file_set(file_set)
