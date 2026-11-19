# SPDX-License-Identifier: MIT
"""Upload → stored dataset through the API for the formats public datasets ship in.

ZIPs of tables plus documentation (ADR004), ZIPs of one file per patient (PhysioNet), a ZIP
inside a ZIP (UCI), SAS transport (.xpt, NHANES) and legacy .xls. Every file is synthetic.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import io
import zipfile
from pathlib import Path

import polars as pl
import pyreadstat
import pytest
import xlwt
from fastapi.testclient import TestClient


def _zip(files: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as zf:  # stored: above the 1 KB minimum
        for name, content in files.items():
            zf.writestr(name, content)
    return buf.getvalue()


def _cohort_csv(n: int = 120) -> bytes:
    rows = [f"P{i:03d},{30 + i % 40},{'yes' if i % 3 else 'no'}" for i in range(n)]
    return ("\n".join(["patient_id,age,covid_hosp", *rows]) + "\n").encode()


_README = b"Synthetic cohort. covid_hosp: hospitalised with COVID-19 (yes/no). age: years at diagnosis.\n" * 3


def _upload(client: TestClient, name: str, data: bytes, dataset_name: str) -> dict:
    response = client.post(
        "/api/datasets/upload",
        files={"file": (name, data, "application/octet-stream")},
        data={"dataset_name": dataset_name},
    )
    body = response.json()
    body["_status_code"] = response.status_code
    return body


def _stored(upload_id: str) -> tuple[pl.DataFrame, dict]:
    from clinical_analytics.ui.storage.user_datasets import UserDatasetStorage

    storage = UserDatasetStorage()
    frame = storage.get_upload_data(upload_id)
    assert frame is not None
    return frame.collect(), storage.get_upload_metadata(upload_id) or {}


def test_uploadDataset_zipWithTableAndReadme_readyAndDocsReachContext(test_client: TestClient, isolated_data_home):
    # Arrange: PhysioNet's COVID-MS package shape: one CSV, a README and a LICENSE
    data = _zip({"cohort.csv": _cohort_csv(), "README.txt": _README, "LICENSE.txt": b"CC BY 4.0\n" * 40})

    # Act
    body = _upload(test_client, "covid_ms.zip", data, "covid_ms")
    frame, meta = _stored(body["upload_id"])

    # Assert
    assert body["_status_code"] == 200, body
    assert body["status"] == "ready"
    assert frame.height == 120
    assert frame.schema["age"] == pl.Int64
    assert "covid_hosp: hospitalised with COVID-19" in meta["doc_context"]


def _patient_file(hours: int, gender: int) -> bytes:
    rows = [f"{80 + h}|{'NaN' if h % 2 else '2.1'}|{gender}|{int(h == hours - 1)}" for h in range(hours)]
    return ("\n".join(["HR|Lactate|Gender|SepsisLabel", *rows]) + "\n").encode()


def test_uploadDataset_zipOfPerPatientPsv_oneTable_patientsFromFileNames(test_client: TestClient, isolated_data_home):
    # Arrange: Sepsis Challenge 2019 shape: two hospital folders, one PSV per patient, a PDF
    files = {f"training_setA/p{i:06d}.psv": _patient_file(3 + i % 4, i % 2) for i in range(1, 16)}
    files.update({f"training_setB/p{100000 + i:06d}.psv": _patient_file(2 + i % 3, i % 2) for i in range(1, 11)})
    files["physionet_project_page.txt"] = b"Hourly vital signs and labs for ICU patients; SepsisLabel 0/1.\n" * 3

    # Act
    body = _upload(test_client, "challenge-2019.zip", _zip(files), "sepsis")
    frame, _meta = _stored(body["upload_id"])

    # Assert
    assert body["_status_code"] == 200, body
    assert frame["patient_id"].n_unique() == 25
    assert frame.group_by("source_folder").agg(pl.col("patient_id").n_unique()).sort("source_folder").rows() == [
        ("training_setA", 15),
        ("training_setB", 10),
    ]
    assert frame.schema["Lactate"] == pl.Float64
    assert any(note.startswith("patient_id: added from the file names: 25 files") for note in body["type_notes"])


def test_uploadDataset_zipInsideZip_tableRead(test_client: TestClient, isolated_data_home):
    # Arrange: UCI ships some datasets as a ZIP holding the data ZIP
    inner = _zip({"cohort.csv": _cohort_csv()})

    # Act
    body = _upload(test_client, "outer.zip", _zip({"dataset.zip": inner}), "nested")
    frame, _meta = _stored(body["upload_id"])

    # Assert
    assert body["_status_code"] == 200, body
    assert frame.height == 120


def test_uploadDataset_zipWithStackedCodebookCsv_codebookReadAsDocs(test_client: TestClient, isolated_data_home):
    # Arrange: UCI diabetes shape: the data table plus a CSV of three codebooks split by blank rows
    codebook = b"admission_type_id,description\n1,Emergency\n2,Urgent\n,\ndischarge_id,description\n1,Home\n"

    # Act
    body = _upload(test_client, "diabetes.zip", _zip({"data.csv": _cohort_csv(), "mapping.csv": codebook}), "dm")
    frame, meta = _stored(body["upload_id"])

    # Assert
    assert body["_status_code"] == 200, body
    assert frame.height == 120
    assert "mapping.csv was kept as documentation, not read as a table" in body["message"]
    assert any(note.startswith("mapping.csv was kept as documentation") for note in body["cohort_notes"])
    assert "discharge_id" in meta["doc_context"]


@pytest.mark.parametrize(
    "labs",
    [
        pytest.param(b"patient_id,alt\nP000,31\nP001,40\n,\nP002,22\n", id="blank_row_then_numbers"),
        pytest.param(b"patient_id,Alt,alt\nP000,31,30\nP001,40,41\n", id="case_twin_headers"),
    ],
)
def test_uploadDataset_zipWithBrokenDataTable_refused400NeverKeptAsDocs(
    test_client: TestClient, isolated_data_home, labs: bytes
):
    # Arrange: a data table that staging refuses. Kept as documentation, its patient rows would
    # reach doc_context and the model's prompts; only a codebook may become documentation.

    # Act
    body = _upload(test_client, "labs.zip", _zip({"cohort.csv": _cohort_csv(), "labs.csv": labs}), "labs")

    # Assert
    assert body["_status_code"] == 400, body
    assert body["detail"].startswith("Table 'labs':")


def test_uploadDataset_zipWithCorruptSav_failsWithReadableReason(test_client: TestClient, isolated_data_home):
    # Arrange: the bytes of a .sav that is not one
    data = _zip({"cohort.csv": _cohort_csv(), "extra.sav": b"$FL2" + b"\x00" * 2000})

    # Act
    body = _upload(test_client, "sav.zip", data, "sav")

    # Assert: the user's file to fix (400) with a user message, not a server fault
    assert body["_status_code"] == 400
    assert "SPSS (.sav) file could not be read" in body["detail"]


def test_uploadDataset_zipOfTablesWithNoSharedKey_saysWhyInPlainWords(test_client: TestClient, isolated_data_home):
    # Arrange: two cohorts with the same columns and no ID (UCI sepsis_survival shape)
    rows = "\n".join(f"{40 + i % 50},{i % 2},{i % 5 + 1}" for i in range(120))
    cohort = f"age_years,sex_0male_1female,episode_number\n{rows}\n".encode()

    # Act
    body = _upload(test_client, "two.zip", _zip({"primary.csv": cohort, "validation.csv": cohort}), "two")

    # Assert: the files cannot be combined: the user's to fix (400), never a server fault
    assert body["_status_code"] == 400
    assert "cannot be joined into one cohort" in body["detail"]


def test_uploadDataset_zipWithoutTables_rejectedWithAcceptedFormats(test_client: TestClient, isolated_data_home):
    # Arrange
    data = _zip({"README.txt": _README, "page.htm": b"<html></html>" * 100})

    # Act
    body = _upload(test_client, "docs_only.zip", data, "docs")

    # Assert
    assert body["_status_code"] == 400
    assert "No table files in ZIP" in body["detail"]
    assert ".xpt" in body["detail"]


def _xpt(tmp_path: Path, df: pl.DataFrame, name: str) -> bytes:
    path = tmp_path / f"{name}.xpt"
    pyreadstat.write_xport(df, str(path), file_format_version=5, table_name=name)
    return path.read_bytes()


def _nhanes_like(tmp_path: Path) -> tuple[bytes, bytes]:
    """Demographics for everyone, a lab file for a subset, joined 1:1 on SEQN (NHANES shape)."""
    demo = pl.DataFrame(
        {
            "SEQN": [float(93700 + i) for i in range(80)],
            "RIAGENDR": [float(1 + i % 2) for i in range(80)],
            "RIDAGEYR": [float(20 + i % 60) for i in range(80)],
        }
    )
    lab = pl.DataFrame(
        {"SEQN": [float(93700 + i) for i in range(0, 80, 2)], "LBXSATSI": [10.5 + i % 30 for i in range(40)]}
    )
    return _xpt(tmp_path, demo, "DEMO_J"), _xpt(tmp_path, lab, "BIOPRO_J")


def test_uploadDataset_xpt_readyWithSasCodesAsIntegers(test_client: TestClient, isolated_data_home, tmp_path: Path):
    # Arrange
    demo, _lab = _nhanes_like(tmp_path)

    # Act
    body = _upload(test_client, "DEMO_J.xpt", demo, "demo")
    frame, _meta = _stored(body["upload_id"])

    # Assert
    assert body["_status_code"] == 200, body
    assert frame.height == 80
    assert frame.schema["RIAGENDR"] == pl.Int64


def test_uploadDataset_zipOfXptFiles_cohortHoldsEveryone(test_client: TestClient, isolated_data_home, tmp_path: Path):
    # Arrange
    demo, lab = _nhanes_like(tmp_path)

    # Act
    body = _upload(test_client, "nhanes.zip", _zip({"DEMO_J.xpt": demo, "BIOPRO_J.xpt": lab}), "nhanes")
    frame, _meta = _stored(body["upload_id"])

    # Assert: anchored on demographics (80 people), not on the lab subset (40)
    assert body["_status_code"] == 200, body
    assert frame.height == 80
    assert frame["LBXSATSI"].null_count() == 40


@pytest.mark.parametrize("raw_biff", [False, True], ids=["ole2_container", "raw_biff_stream"])
def test_uploadDataset_legacyXls_ready(test_client: TestClient, isolated_data_home, raw_biff: bool):
    # Arrange
    wb = xlwt.Workbook()
    ws = wb.add_sheet("Sheet 1")
    for c, name in enumerate(["patno", "rx", "sdate", "visit"]):
        ws.write(0, c, name)
    for r in range(1, 41):
        ws.write(r, 0, r)
        ws.write(r, 1, "placebo" if r % 2 else "0.2 mg estrogen")
        ws.write(r, 2, 2700 + r)
        ws.write(r, 3, dt.date(2020, 1, 1 + r % 28), xlwt.easyxf(num_format_str="YYYY-MM-DD"))
    buf = io.BytesIO()
    wb.save(buf)
    data = bytes(wb.get_biff_data()) if raw_biff else buf.getvalue()

    # Act
    body = _upload(test_client, "prostate.xls", data, "prostate")
    frame, _meta = _stored(body["upload_id"])

    # Assert
    assert body["_status_code"] == 200, body
    assert frame.height == 40
    assert frame.schema["sdate"] == pl.Int64
    assert frame.schema["visit"] == pl.Date


# --- Upload failures the user can fix: 400 with what to do, never a 500 ---------------------------


def test_uploadDataset_csvWithBlankHeader_readyWithNamedColumn(test_client: TestClient, isolated_data_home):
    # Arrange: R's write.csv row-name column under a "" header (UCI hcvdat0.csv shape). It used to
    # fail the query-view probe ("column types differ ... (, C0)") with a 500.
    rows = [f'"{i}",P{i:03d},{30 + i % 40},{38.5 + i % 7}' for i in range(1, 121)]
    data = ("\n".join(['"","patient_id","Age","ALB"', *rows]) + "\n").encode()

    # Act
    body = _upload(test_client, "hcvdat0.csv", data, "hcv")
    frame, _meta = _stored(body["upload_id"])

    # Assert
    assert body["_status_code"] == 200, body
    assert body["status"] == "ready"
    assert frame.height == 120
    assert "column_0" in frame.columns
    assert "" not in frame.columns


def _no_id_cohort(n: int = 150) -> bytes:
    """UCI sepsis_survival shape: age, sex, episode and outcome; no column identifies a patient."""
    rows = [f"{40 + i % 50},{i % 2},{i % 5 + 1},{i % 2}" for i in range(n)]
    return (
        "\n".join(["age_years,sex_0male_1female,episode_number,hospital_outcome_1alive_0dead", *rows]) + "\n"
    ).encode()


@pytest.mark.parametrize("as_zip", [False, True], ids=["csv", "zip_of_one_csv"])
def test_uploadDataset_noPatientIdColumn_uploadsAndSaysRowsNotPatients(
    test_client: TestClient, isolated_data_home, as_zip: bool
):
    # Arrange: before #103 and this branch, "Error saving upload: ValueError" with a 500
    data = _zip({"primary_cohort.csv": _no_id_cohort()}) if as_zip else _no_id_cohort()
    name = "sepsis.zip" if as_zip else "sepsis.csv"

    # Act
    body = _upload(test_client, name, data, "sepsis")

    # Assert: stored with a row number, never counted as patients, and the upload says so
    assert body["_status_code"] == 200, body
    assert body["status"] == "ready"
    assert body["unit_note"].startswith("No column in this file identifies a patient")
    assert "Answers count rows, not patients." in body["unit_note"]
    assert "add a column with one ID for each patient" in body["unit_note"]


def test_uploadDataset_headersDifferOnlyInCase_refused400NamesBoth(test_client: TestClient, isolated_data_home):
    # Arrange: the query engine reads column names without case, so "Age" and "age" would be one
    rows = [f"P{i:03d},{30 + i % 40},{31 + i % 40}" for i in range(120)]
    data = ("\n".join(["patient_id,Age,age", *rows]) + "\n").encode()

    # Act
    body = _upload(test_client, "twins.csv", data, "twins")

    # Assert
    assert body["_status_code"] == 400, body
    assert "'Age' and 'age' differ only in upper and lower case" in body["detail"]


def _deflated_zip(files: dict[str, bytes]) -> bytearray:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, content in files.items():
            zf.writestr(name, content)
    return bytearray(buf.getvalue())


def _set_central_field(data: bytearray, offset: int, value: int, *, local_offset: int | None = None) -> bytes:
    """Overwrite a 2-byte field of every central-directory entry (and the local header's)."""
    start = 0
    while (i := data.find(b"PK\x01\x02", start)) != -1:
        data[i + offset : i + offset + 2] = value.to_bytes(2, "little")
        start = i + 4
    if local_offset is not None:
        start = 0
        while (i := data.find(b"PK\x03\x04", start)) != -1:
            data[i + local_offset : i + local_offset + 2] = value.to_bytes(2, "little")
            start = i + 4
    return bytes(data)


def _damaged_member(files: dict[str, bytes], member: str) -> bytes:
    """Flip bytes inside one member's compressed data (its CRC or inflate then fails)."""
    data = _deflated_zip(files)
    header = data.find(b"PK\x03\x04")  # first local header
    while header != -1:
        name_len = int.from_bytes(data[header + 26 : header + 28], "little")
        extra_len = int.from_bytes(data[header + 28 : header + 30], "little")
        name = bytes(data[header + 30 : header + 30 + name_len]).decode()
        body = header + 30 + name_len + extra_len
        if name == member:
            for k in range(body + 2, body + 12):
                data[k] ^= 0xFF
            return bytes(data)
        header = data.find(b"PK\x03\x04", header + 4)
    raise AssertionError(f"{member} not in the ZIP")


# Hex digests barely compress, so the deflated ZIP stays above the route's 1 KB minimum.
_TABLE_AND_DOC = {
    "cohort.csv": _cohort_csv(),
    "README.txt": _README + b"".join(hashlib.sha256(str(i).encode()).hexdigest().encode() for i in range(40)),
}


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        pytest.param(
            _set_central_field(_deflated_zip(_TABLE_AND_DOC), 8, 0x1, local_offset=6),
            "is password-protected",
            id="encrypted",
        ),
        pytest.param(
            _set_central_field(_deflated_zip(_TABLE_AND_DOC), 10, 9, local_offset=8),
            "compression method this app cannot read",
            id="deflate64",
        ),
        pytest.param(_damaged_member(_TABLE_AND_DOC, "cohort.csv"), "cohort.csv in the ZIP is damaged", id="bad_table"),
        pytest.param(_damaged_member(_TABLE_AND_DOC, "README.txt"), "README.txt in the ZIP is damaged", id="bad_doc"),
        pytest.param(
            _set_central_field(_deflated_zip(_TABLE_AND_DOC), 6, 140), "Corrupted ZIP file", id="unknown_zip_version"
        ),
    ],
)
def test_uploadDataset_damagedOrLockedZip_refused400NamesTheFix(
    test_client: TestClient, isolated_data_home, data: bytes, expected: str
):
    # Arrange: done in the parameters (each case used to end in a 500, or for a damaged README,
    # in a "ready" upload whose documentation was dropped in silence)

    # Act
    body = _upload(test_client, "package.zip", data, "package")

    # Assert
    assert body["_status_code"] == 400, body
    assert expected in body["detail"]
    assert "upload the new ZIP" in body["detail"]


def test_uploadDataset_zipWithTooManyEntries_refused400(
    test_client: TestClient, isolated_data_home, monkeypatch: pytest.MonkeyPatch
):
    # Arrange: a low cap stands in for an archive of millions of empty entries
    import clinical_analytics.ui.storage.user_datasets as user_datasets

    monkeypatch.setattr(user_datasets, "MAX_ZIP_ENTRIES", 3)
    data = _zip({"cohort.csv": _cohort_csv(), **{f"empty_{i}.txt": b"" for i in range(4)}})

    # Act
    body = _upload(test_client, "many.zip", data, "many")

    # Assert
    assert body["_status_code"] == 400, body
    assert "The ZIP holds more than 3 files" in body["detail"]


# --- Review round: docs in a nested ZIP, docs that share a name, workbooks in a ZIP ---------------


def test_uploadDataset_zipInsideZipWithReadme_docsReachContext(test_client: TestClient, isolated_data_home):
    # Arrange: the README sits next to the table inside the nested ZIP (UCI shape)
    inner = _zip({"cohort.csv": _cohort_csv(), "README.txt": _README})

    # Act
    body = _upload(test_client, "outer.zip", _zip({"dataset.zip": inner}), "nested_docs")
    _frame, meta = _stored(body["upload_id"])

    # Assert
    assert body["_status_code"] == 200, body
    assert "covid_hosp: hospitalised with COVID-19" in meta["doc_context"]


def test_uploadDataset_docsSharingAName_bothReachContext(test_client: TestClient, isolated_data_home):
    # Arrange: README.txt in two folders; extraction once wrote both to one temp path
    data = _zip(
        {
            "cohort.csv": _cohort_csv(),
            "siteA/README.txt": b"Site A dictionary: alpha_marker means the first assay.\n" * 4,
            "siteB/README.txt": b"Site B dictionary: beta_marker means the second assay.\n" * 4,
        }
    )

    # Act
    body = _upload(test_client, "sites.zip", data, "sites")
    _frame, meta = _stored(body["upload_id"])

    # Assert
    assert body["_status_code"] == 200, body
    assert "alpha_marker" in meta["doc_context"]
    assert "beta_marker" in meta["doc_context"]


def _xlsx(sheets: dict[str, list[list[object]]]) -> bytes:
    import openpyxl

    wb = openpyxl.Workbook()
    wb.remove(wb.active)
    for title, rows in sheets.items():
        ws = wb.create_sheet(title)
        for row in rows:
            ws.append(row)
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def test_uploadDataset_zipWithTwoWorkbooks_eachUnreadSheetNamedWithItsFile(test_client: TestClient, isolated_data_home):
    # Arrange: only the second workbook has a sheet left unread; the note once came from the first
    first = _xlsx({"Baseline": [["patient_id", "bmi"], *[[f"P{i:03d}", 20 + i % 9] for i in range(120)]]})
    second = _xlsx(
        {
            "Labs": [["patient_id", "alt"], *[[f"P{i:03d}", 30 + i % 11] for i in range(0, 120, 2)]],
            "FollowUp": [["patient_id", "alt_6m"], ["P000", 33]],
        }
    )

    # Act
    body = _upload(test_client, "books.zip", _zip({"a.xlsx": first, "b.xlsx": second}), "books")

    # Assert
    assert body["_status_code"] == 200, body
    assert any(n.startswith("b.xlsx: Read 1 of 2 sheets") and "FollowUp" in n for n in body["cohort_notes"])
    assert not any(n.startswith("a.xlsx") for n in body["cohort_notes"])


@pytest.mark.parametrize("extension", [".txt", ".tsv"])
def test_uploadDataset_wonderExport_readyNotesAsDocsNeverRows(
    test_client: TestClient, isolated_data_home, make_wonder_export, extension: str
):
    # Arrange: a synthetic CDC WONDER export as downloaded (tab-separated .txt)
    data = make_wonder_export(states=40)

    # Act
    body = _upload(test_client, f"wonder{extension}", data, f"wonder{extension[1:]}")
    frame, meta = _stored(body["upload_id"])

    # Assert: uploaded; the notes block is documentation (doc_context), not rows
    assert body["_status_code"] == 200, body
    assert frame.height == 41  # 40 states and the Total row
    assert "Query Parameters" in meta.get("doc_context", "")
    assert any("were kept as the dataset's documentation, not as rows" in n for n in body["cohort_notes"])
    assert frame["Deaths"].min() > 0  # "Suppressed" is never 0


def test_uploadDataset_txtNotTabSeparated_refused400SaysHowToFix(test_client: TestClient, isolated_data_home):
    # Arrange: prose, not a table
    data = ("Some notes about a study, written as prose and not as a table. " * 30).encode()

    # Act
    body = _upload(test_client, "notes.txt", data, "notes")

    # Assert
    assert body["_status_code"] == 400, body
    assert "not tab-separated" in body["detail"]


@pytest.mark.parametrize(
    "data",
    [
        pytest.param("patient_id\tage\nP1\t40\nP2\t50\n".encode("utf-16") * 40, id="utf16_unicodeText"),
        pytest.param(
            b"Study protocol\tversion 2\n" + b"This study enrolled adults with a long history. " * 40, id="prose"
        ),
    ],
)
def test_uploadDataset_txtNotUtf8Table_refused400(test_client: TestClient, isolated_data_home, data: bytes):
    # Act
    body = _upload(test_client, "export.txt", data, "export")

    # Assert: never stored as a garbled or prose "table"
    assert body["_status_code"] == 400, body
    assert "upload it again" in body["detail"]


def test_uploadDataset_zipOfNumberedWonderExports_neverJoinedAsOneSet(
    test_client: TestClient, isolated_data_home, make_wonder_export
):
    # Arrange: numbered WONDER files are read one by one; joined as a file set (review of #123),
    # their Total rows and notes lines became data rows and sums doubled
    data = _zip({"deaths_1.tsv": make_wonder_export(states=29), "deaths_2.tsv": make_wonder_export(states=29)})

    # Act
    body = _upload(test_client, "wonder.zip", data, "wonderzip")

    # Assert: two separate tables share no ID, so the ZIP is refused plainly, never mixed
    assert body["_status_code"] == 400, body
    assert "Upload the table you want to ask about as its own file." in body["detail"]


def test_uploadDataset_zipOfOneWonderExport_notesAsDocsNeverRows(
    test_client: TestClient, isolated_data_home, make_wonder_export
):
    # Arrange
    data = _zip({"deaths.tsv": make_wonder_export(states=29)})

    # Act
    body = _upload(test_client, "wonder1.zip", data, "wonder1")
    frame, meta = _stored(body["upload_id"])

    # Assert
    assert body["_status_code"] == 200, body
    assert frame.height == 30  # 29 states and the Total row
    assert "Query Parameters" in meta.get("doc_context", "")
    assert any(n.startswith("deaths.tsv: The notes under the data") for n in body["cohort_notes"])
