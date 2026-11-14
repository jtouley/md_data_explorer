# SPDX-License-Identifier: MIT
"""Upload → stored dataset through the API for the formats public datasets ship in.

ZIPs of tables plus documentation (ADR004), ZIPs of one file per patient (PhysioNet), a ZIP
inside a ZIP (UCI), SAS transport (.xpt, NHANES) and legacy .xls. Every file is synthetic.
"""

from __future__ import annotations

import datetime as dt
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
    assert "mapping.csv was read as documentation, not as a table" in body["message"]
    assert "discharge_id" in meta["doc_context"]


def test_uploadDataset_zipOfTablesWithNoSharedKey_saysWhyInPlainWords(test_client: TestClient, isolated_data_home):
    # Arrange: two cohorts with the same columns and no ID (UCI sepsis_survival shape)
    rows = "\n".join(f"{40 + i % 50},{i % 2},{i % 5 + 1}" for i in range(120))
    cohort = f"age_years,sex_0male_1female,episode_number\n{rows}\n".encode()

    # Act
    body = _upload(test_client, "two.zip", _zip({"primary.csv": cohort, "validation.csv": cohort}), "two")

    # Assert
    assert body["_status_code"] == 500
    assert "cannot be joined into one cohort" in body["detail"]


def test_uploadDataset_zipWithoutTables_rejectedWithAcceptedFormats(test_client: TestClient, isolated_data_home):
    # Arrange
    data = _zip({"README.txt": _README, "page.htm": b"<html></html>" * 100})

    # Act
    body = _upload(test_client, "docs_only.zip", data, "docs")

    # Assert
    assert body["_status_code"] == 500
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
