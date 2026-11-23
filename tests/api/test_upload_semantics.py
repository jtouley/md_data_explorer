# SPDX-License-Identifier: MIT
"""Upload through the API: the contract's semantics decide what the file means (ADR017).

Synthetic files only (a glaucoma clinic with its dictionary, a sleep-study record list).
"""

from __future__ import annotations

import io
import zipfile
from collections.abc import Callable
from typing import Any

import polars as pl
import pytest
from fastapi.testclient import TestClient

pytestmark = [pytest.mark.slow, pytest.mark.integration, pytest.mark.usefixtures("isolated_data_home")]

MONTHS = ("January", "February", "March", "April", "May", "June")
DICTIONARY = """GLAUCOMA IOP CLINIC - DATA DICTIONARY
Row = one clinic visit. eye_owner is the patient identifier.

eye_owner  Clinic subject
exam_no    Exam number (1 = first exam)
exam_date   Visit date
iop_mmhg   Intraocular pressure, mmHg (5-60). 999 = not measured
drops      Drops prescribed:
             1 = prostaglandin (latanoprost, nightly)
             2 = beta-blocker / timolol
             3 = none
"""


def _glaucoma() -> pl.DataFrame:
    """60 patients with 1-3 visits; three visits had no pressure taken (999)."""
    rows: list[dict[str, Any]] = []
    for p in range(60):
        for k in range(1 + p % 3):
            iop = 999 if (p, k) in ((4, 1), (10, 0), (32, 2)) else 14 + (p + k) % 12
            month = MONTHS[(p + k) % 6]
            rows.append(
                {
                    "eye_owner": f"G{p:03d}",
                    "exam_no": k + 1,
                    "exam_date": f"{month} {1 + p % 27}, 2022",
                    "iop_mmhg": iop,
                    "drops": 1 + p % 3,
                }
            )
    return pl.DataFrame(rows)


def _zip(df: pl.DataFrame) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("iop_visits.csv", df.write_csv())
        zf.writestr("dictionary.txt", DICTIONARY)
    return buf.getvalue()


def _post_zip(client: TestClient, name: str) -> dict[str, Any]:
    response = client.post(
        "/api/datasets/upload",
        files={"file": ("iop.zip", _zip(_glaucoma()), "application/zip")},
        data={"dataset_name": name},
    )
    assert response.status_code == 200, response.text
    body: dict[str, Any] = response.json()
    return body


def test_uploadDataset_visitLevelWithDictionary_readyFactsInClinicalWords(test_client: TestClient) -> None:
    # Act
    body = _post_zip(test_client, "iop")
    semantics = test_client.get(f"/api/datasets/{body['upload_id']}/semantics").json()["semantics"]

    # Assert: a correct visit-level file is ready, never "needs review" for repeated patient IDs
    assert (body["status"], body["failed_checks"]) == ("ready", [])
    assert (
        body["facts"][0] == "Each line is one clinic visit; 60 patients (eye_owner names the patient) over 120 lines."
    )
    assert "999 in iop_mmhg means not recorded (not measured)." in body["facts"]
    assert "exam_no numbers each patient's lines in order (1 = first)." in body["facts"]
    assert body["quality"]["facts"] == body["facts"]
    assert body["one_row_per_patient"] is None
    assert (semantics["patient_key"]["column"], semantics["patient_key"]["source"]) == ("eye_owner", "dictionary")
    assert semantics["columns"]["drops"]["labels"]["1"] == "prostaglandin (latanoprost, nightly)"
    assert semantics["columns"]["exam_date"]["date"]["format"] == "%B %d, %Y"
    for jargon in ("null", "grain", "distinct", "sql"):
        assert jargon not in " ".join(body["facts"]).lower()


def test_uploadDataset_documentedCode_storedAsMissingForEveryReader(test_client: TestClient) -> None:
    # Arrange
    from clinical_analytics.api import dependencies

    body = _post_zip(test_client, "iop_codes")

    # Act
    with dependencies.lease_semantic_layer(body["upload_id"]) as lease:
        view = lease.layer.get_base_view().to_polars()
    dependencies.invalidate_semantic_layer_cache(body["upload_id"])

    # Assert: 999 is not a pressure; dates are dates
    assert view.get_column("iop_mmhg").null_count() == 3
    assert view.get_column("iop_mmhg").max() < 999
    assert view.schema["exam_date"] == pl.Date


def test_uploadDataset_sameFileTwice_identicalSemantics(test_client: TestClient) -> None:
    # Arrange
    first = _post_zip(test_client, "iop_a")
    second = _post_zip(test_client, "iop_b")

    # Act
    a = test_client.get(f"/api/datasets/{first['upload_id']}/semantics").json()
    b = test_client.get(f"/api/datasets/{second['upload_id']}/semantics").json()

    # Assert: same data, same version, same semantics (safe to run twice)
    assert a["dataset_version"] == b["dataset_version"]
    assert a["semantics"] == b["semantics"]


def test_uploadDataset_uniqueRecordOnly_asksFromContractQuestion(
    upload: Callable[..., Any], test_client: TestClient
) -> None:
    # Arrange: sleep studies listed by a unique record code; nothing names a patient
    df = pl.DataFrame({"record_code": [f"R{i:04d}" for i in range(200)], "ahi": [5.5 + i % 40 for i in range(200)]})

    # Act
    body = upload(df, "sleep").json()
    semantics = test_client.get(f"/api/datasets/{body['upload_id']}/semantics").json()["semantics"]

    # Assert: asked only because the contract holds the open question
    assert body["one_row_per_patient"] is not None
    assert [q["kind"] for q in semantics["open_questions"]] == ["rows_are_patients"]
    assert "You can confirm that each line is one patient." in body["facts"][0]


def test_governedSqlCatalog_contractSemantics_idIsTheFilesKey(test_client: TestClient) -> None:
    # Arrange
    from clinical_analytics.contracts.service import read_saved_contract
    from clinical_analytics.contracts.store import contract_path
    from clinical_analytics.core.data_home import uploads_dir
    from clinical_analytics.sql.catalog import SqlCatalog
    from clinical_analytics.ui.storage.user_datasets import UserDatasetStorage

    body = _post_zip(test_client, "iop_sql")
    metadata = UserDatasetStorage().get_upload_metadata(body["upload_id"]) or {}
    contract = read_saved_contract(contract_path(uploads_dir().parent, body["upload_id"], metadata["dataset_version"]))

    # Act
    catalog = SqlCatalog.from_contract(contract, uploads_dir())

    # Assert: the patient is eye_owner; the app's patient_id copy is restricted, never the ID
    assert catalog.id_column is not None and catalog.id_column.name == "eye_owner"
    copy = catalog.column("patient_id")
    assert copy is not None and (copy.is_id, copy.phi) == (False, True)


def _post_zip_of(client: TestClient, name: str, table: pl.DataFrame, dictionary: str) -> dict[str, Any]:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("table.csv", table.write_csv())
        zf.writestr("dictionary.txt", dictionary)
    response = client.post(
        "/api/datasets/upload",
        files={"file": (f"{name}.zip", buf.getvalue(), "application/zip")},
        data={"dataset_name": name},
    )
    assert response.status_code == 200, response.text
    body: dict[str, Any] = response.json()
    return body


def test_uploadDataset_floatScoreWithIntegerCode_readyCodeStoredMissing(test_client: TestClient) -> None:
    # Arrange: fibromyalgia FIQ totals with decimals; the dictionary documents 999 as an integer
    table = pl.DataFrame(
        {
            "mrn": [f"F{i % 90:03d}" for i in range(180)],
            "fiq_total": [12.5 + i % 60 if i % 30 else 999.0 for i in range(180)],
        }
    )

    # Act
    body = _post_zip_of(test_client, "fiq", table, "fiq_total  FIQ total (0-100). 999 = not done\n")

    # Assert: no 500; the code is a missing value for every reader
    assert body["status"] == "ready"
    assert "999 in fiq_total means not recorded (not done)." in body["facts"]


def test_semanticsEndpoint_versionWithoutSemantics_sameReadOnlyPass(test_client: TestClient) -> None:
    # Arrange: strip the semantics from the stored contract, as for a version stored before ADR017
    import yaml

    from clinical_analytics.contracts.store import contract_path
    from clinical_analytics.core.data_home import uploads_dir
    from clinical_analytics.ui.storage.user_datasets import UserDatasetStorage

    body = _post_zip(test_client, "iop_legacy")
    version = (UserDatasetStorage().get_upload_metadata(body["upload_id"]) or {})["dataset_version"]
    path = contract_path(uploads_dir().parent, body["upload_id"], version)
    document = yaml.safe_load(path.read_text())
    document["customProperties"] = [p for p in document["customProperties"] if p["property"] != "semantics"]
    path.write_text(yaml.safe_dump(document))

    # Act
    response = test_client.get(f"/api/datasets/{body['upload_id']}/semantics")

    # Assert: answered from the stored data (no dictionary any more), never a 404; nothing written
    assert response.status_code == 200, response.text
    assert response.json()["semantics"]["patient_key"]["column"] == "eye_owner"
    assert "semantics" not in path.read_text()


def test_missingCodeAnswer_yes_answersReadItAsMissing(test_client: TestClient) -> None:
    # Arrange: chest-pain unit troponin with an undocumented 999 the upload asks about
    from clinical_analytics.api import dependencies

    table = pl.DataFrame(
        {
            "mrn": [f"C{i:03d}" for i in range(200)],
            "trop_ng_l": [999.0 if i % 20 == 0 else 4.0 + i % 30 for i in range(200)],
        }
    )
    body = _post_zip_of(test_client, "trop", table, "mrn  Medical record number (patient identifier)\n")
    assert "Question for you: Does 999 in trop_ng_l mean not recorded?" in body["facts"]

    # Act
    answer = test_client.post(
        f"/api/datasets/{body['upload_id']}/missing-codes",
        json={"column": "trop_ng_l", "value": 999.0, "not_recorded": True},
    )
    with dependencies.lease_semantic_layer(body["upload_id"]) as lease:
        view = lease.layer.get_base_view().to_polars()
    dependencies.invalidate_semantic_layer_cache(body["upload_id"])

    # Assert
    assert answer.status_code == 200, answer.text
    assert "999 in trop_ng_l means not recorded." in answer.json()["facts"]
    assert view.get_column("trop_ng_l").null_count() == 10


def test_missingCodeAnswer_valueNotAsked_409NothingStored(test_client: TestClient) -> None:
    # Arrange
    body = _post_zip(test_client, "iop_not_asked")

    # Act
    response = test_client.post(
        f"/api/datasets/{body['upload_id']}/missing-codes",
        json={"column": "iop_mmhg", "value": 14, "not_recorded": True},
    )

    # Assert
    assert response.status_code == 409


def test_upload_codebookCodedIntegerItem_typedCategoricalNationalQuestionDeclined(test_client: TestClient) -> None:
    # Arrange: a synthetic wellbeing survey with its SAS-style codebook
    from clinical_analytics.api import dependencies
    from clinical_analytics.core.query_service import QueryService
    from tests.contracts.test_semantics_codebooks import CODEBOOK, _survey

    body = _post_zip_of(test_client, "wellbeing", _survey(), CODEBOOK)

    # Act
    with dependencies.lease_semantic_layer(body["upload_id"]) as lease:
        types = lease.layer.config["variable_types"]
        national = QueryService(lease.layer).ask(
            "what share of adults sleep well nationally", dataset_id=body["upload_id"], dataset_version="v1"
        )
    dependencies.invalidate_semantic_layer_cache(body["upload_id"])

    # Assert
    assert types["SLEEPOK"]["type"] == "categorical"
    assert body["unit_note"] == ""
    assert national.analysis is None
    assert "not weighted population estimates" in (national.failure_reason or "")
