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
Row = one clinic visit. subj is the patient identifier.

subj       Clinic subject
visit_no   Visit number (1 = first visit)
vis_date   Visit date
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
                    "subj": f"G{p:03d}",
                    "visit_no": k + 1,
                    "vis_date": f"{month} {1 + p % 27}, 2022",
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
    assert body["facts"][0] == "Each line is one clinic visit; 60 patients (subj names the patient) over 120 lines."
    assert "999 in iop_mmhg means not recorded (not measured)." in body["facts"]
    assert "visit_no numbers each patient's visits in order (1 = first)." in body["facts"]
    assert body["quality"]["facts"] == body["facts"]
    assert body["one_row_per_patient"] is None
    assert (semantics["patient_key"]["column"], semantics["patient_key"]["source"]) == ("subj", "dictionary")
    assert semantics["columns"]["drops"]["labels"]["1"] == "prostaglandin (latanoprost, nightly)"
    assert semantics["columns"]["vis_date"]["date"]["format"] == "%B %d, %Y"
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
    assert view.schema["vis_date"] == pl.Date


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

    # Assert: the patient is subj; the app's patient_id copy is restricted, never the ID
    assert catalog.id_column is not None and catalog.id_column.name == "subj"
    copy = catalog.column("patient_id")
    assert copy is not None and (copy.is_id, copy.phi) == (False, True)
