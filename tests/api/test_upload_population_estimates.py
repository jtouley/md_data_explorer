# SPDX-License-Identifier: MIT
"""Upload with the population-estimates mark (ADR014 §7): only the exact acknowledgement value
declares it, it is bound to the version stored, and a declared file that doesn't look like
estimates says so."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from clinical_analytics.contracts.config import POPULATION_ESTIMATES_FORM_VALUE, POPULATION_ESTIMATES_NOT_DETECTED


def _estimates_csv() -> bytes:
    rows = [
        f"{place},{sex},{year},{100 + i},{90 + i},{110 + i}"
        for i, (place, sex, year) in enumerate(
            (p, s, y)
            for p in ("Colorado", "Baca County", "Otero County")
            for s in ("Both", "Male", "Female")
            for y in range(2005, 2015)
        )
    ]
    return ("\n".join(["location_name,sex,year_id,mx,lower,upper", *rows]) + "\n").encode()


def _patients_csv() -> bytes:
    rows = [f"P{i:03d},{30 + i % 40},{'F' if i % 2 else 'M'}" for i in range(200)]
    return ("\n".join(["patient_id,age,sex", *rows]) + "\n").encode()


def _upload(client: TestClient, body: bytes, mark: str | None):
    form = {"dataset_name": "estimates"} | ({"population_estimates": mark} if mark is not None else {})
    return client.post("/api/datasets/upload", files={"file": ("est.csv", body, "text/csv")}, data=form)


@pytest.mark.parametrize(
    ("mark", "declared"),
    [(POPULATION_ESTIMATES_FORM_VALUE, True), (None, False), ("", False)],
    ids=["exact_value", "absent", "empty"],
)
def test_uploadDataset_populationEstimatesMark_storedForThisVersionOnly(
    test_client: TestClient, isolated_data_home, mark, declared
):
    # Arrange
    from clinical_analytics.api import dependencies
    from clinical_analytics.contracts.declarations import FileDeclarationStore
    from clinical_analytics.core.data_home import uploads_dir
    from clinical_analytics.ui.storage.user_datasets import UserDatasetStorage, active_version_id

    # Act
    response = _upload(test_client, _estimates_csv(), mark)
    body = response.json()
    assert response.status_code == 200, body
    stored = FileDeclarationStore(uploads_dir().parent).load(body["upload_id"])
    version = active_version_id(UserDatasetStorage().get_upload_metadata(body["upload_id"]))

    # Assert
    assert body["population_estimates"] is declared
    assert (stored.population_estimates_for(version) is not None) is declared
    assert stored.population_estimates_for("another-version") is None
    assert body["population_estimates_note"] is None
    dependencies.invalidate_semantic_layer_cache(body["upload_id"])


@pytest.mark.parametrize("mark", ["true", "on", "1", "yes", "True", POPULATION_ESTIMATES_FORM_VALUE + " "])
def test_uploadDataset_genericTruthyMark_rejectedNotDeclared(test_client: TestClient, isolated_data_home, mark):
    # Act
    response = _upload(test_client, _estimates_csv(), mark)

    # Assert
    assert response.status_code == 400
    assert POPULATION_ESTIMATES_FORM_VALUE in response.json()["detail"]


def test_uploadDataset_declaredPatientFile_noteSaysRulesStillApply(test_client: TestClient, isolated_data_home):
    # Act
    response = _upload(test_client, _patients_csv(), POPULATION_ESTIMATES_FORM_VALUE)
    body = response.json()

    # Assert
    assert response.status_code == 200, body
    assert body["population_estimates_note"] == POPULATION_ESTIMATES_NOT_DETECTED


def test_uploadDataset_declaredGbdTable_valueColumnKeptRowsNumbered(
    test_client: TestClient, isolated_data_home, make_public_table
):
    # Arrange: GBD's 'val' is nearly unique per row; it must never become a patient ID
    import io

    from clinical_analytics.core.identifier_rules import ROW_NUMBER_COLUMN
    from clinical_analytics.ui.storage.user_datasets import UserDatasetStorage

    buffer = io.BytesIO()
    make_public_table("gbd", locations=("Michigan", "Ohio", "Indiana"), years=(2010, 2023)).write_csv(buffer)

    # Act
    response = _upload(test_client, buffer.getvalue(), POPULATION_ESTIMATES_FORM_VALUE)
    body = response.json()
    storage = UserDatasetStorage()
    metadata = storage.get_upload_metadata(body["upload_id"])
    stored = storage.get_upload_data(body["upload_id"]).collect()

    # Assert: rows are numbered (counted as rows); no patient_id is made from the estimate key
    assert response.status_code == 200, body
    assert {"val", "lower", "upper", ROW_NUMBER_COLUMN} <= set(stored.columns)
    assert "patient_id" not in stored.columns
    assert metadata["synthetic_id_metadata"]["patient_id"]["patient_id_source"] == "row_index"
    assert body["population_estimates"] is True
    assert body["one_row_per_patient"] is None  # a public table is never asked "Is each row one patient?"


@pytest.mark.parametrize(
    "id_column",
    ["mrn", "patient_id", "subject", "pid", "Patient ID"],
    ids=["mrn", "patient_id", "subject", "pid", "spaced"],
)
def test_uploadDataset_declaredVisitFile_patientIdKeptNotEstimates(
    test_client: TestClient, isolated_data_home, id_column
):
    # Arrange: repeated visits per patient with a rate-named value, declared by mistake
    import io

    import polars as pl

    from clinical_analytics.ui.storage.user_datasets import UserDatasetStorage

    frame = pl.DataFrame(
        {
            id_column: [f"MRN{p:04d}" for p in range(1, 41) for _ in range(3)],
            "visit_month": ["2024-01", "2024-02", "2024-03"] * 40,
            "sex": ["F" if p % 7 == 0 else "M" for p in range(1, 41) for _ in range(3)],
            "hiv_viral_load_rate": [55.0 + (i % 50) for i in range(120)],
        }
    )
    buffer = io.BytesIO()
    frame.write_csv(buffer)

    # Act
    response = _upload(test_client, buffer.getvalue(), POPULATION_ESTIMATES_FORM_VALUE)
    body = response.json()
    metadata = UserDatasetStorage().get_upload_metadata(body["upload_id"])

    # Assert: never treated as estimates; the real ID stays the patient key
    assert response.status_code == 200, body
    assert body["population_estimates_note"] == POPULATION_ESTIMATES_NOT_DETECTED
    assert metadata["synthetic_id_metadata"]["patient_id"]["patient_id_source"] != "row_index"


@pytest.mark.parametrize(
    ("column", "ident"),
    [
        ("category", lambda p: f"Subject {p:02d}"),
        ("indicator", lambda p: f"{['LOPEZ', 'SMITH', 'NGUYEN'][p % 3]}, {['ANA', 'JOHN', 'MAI'][p // 3 % 3]}{p}"),
    ],
    ids=["subject_labels", "last_first_labels"],
)
def test_uploadDataset_declaredPeopleAsLabels_patientRulesStillApply(
    test_client: TestClient, isolated_data_home, column, ident
):
    # Arrange: probe g5, people disguised as a label column beside a month
    rows = [f'"{ident(p)}",{m},{5 + p % 6}.{p % 10}' for p in range(1, 31) for m in ("2024-01", "2024-02", "2024-03")]
    body = ("\n".join([f"{column},visit_month,hba1c_pct", *rows]) + "\n").encode()

    # Act
    response = _upload(test_client, body, POPULATION_ESTIMATES_FORM_VALUE)
    payload = response.json()

    # Assert
    assert response.status_code == 200, payload
    assert payload["population_estimates_note"] == POPULATION_ESTIMATES_NOT_DETECTED
