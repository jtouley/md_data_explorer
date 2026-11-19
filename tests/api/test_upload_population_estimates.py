"""Upload with the population-estimates mark (ADR014): stored per dataset, never inferred."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


def _estimates_csv() -> bytes:
    rows = [
        f"{place},{sex},{year},{100 + i},{90 + i},{110 + i}"
        for i, (place, sex, year) in enumerate(
            (p, s, y)
            for p in ("State", "A County", "B County")
            for s in ("Both", "Male", "Female")
            for y in range(2005, 2015)
        )
    ]
    return ("\n".join(["location_name,sex,year_id,mx,lower,upper", *rows]) + "\n").encode()


@pytest.mark.parametrize(("ticked", "declared"), [("true", True), (None, False)], ids=["ticked", "not_ticked"])
def test_uploadDataset_populationEstimatesMark_storedOnlyWhenTicked(
    test_client: TestClient, isolated_data_home, ticked, declared
):
    # Arrange
    from clinical_analytics.api import dependencies
    from clinical_analytics.contracts.declarations import FileDeclarationStore
    from clinical_analytics.core.data_home import uploads_dir

    form = {"dataset_name": "estimates"} | ({"population_estimates": ticked} if ticked else {})

    # Act
    response = test_client.post(
        "/api/datasets/upload", files={"file": ("est.csv", _estimates_csv(), "text/csv")}, data=form
    )
    body = response.json()
    assert response.status_code == 200, body
    stored = FileDeclarationStore(uploads_dir().parent).load(body["upload_id"])

    # Assert
    assert body["population_estimates"] is declared
    assert (stored.population_estimates is not None) is declared
    dependencies.invalidate_semantic_layer_cache(body["upload_id"])
