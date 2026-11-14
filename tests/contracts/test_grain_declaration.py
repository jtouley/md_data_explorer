"""Grain declarations: a visit grain names its key; a patient grain is one patient per row."""

from __future__ import annotations

from typing import Any

import pydantic
import pytest

from clinical_analytics.contracts.declarations import Declarations, GrainDeclaration
from clinical_analytics.core.answer_checks import Grain, declared_visits


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        ({"grain": "visit", "key": None, "rows": 20, "patients": 10}, "names the patient key"),
        ({"grain": "visit", "key": "", "rows": 20, "patients": 10}, "names the patient key"),
        ({"grain": "patient", "key": None, "rows": 20, "patients": 19}, "rows == patients"),
        ({"grain": "row", "key": "patient_id", "rows": 1, "patients": 1}, "grain"),
        ({"grain": "patient", "key": None, "rows": -1, "patients": -1}, "greater than or equal"),
    ],
)
def test_grainDeclaration_inconsistentFields_rejected(fields: dict[str, Any], message: str) -> None:
    # Arrange: the fields (parametrized)

    # Act / Assert
    with pytest.raises(pydantic.ValidationError, match=message):
        GrainDeclaration(**fields)


@pytest.mark.parametrize("key", [None, "patient_id"])
def test_grainDeclaration_patientGrain_roundTripsAndIsNeverReadAsVisits(key: str | None) -> None:
    # Arrange
    grain = GrainDeclaration(grain="patient", key=key, rows=300, patients=300, dataset_version="v1")
    stored = Declarations().with_grain(grain)

    # Act
    loaded = Declarations.model_validate_json(stored.model_dump_json())

    # Assert: kept per version, and the question box's visit check does not take it for visits
    assert loaded.grain_for("v1") == grain
    assert loaded.grain_for("v2") is None
    assert not declared_visits(Grain(key="patient_id", rows=300, patients=300), loaded, "v1")


def test_grainDeclaration_legacyVisitFile_stillReads() -> None:
    # Arrange: a file written before the grain field had a second value
    text = '{"grains": [{"key": "patient_id", "rows": 20, "patients": 10, "dataset_version": "v1"}]}'

    # Act
    loaded = Declarations.model_validate_json(text)

    # Assert
    assert loaded.grain_for("v1") == GrainDeclaration(key="patient_id", rows=20, patients=10, dataset_version="v1")
