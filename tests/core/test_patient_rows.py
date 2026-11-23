# SPDX-License-Identifier: MIT
""" "Is each row one patient?": when it is asked, and what each answer stores (synthetic only)."""

from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import polars as pl
import pytest

from clinical_analytics.contracts.declarations import (
    GrainDeclaration,
    MemoryDeclarationStore,
    confirmation_token,
)
from clinical_analytics.contracts.inference import id_named
from clinical_analytics.core.answer_checks import Grain, declared_patient_rows
from clinical_analytics.core.identifier_rules import PATIENT_ROWS_ASKABLE_CONFIG_FLAG, rows_may_be_patients
from clinical_analytics.core.patient_rows import (
    PatientRowsNotAskedError,
    PatientRowsStaleError,
    confirmed_patient_rows,
    pending_token,
    question_token,
    record_answer,
)

N = 120
DATASET = "ds_rows"


def _metadata(version: str = "v1", rows: int = N) -> dict[str, Any]:
    """Upload metadata of a version: the version and its row count (what the token binds)."""
    return {"dataset_version": version, "row_count": rows}


def _frame(**columns: list[Any]) -> pl.DataFrame:
    base: dict[str, list[Any]] = {"sex": ["female" if i % 2 else "male" for i in range(N)]}
    return pl.DataFrame({**columns, **base})


@pytest.fixture
def asked(make_semantics: Callable[..., Any]) -> Any:
    """Semantics of a file whose rows are told apart by an unnamed unique column: asked."""
    return make_semantics(_frame(record_code=[f"R{i:04d}" for i in range(N)]))


@pytest.fixture
def not_asked(make_semantics: Callable[..., Any]) -> Any:
    """Semantics of visit rows whose person_id repeats: never asked."""
    return make_semantics(_frame(person_id=[f"P{i % 40:03d}" for i in range(N)]))


@pytest.mark.parametrize(
    ("df", "asked"),
    [
        pytest.param(_frame(record_code=[f"R{i:04d}" for i in range(N)]), True, id="unique_unnamed_column"),
        pytest.param(_frame(), True, id="no_id_at_all_row_number"),
        pytest.param(
            _frame(visit_code=[f"V{i:04d}" for i in range(N)], person_id=[f"P{i % 40:03d}" for i in range(N)]),
            False,
            id="visit_rows_person_id_repeats",
        ),
        pytest.param(
            _frame(caseId=[f"C{i % 60:03d}" for i in range(N)]),
            False,
            id="camel_case_id_repeats_row_number",
        ),
        pytest.param(
            _frame(subject_id=[f"S{i % 60:03d}" for i in range(N)]),
            False,
            id="patient_key_name_repeats",
        ),
        pytest.param(_frame(patient_id=[f"P{i:04d}" for i in range(N)]), False, id="patient_id_names_patients"),
        pytest.param(
            _frame(record_code=[f"R{i:04d}" for i in range(N)], person_id=[float(i % 50) for i in range(N)]),
            False,
            id="spss_float_person_id_repeats",
        ),
        pytest.param(
            _frame(record_code=[f"R{i:04d}" for i in range(N)], **{"Hosp No": [f"H{i % 70:03d}" for i in range(N)]}),
            False,
            id="hosp_no_repeats",
        ),
        pytest.param(
            _frame(record_code=[f"R{i:04d}" for i in range(N)], site_id=[f"S{i % 5}" for i in range(N)]),
            False,
            id="site_id_repeats_not_asked_by_design",
        ),
        pytest.param(
            _frame(
                record_code=[f"R{i:04d}" for i in range(N)],
                study_id=["TRIAL-1"] * N,
                person_id=[None if i % 3 else f"P{i:03d}" for i in range(N)],
            ),
            True,
            id="one_value_study_code_and_unique_sparse_id",
        ),
    ],
)
def test_semantics_fileShape_askedOnlyWhenNoIdRepeats(
    make_semantics: Callable[..., Any], df: pl.DataFrame, asked: bool
) -> None:
    # Arrange
    metadata: dict[str, Any] = {"dataset_version": "v1", "row_count": df.height}

    # Act
    semantics = make_semantics(df)

    # Assert
    assert rows_may_be_patients(semantics) is asked
    assert (question_token(metadata, semantics) is not None) is asked


@pytest.mark.parametrize(
    "df",
    [
        pytest.param(_frame(a=[f"A{i % 12}" for i in range(N)], b=[i // 12 for i in range(N)]), id="composite"),
        pytest.param(_frame(person_id=[f"P{i % 40:03d}" for i in range(N)]), id="an_id_repeats"),
        pytest.param(_frame(mrn=[f"M{i:04d}" for i in range(N)]), id="named_patient_key"),
        pytest.param(_frame(patient_id=[f"P{i:04d}" for i in range(N)]), id="existing_patient_id"),
    ],
)
def test_rowsMayBePatients_notAskedShapes_false(make_semantics: Callable[..., Any], df: pl.DataFrame) -> None:
    # Arrange
    semantics = make_semantics(df)

    # Act
    result = rows_may_be_patients(semantics)

    # Assert
    assert result is False


@pytest.mark.parametrize(
    ("rows", "version"),
    [
        pytest.param(0, "v1", id="no_rows"),
        pytest.param(True, "v1", id="bool_rows"),
        pytest.param(N, "", id="no_version"),
    ],
)
def test_questionToken_badCountsOrVersion_notAsked(asked: Any, rows: Any, version: str) -> None:
    # Arrange
    metadata = _metadata(version=version, rows=rows)

    # Act
    token = question_token(metadata, asked)

    # Assert
    assert token is None


@pytest.mark.parametrize(
    ("column", "expected"),
    [
        ("person_id", True),
        ("caseId", True),
        ("Encounter ID", True),
        ("subject", False),  # a broad convention is "subject_id"; "subject" alone is asked
        ("covid19_admission_hospital", False),
        ("paid", False),
        ("record_code", False),
        ("Hosp No", True),
        ("Record Number", True),
        ("", False),
    ],
)
def test_idNamed_columnNames_tokenMatchOnly(column: str, expected: bool) -> None:
    # Arrange / Act
    result = id_named(column)

    # Assert
    assert result is expected


def test_recordAnswer_pendingThenYes_storesPatientGrainAndStopsAsking(asked: Any) -> None:
    # Arrange
    metadata, store = _metadata(), MemoryDeclarationStore()
    token = pending_token(metadata, asked, store.load(DATASET))

    # Act
    counts = record_answer(DATASET, metadata, asked, store, confirmation=str(token), one_row_per_patient=True)

    # Assert: the same declaration governed SQL stores, under the same token
    assert token == confirmation_token("v1", None, N)
    assert counts == "patients"
    assert store.load(DATASET).grain_for("v1") == GrainDeclaration(
        grain="patient", key=None, rows=N, patients=N, dataset_version="v1"
    )
    assert pending_token(metadata, asked, store.load(DATASET)) is None
    assert confirmed_patient_rows(metadata, asked, store.load(DATASET))


def test_recordAnswer_yesTwice_sameDeclaration(asked: Any) -> None:
    # Arrange
    metadata, store = _metadata(), MemoryDeclarationStore()
    token = str(question_token(metadata, asked))
    record_answer(DATASET, metadata, asked, store, confirmation=token, one_row_per_patient=True)
    first = store.load(DATASET)

    # Act
    record_answer(DATASET, metadata, asked, store, confirmation=token, one_row_per_patient=True)

    # Assert
    assert store.load(DATASET) == first


def test_recordAnswer_noAfterYes_rowsAgainVisitGrainOfOtherVersionKept(asked: Any) -> None:
    # Arrange
    metadata, store = _metadata(), MemoryDeclarationStore()
    token = str(question_token(metadata, asked))
    other = GrainDeclaration(key="person_id", rows=10, patients=5, dataset_version="v0")
    store.update(DATASET, lambda d: d.with_grain(other))
    record_answer(DATASET, metadata, asked, store, confirmation=token, one_row_per_patient=True)

    # Act
    counts = record_answer(DATASET, metadata, asked, store, confirmation=token, one_row_per_patient=False)

    # Assert
    assert counts == "rows"
    assert store.load(DATASET).grain_for("v1") is None
    assert store.load(DATASET).grain_for("v0") == other
    assert pending_token(metadata, asked, store.load(DATASET)) == token


@pytest.mark.parametrize(
    "token",
    [
        pytest.param(confirmation_token("v0", None, N), id="earlier_version"),
        pytest.param(confirmation_token("v1", None, N + 1), id="other_row_count"),
        pytest.param(confirmation_token("v1", "record_code", N), id="other_id_column"),
        pytest.param("0" * 32, id="forged"),
        pytest.param("", id="empty"),
        pytest.param("é" * 32, id="non_ascii"),
    ],
)
def test_recordAnswer_staleToken_refusedNothingStored(asked: Any, token: str) -> None:
    # Arrange
    metadata, store = _metadata(), MemoryDeclarationStore()

    # Act / Assert
    with pytest.raises(PatientRowsStaleError):
        record_answer(DATASET, metadata, asked, store, confirmation=token, one_row_per_patient=True)
    assert store.load(DATASET).grains == ()


def test_recordAnswer_overwriteNewVersion_asksAgainOldTokenRefused(asked: Any) -> None:
    # Arrange: "yes" on v1, then the file is replaced (v2)
    v1, v2, store = _metadata("v1"), _metadata("v2", rows=N + 5), MemoryDeclarationStore()
    old = str(question_token(v1, asked))
    record_answer(DATASET, v1, asked, store, confirmation=old, one_row_per_patient=True)

    # Act
    pending = pending_token(v2, asked, store.load(DATASET))

    # Assert: asked again with a new token; v1's answer never reaches v2's rows
    assert pending is not None and pending != old
    assert not confirmed_patient_rows(v2, asked, store.load(DATASET))
    with pytest.raises(PatientRowsStaleError):
        record_answer(DATASET, v2, asked, store, confirmation=old, one_row_per_patient=True)
    assert store.load(DATASET).grain_for("v2") is None


def test_recordAnswer_notAsked_refusedNothingStored(not_asked: Any) -> None:
    # Arrange: IDs repeat, so the question is never offered, whatever token is sent
    metadata, store = _metadata(), MemoryDeclarationStore()

    # Act / Assert
    with pytest.raises(PatientRowsNotAskedError):
        record_answer(
            DATASET,
            metadata,
            not_asked,
            store,
            confirmation=confirmation_token("v1", None, N),
            one_row_per_patient=True,
        )
    assert store.load(DATASET).grains == ()


@pytest.mark.parametrize(
    ("askable", "grain", "stored", "expected"),
    [
        pytest.param(True, Grain(key=None, rows=N, patients=None), (None, N, "v1"), True, id="confirmed"),
        pytest.param(
            False, Grain(key=None, rows=N, patients=None), (None, N, "v1"), False, id="ids_repeat_not_askable"
        ),
        pytest.param(None, Grain(key=None, rows=N, patients=None), (None, N, "v1"), False, id="flag_absent"),
        pytest.param(True, Grain(key=None, rows=N, patients=None), (None, N - 1, "v1"), False, id="other_row_count"),
        pytest.param(True, Grain(key=None, rows=N, patients=None), (None, N, "v0"), False, id="other_version"),
        pytest.param(
            True,
            Grain(key="patient_id", rows=N, patients=N - 2, missing_keys=2),
            ("patient_id", N, "v1"),
            False,
            id="keyed_rows",
        ),
        pytest.param(True, Grain(key=None, rows=N, patients=None), None, False, id="nothing_stored"),
    ],
)
def test_declaredPatientRows_storedGrain_onlyAskableVersionAndCount(
    askable: bool | None, grain: Grain, stored: tuple[str | None, int, str] | None, expected: bool
) -> None:
    # Arrange
    layer = SimpleNamespace(config={} if askable is None else {PATIENT_ROWS_ASKABLE_CONFIG_FLAG: askable})
    store = MemoryDeclarationStore()
    if stored is not None:
        key, rows, version = stored
        declared = GrainDeclaration(grain="patient", key=key, rows=rows, patients=rows, dataset_version=version)
        store.update(DATASET, lambda d: d.with_grain(declared))

    # Act
    result = declared_patient_rows(layer, grain, store.load(DATASET), "v1")

    # Assert
    assert result is expected


@pytest.mark.parametrize(("source_format", "expected"), [("cdc_wonder", False), ("", True)], ids=["wonder", "other"])
def test_rowsMayBePatients_publishedExport_neverAsked(
    make_semantics: Callable[..., Any], source_format: str, expected: bool
) -> None:
    # Arrange: rows told apart by one unique column, no ID-like column repeating
    df = _frame(**{"State Code": [f"{i:02d}" for i in range(N)]})

    # Act
    result = rows_may_be_patients(make_semantics(df, source_format=source_format))

    # Assert
    assert result is expected
