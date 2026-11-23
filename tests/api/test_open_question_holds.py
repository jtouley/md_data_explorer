# SPDX-License-Identifier: MIT
"""One hold rule for every open question (ADR017, ``contracts.holds``), through all three readers:
the question path, governed SQL and the aggregate path. Synthetic data only.
"""

from __future__ import annotations

import io
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any, get_args

import polars as pl
import pytest
from fastapi.testclient import TestClient

from clinical_analytics.api import dependencies
from clinical_analytics.api.layer_cache import LayerCache
from clinical_analytics.contracts.config import POPULATION_ESTIMATES_FORM_VALUE
from clinical_analytics.contracts.holds import COLUMN_KINDS, COUNT_KINDS, TABLE_KINDS, open_question_holds
from clinical_analytics.contracts.semantics import (
    LAYER_DECLARED_KEY,
    DatasetSemantics,
    Grain,
    OpenQuestion,
    QuestionKind,
)
from clinical_analytics.contracts.store import write_contract
from clinical_analytics.core import nl_query_config
from clinical_analytics.core.data_home import uploads_dir
from tests.api.conftest import FakeLayer
from tests.sql.conftest import synthetic_cohort

pytestmark = [pytest.mark.slow, pytest.mark.integration]

KINDS: tuple[str, ...] = get_args(QuestionKind)
DATASET = "ds_holds_1"
VERSION = "v1"


def _asking(kind: str, column: str = "age", answered: bool = False) -> DatasetSemantics:
    """Semantics with one open question of ``kind`` (on ``column`` when it concerns a column)."""
    question = OpenQuestion(
        id=f"{kind}:{column}", kind=kind, column=column, question=f"Question of kind {kind}?", answered=answered,  # type: ignore[arg-type]
        options=("yes", "no"),
    )  # fmt: skip
    return DatasetSemantics(
        rows=10, grain=Grain(kind="patient", noun="patient", one_row_per_patient=True, rows=10, source="structure"),
        open_questions=(question,), dataset_version=VERSION,
    )  # fmt: skip


def test_holdRules_everyQuestionKind_hasOneRule() -> None:
    # Assert: no kind can be added without saying what it holds
    assert set(KINDS) == COLUMN_KINDS | TABLE_KINDS | COUNT_KINDS


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("answered", [False, True], ids=["open", "answered"])
def test_openQuestionHolds_everyKind_holdsUntilAnswered(kind: str, answered: bool) -> None:
    # Act
    holds = open_question_holds(_asking(kind, answered=answered), columns={"age"}, counts_lines=True)

    # Assert
    assert [h.kind for h in holds] == ([] if answered else [kind])


@pytest.mark.parametrize("kind", sorted(COLUMN_KINDS))
def test_openQuestionHolds_columnKindOtherColumn_answered(kind: str) -> None:
    # Act / Assert
    assert open_question_holds(_asking(kind), columns={"ward"}, counts_lines=True) == []


# --------------------------------------------------------------------------- the question path
@pytest.mark.parametrize("kind", KINDS)
def test_questionPath_everyKind_heldNoNumber(kind: str, make_semantic_layer: Callable[..., Any]) -> None:
    # Arrange
    from clinical_analytics.core.query_service import QueryService

    layer = make_semantic_layer(
        data=synthetic_cohort(), config_overrides={LAYER_DECLARED_KEY: _asking(kind).model_dump(mode="json")}
    )
    question = "how many patients?" if kind in COUNT_KINDS else "average age"

    # Act
    result = QueryService(layer).ask(question, dataset_id="ds_q", dataset_version=VERSION)

    # Assert: no answer, and no confirmation can release it
    assert result.analysis is None
    assert result.requires_confirmation is False
    assert f"Question of kind {kind}?" in (result.failure_reason or "")


# --------------------------------------------------------------------------- governed SQL
@pytest.fixture
def sql_layer(
    isolated_data_home: Path, make_contract: Callable[..., Any], monkeypatch: pytest.MonkeyPatch
) -> Iterator[Callable[[DatasetSemantics, bool], None]]:
    """Factory: store the synthetic cohort (with or without its ID) and serve a layer whose own
    semantics carry ``semantics``."""
    monkeypatch.setattr(nl_query_config, "ENABLE_SQL_QUERIES", True)
    caches: list[LayerCache[FakeLayer]] = []

    def serve(semantics: DatasetSemantics, no_id: bool) -> None:
        data_dir = uploads_dir().parent
        parquet = data_dir / "parquet" / f"{DATASET}_cohort_{VERSION}.parquet"
        parquet.parent.mkdir(parents=True, exist_ok=True)
        df = synthetic_cohort().drop("patient_id") if no_id else synthetic_cohort()
        df.write_parquet(parquet)
        contract = make_contract(
            df, upload_id=DATASET, dataset_version=VERSION, id_column=None if no_id else "patient_id"
        )
        write_contract(data_dir, DATASET, VERSION, contract)

        def build(dataset_id: str) -> FakeLayer:
            layer = FakeLayer(dataset_id, 1, loaded_version=VERSION)
            layer.config = {  # type: ignore[attr-defined]
                "init_params": {"source_path": str(parquet)},
                LAYER_DECLARED_KEY: semantics.model_dump(mode="json"),
            }
            return layer

        cache: LayerCache[FakeLayer] = LayerCache(2, build)
        caches.append(cache)
        monkeypatch.setattr(dependencies, "_layer_cache", cache)

    yield serve
    for cache in caches:
        cache.shutdown()


@pytest.mark.parametrize("kind", KINDS)
def test_governedSql_everyKind_heldNoRows(
    kind: str, test_client: TestClient, sql_layer: Callable[[DatasetSemantics, bool], None]
) -> None:
    # Arrange: a count of lines on a file without IDs for "what is a line" questions (SQL's own gate)
    counts = kind in COUNT_KINDS
    sql_layer(_asking(kind), counts)
    sql = "SELECT count(*) AS n FROM unified" if counts else "SELECT avg(age) AS mean_age FROM unified"

    # Act
    body = test_client.post("/api/queries/sql", json={"dataset_id": DATASET, "sql": sql}).json()

    # Assert: no rows; held for its question, or for SQL's one-row-per-patient confirmation
    assert body["rows"] == []
    if counts:
        assert body["needs_confirmation"] == "one_row_per_patient"
    else:
        assert body["status"] == "held"
        assert body["open_questions"] == [f"{kind}:age"]


# --------------------------------------------------------------------------- the aggregate path
@pytest.mark.parametrize("kind", sorted(COLUMN_KINDS | TABLE_KINDS))
def test_aggregatePath_everyKind_heldNoTable(
    kind: str,
    test_client: TestClient,
    isolated_data_home: Path,
    make_public_table: Callable[..., pl.DataFrame],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange: declared public estimates whose contract asks a question
    from clinical_analytics.api.routes import aggregate

    buffer = io.BytesIO()
    make_public_table("atlasplus").write_csv(buffer)
    upload_id = test_client.post(
        "/api/datasets/upload",
        files={"file": ("a.csv", buffer.getvalue(), "text/csv")},
        data={"dataset_name": "AtlasPlus", "population_estimates": POPULATION_ESTIMATES_FORM_VALUE},
    ).json()["upload_id"]
    monkeypatch.setattr(aggregate, "_version_semantics", lambda *_a, **_k: _asking(kind))

    # Act
    body = test_client.post(
        "/api/aggregate/questions",
        json={"dataset_ids": [upload_id], "question": "AtlasPlus syphilis rate in Michigan, 2023"},
    ).json()
    dependencies.invalidate_semantic_layer_cache(upload_id)

    # Assert: line-counting questions never apply to estimates, whose lines are never patients
    assert body["status"] == "held"
    assert body["table"] is None
    assert f"Question of kind {kind}?" in body["message"]
