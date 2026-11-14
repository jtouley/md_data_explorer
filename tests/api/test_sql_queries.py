"""POST /api/queries/sql: behind its flag, on a leased layer, against the saved contract."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from clinical_analytics.api import dependencies
from clinical_analytics.api.layer_cache import LayerCache
from clinical_analytics.contracts.declarations import FileDeclarationStore
from clinical_analytics.contracts.store import contract_path, contracts_root, write_contract
from clinical_analytics.core import nl_query_config
from clinical_analytics.core.data_home import uploads_dir
from clinical_analytics.core.result_presentation import SECONDARY_SUPPRESSED
from clinical_analytics.sql import config as sql_config
from clinical_analytics.sql import sandbox as sql_sandbox
from clinical_analytics.sql.service import DECLARATIONS_UNREADABLE, NO_CONTRACT, NOT_STORED
from tests.api.conftest import FakeLayer
from tests.sql.conftest import synthetic_cohort

DATASET = "ds_sql_1"
VERSION = "v1"


@pytest.fixture
def stored_dataset(isolated_data_home: Path, make_contract: Callable[..., Any], request: pytest.FixtureRequest) -> Path:
    """The synthetic cohort as stored Parquet with its contract saved, as an upload leaves it.
    Indirect parameter ``"no_id"``: the same cohort with no patient ID column."""
    data_dir = uploads_dir().parent
    parquet = data_dir / "parquet" / f"{DATASET}_cohort_{VERSION}.parquet"
    parquet.parent.mkdir(parents=True, exist_ok=True)
    no_id = getattr(request, "param", None) == "no_id"
    df = synthetic_cohort().drop("patient_id") if no_id else synthetic_cohort()
    df.write_parquet(parquet)
    contract = make_contract(df, upload_id=DATASET, dataset_version=VERSION, id_column=None if no_id else "patient_id")
    write_contract(data_dir, DATASET, VERSION, contract)
    return parquet


@pytest.fixture
def layer_cache(monkeypatch: pytest.MonkeyPatch, stored_dataset: Path) -> Iterator[LayerCache[FakeLayer]]:
    """The API's layer cache, building a layer that reads the stored Parquet."""

    def build(dataset_id: str) -> FakeLayer:
        layer = FakeLayer(dataset_id, 1, loaded_version=VERSION)
        layer.config = {"init_params": {"source_path": str(stored_dataset)}}  # type: ignore[attr-defined]
        return layer

    cache: LayerCache[FakeLayer] = LayerCache(2, build)
    monkeypatch.setattr(dependencies, "_layer_cache", cache)
    yield cache
    cache.shutdown()


@pytest.fixture
def sql_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(nl_query_config, "ENABLE_SQL_QUERIES", True)


def _post(client: TestClient, sql: str, dataset_id: str = DATASET, **extra: Any) -> Any:
    return client.post("/api/queries/sql", json={"dataset_id": dataset_id, "sql": sql, **extra})


def test_sqlRoute_flagOff_notFound(test_client: TestClient, layer_cache: LayerCache[FakeLayer]) -> None:
    # Arrange: the default configuration
    assert nl_query_config.ENABLE_SQL_QUERIES is False

    # Act
    response = _post(test_client, "SELECT count(*) AS n FROM unified")

    # Assert
    assert response.status_code == 404
    assert layer_cache.cached_ids() == []  # nothing was loaded


def test_sqlRoute_validQuery_answeredWithSqlShownAndLeaseReleased(
    test_client: TestClient, layer_cache: LayerCache[FakeLayer], sql_enabled: None
) -> None:
    # Arrange
    sql = "SELECT ward, count(*) AS n FROM unified GROUP BY ward"

    # Act
    response = _post(test_client, sql)

    # Assert
    body = response.json()
    assert response.status_code == 200
    assert body["status"] == "answered"
    assert body["sql"].startswith("SELECT")
    assert body["columns"] == ["ward", "n"]
    assert "ICU" not in {row["ward"] for row in body["rows"]}
    assert any(row["n"] == SECONDARY_SUPPRESSED for row in body["rows"])
    assert layer_cache._entries[DATASET].leases == 0


@pytest.mark.parametrize(
    "sql", ["SELECT * FROM unified", "COPY unified TO 'x.csv'", "SELECT count(*) FROM read_csv('/etc/passwd')"]
)
def test_sqlRoute_refusedQuery_reasonAndNoRows(
    test_client: TestClient, layer_cache: LayerCache[FakeLayer], sql_enabled: None, sql: str
) -> None:
    # Arrange: the query (parametrized)

    # Act
    body = _post(test_client, sql).json()

    # Assert
    assert body["status"] == "refused"
    assert body["reason"]
    assert body["rows"] == []
    assert layer_cache._entries[DATASET].leases == 0


def test_sqlRoute_invalidDatasetId_badRequest(
    test_client: TestClient, layer_cache: LayerCache[FakeLayer], sql_enabled: None
) -> None:
    # Arrange: a path-like id

    # Act
    response = _post(test_client, "SELECT count(*) FROM unified", dataset_id="../etc")

    # Assert
    assert response.status_code == 400


@pytest.mark.parametrize(
    ("change", "reason"),
    [("no_contract", NO_CONTRACT), ("in_memory", NOT_STORED), ("damaged", "can't be read")],
)
def test_sqlRoute_datasetWithoutCheckableData_refused(
    test_client: TestClient,
    layer_cache: LayerCache[FakeLayer],
    sql_enabled: None,
    monkeypatch: pytest.MonkeyPatch,
    change: str,
    reason: str,
) -> None:
    # Arrange
    path = contract_path(uploads_dir().parent, DATASET, VERSION)
    if change == "no_contract":
        path.unlink()
    elif change == "damaged":
        path.write_text("not: [a contract")
    else:
        monkeypatch.setattr(layer_cache, "_build", lambda dataset_id: FakeLayer(dataset_id, 1, loaded_version=VERSION))

    # Act
    body = _post(test_client, "SELECT count(*) AS n FROM unified").json()

    # Assert
    assert body["status"] == "refused"
    assert reason in body["reason"]


@pytest.mark.parametrize("stored_dataset", ["no_id"], indirect=True)
def test_sqlRoute_noIdColumn_refusedUntilConfirmedThenRemembered(
    test_client: TestClient, layer_cache: LayerCache[FakeLayer], sql_enabled: None
) -> None:
    # Arrange
    sql = "SELECT ward, count(*) AS n FROM unified GROUP BY ward"
    store = FileDeclarationStore(uploads_dir().parent)

    # Act
    refused = _post(test_client, sql).json()
    confirmed = _post(test_client, sql, confirm_one_row_per_patient=True).json()
    remembered = _post(test_client, sql).json()

    # Assert: refused with what to confirm; the confirmation is the version's patient grain
    assert refused["status"] == "refused"
    assert refused["needs_confirmation"] == "one_row_per_patient"
    assert confirmed["status"] == "answered"
    assert "ICU" not in {row["ward"] for row in confirmed["rows"]}
    assert remembered["status"] == "answered"
    grain = store.load(DATASET).grain_for(VERSION)
    assert grain is not None
    assert (grain.grain, grain.key, grain.rows, grain.patients) == ("patient", None, 300, 300)
    assert store.load(DATASET).grain_for("v2") is None  # a re-upload asks again


@pytest.mark.parametrize("stored_dataset", ["no_id"], indirect=True)
def test_sqlRoute_confirmationWithRefusedQuery_notStored(
    test_client: TestClient, layer_cache: LayerCache[FakeLayer], sql_enabled: None
) -> None:
    # Arrange: an invalid query, so nothing was asked of the data

    # Act
    body = _post(test_client, "SELECT * FROM unified", confirm_one_row_per_patient=True).json()

    # Assert
    assert body["status"] == "refused"
    assert body["needs_confirmation"] is None
    assert FileDeclarationStore(uploads_dir().parent).load(DATASET).grains == ()


def test_sqlRoute_unreadableDeclarations_refusedNot500(
    test_client: TestClient, layer_cache: LayerCache[FakeLayer], sql_enabled: None
) -> None:
    # Arrange
    path = contracts_root(uploads_dir().parent) / DATASET / "declarations.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not json")

    # Act
    response = _post(test_client, "SELECT count(*) AS n FROM unified")

    # Assert
    assert response.status_code == 200
    assert response.json()["reason"] == DECLARATIONS_UNREADABLE


def test_sqlRoute_sandboxBusy_serviceUnavailable(
    test_client: TestClient, layer_cache: LayerCache[FakeLayer], sql_enabled: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange: every slot taken
    monkeypatch.setattr(sql_sandbox, "_SLOTS", sql_sandbox.threading.BoundedSemaphore(1))
    monkeypatch.setattr(sql_config, "TIMEOUT_SECONDS", 0.05)
    sql_sandbox._SLOTS.acquire()

    # Act
    response = _post(test_client, "SELECT count(*) AS n FROM unified")

    # Assert
    assert response.status_code == 503
    assert response.headers["Retry-After"]
    assert layer_cache._entries[DATASET].leases == 0


@pytest.mark.parametrize(
    ("length", "status_code"),
    [(sql_config.MAX_SQL_CHARS + 1, 200), (sql_config.MAX_SQL_REQUEST_CHARS + 1, 422)],
)
def test_sqlRoute_longQuery_plainRefusalUntilTheRequestCap(
    test_client: TestClient, layer_cache: LayerCache[FakeLayer], sql_enabled: None, length: int, status_code: int
) -> None:
    # Arrange
    head = "SELECT count(*) AS n FROM unified -- "
    sql = head + "x" * (length - len(head))

    # Act
    response = _post(test_client, sql)

    # Assert
    assert response.status_code == status_code
    if status_code == 200:
        assert f"longer than {sql_config.MAX_SQL_CHARS}" in response.json()["reason"]
