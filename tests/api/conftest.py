"""
Pytest configuration and shared fixtures for API integration tests.

Provides real FastAPI server, database, and dataset fixtures for integration testing.
"""

import socket
import threading
import time
from collections.abc import Callable, Iterator
from datetime import UTC, datetime
from typing import Any

import polars as pl
import pytest
import requests
import uvicorn
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool


@pytest.fixture
def isolated_data_home(tmp_path, monkeypatch):
    """Own data home per test, with the session tables created. Returns the data home path.

    xdist workers share the default data home, and its DuckDB file takes one writer. The plain
    test_client does not run the lifespan that creates the session tables; workers share the
    session DB, so a parallel worker may win the CREATE race and the retry then no-ops.
    """
    from sqlalchemy.exc import OperationalError

    from clinical_analytics.api.db.database import create_tables
    from clinical_analytics.core.data_home import DATA_HOME_ENV

    home = tmp_path / "home"
    monkeypatch.setenv(DATA_HOME_ENV, str(home))
    try:
        create_tables()
    except OperationalError:
        create_tables()
    return home


@pytest.fixture
def test_client():
    """Plain FastAPI test client with no service mocking.

    Shared across tests that don't need to patch any route dependency
    (e.g. CORS, health, dataset listing). Tests that need mocked services
    (e.g. query/enrichment routes) define their own specialized `test_client`
    in their own file, which takes precedence for that file.
    """
    from clinical_analytics.api.main import app

    return TestClient(app)


@pytest.fixture(scope="module")
def real_fastapi_app():
    """Real FastAPI app with lifecycle hooks (runs once per module)."""
    from clinical_analytics.api.main import app

    return app


@pytest.fixture
def real_server(real_fastapi_app):
    """Start real uvicorn server for integration tests.

    Uses dynamic port allocation to avoid conflicts.
    Waits for server to be ready before yielding.
    """
    # Find available port
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()

    # Configure server
    # log_config=None as in production (api/serve.py): uvicorn must not replace the redacting handler.
    config = uvicorn.Config(real_fastapi_app, host="127.0.0.1", port=port, log_level="error", log_config=None)
    server = uvicorn.Server(config)

    # Start server in background thread
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    # Wait for server to be ready (timeout after 5s)
    base_url = f"http://127.0.0.1:{port}"
    for _ in range(50):
        try:
            requests.get(f"{base_url}/health", timeout=0.1)
            break
        except Exception:
            time.sleep(0.1)
    else:
        raise RuntimeError("Server failed to start within 5 seconds")

    yield base_url

    # Cleanup
    server.should_exit = True
    thread.join(timeout=2)


@pytest.fixture(scope="function")
def test_db():
    """Create test database for integration tests.

    Uses SQLite in-memory for fast, isolated tests.
    Creates tables using SQLAlchemy models.
    """
    from clinical_analytics.api.models.database import Base

    # Create in-memory database
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    # Create tables
    Base.metadata.create_all(bind=engine)

    yield engine

    # Cleanup
    Base.metadata.drop_all(bind=engine)


@pytest.fixture
def uploaded_test_dataset(tmp_path):
    """Create real uploaded dataset with DuckDB storage.

    Includes metrics configuration to match production datasets.
    """
    from clinical_analytics.ui.storage.user_datasets import UserDatasetStorage

    storage = UserDatasetStorage(upload_dir=tmp_path / "uploads")

    # Create test dataset
    df = pl.DataFrame(
        {
            "patient_id": [f"P{i:03d}" for i in range(100)],
            "age": [25 + (i % 50) for i in range(100)],
            "outcome": [i % 2 for i in range(100)],
        }
    )

    upload_id = "test_dataset_001"
    tables = [{"name": "patients", "data": df}]

    storage.save_dataset(
        upload_id=upload_id,
        tables=tables,
        metadata={
            "dataset_name": "Test Dataset",
            "upload_timestamp": datetime.now(UTC).isoformat(),
        },
        config_overrides={
            "metrics": {
                "patient_count": {
                    "expression": "count()",
                    "type": "count",
                    "label": "Patient Count",
                    "description": "Total number of patients",
                }
            }
        },
        dataset_version="v1",
    )

    return upload_id, storage


@pytest.fixture
def fresh_layer_cache():
    """The API's semantic-layer cache, empty and private to the test (4 layers, real builder).

    Yields the LayerCache; pending background closes are drained and its thread stopped after.
    """
    from unittest.mock import patch

    from clinical_analytics.api import dependencies
    from clinical_analytics.api.layer_cache import LayerCache

    cache = LayerCache(4, dependencies._build_semantic_layer)
    with patch.object(dependencies, "_layer_cache", cache):
        yield cache
    cache.shutdown()


@pytest.fixture
def seed_layer_cache(fresh_layer_cache):
    """Put a layer in the private cache as if a question had built it: seed(dataset_id, layer)."""

    def _seed(dataset_id, layer):
        build = fresh_layer_cache._build
        fresh_layer_cache._build = lambda _id: layer
        try:
            fresh_layer_cache.lease(dataset_id).release()
        finally:
            fresh_layer_cache._build = build
        return layer

    return _seed


class FakeLayer:
    """A layer with the parts the layer cache and the query routes touch: a DuckDB lock,
    close(), and the upload version its rows were read with."""

    def __init__(self, dataset_id: str, number: int, loaded_version: str | None = None) -> None:
        self.dataset_id = dataset_id
        self.number = number
        self.loaded_version = loaded_version
        self._con_lock = threading.RLock()
        self.closed = False
        self.closed_on: str | None = None

    def close(self) -> None:
        with self._con_lock:
            self.closed = True
            self.closed_on = threading.current_thread().name


class Builds:
    """A layer builder for LayerCache: counts builds per dataset and hands out a new FakeLayer
    each time. ``versions[dataset_id]`` is the upload version the next build of it reads;
    ``before_return`` runs inside each build (a re-upload or a stall mid-build)."""

    def __init__(self) -> None:
        self.count: dict[str, int] = {}
        self.layers: list[FakeLayer] = []
        self.by_dataset: dict[str, list[FakeLayer]] = {}
        self.versions: dict[str, str] = {}
        self.before_return: Callable[[str], None] | None = None

    def __call__(self, dataset_id: str) -> FakeLayer:
        self.count[dataset_id] = self.count.get(dataset_id, 0) + 1
        layer = FakeLayer(dataset_id, self.count[dataset_id], self.versions.get(dataset_id))
        if self.before_return is not None:
            self.before_return(dataset_id)
        self.layers.append(layer)
        self.by_dataset.setdefault(dataset_id, []).append(layer)
        return layer


@pytest.fixture
def builds() -> Builds:
    """A fresh FakeLayer builder (see Builds)."""
    return Builds()


@pytest.fixture
def make_layer_lease() -> Iterator[Callable[..., Any]]:
    """Factory: make_layer_lease(layer) -> a real LayerLease on ``layer`` from a private cache.

    For service tests that need a lease (submit_query and execute_query require one) without the
    API's cache. Each call returns a new lease; the caches are shut down after the test.
    """
    from clinical_analytics.api.layer_cache import LayerCache

    caches: list[LayerCache[Any]] = []

    def _make(layer: Any, dataset_id: str = "ds") -> Any:
        if not hasattr(layer, "close"):
            layer.close = lambda: None
        cache: LayerCache[Any] = LayerCache(1, lambda _id: layer)
        caches.append(cache)
        return cache.lease(dataset_id)

    yield _make
    for cache in caches:
        cache.shutdown()
