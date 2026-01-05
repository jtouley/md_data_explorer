"""
Pytest configuration and shared fixtures for API integration tests.

Provides real FastAPI server, database, and dataset fixtures for integration testing.
"""

import socket
import threading
import time
from datetime import UTC, datetime

import polars as pl
import pytest
import requests
import uvicorn
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool


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
    config = uvicorn.Config(real_fastapi_app, host="127.0.0.1", port=port, log_level="error")
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
