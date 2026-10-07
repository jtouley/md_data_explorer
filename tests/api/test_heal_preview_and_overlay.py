"""Heal witnesses for pandas previews and the enrichment overlay path."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
from clinical_analytics.api.main import app
from clinical_analytics.api.routes import enrichments as enrichment_routes
from clinical_analytics.api.routes.enrichments import (
    DEFAULT_OVERLAY_DIR,
    get_enrichment_service,
    get_overlay_store,
)
from clinical_analytics.api.services.query_service import AsyncQueryService
from clinical_analytics.core.overlay_store import OverlayStore
from fastapi.testclient import TestClient


def test_preview_pandas_execute_frame_is_a_table() -> None:
    service = object.__new__(AsyncQueryService)
    frame = pd.DataFrame({"mean": [174.5]})

    preview = service._get_result_preview({"success": True, "result": frame, "run_key": "k"})

    assert preview is not None
    assert preview["result"]["table"]["columns"] == ["mean"]
    assert preview["result"]["table"]["rows"] == [{"mean": 174.5}]


def test_default_overlay_dir_matches_store_contract() -> None:
    store = OverlayStore(base_dir=DEFAULT_OVERLAY_DIR)
    path = store.get_overlay_path("upl", "deadbeefdeadbeef")
    assert path == Path("data/uploads/metadata/overlays/upl/deadbeefdeadbeef")
    assert "overlays/overlays" not in path.as_posix()


def test_pending_route_passes_dataset_version_not_v1() -> None:
    service = MagicMock()
    service.get_pending_suggestions.return_value = []
    app.dependency_overrides[get_enrichment_service] = lambda: service
    app.dependency_overrides[get_overlay_store] = lambda: MagicMock()
    app.dependency_overrides[enrichment_routes.get_dataset_version] = lambda: "deadbeefdeadbeef"
    try:
        client = TestClient(app)
        response = client.get("/api/datasets/upl/enrichments/pending")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    service.get_pending_suggestions.assert_called_once_with(
        upload_id="upl",
        version="deadbeefdeadbeef",
    )
