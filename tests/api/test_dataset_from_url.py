# SPDX-License-Identifier: MIT
"""POST /api/datasets/from-url: a public link becomes a dataset through the upload path.

Downloads come from the local HTTPS server (``local_https_source``); data is synthetic.
"""

from __future__ import annotations

import hashlib
from typing import get_args

import pytest
from fastapi.testclient import TestClient
from fixtures.https_server import CannedResponse

from clinical_analytics.api.routes import dataset_url
from clinical_analytics.storage.url_fetch import FetchFailureKind, UrlFetchError

ROUTE = "/api/datasets/from-url"
STATE_RATES_CSV = (
    "state,year,sex,hcv_deaths,population\n"
    + "".join(
        f"S{i % 50:02d},{2010 + i % 14},{'female' if i % 2 else 'male'},{i % 97},{100000 + i}\n" for i in range(300)
    )
).encode()


def _serve_rates(source, path: str = "/api/views/hcv/rows.csv?accessType=DOWNLOAD") -> str:
    source.routes[path] = CannedResponse(
        headers={"Content-Type": "text/csv", "Content-Disposition": 'attachment; filename="hcv_deaths.csv"'},
        body=STATE_RATES_CSV,
    )
    return source.url(path)


def test_loadFromUrl_publicCsv_readyWithSourceRecorded(test_client: TestClient, isolated_data_home, local_https_source):
    # Arrange
    from clinical_analytics.api import dependencies

    url = _serve_rates(local_https_source)

    # Act
    response = test_client.post(ROUTE, json={"url": url, "dataset_name": "HCV deaths by state"})
    body = response.json()
    detail = test_client.get(f"/api/datasets/{body['upload_id']}").json()

    # Assert: the upload pipeline ran (ready, quality) and the source came back and was stored
    assert response.status_code == 200, body
    assert body["status"] == "ready"
    assert body["dataset_name"] == "HCV deaths by state"
    assert body["quality"] is not None
    source = body["source"]
    assert source["url"] == url
    assert source["sha256"] == hashlib.sha256(STATE_RATES_CSV).hexdigest()
    assert source["size_bytes"] == len(STATE_RATES_CSV)
    assert source["filename"] == "hcv_deaths.csv"
    assert source["methods_text"].startswith(f"Data downloaded from {url} on ")
    assert body["source_recorded"] is True
    stored = detail["metadata"]["source"]
    assert stored["sha256"] == source["sha256"]
    assert stored["fetched_at"] == source["fetched_at"]
    assert stored["dataset_version"]
    assert detail["tables"][0]["row_count"] == 300
    dependencies.invalidate_semantic_layer_cache(body["upload_id"])


def test_loadFromUrl_sameLinkTwiceSameName_secondRefusedAsDuplicate(
    test_client: TestClient, isolated_data_home, local_https_source
):
    # Arrange: running it twice must not silently replace the first dataset
    from clinical_analytics.api import dependencies

    url = _serve_rates(local_https_source)
    first = test_client.post(ROUTE, json={"url": url, "dataset_name": "rates"})

    # Act
    second = test_client.post(ROUTE, json={"url": url, "dataset_name": "rates"})

    # Assert: refused before downloading again, with what to do
    assert first.status_code == 200
    assert second.status_code == 409
    assert "already exists" in second.json()["detail"]
    assert "Type a different name" in second.json()["detail"]
    assert len(local_https_source.requests) == 1
    dependencies.invalidate_semantic_layer_cache(first.json()["upload_id"])


def test_loadFromUrl_fileUnderUploadMinimum_uploadRuleApplies(
    test_client: TestClient, isolated_data_home, local_https_source
):
    # Arrange: the upload's 1 KB minimum applies unchanged to downloaded files
    local_https_source.routes["/tiny.csv"] = CannedResponse(headers={"Content-Type": "text/csv"}, body=b"a,b\n1,2\n")

    # Act
    response = test_client.post(ROUTE, json={"url": local_https_source.url("/tiny.csv")})

    # Assert
    assert response.status_code >= 400
    assert "too small" in response.json()["detail"]


@pytest.mark.parametrize("url", ["https://127.0.0.1/a.csv", "https://169.254.169.254/latest/meta-data/"])
def test_loadFromUrl_privateAddress_400PlainMessage(test_client: TestClient, isolated_data_home, url: str):
    # Arrange: url

    # Act
    response = test_client.post(ROUTE, json={"url": url})

    # Assert
    assert response.status_code == 400
    assert "private network" in response.json()["detail"]


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"url": ""},
        {"url": "https://a.example.test/x.csv", "extra": 1},
        {"url": "https://a.example.test/" + "a" * 2100},
    ],
)
def test_loadFromUrl_malformedRequest_422(test_client: TestClient, payload: dict[str, object]):
    # Arrange: payload

    # Act
    response = test_client.post(ROUTE, json=payload)

    # Assert
    assert response.status_code == 422


@pytest.mark.parametrize("kind", get_args(FetchFailureKind))
def test_loadFromUrl_eachFetchFailure_mappedStatusAndUserMessage(
    test_client: TestClient, monkeypatch, kind: FetchFailureKind
):
    # Arrange
    def failing(*_args: object, **_kwargs: object) -> None:
        raise UrlFetchError(kind, f"Plain words for {kind}.")

    monkeypatch.setattr(dataset_url, "fetch_public_file", failing)

    # Act
    response = test_client.post(ROUTE, json={"url": "https://data.example.test/a.csv"})

    # Assert
    assert response.status_code == dataset_url.FETCH_FAILURE_STATUS[kind]
    assert response.json() == {"detail": f"Plain words for {kind}."}


def test_loadFromUrl_sourceNotSaved_stillReadyAndSaysSo(
    test_client: TestClient, isolated_data_home, local_https_source, monkeypatch
):
    # Arrange
    from clinical_analytics.api import dependencies
    from clinical_analytics.ui.storage.user_datasets import UserDatasetStorage

    url = _serve_rates(local_https_source)
    monkeypatch.setattr(UserDatasetStorage, "update_metadata", lambda *_a, **_k: (False, "disk full"))

    # Act
    response = test_client.post(ROUTE, json={"url": url})

    # Assert
    body = response.json()
    assert response.status_code == 200, body
    assert body["status"] == "ready"
    assert "couldn't be saved" in body["message"]
    assert body["source_recorded"] is False
    assert body["source"]["url"] == url
    dependencies.invalidate_semantic_layer_cache(body["upload_id"])


def test_loadFromUrl_defaultNameTaken_409AfterDownload(test_client: TestClient, isolated_data_home, local_https_source):
    # Arrange: two portal exports that are both called rows.csv
    from clinical_analytics.api import dependencies

    for path in ("/a/rows.csv", "/b/rows.csv"):
        local_https_source.routes[path] = CannedResponse(headers={"Content-Type": "text/csv"}, body=STATE_RATES_CSV)
    first = test_client.post(ROUTE, json={"url": local_https_source.url("/a/rows.csv")})

    # Act
    second = test_client.post(ROUTE, json={"url": local_https_source.url("/b/rows.csv"), "dataset_name": "  "})

    # Assert
    assert first.status_code == 200, first.text
    assert first.json()["dataset_name"] == "rows"
    assert second.status_code == 409
    assert "“rows”" in second.json()["detail"]
    dependencies.invalidate_semantic_layer_cache(first.json()["upload_id"])


@pytest.mark.parametrize("error", [OSError("unreadable"), ValueError("bad json")])
def test_loadFromUrl_metadataUnreadableWhenRecording_stillReadyNotRecorded(
    test_client: TestClient, isolated_data_home, local_https_source, monkeypatch, error: Exception
):
    # Arrange
    from clinical_analytics.api import dependencies

    url = _serve_rates(local_https_source)

    class Storage(dataset_url.UserDatasetStorage):
        def get_upload_metadata(self, upload_id: str):  # type: ignore[override]
            raise error

    monkeypatch.setattr(dataset_url, "UserDatasetStorage", Storage)

    # Act
    response = test_client.post(ROUTE, json={"url": url})

    # Assert
    body = response.json()
    assert response.status_code == 200, body
    assert body["source_recorded"] is False
    dependencies.invalidate_semantic_layer_cache(body["upload_id"])
