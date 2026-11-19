"""Fixtures for the acceptance-corpus fetcher and scorer (synthetic bytes only, no real data)."""

from __future__ import annotations

import hashlib
import importlib.util
import io
import zipfile
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from clinical_analytics.eval.corpus import run as run_module
from clinical_analytics.eval.corpus.fetch import CorpusDataset, html_section_text

PAGE_URL = "https://example.org/page.html"
# A live page: the "description" section is the dictionary, the "views" section changes daily.
SYNTHETIC_PAGE = b'<h2 id="description">Data</h2><p>sex: biological sex</p><h2 id="views">Views: 12</h2>'


@pytest.fixture
def make_corpus_dataset() -> Callable[..., tuple[CorpusDataset, dict[str, bytes]]]:
    """Factory: a one-table corpus entry over synthetic bytes, and the URL -> bytes it serves."""

    def make(table: bytes = b"id,sex\n1,female\n2,male\n", page: bytes = SYNTHETIC_PAGE, **pkg: Any):
        urls = {"https://example.org/table.csv": table, PAGE_URL: page}
        section = html_section_text(SYNTHETIC_PAGE, "description", PAGE_URL).encode()
        dataset = CorpusDataset.model_validate(
            {
                "name": "synthetic",
                "files": [
                    {
                        "path": "table.csv",
                        "url": "https://example.org/table.csv",
                        "kind": "table",
                        "sha256": hashlib.sha256(table).hexdigest(),
                    },
                    {
                        "path": "dictionary.txt",
                        "url": PAGE_URL,
                        "kind": "dictionary",
                        "extract_section": "description",
                        "sha256": hashlib.sha256(section).hexdigest(),
                    },
                ],
                "upload_package": {
                    "zip": "synthetic.zip",
                    "contents": [
                        {"name": "table.csv", "from": "table.csv", "make": "copy"},
                        {"name": "data_dictionary.txt", "from": "dictionary.txt", "make": "copy"},
                    ],
                    **pkg,
                },
            }
        )
        return dataset, urls

    return make


def _zip_bytes(members: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, body in members.items():
            archive.writestr(name, body)
    return buffer.getvalue()


@pytest.fixture
def make_archive_dataset() -> Callable[..., tuple[CorpusDataset, dict[str, bytes], bytes]]:
    """Factory: a corpus entry whose table ships inside a ZIP inside a ZIP (as UCI ships some
    sets), uploaded on its own; the URL -> bytes it serves; and the table's bytes."""

    def make(table: bytes = b"id,sex\n1,female\n", member_sha256: str | None = None, **pkg: Any):
        archive = _zip_bytes({"inner.zip": _zip_bytes({"table.csv": table, "other.csv": b"x\n"})})
        url = "https://example.org/archive.zip"
        dataset = CorpusDataset.model_validate(
            {
                "name": "synthetic_archive",
                "files": [
                    {
                        "path": "archive.zip",
                        "url": url,
                        "kind": "archive",
                        "sha256": hashlib.sha256(archive).hexdigest(),
                    }
                ],
                "upload_package": {
                    "upload": "file",
                    "contents": [
                        {
                            "name": "table.csv",
                            "from": "archive.zip",
                            "make": "extract",
                            "member": "inner.zip!table.csv",
                            "sha256": member_sha256 or hashlib.sha256(table).hexdigest(),
                        }
                    ],
                    **pkg,
                },
                "oracle_input": "table.csv",
            }
        )
        return dataset, {url: archive}, table

    return make


@pytest.fixture
def make_downloader() -> Callable[[dict[str, bytes]], tuple[Callable[[str], bytes], list[str]]]:
    """Factory: a downloader serving fixed bytes, and the list of URLs it was asked for."""

    def make(urls: dict[str, bytes]) -> tuple[Callable[[str], bytes], list[str]]:
        calls: list[str] = []

        def download(url: str) -> bytes:
            calls.append(url)
            return urls[url]

        return download, calls

    return make


@pytest.fixture
def make_count_table() -> Callable[..., dict[str, Any]]:
    """Factory: a presented {claim, table} grouped-count answer, as the API returns it."""

    def make(column: str, rows: list[tuple[str, Any]], value_column: str = "count", claim: str = "x") -> dict[str, Any]:
        return {
            "claim": claim,
            "table": {"columns": [column, value_column], "rows": [{column: k, value_column: v} for k, v in rows]},
        }

    return make


@pytest.fixture
def covid_ms_oracles() -> ModuleType:
    """The covid_ms oracle module, loaded by path as the harness loads it."""
    path = Path(__file__).parent / "oracles" / "covid_ms.py"
    spec = importlib.util.spec_from_file_location("covid_ms_oracle_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeClock:
    """Stands in for the runner's ``time`` module: ``sleep`` advances ``monotonic``."""

    def __init__(self) -> None:
        self.now = 0.0

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += max(seconds, 1.0)


class FakeApi:
    """The runner's view of the API: queries finish at a set clock time and log when they do.

    ``finish_at`` maps query text to the seconds its query takes (default: none); the queries in
    ``falls_back`` log a model-tier fallback when they finish; ``fail`` maps "POST /api/sessions"
    (first call only) or "POST /api/queries" to the response or error to give instead.
    """

    def __init__(self, clock: FakeClock, finish_at: dict[str, float], falls_back: set[str], fail: dict[str, Any]):
        self.clock, self.finish_at, self.falls_back, self.fail = clock, finish_at, falls_back, fail
        self.pending: str | None = None
        self.log = ""
        self.queries: dict[str, tuple[str, float]] = {}
        self.session_calls = 0
        self.logged: set[str] = set()

    def post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        failure = self.fail.get(f"POST {path}")
        if path == "/api/sessions":
            self.session_calls += 1
            if failure is not None and self.session_calls == 1:
                if isinstance(failure, Exception):
                    raise failure
                return dict(failure)
            return {"session_id": f"s{self.session_calls}"}
        if failure is not None:
            return dict(failure)
        query_id = f"q{len(self.queries) + 1}"
        text = payload["query_text"]
        self.queries[query_id] = (text, self.clock.now + self.finish_at.get(text, 0.0))
        return {"query_id": query_id}

    def request(self, method: str, path: str) -> dict[str, Any]:
        query_id = path.rsplit("/", 1)[-1]
        text, done_at = self.queries[query_id]
        if self.clock.now < done_at:
            return {"status": "running"}
        if query_id not in self.logged:
            self.logged.add(query_id)
            self.log += f"t [info     ] parse_outcome  llm_called=True query={text}\n"
            if text in self.falls_back:
                self.log += f"t [warning  ] llm_parse_exception_fallback_to_stub error_type=X query={text}\n"
        return {"status": "completed", "result_data": {"claim": "Total count: 1141 patients.", "table": None}}

    def log_offset(self) -> int:
        return len(self.log)

    def log_since(self, offset: int) -> str:
        return self.log[offset:]


@pytest.fixture
def make_fake_api(monkeypatch: pytest.MonkeyPatch) -> Callable[..., FakeApi]:
    """Factory: a FakeApi on a fake clock patched into the runner (no server, no sleeping)."""

    def make(
        finish_at: dict[str, float] | None = None,
        falls_back: set[str] | None = None,
        fail: dict[str, Any] | None = None,
    ) -> FakeApi:
        clock = FakeClock()
        monkeypatch.setattr(run_module, "time", clock)
        return FakeApi(clock, finish_at or {}, falls_back or set(), fail or {})

    return make
