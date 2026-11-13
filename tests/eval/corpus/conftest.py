"""Fixtures for the acceptance-corpus fetcher and scorer (synthetic bytes only, no real data)."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from typing import Any

import pytest

from clinical_analytics.eval.corpus.fetch import CorpusDataset


@pytest.fixture
def make_corpus_dataset() -> Callable[..., tuple[CorpusDataset, dict[str, bytes]]]:
    """Factory: a one-table corpus entry over synthetic bytes, and the URL -> bytes it serves."""

    def make(table: bytes = b"id,sex\n1,female\n2,male\n", page: bytes = b"<p>sex: biological sex</p>", **pkg: Any):
        urls = {"https://example.org/table.csv": table, "https://example.org/page.html": page}
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
                        "path": "page.html",
                        "url": "https://example.org/page.html",
                        "kind": "dictionary",
                        "sha256": hashlib.sha256(page).hexdigest(),
                    },
                ],
                "upload_package": {
                    "zip": "synthetic.zip",
                    "contents": [
                        {"name": "table.csv", "from": "table.csv", "make": "copy"},
                        {"name": "data_dictionary.txt", "from": "page.html", "make": "html_to_text"},
                    ],
                    **pkg,
                },
            }
        )
        return dataset, urls

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
