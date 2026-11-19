"""The harness uploads an aggregate question set with the population-estimates mark (ADR014)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from clinical_analytics.eval.corpus import run as run_module


@pytest.mark.parametrize(("declared", "expected"), [(True, True), (False, False)], ids=["aggregate", "patient"])
def test_upload_populationEstimates_formFieldOnlyWhenDeclared(tmp_path: Path, declared, expected):
    # Arrange: an Api without a server; request() records what would be sent
    api = object.__new__(run_module.Api)
    sent: dict[str, Any] = {}

    def request(method: str, path: str, body: bytes | None = None, ctype: str | None = None, timeout: float = 0):
        sent["body"] = body or b""
        return {"upload_id": "u1"}

    api.request = request  # type: ignore[method-assign]
    table = tmp_path / "t.csv"
    table.write_text("a,b\n1,2\n", encoding="utf-8")

    # Act
    upload_id = api.upload(table, "t", declared)

    # Assert
    assert upload_id == "u1"
    assert (b'name="population_estimates"' in sent["body"]) is expected
