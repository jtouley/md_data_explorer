# SPDX-License-Identifier: MIT
"""DuckDB limits: parsed and validated when the config is imported, applied to every layer."""

from __future__ import annotations

import os
import subprocess
import sys

import duckdb
import ibis
import pytest

from clinical_analytics.core import nl_query_config, semantic
from clinical_analytics.core.nl_query_config import (
    DuckdbMemoryLimit,
    parse_duckdb_memory_limit,
    per_layer_memory_limit,
    resolve_duckdb_threads,
    resolve_semantic_layer_cache_size,
)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("20%", DuckdbMemoryLimit(20.0, "%")),
        ("100%", DuckdbMemoryLimit(100.0, "%")),
        ("2GB", DuckdbMemoryLimit(2.0, "GB")),
        (" 1.5 gb ", DuckdbMemoryLimit(1.5, "GB")),
        ("512mib", DuckdbMemoryLimit(512.0, "MiB")),
        ("0.5%", DuckdbMemoryLimit(0.5, "%")),
    ],
)
def test_parseDuckdbMemoryLimit_validValue_numberAndUnit(value: str, expected: DuckdbMemoryLimit) -> None:
    # Arrange / Act
    parsed = parse_duckdb_memory_limit(value)

    # Assert
    assert parsed == expected


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ("0GB", "above 0"),
        ("0%", "above 0"),
        ("0.0 MB", "above 0"),
        ("500%", "at most 100%"),
        ("100.1%", "at most 100%"),
        ("lots", "a size such as"),
        ("1GB'; DROP TABLE x; --", "a size such as"),
        ("", "a size such as"),
        ("-1GB", "a size such as"),
        ("1e3GB", "a size such as"),
        (None, "a size such as"),
        (20, "a size such as"),
    ],
)
def test_parseDuckdbMemoryLimit_badValue_raisesValueError(value: object, message: str) -> None:
    # Arrange / Act / Assert
    with pytest.raises(ValueError, match=message):
        parse_duckdb_memory_limit(value)


@pytest.mark.parametrize(
    ("limit", "sql"),
    [
        (DuckdbMemoryLimit(20.0, "%"), "20%"),
        (DuckdbMemoryLimit(1.5, "GB"), "1.5GB"),
        (DuckdbMemoryLimit(512.0, "MiB"), "512MiB"),
    ],
)
def test_duckdbMemoryLimit_sql_numberThenUnit(limit: DuckdbMemoryLimit, sql: str) -> None:
    # Arrange / Act / Assert
    assert limit.sql == sql


@pytest.mark.parametrize(("value", "expected"), [(1, 1), (8, 8), ("3", 3)])
def test_resolveDuckdbThreads_positive_keptAsGiven(value: object, expected: int) -> None:
    # Arrange / Act / Assert
    assert resolve_duckdb_threads(value) == expected


def test_resolveDuckdbThreads_zero_halfTheCpusAtLeastOne() -> None:
    # Arrange / Act
    threads = resolve_duckdb_threads(0)

    # Assert
    assert threads == max(1, (os.cpu_count() or 2) // 2)


@pytest.mark.parametrize("value", [-1, True, "many", 2.5, None])
def test_resolveDuckdbThreads_badValue_raisesValueError(value: object) -> None:
    # Arrange / Act / Assert
    with pytest.raises(ValueError, match="duckdb_threads"):
        resolve_duckdb_threads(value)


def _import_config_with(env_overrides: dict[str, str]) -> subprocess.CompletedProcess[str]:
    env = {key: value for key, value in os.environ.items() if not key.startswith("DUCKDB_")}
    env.update(env_overrides)
    return subprocess.run(
        [sys.executable, "-c", "import clinical_analytics.core.nl_query_config"],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.parametrize(
    ("budget", "size", "expected"),
    [
        (DuckdbMemoryLimit(60.0, "%"), 4, DuckdbMemoryLimit(15.0, "%")),
        (DuckdbMemoryLimit(50.0, "%"), 1, DuckdbMemoryLimit(50.0, "%")),
        (DuckdbMemoryLimit(2.0, "GB"), 4, DuckdbMemoryLimit(0.5, "GB")),
        (DuckdbMemoryLimit(100.0, "%"), 3, DuckdbMemoryLimit(100.0 / 3, "%")),
    ],
)
def test_perLayerMemoryLimit_totalBudget_splitEvenlyAcrossTheCache(
    budget: DuckdbMemoryLimit, size: int, expected: DuckdbMemoryLimit
) -> None:
    # Arrange / Act
    limit = per_layer_memory_limit(budget, size)

    # Assert
    assert limit == expected
    assert limit.amount * size == pytest.approx(budget.amount)


@pytest.mark.parametrize(
    "budget",
    [DuckdbMemoryLimit(100.0, "MB"), DuckdbMemoryLimit(0.2, "GiB"), DuckdbMemoryLimit(1.0, "%")],
)
def test_perLayerMemoryLimit_shareTooSmallToAnswer_raisesValueError(budget: DuckdbMemoryLimit) -> None:
    # Arrange / Act / Assert: under 64 MiB, or under 1% of RAM, per layer
    with pytest.raises(ValueError, match="duckdb_memory_budget"):
        per_layer_memory_limit(budget, 4)


@pytest.mark.parametrize(("value", "expected"), [(1, 1), (4, 4), ("6", 6)])
def test_resolveSemanticLayerCacheSize_positive_keptAsGiven(value: object, expected: int) -> None:
    # Arrange / Act / Assert
    assert resolve_semantic_layer_cache_size(value) == expected


@pytest.mark.parametrize("value", [0, -1, True, False, "many", 2.5, None])
def test_resolveSemanticLayerCacheSize_badValue_raisesValueError(value: object) -> None:
    # Arrange / Act / Assert: 0 would mean "close every layer at once", never "unbounded"
    with pytest.raises(ValueError, match="semantic_layer_cache_size"):
        resolve_semantic_layer_cache_size(value)


def test_configImport_defaults_perLayerLimitIsBudgetOverCacheSize() -> None:
    # Arrange / Act
    expected = per_layer_memory_limit(nl_query_config.DUCKDB_MEMORY_BUDGET, nl_query_config.SEMANTIC_LAYER_CACHE_SIZE)

    # Assert
    assert nl_query_config.DUCKDB_MEMORY_LIMIT == expected
    assert nl_query_config.SEMANTIC_LAYER_CACHE_SIZE == 4


@pytest.mark.parametrize(
    ("env", "message"),
    [
        ({"DUCKDB_MEMORY_BUDGET": "0GB"}, "duckdb_memory_budget"),
        ({"DUCKDB_MEMORY_BUDGET": "500%"}, "duckdb_memory_budget"),
        ({"DUCKDB_MEMORY_BUDGET": "100MB"}, "duckdb_memory_budget"),
        ({"SEMANTIC_LAYER_CACHE_SIZE": "0"}, "semantic_layer_cache_size"),
        # The old per-layer setting: silently ignoring it would change the memory used.
        ({"DUCKDB_MEMORY_LIMIT": "20%"}, "duckdb_memory_budget"),
    ],
)
def test_configImport_badBudgetOrCacheSize_failsAtStartup(env: dict[str, str], message: str) -> None:
    # Arrange / Act
    completed = _import_config_with(env)

    # Assert
    assert completed.returncode != 0
    assert message in completed.stderr


@pytest.mark.parametrize("value", ["0GB", "0%", "500%"])
def test_configImport_badMemoryLimit_failsAtStartup(value: str) -> None:
    # Arrange
    env = {**os.environ, "DUCKDB_MEMORY_BUDGET": value}

    # Act
    completed = subprocess.run(
        [sys.executable, "-c", "import clinical_analytics.core.nl_query_config"],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    # Assert
    assert completed.returncode != 0
    assert "duckdb_memory_budget" in completed.stderr


def test_semanticLayerInit_anyPath_limitsApplied(make_semantic_layer, monkeypatch) -> None:
    # Arrange
    monkeypatch.setattr(nl_query_config, "DUCKDB_MEMORY_LIMIT", DuckdbMemoryLimit(1.0, "GiB"))
    monkeypatch.setattr(nl_query_config, "DUCKDB_THREADS", 2)

    # Act
    layer = make_semantic_layer()

    # Assert
    raw = layer.con.con
    assert raw.execute("SELECT current_setting('threads')").fetchone()[0] == 2
    memory = raw.execute("SELECT current_setting('memory_limit')").fetchone()[0]
    assert memory == "1.0 GiB"


def test_semanticLayerInit_configureFails_connectionClosed(make_semantic_layer, monkeypatch) -> None:
    # Arrange
    opened: list[object] = []
    connect = ibis.duckdb.connect

    def recording_connect(*args: object, **kwargs: object) -> object:
        backend = connect(*args, **kwargs)
        opened.append(backend)
        return backend

    def failing_limits(raw: object) -> None:
        raise duckdb.InvalidInputException("bad limit")

    monkeypatch.setattr(semantic.ibis.duckdb, "connect", recording_connect)
    monkeypatch.setattr(semantic, "_apply_duckdb_limits", failing_limits)

    # Act
    with pytest.raises(duckdb.InvalidInputException, match="bad limit"):
        make_semantic_layer()

    # Assert
    assert len(opened) == 1
    with pytest.raises(duckdb.ConnectionException):
        opened[0].con.execute("SELECT 1")  # type: ignore[attr-defined]
