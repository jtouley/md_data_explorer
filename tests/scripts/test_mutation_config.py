# SPDX-License-Identifier: MIT
"""The mutation-testing config ([tool.mutmut]) is read by every mutmut the dev extra allows.

``process_isolation`` and ``forkserver_warmup`` keep mutant workers off a process that has
imported polars (a fork from one deadlocks). mutmut reads them from 3.8 on; an older one ignores
them and forks plainly, so the dev floor must not let one in.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

from packaging.requirements import Requirement
from packaging.version import Version

PYPROJECT = Path(__file__).resolve().parents[2] / "pyproject.toml"
FORKSERVER_KEYS = {"process_isolation", "forkserver_warmup"}
FIRST_MUTMUT_WITH_FORKSERVER = Version("3.8")


def test_devMutmutFloor_forkserverKeysSet_floorReadsThem() -> None:
    # Arrange
    project = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    dev = [Requirement(r) for r in project["project"]["optional-dependencies"]["dev"]]
    mutmut = next(r for r in dev if r.name == "mutmut")

    # Act
    uses_forkserver = bool(FORKSERVER_KEYS & set(project["tool"]["mutmut"]))
    older_allowed = mutmut.specifier.contains(Version("3.7.99"), prereleases=True)

    # Assert
    assert uses_forkserver, "the forkserver keys are gone: this floor check can go too"
    assert not older_allowed, f"{mutmut} allows a mutmut before {FIRST_MUTMUT_WITH_FORKSERVER}"
    assert mutmut.specifier.contains(FIRST_MUTMUT_WITH_FORKSERVER)
