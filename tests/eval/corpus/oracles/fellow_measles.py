"""Ground truth for questions/fellow_measles.yaml: the shared fellow-scenario oracles, for this dataset."""

from __future__ import annotations

import importlib.util
from pathlib import Path

from clinical_analytics.eval.corpus.score import OracleAnswer

_spec = importlib.util.spec_from_file_location(
    "fellow_hospital_oracles", Path(__file__).with_name("fellow_hospital.py")
)
assert _spec is not None and _spec.loader is not None
_shared = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_shared)

# This dataset's oracles by question id, as every oracle module exposes them.
ORACLES = _shared.ORACLES["fellow_measles"]


def answers(csv_path: Path) -> dict[str, OracleAnswer]:
    return _shared.answers(csv_path, "fellow_measles")
