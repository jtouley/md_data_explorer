"""Ground truth for tests/eval/corpus/questions/cgd_infections.yaml (R survival::cgd).

Written from the R survival documentation, not from the app's code. One row is a follow-up
interval of one patient (counting-process format: tstart, tstop, enum); status 1 means the
interval ended in a serious infection. 203 rows, 128 patients: a question about patients
counts distinct ``id``s, and a patient attribute (age, arm) repeated on each row counts once.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from clinical_analytics.eval.corpus import patient_kit as kit
from clinical_analytics.eval.corpus.score import OracleAnswer

ID = "id"
INFECTION = kit.text("status") == "1"

ORACLES: dict[str, kit.Oracle] = {
    "cgd_infections-total-001": kit.count(patient=ID),
    "cgd_infections-events-002": kit.count(INFECTION, unit="visits", patient=ID),
    "cgd_infections-arm-003": kit.count_by("treat", patient=ID),
    "cgd_infections-infected-004": kit.count_by("treat", INFECTION, patient=ID),
    "cgd_infections-rate-005": kit.rates(INFECTION, "treat", patient=ID),
    "cgd_infections-center-006": kit.count_by("center", patient=ID),
    "cgd_infections-age-007": kit.stat_by("median", "age", "treat", patient=ID),
    "cgd_infections-inherit-008": kit.count_by("inherit", patient=ID),
    "cgd_infections-listing-009": kit.held("listing", INFECTION, patient=ID),
}


def answers(path: Path, oracles: dict[str, kit.Oracle] | None = None) -> dict[str, OracleAnswer]:
    return kit.answers(pl.scan_csv(path, infer_schema=False), oracles or ORACLES)
