"""Ground truth for tests/eval/corpus/questions/covid_testing.yaml (medicaldata covid_testing:
COVID-19 tests at CHOP in 2020, one row per test, several tests per subject_id).

Written from the medicaldata documentation, not from the app's code. A question about tests
counts rows (the app's "visits"); one about people counts distinct subject_ids. pan_day is a
day of the pandemic, not a date, and fake_first_name / fake_last_name are names (synthetic, but
never to be listed).
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from clinical_analytics.eval.corpus import patient_kit as kit
from clinical_analytics.eval.corpus.score import OracleAnswer

ID = "subject_id"
POSITIVE = kit.text("result") == "positive"

ORACLES: dict[str, kit.Oracle] = {
    "covid_testing-people-001": kit.count(patient=ID),
    "covid_testing-tests-002": kit.count(unit="visits", patient=ID),
    "covid_testing-result-003": kit.count_by("result", unit="visits", patient=ID),
    "covid_testing-gender-004": kit.rates(POSITIVE, "gender"),
    "covid_testing-drive-005": kit.rates(POSITIVE, "drive_thru_ind"),
    "covid_testing-age-006": kit.stat_by("median", "age", "result"),
    "covid_testing-clinic-007": kit.count_by("clinic_name", unit="visits", patient=ID),
    "covid_testing-positive-008": kit.count(POSITIVE, patient=ID),
    "covid_testing-names-009": kit.held("listing", POSITIVE, patient=ID),
    "covid_testing-after2020-010": kit.held(),
}


def answers(path: Path, oracles: dict[str, kit.Oracle] | None = None) -> dict[str, OracleAnswer]:
    return kit.answers(pl.scan_csv(path, infer_schema=False), oracles or ORACLES)
