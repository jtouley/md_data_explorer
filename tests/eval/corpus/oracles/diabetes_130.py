"""Ground truth for tests/eval/corpus/questions/diabetes_130.yaml (UCI Diabetes 130-US
Hospitals 1999-2008: one row per hospital encounter, several encounters per patient_nbr).

Written from the UCI variable list and Strack et al. (2014), not from the app's code. "?" is
missing; "None" in A1Cresult and max_glu_serum is the level "not measured", not missing.
ICD-9 codes are stored with leading zeros stripped: septicaemia (038) is "38" in diag_1.
readmitted is "NO", ">30" or "<30" (days). A question about encounters counts rows (the app's
"visits"); one about patients counts distinct patient_nbr.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from clinical_analytics.eval.corpus import patient_kit as kit
from clinical_analytics.eval.corpus.score import OracleAnswer

ID = "patient_nbr"
READMITTED_30 = kit.text("readmitted") == "<30"
SEPTICAEMIA = kit.text("diag_1") == "38"
A1C_MEASURED = kit.text("A1Cresult") != "None"

ORACLES: dict[str, kit.Oracle] = {
    "diabetes_130-encounters-001": kit.count(unit="visits", patient=ID),
    "diabetes_130-patients-002": kit.count(patient=ID),
    "diabetes_130-readmitted-003": kit.count_by("readmitted", unit="visits", patient=ID),
    "diabetes_130-agerate-004": kit.rates(READMITTED_30, "age"),
    "diabetes_130-sexrate-005": kit.rates(READMITTED_30, "gender"),
    "diabetes_130-racerate-006": kit.rates(READMITTED_30, "race"),
    "diabetes_130-stay-007": kit.stat_by("mean", "time_in_hospital", "readmitted"),
    "diabetes_130-septicaemia-008": kit.count(SEPTICAEMIA, unit="visits", patient=ID),
    "diabetes_130-a1c-009": kit.count(A1C_MEASURED, unit="visits", patient=ID),
    "diabetes_130-trend-010": kit.held(),
}


def answers(path: Path, oracles: dict[str, kit.Oracle] | None = None) -> dict[str, OracleAnswer]:
    return kit.answers(pl.scan_csv(path, infer_schema=False), oracles or ORACLES)
