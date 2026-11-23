# SPDX-License-Identifier: MIT
"""Synthetic tables for the model-written query tests. No real data.

Each is generated from the row number, so the tests' Polars oracles recompute every expected
value independently of the SQL the compiler writes.

``clinic``: visits (patients repeat, a sequence numbers each patient's visits, a dated visit), with
a data dictionary that labels its codes. ``cohort``: one row per patient. ``ledger``: lines that
no column names a patient.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

import polars as pl

CLINIC_PATIENTS = 40
CLINIC_DICTIONARY = """\
HYPERTENSION CLINIC FOLLOW-UP
One row per clinic visit; pt_ref is the patient.

pt_ref        Patient reference number
visit_seq     Visit number within the patient (1 = first visit)
visit_on      Visit date
age_at_visit  Age in years at the visit
sex           F = female, M = male
systolic_mmhg Systolic blood pressure, mmHg
smoker        Current smoker at the visit (Y/N)
arm           Treatment arm: 1 = ACE inhibitor, 2 = calcium blocker, 3 = diuretic
"""


def clinic() -> pl.DataFrame:
    """40 patients with 1-4 visits each; smoking and the arm change between some visits."""
    rows = []
    for p in range(CLINIC_PATIENTS):
        for v in range(1, 1 + 1 + p % 4):
            rows.append(
                {
                    "pt_ref": f"H{p:03d}",
                    "visit_seq": v,
                    "visit_on": date(2022, 1, 3) + timedelta(days=60 * v + p),
                    "age_at_visit": 45 + p % 30 + v // 3,
                    "sex": "F" if p % 3 else "M",
                    "systolic_mmhg": float(120 + (p * 7 + v * 11) % 50) if (p + v) % 9 else None,
                    "smoker": "Y" if (p + v) % 4 == 0 else "N",
                    "arm": 1 + (p + (v if p % 5 == 0 else 0)) % 3,
                }
            )
    return pl.DataFrame(rows)


COHORT_DICTIONARY = """\
LUNG SURGERY COHORT. One row per patient.

subject_no  Study subject number (patient)
age         Age in years at surgery
sex         female / male
stage       Tumour stage: I, II or III
died_1y     Died within one year of surgery (yes/no)
fev_pct     Pre-operative FEV1, percent predicted; blank = not measured
"""


def cohort() -> pl.DataFrame:
    """60 patients, one row each."""
    return pl.DataFrame(
        {
            "subject_no": [f"S{i:03d}" for i in range(60)],
            "age": [40 + (i * 13) % 45 for i in range(60)],
            "sex": ["female" if i % 2 else "male" for i in range(60)],
            "stage": [("I", "II", "III")[i % 3] for i in range(60)],
            "died_1y": ["yes" if i % 7 == 0 else "no" for i in range(60)],
            "fev_pct": [None if i % 10 == 0 else float(50 + (i * 17) % 50) for i in range(60)],
        }
    )


def ledger() -> pl.DataFrame:
    """30 numbered lines of supply use: nothing names a patient."""
    return pl.DataFrame(
        {
            "line_no": [f"L{i:03d}" for i in range(30)],
            "ward": [("north", "south", "east")[i % 3] for i in range(30)],
            "qty": [(i * 7) % 11 + 1 for i in range(30)],
        }
    )


# A model reply with every slot of the plan; tests override the slots they are about.
EMPTY_REPLY: dict[str, Any] = {
    "answer": "count",
    "count_of": "patients",
    "filters": [],
    "numerator": [],
    "measure": None,
    "group_by": None,
    "period": "none",
    "cut_points": [],
    "which_visit": "any",
}


def reply(**slots: Any) -> dict[str, Any]:
    return {**EMPTY_REPLY, **slots}


def cond(column: str, op: str, value: Any = None, values: list[Any] | None = None) -> dict[str, Any]:
    """One condition slot of a reply."""
    return {"column": column, "op": op, "value": value, "values": values or []}


def injuries() -> pl.DataFrame:
    """A summary table: road injuries counted per district, year (60 years) and road user. No person."""
    rows = []
    for d, district in enumerate(("delta", "ridge", "valley")):
        for year in range(1965, 2025):
            for u, user in enumerate(("pedestrian", "cyclist", "driver")):
                rows.append(
                    {"district": district, "year": year, "road_user": user, "injury_cases": (d * 7 + year + u * 3) % 40}
                )
    return pl.DataFrame(rows)
