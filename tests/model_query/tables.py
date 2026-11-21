# SPDX-License-Identifier: MIT
"""Synthetic tables for the model-written query tests and golden samples. No real data.

``patients``: one row per patient, small enough to check every expected value by hand.
``visits``: rows are visits (patients repeat, some rows are exact duplicates, a titer is recorded
in two units), generated from the row number so the oracles in the tests can recompute it.
"""

from __future__ import annotations

import polars as pl

_PATIENTS = [
    # id, sex, dx, admitted, confirmed, onset_year, comorbid, age_cat
    ("P01", "female", "relapsing", "No", "Yes", 2005, "Yes", 1),
    ("P02", "female", "relapsing", "No", "No", 2010, "No", 1),
    ("P03", "female", "relapsing", "Yes", "Yes", 2012, "Yes", 2),
    ("P04", "female", "progressive", "No", "No", 1999, "No", 2),
    ("P05", "female", "other", "No", "No", 2015, "No", 1),
    ("P06", "female", "relapsing", "No", "Yes", 2008, "No", 1),
    ("P07", "male", "relapsing", "No", "No", 2011, "No", 1),
    ("P08", "male", "relapsing", "Yes", "Yes", 2001, "Yes", 2),
    ("P09", "male", "progressive", "No", "No", 1995, "Yes", 2),
    ("P10", "male", "other", "No", "No", 2018, "No", 1),
    ("P11", "female", "relapsing", "No", "No", 2003, "No", 2),
    ("P12", "male", "relapsing", "No", "No", 2007, "Yes", 1),
]
PATIENT_COLUMNS = ["patient_id", "sex", "dx", "admitted", "confirmed", "onset_year", "comorbid", "age_cat"]
VISIT_ROWS = 240
VISIT_PATIENTS = 150


def patients() -> pl.DataFrame:
    return pl.DataFrame(_PATIENTS, schema=PATIENT_COLUMNS, orient="row")


def visits() -> pl.DataFrame:
    """240 visits of 150 patients, every 20th row written twice (an extract duplicate)."""
    rows = []
    for i in range(VISIT_ROWS):
        patient = i % VISIT_PATIENTS
        in_iu = i % 4 == 0
        level = 50 + (i * 37) % 400  # mIU/mL
        row = {
            "patient_id": f"V{patient:03d}",
            "encounter_id": f"E{i:04d}",
            "visit_year": 2021 + i % 4,
            "age_years": 40 + (patient * 7) % 50,
            "sex": "female" if patient % 2 == 0 else "male",
            "region": ("North", "South", "East")[patient % 3],
            "result": "positive" if i % 5 == 0 or i % 7 == 0 else "negative",
            "abx": "yes" if i % 3 == 0 or i % 10 == 0 else "no",
            "screened": "yes" if i % 4 == 1 else "no",
            "titer": None if i % 11 == 0 else (level / 1000 if in_iu else float(level)),
            "titer_unit": None if i % 11 == 0 else ("IU/mL" if in_iu else "mIU/mL"),
        }
        rows.append(row)
        if i % 20 == 0:
            rows.append(dict(row))
    return pl.DataFrame(rows)


VISITS_DICTIONARY = """\
Synthetic clinic: one row per test visit; a patient can have several visits. Some visits appear
twice (duplicate extract rows); count patients by patient_id and tests by encounter_id.

patient_id    Patient key.
encounter_id  Test visit key.
visit_year    Calendar year of the visit.
age_years     Age in years.
result        positive or negative.
abx           yes if an antibiotic was dispensed before the visit.
screened      yes if screened at this visit.
titer         Antibody titer; empty when not drawn. The unit is in titer_unit.
titer_unit    mIU/mL or IU/mL (1 IU/mL = 1000 mIU/mL).
"""


def screening() -> pl.DataFrame:
    """One row per worker: doses on record and an antibody level, the level missing for some.

    "Fewer than two doses and a level under 120": true for W01 and W03, unknown for W05 and W06
    (level missing), false for the rest (W07's two doses decide it without the level).
    """
    return pl.DataFrame(
        {
            "patient_id": [f"W{i:02d}" for i in range(1, 9)],
            "doses": [0, 2, 1, 1, 0, 1, 2, 0],
            "igg": [50.0, 30.0, 90.0, 300.0, None, None, None, 150.0],
        }
    )
