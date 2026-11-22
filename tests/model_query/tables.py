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


def clinic_visits() -> pl.DataFrame:
    """A made-up allergy clinic: one row per clinic visit, 6 patients with 2 or 3 visits each.

    ``gender`` is coded (1 = male, 2 = female, per ``CLINIC_DICTIONARY``); ``height_cm`` is the
    same on every visit of a patient, ``eos_baseline`` and ``eos_month6`` are one measure at two
    times, and ``ige`` differs between a patient's visits.
    """
    rows = [
        # pid, gender, height, eos0, eos6, ige, year, shots
        ("A1", 1, 170.0, 300.0, 200.0, 100.0, 2022, "yes"),
        ("A1", 1, 170.0, 300.0, 200.0, 140.0, 2023, "no"),
        ("A2", 2, 160.0, 500.0, 250.0, 80.0, 2022, "yes"),
        ("A2", 2, 160.0, 500.0, 250.0, 80.0, 2023, "yes"),
        ("A2", 2, 160.0, 500.0, 250.0, 60.0, 2024, "no"),
        ("A3", 2, 150.0, 200.0, 100.0, 300.0, 2023, "no"),
        ("A3", 2, 150.0, 200.0, 100.0, 320.0, 2024, "no"),
        ("A4", 1, 180.0, 400.0, 300.0, 50.0, 2022, "yes"),
        ("A4", 1, 180.0, 400.0, 300.0, 70.0, 2024, "yes"),
        ("A5", 2, 165.0, 100.0, 90.0, 210.0, 2024, "no"),
        ("A5", 2, 165.0, 100.0, 90.0, 230.0, 2024, "yes"),
        ("A6", 1, 175.0, 350.0, 150.0, 90.0, 2023, "no"),
        ("A6", 1, 175.0, 350.0, 150.0, 95.0, 2024, "no"),
    ]
    columns = ["pt_id", "gender", "height_cm", "eos_baseline", "eos_month6", "ige", "visit_year", "shots"]
    return pl.DataFrame(rows, schema=columns, orient="row")


CLINIC_DICTIONARY = """\
Made-up allergy clinic: one row per clinic visit.

pt_id         Patient identifier.
gender        Patient gender: 1 = male, 2 = female
height_cm     Height in centimetres.
eos_baseline  Blood eosinophil count at enrolment (cells/uL).
eos_month6    Blood eosinophil count six months after enrolment (cells/uL).
ige           Total IgE at the visit (kU/L).
visit_year    Calendar year of the visit.
shots         yes if allergy shots were given at the visit.
"""


def eye_visits() -> pl.DataFrame:
    """A made-up glaucoma clinic: one row per visit, 4 patients with 2 to 4 visits.

    ``visit_no`` numbers each patient's visits (1, 2, ...): it repeats across patients and is not a
    visit ID. ``drops_used`` is a Y/N flag, ``iop_mmhg`` the eye pressure, ``adherence_pct`` a
    percentage, ``grade`` a field grade recorded on some visits only.
    """
    import datetime as dt

    rows = [
        # pat_no, visit_no, date, iop, drops, adherence, grade, sex
        ("G1", 1, dt.date(2022, 1, 10), 25.0, "N", 40.0, None, "F"),
        ("G1", 2, dt.date(2022, 3, 10), 22.0, "Y", 70.0, 2, "F"),
        ("G1", 3, dt.date(2023, 1, 5), 19.0, "Y", 90.0, 1, "F"),
        ("G2", 1, dt.date(2022, 2, 1), 18.0, "N", 100.0, None, "M"),
        ("G2", 2, dt.date(2023, 2, 1), 17.0, "N", 100.0, 1, "M"),
        ("G3", 1, dt.date(2022, 5, 5), 30.0, "Y", 50.0, 3, "F"),
        ("G3", 2, dt.date(2022, 6, 4), 26.0, "Y", 60.0, None, "F"),
        ("G3", 3, dt.date(2023, 6, 4), 20.0, "Y", 80.0, 2, "F"),
        ("G3", 4, dt.date(2024, 6, 3), 19.0, "Y", 95.0, 1, "F"),
        ("G4", 1, dt.date(2023, 9, 9), 20.0, "N", 85.0, None, "M"),
        ("G4", 2, dt.date(2024, 9, 8), 19.0, "N", 90.0, 1, "M"),
    ]
    columns = ["pat_no", "visit_no", "visit_date", "iop_mmhg", "drops_used", "adherence_pct", "grade", "sex"]
    return pl.DataFrame(rows, schema=columns, orient="row")


EYE_DICTIONARY = """\
Made-up glaucoma clinic: one row per clinic visit.

pat_no         Patient number.
visit_no       Number of the visit for this patient (1 = first visit).
visit_date     Date of the visit.
iop_mmhg       Intraocular pressure (mmHg).
drops_used     Y if pressure-lowering drops were in use at the visit, N if not.
adherence_pct  Share of prescribed drop doses taken since the last visit (percent).
grade          Optic disc grade, 1 to 3; empty when not graded.
sex            F or M.
"""


def rehab_stays() -> pl.DataFrame:
    """A made-up rehabilitation unit: one row per stay, each patient once; ``therapy_sessions``
    counts the sessions of the stay."""
    rows = [
        ("R1", 2022, "home", 10), ("R2", 2022, "home", 14), ("R3", 2023, "care home", 20),
        ("R4", 2023, "home", 8), ("R5", 2023, "hospital", 5), ("R6", 2024, "home", 12),
    ]  # fmt: skip
    return pl.DataFrame(rows, schema=["patient_id", "stay_year", "discharge_to", "therapy_sessions"], orient="row")
