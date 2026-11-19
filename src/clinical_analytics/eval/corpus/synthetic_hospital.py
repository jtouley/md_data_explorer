# SPDX-License-Identifier: MIT
"""A synthetic hospital for the fellow case-study acceptance scenarios. No real patient data.

The four extracts are what an ID fellow pulls from a hospital data warehouse to fill the
[blanks] in the case-study abstracts of the fellows' research guide (C. diff in adults 65+,
hepatitis B screening, pneumonia pathogens by vaccination, measles immunity gaps). Every row is
made up by a seeded ``random.Random`` from one shared patient registry, so a patient key can
appear in several extracts, as it would in a real warehouse.

Rows are visits (one row per test, clinic visit, admission or immunity check), each with a
patient key. Each extract carries deliberate traps that a careless answer gets wrong:

- duplicate visits: exact duplicate rows (a lab order joined twice), repeat visits by the same
  patient, and, for pneumonia, one row per pathogen when two are found in one admission;
- small subgroups: groups of 1-10 patients that must never be shown (a partial first year,
  a rare birthplace region, Legionella, a small risk group);
- mixed units: measles IgG reported in mIU/mL by one lab and IU/mL by another.

The bytes depend only on the seed and this code: ``random.Random`` is stable across Python
versions for ``random()``, ``randint`` and ``choices`` with weights, and every value is written
as text. ``corpus.yaml`` pins each file's sha256, so a change here that alters the data fails
the fetch until the pin and the oracles are reviewed together.

    uv run python -m clinical_analytics.eval.corpus.synthetic_hospital cdiff table > cdiff.csv
"""

from __future__ import annotations

import io
import math
import random
import sys
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Final, Literal, TypeAlias

import polars as pl

Extract: TypeAlias = Literal["cdiff", "hepb", "pneumonia", "measles"]
Part: TypeAlias = Literal["table", "dictionary"]
EXTRACTS: Final[tuple[Extract, ...]] = ("cdiff", "hepb", "pneumonia", "measles")
DEFAULT_SEED: Final = 20261119
REGISTRY_SIZE: Final = 14000

YES, NO, UNKNOWN = "yes", "no", "unknown"


class SyntheticDataError(ValueError):
    """An unknown extract or part was asked for."""


@dataclass(frozen=True)
class Patient:
    patient_id: str
    sex: str
    birth_date: date
    birth_region: str


# Birthplace regions and their share of the registry. Pacific Islands is the rare one.
REGIONS: Final[tuple[tuple[str, float], ...]] = (
    ("United States", 0.70),
    ("East Asia", 0.06),
    ("Southeast Asia", 0.05),
    ("Sub-Saharan Africa", 0.05),
    ("Latin America", 0.06),
    ("Eastern Europe", 0.04),
    ("Middle East", 0.0388),
    ("Pacific Islands", 0.0012),
)


def _logistic(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _age(born: date, on: date) -> int:
    return on.year - born.year - ((on.month, on.day) < (born.month, born.day))


def _day(rng: random.Random, start: date, end: date) -> date:
    return start + timedelta(days=rng.randint(0, (end - start).days))


def registry(seed: int) -> list[Patient]:
    """The hospital's patients: key, sex, date of birth, birthplace region."""
    rng = random.Random(f"{seed}:registry")
    names = [name for name, _ in REGIONS]
    weights = [weight for _, weight in REGIONS]
    patients = []
    for i in range(REGISTRY_SIZE):
        born = _day(rng, date(1925, 1, 1), date(2006, 12, 31))
        region = rng.choices(names, weights)[0]
        patients.append(Patient(f"LH{100000 + i * 7:06d}", rng.choice(("female", "male")), born, region))
    return patients


def _csv(rows: list[dict[str, str]], columns: list[str]) -> bytes:
    """The rows as CSV text; an empty value is written as an empty field (a missing value)."""
    data = {c: [row[c] or None for row in rows] for c in columns}
    frame = pl.DataFrame(data, schema=dict.fromkeys(columns, pl.String))
    buffer = io.BytesIO()
    frame.write_csv(buffer)
    return buffer.getvalue()


def _with_duplicates(rng: random.Random, rows: list[dict[str, str]], share: float) -> list[dict[str, str]]:
    """Exact duplicate rows for about ``share`` of visits, each right after its original."""
    out: list[dict[str, str]] = []
    for row in rows:
        out.append(row)
        if rng.random() < share:
            out.append(dict(row))
    return out


# ---------------------------------------------------------------------------- C. diff


CDIFF_COLUMNS: Final = [
    "patient_id",
    "encounter_id",
    "test_date",
    "test_year",
    "age_years",
    "sex",
    "cdiff_result",
    "outpatient_abx_90d",
    "ppi_use",
]


def cdiff_rows(seed: int) -> list[dict[str, str]]:
    """Adult encounters with a C. difficile test sent within 72 hours of arrival (community
    onset), 1 Oct 2017 to 31 Dec 2025. Positivity rises after 2020 in patients 65+; outpatient
    antibiotics in the prior 90 days raise the odds of a positive test."""
    rng = random.Random(f"{seed}:cdiff")
    patients = registry(seed)
    rows: list[dict[str, str]] = []
    encounter = 500000
    # Tests per year; 2017 is a partial year (the extract starts on 1 Oct 2017).
    per_year = {2017: 55, 2018: 560, 2019: 590, 2020: 470, 2021: 610, 2022: 630, 2023: 650, 2024: 660, 2025: 680}
    repeaters: list[Patient] = []
    for year, n_tests in per_year.items():
        start = date(2017, 10, 1) if year == 2017 else date(year, 1, 1)
        for _ in range(n_tests):
            on = _day(rng, start, date(year, 12, 31))
            # One in eight tests is a repeat visit by a patient already tested.
            if repeaters and rng.random() < 0.125:
                patient = rng.choice(repeaters)
            else:
                patient = rng.choice(patients)
            age = _age(patient.birth_date, on)
            if age < 18:
                continue
            repeaters.append(patient)
            older = age >= 65
            abx = rng.random() < (0.42 if older else 0.30)
            ppi = rng.random() < (0.35 if older else 0.20)
            later = year >= 2021
            logit = -2.6 + 0.95 * abx + 0.35 * ppi + 0.30 * older + (0.55 if later and older else 0.0)
            positive = rng.random() < _logistic(logit)
            encounter += rng.randint(1, 9)
            rows.append(
                {
                    "patient_id": patient.patient_id,
                    "encounter_id": f"E{encounter}",
                    "test_date": on.isoformat(),
                    "test_year": str(year),
                    "age_years": str(age),
                    "sex": patient.sex,
                    "cdiff_result": "positive" if positive else "negative",
                    "outpatient_abx_90d": YES if abx else NO,
                    "ppi_use": YES if ppi else NO,
                }
            )
    rows.sort(key=lambda row: (row["test_date"], row["encounter_id"]))
    return _with_duplicates(rng, rows, 0.03)


CDIFF_DICTIONARY: Final = """\
Lakeside General Hospital (synthetic): C. difficile tests, community onset, Oct 2017 - Dec 2025

One row per adult encounter with a C. difficile toxin/PCR test collected within 72 hours of
arrival (community onset). The extract starts on 1 October 2017. Some encounters appear twice
(the lab order was joined twice); count encounters by encounter_id and patients by patient_id.

patient_id          Hospital patient key.
encounter_id        Encounter key.
test_date           Date the test was collected (YYYY-MM-DD).
test_year           Calendar year of test_date.
age_years           Age in years at the encounter.
sex                 female or male.
cdiff_result        positive or negative.
outpatient_abx_90d  yes if an outpatient antibiotic was dispensed in the 90 days before arrival.
ppi_use             yes if a proton pump inhibitor was on the home medication list.
"""


# ---------------------------------------------------------------------------- hepatitis B


HEPB_COLUMNS: Final = [
    "patient_id",
    "visit_id",
    "visit_date",
    "age_years",
    "sex",
    "birth_region",
    "hbv_screen_done",
    "hbsag_result",
    "new_hbv_diagnosis",
]

# Chance a patient was ever screened, and chance a screened patient is HBsAg-reactive.
HEPB_SCREENED: Final = {
    "United States": 0.22,
    "East Asia": 0.58,
    "Southeast Asia": 0.52,
    "Sub-Saharan Africa": 0.47,
    "Latin America": 0.30,
    "Eastern Europe": 0.33,
    "Middle East": 0.31,
    "Pacific Islands": 0.45,
}
HEPB_REACTIVE: Final = {
    "United States": 0.003,
    "East Asia": 0.06,
    "Southeast Asia": 0.05,
    "Sub-Saharan Africa": 0.07,
    "Latin America": 0.006,
    "Eastern Europe": 0.03,
    "Middle East": 0.02,
    "Pacific Islands": 0.10,
}


def hepb_rows(seed: int) -> list[dict[str, str]]:
    """Adults with a primary-care visit, 2023-2025: one row per visit. A screened patient has
    the HBsAg result on one visit; a reactive result found for the first time is a new
    diagnosis, flagged on that visit only."""
    rng = random.Random(f"{seed}:hepb")
    patients = [p for p in registry(seed) if _age(p.birth_date, date(2023, 1, 1)) >= 18]
    chosen = [p for p in patients if p.birth_region != "United States" or rng.random() < 0.45]
    rows: list[dict[str, str]] = []
    visit = 700000
    for patient in chosen:
        n_visits = rng.choices((1, 2, 3, 4), (0.45, 0.30, 0.17, 0.08))[0]
        dates = sorted(_day(rng, date(2023, 1, 1), date(2025, 12, 31)) for _ in range(n_visits))
        screened = rng.random() < HEPB_SCREENED[patient.birth_region]
        screen_at = rng.randrange(n_visits) if screened else -1
        reactive = screened and rng.random() < HEPB_REACTIVE[patient.birth_region]
        known = reactive and rng.random() < 0.35  # already known: reactive, but not a new diagnosis
        for i, on in enumerate(dates):
            visit += rng.randint(1, 5)
            here = i == screen_at
            rows.append(
                {
                    "patient_id": patient.patient_id,
                    "visit_id": f"V{visit}",
                    "visit_date": on.isoformat(),
                    "age_years": str(_age(patient.birth_date, on)),
                    "sex": patient.sex,
                    "birth_region": patient.birth_region,
                    "hbv_screen_done": YES if here else NO,
                    "hbsag_result": ("reactive" if reactive else "nonreactive") if here else "",
                    "new_hbv_diagnosis": YES if here and reactive and not known else NO,
                }
            )
    rows.sort(key=lambda row: (row["visit_date"], row["visit_id"]))
    return _with_duplicates(rng, rows, 0.02)


HEPB_DICTIONARY: Final = """\
Lakeside General Hospital (synthetic): adult primary-care visits and hepatitis B screening, 2023-2025

One row per visit; a patient can have several visits. Some visits appear twice (duplicate
extract rows); count patients by patient_id. A patient counts as screened if any of their
visits has hbv_screen_done = yes.

patient_id          Hospital patient key.
visit_id            Visit key.
visit_date          Date of the visit (YYYY-MM-DD).
age_years           Age in years at the visit.
sex                 female or male.
birth_region        Region of birth (United States, East Asia, Southeast Asia, Sub-Saharan Africa,
                    Latin America, Eastern Europe, Middle East, Pacific Islands).
hbv_screen_done     yes if an HBsAg screening test was resulted at this visit.
hbsag_result        reactive or nonreactive; empty when no test at this visit.
new_hbv_diagnosis   yes if this visit's reactive HBsAg was a new hepatitis B diagnosis.
"""


# ---------------------------------------------------------------------------- pneumonia


PNEUMONIA_COLUMNS: Final = [
    "patient_id",
    "encounter_id",
    "admit_date",
    "admit_year",
    "age_years",
    "sex",
    "pathogen",
    "pneumococcal_vaccine",
    "flu_vaccine_this_season",
    "icu_admission",
    "died_in_hospital",
]

# Pathogen identified for an admission (None identified is the most common result).
PATHOGENS: Final[tuple[tuple[str, float], ...]] = (
    ("None identified", 0.46),
    ("Influenza", 0.15),
    ("S. aureus", 0.10),
    ("S. pneumoniae", 0.11),
    ("RSV", 0.06),
    ("H. influenzae", 0.05),
    ("Pseudomonas", 0.05),
    ("Legionella", 0.007),
)


def _vaccine(rng: random.Random, p_yes: float) -> str:
    draw = rng.random()
    return YES if draw < p_yes else (UNKNOWN if draw < p_yes + 0.12 else NO)


def pneumonia_rows(seed: int) -> list[dict[str, str]]:
    """Admissions with pneumonia in adults 50+, 2022-2025. Patients with S. pneumoniae are less
    often pneumococcal-vaccinated, and with influenza less often flu-vaccinated. One admission in
    twelve has a second pathogen, written as a second row with the same encounter_id."""
    rng = random.Random(f"{seed}:pneumonia")
    patients = [p for p in registry(seed) if _age(p.birth_date, date(2022, 1, 1)) >= 50]
    names = [name for name, _ in PATHOGENS]
    weights = [weight for _, weight in PATHOGENS]
    rows: list[dict[str, str]] = []
    encounter = 900000
    for _ in range(1500):
        patient = rng.choice(patients)
        on = _day(rng, date(2022, 1, 1), date(2025, 12, 31))
        age = _age(patient.birth_date, on)
        found = [rng.choices(names, weights)[0]]
        if found[0] != "None identified" and rng.random() < 1 / 12:
            second = rng.choices(names[1:], weights[1:])[0]
            if second != found[0]:
                found.append(second)
        pneumo = _vaccine(rng, 0.38 if "S. pneumoniae" in found else 0.62)
        flu = _vaccine(rng, 0.30 if "Influenza" in found else 0.55)
        icu = rng.random() < 0.22
        died = rng.random() < (0.16 if icu else 0.06)
        encounter += rng.randint(1, 9)
        for pathogen in found:
            rows.append(
                {
                    "patient_id": patient.patient_id,
                    "encounter_id": f"A{encounter}",
                    "admit_date": on.isoformat(),
                    "admit_year": str(on.year),
                    "age_years": str(age),
                    "sex": patient.sex,
                    "pathogen": pathogen,
                    "pneumococcal_vaccine": pneumo,
                    "flu_vaccine_this_season": flu,
                    "icu_admission": YES if icu else NO,
                    "died_in_hospital": YES if died else NO,
                }
            )
    rows.sort(key=lambda row: (row["admit_date"], row["encounter_id"], row["pathogen"]))
    return rows


PNEUMONIA_DICTIONARY: Final = """\
Lakeside General Hospital (synthetic): pneumonia admissions in adults 50 and older, 2022-2025

One row per admission and pathogen: an admission with two pathogens found has two rows with the
same encounter_id. Count admissions by encounter_id and patients by patient_id.

patient_id               Hospital patient key.
encounter_id             Admission key.
admit_date               Admission date (YYYY-MM-DD).
admit_year               Calendar year of admit_date.
age_years                Age in years at admission.
sex                      female or male.
pathogen                 Pathogen identified (Influenza, S. aureus, S. pneumoniae, RSV,
                         H. influenzae, Pseudomonas, Legionella, or None identified).
pneumococcal_vaccine     yes if any pneumococcal vaccine is documented; no; unknown.
flu_vaccine_this_season  yes if an influenza vaccine is documented for this season; no; unknown.
icu_admission            yes or no.
died_in_hospital         yes or no.
"""


# ---------------------------------------------------------------------------- measles


MEASLES_COLUMNS: Final = [
    "patient_id",
    "visit_id",
    "visit_date",
    "risk_group",
    "age_years",
    "mmr_doses_documented",
    "measles_igg",
    "measles_igg_unit",
]
# Patients per risk group; Childcare worker is the small one.
RISK_GROUPS: Final = {"Healthcare worker": 420, "Pregnant": 300, "International traveler": 160, "Childcare worker": 8}
PROTECTIVE_MIU_PER_ML: Final = 120.0


def measles_rows(seed: int) -> list[dict[str, str]]:
    """Measles immunity checks at employee health, prenatal and travel clinics, 2024-2025: one
    row per check. One lab reports IgG in mIU/mL, the other in IU/mL (1 IU/mL = 1000 mIU/mL).
    One patient in six had a second check (a titer redrawn)."""
    rng = random.Random(f"{seed}:measles")
    adults = [p for p in registry(seed) if 18 <= _age(p.birth_date, date(2024, 1, 1)) <= 70]
    rng.shuffle(adults)
    rows: list[dict[str, str]] = []
    visit = 300000
    taken = 0
    for group, size in RISK_GROUPS.items():
        members = adults[taken : taken + size]
        taken += size
        for patient in members:
            if group == "Pregnant" and patient.sex != "female":
                continue
            doses = rng.choices((0, 1, 2), (0.22, 0.20, 0.58) if group == "Healthcare worker" else (0.33, 0.25, 0.42))[
                0
            ]
            protective = rng.random() < (0.86 if doses == 2 else 0.68)
            n_checks = 2 if rng.random() < 1 / 6 else 1
            for _ in range(n_checks):
                on = _day(rng, date(2024, 1, 1), date(2025, 12, 31))
                tested = rng.random() < 0.85
                level = rng.uniform(150.0, 2500.0) if protective else rng.uniform(5.0, 110.0)
                second_lab = rng.random() < 0.35
                visit += rng.randint(1, 5)
                rows.append(
                    {
                        "patient_id": patient.patient_id,
                        "visit_id": f"M{visit}",
                        "visit_date": on.isoformat(),
                        "risk_group": group,
                        "age_years": str(_age(patient.birth_date, on)),
                        "mmr_doses_documented": str(doses),
                        "measles_igg": (f"{level / 1000:.3f}" if second_lab else f"{level:.0f}") if tested else "",
                        "measles_igg_unit": ("IU/mL" if second_lab else "mIU/mL") if tested else "",
                    }
                )
    rows.sort(key=lambda row: (row["visit_date"], row["visit_id"]))
    return _with_duplicates(rng, rows, 0.02)


MEASLES_DICTIONARY: Final = """\
Lakeside General Hospital (synthetic): measles immunity checks in at-risk groups, 2024-2025

One row per immunity check; a patient whose titer was redrawn has two rows. Some checks appear
twice (duplicate extract rows); count patients by patient_id.

patient_id            Hospital patient key.
visit_id              Visit key.
visit_date            Date of the check (YYYY-MM-DD).
risk_group            Healthcare worker, Pregnant, International traveler, or Childcare worker.
age_years             Age in years at the check.
mmr_doses_documented  MMR doses documented in the record (0, 1 or 2).
measles_igg           Measles IgG level; empty when not drawn. The unit is in measles_igg_unit.
measles_igg_unit      mIU/mL or IU/mL (1 IU/mL = 1000 mIU/mL); the two labs report differently.

Evidence of immunity for this project: two documented MMR doses, or a measles IgG of at least
120 mIU/mL on any check.
"""


# ---------------------------------------------------------------------------- entry points


_TABLES: Final[dict[Extract, tuple[Callable[[int], list[dict[str, str]]], list[str], str]]] = {
    "cdiff": (cdiff_rows, CDIFF_COLUMNS, CDIFF_DICTIONARY),
    "hepb": (hepb_rows, HEPB_COLUMNS, HEPB_DICTIONARY),
    "pneumonia": (pneumonia_rows, PNEUMONIA_COLUMNS, PNEUMONIA_DICTIONARY),
    "measles": (measles_rows, MEASLES_COLUMNS, MEASLES_DICTIONARY),
}


def generate(extract: str, part: str, seed: int = DEFAULT_SEED) -> bytes:
    """The bytes of one extract's table (CSV) or data dictionary (plain text).

    Raises:
        SyntheticDataError: ``extract`` or ``part`` is not known.
    """
    if extract not in _TABLES:
        raise SyntheticDataError(f"unknown synthetic extract {extract!r}; use one of {', '.join(EXTRACTS)}")
    rows_of, columns, dictionary = _TABLES[extract]  # type: ignore[index]
    if part == "table":
        return _csv(rows_of(seed), columns)
    if part == "dictionary":
        return dictionary.encode("utf-8")
    raise SyntheticDataError(f"unknown part {part!r}; use table or dictionary")


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) not in (2, 3):
        print("usage: synthetic_hospital EXTRACT table|dictionary [SEED]", file=sys.stderr)
        return 2
    try:
        body = generate(args[0], args[1], int(args[2]) if len(args) == 3 else DEFAULT_SEED)
    except (SyntheticDataError, ValueError) as error:
        print(f"synthetic_hospital: {error}", file=sys.stderr)
        return 2
    sys.stdout.buffer.write(body)
    return 0


if __name__ == "__main__":
    sys.exit(main())
