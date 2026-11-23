# SPDX-License-Identifier: MIT
"""Synthetic tables for the model-written query tests. No real data, and no name, shape or
example from any evaluation dataset: every domain and column name here is invented.

Each is generated from the row number, so the tests' Polars oracles recompute every expected
value independently of the SQL the compiler writes.

``shots``: allergen-shot visits (patients repeat, a round numbers each patient's visits, a dated
visit), with a data dictionary that labels its codes. ``cohort``: one row per patient.
``ledger``: lines that no column names a patient. ``vials``: a tally of discarded vials (a table
of groups with a count column). ``coverage_estimates``: published estimates.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

import polars as pl

SHOT_PATIENTS = 40
SHOTS_DICTIONARY = """\
ALLERGEN IMMUNOTHERAPY LOG
One row per shot visit; holder_ref is the patient.

holder_ref     Patient reference number
shot_round     Shot visit number within the patient (1 = first shot visit)
shot_date       Date the shot was given
years_at_shot  Age in years at the shot visit
sex            F = female, M = male
ige_kul        Specific IgE level, kU/L
wheeze         Wheeze at the shot visit (Y/N)
extract        Allergen extract: 1 = grass pollen, 2 = dust mite, 3 = bee venom
"""


def shots() -> pl.DataFrame:
    """40 patients with 1-4 shot visits each; wheeze and the extract change between some visits."""
    rows = []
    for p in range(SHOT_PATIENTS):
        for v in range(1, 1 + 1 + p % 4):
            rows.append(
                {
                    "holder_ref": f"H{p:03d}",
                    "shot_round": v,
                    "shot_date": date(2022, 1, 3) + timedelta(days=60 * v + p),
                    "years_at_shot": 45 + p % 30 + v // 3,
                    "sex": "F" if p % 3 else "M",
                    "ige_kul": float(120 + (p * 7 + v * 11) % 50) if (p + v) % 9 else None,
                    "wheeze": "Y" if (p + v) % 4 == 0 else "N",
                    "extract": 1 + (p + (v if p % 5 == 0 else 0)) % 3,
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


def vials() -> pl.DataFrame:
    """A tally of discarded vaccine vials per depot, stock round (1-60) and vial kind. No person."""
    rows = []
    for d, depot in enumerate(("alder", "birch", "cedar", "dunes")):
        for round_no in range(1, 61):
            for k, kind in enumerate(("single", "multi")):
                rows.append(
                    {
                        "depot": depot,
                        "round_no": round_no,
                        "vial_kind": kind,
                        "discard_count": (d * 7 + round_no * 3 + k * 5) % 40,
                    }
                )
    return pl.DataFrame(rows)


def coverage_estimates() -> pl.DataFrame:
    """Published estimates: vaccination coverage per region and year (never added up)."""
    return pl.DataFrame(
        {
            "region": [r for r in ("north", "south") for _ in range(2015, 2025)],
            "year": [y for _ in ("north", "south") for y in range(2015, 2025)],
            "coverage_pct": [round(80 + ((r * 3 + y) % 15) * 1.1, 1) for r in range(2) for y in range(2015, 2025)],
        }
    )


def frequency_marks(frame: pl.DataFrame, count_column: str = "discard_count") -> dict[str, Any]:
    """The contract's marks for a table of groups with a count column (``aggregate``), set as an
    answered contract holds them, so the tests never depend on how inference names it."""
    from clinical_analytics.contracts.semantics import AggregateInfo

    aggregate = AggregateInfo(kind="frequency", count_column=count_column, source="declared")
    return {"aggregate": aggregate, "case_count_column": count_column, "patient_key": None, "open_questions": ()}
