"""Over-holding regression probes: questions the M2 gate used to hold that have one answer.

Each answer probe was held or refused before the over-holding fixes and must now be answered
with the value computed here with Polars from the question's plain meaning. Each hold probe
is a case where loosening the gate could produce a wrong answer (a stored number bound by a
label digit, a reversed proportion, a dropped filter, statin non-users read as users, an
ambiguous number or list); it must stay held or refused.

Three frames: the committed synthetic cohort, the second review's generated frame
(adversarial_probes.own_frame) and a small seeded frame with a Creatinine column holding 2.2
next to a diabetes type column holding "Type 2".
"""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import polars as pl

c = pl.col
STATIN = "Statin use 1: Yes 2: No"
DEXA = "DEXA result 1: Normal 2: Osteopenia 3: Osteoporosis"
ALL = pl.lit(True)

Truth = Callable[[pl.DataFrame], Any]


@dataclass(frozen=True)
class Probe:
    dataset: str  # "coh", "own" or "creat"
    query: str
    truth: Truth | None  # None: must hold or refuse
    group_by: str | None = None
    stat: str | None = None


def count(where: pl.Expr) -> Truth:
    return lambda frame: frame.filter(where).height


def stat(column: str, how: str, where: pl.Expr = ALL) -> Truth:
    return lambda frame: float(getattr(frame.filter(where).get_column(column).drop_nulls(), how)())


def share(event: pl.Expr, among: pl.Expr = ALL) -> Truth:
    def compute(frame: pl.DataFrame) -> float:
        cohort = frame.filter(among)
        return cohort.filter(event).height / cohort.height

    return compute


def by(group: str, value: Callable[[pl.DataFrame], Any], where: pl.Expr = ALL) -> Truth:
    def compute(frame: pl.DataFrame) -> dict[str, Any]:
        return {str(key[0]): value(rows) for key, rows in frame.filter(where).group_by(group, maintain_order=True)}

    return compute


def size(rows: pl.DataFrame) -> int:
    return rows.height


def measure(column: str, how: str) -> Callable[[pl.DataFrame], float]:
    return lambda rows: float(getattr(rows.get_column(column).drop_nulls(), how)())


def rate(event: pl.Expr) -> Callable[[pl.DataFrame], float]:
    return lambda rows: rows.filter(event).height / rows.height


def creatinine_frame(size_: int = 300) -> pl.DataFrame:
    """Seeded frame where a label digit ("Type 2") and a stored number (Creatinine 2.2) meet."""
    rng = random.Random(11)
    return pl.DataFrame(
        {
            "patient_id": [f"C{i:04d}" for i in range(size_)],
            "age": [rng.randint(30, 90) for _ in range(size_)],
            "sex": [rng.choice(["F", "M"]) for _ in range(size_)],
            "diabetes_type": [rng.choice(["Type 1", "Type 2", "None"]) for _ in range(size_)],
            "Creatinine": [rng.choice([0.8, 1.0, 1.4, 2.2, 3.1]) for _ in range(size_)],
            STATIN: [rng.choice([1, 2]) for _ in range(size_)],
        }
    )


S = c(STATIN)
OWN_STATIN = c("statin")
READMITTED = c("readmitted") == "yes"

ANSWERED: tuple[Probe, ...] = (
    # Grouped comparison computes the asked statistic (median, maximum) per group.
    Probe("coh", "compare median ldl by sex", by("sex", measure("ldl", "median")), group_by="sex"),
    Probe(
        "coh",
        "compare median ldl between the north and the west",
        by("site", measure("ldl", "median"), c("site").is_in(["North", "West"])),
        group_by="site",
    ),
    Probe("coh", "compare maximum bmi by site", by("site", measure("bmi", "max")), group_by="site"),
    # Comparison groups named by their values.
    Probe("coh", "compare ldl between women and men", by("sex", measure("ldl", "mean")), group_by="sex"),
    Probe(
        "coh",
        "compare ldl between the north and the south",
        by("site", measure("ldl", "mean"), c("site").is_in(["North", "South"])),
        group_by="site",
    ),
    Probe(
        "coh",
        "compare ldl by treatment in women",
        by("treatment", measure("ldl", "mean"), c("sex") == "F"),
        group_by="treatment",
    ),
    Probe("coh", "how many patients in each site?", by("site", size), group_by="site"),
    # 0/1 indicator columns and inflections of their names.
    Probe("coh", "how many diabetic patients?", count(c("diabetes") == 1)),
    Probe("coh", "how many diabetics are women?", count((c("diabetes") == 1) & (c("sex") == "F"))),
    Probe("coh", "what is the average ldl of diabetics?", stat("ldl", "mean", c("diabetes") == 1), stat="mean"),
    Probe("coh", "what proportion of women are diabetic?", share(c("diabetes") == 1, c("sex") == "F")),
    Probe("coh", "how many patients without diabetes?", count(c("diabetes") == 0)),
    Probe(
        "coh",
        "what is the average ldl in non-diabetic patients?",
        stat("ldl", "mean", c("diabetes") == 0),
        stat="mean",
    ),
    Probe("coh", "how many patients had the outcome?", count(c("outcome") == 1)),
    Probe("coh", "how many patients with no event?", count(c("event") == 0)),
    Probe("coh", "how many events in women?", count((c("event") == 1) & (c("sex") == "F"))),
    # Negation, "non-users", and lists joined by "or".
    Probe("coh", "average age of statin non-users", stat("age", "mean", S == 2), stat="mean"),
    Probe("coh", "how many statin non-users are women?", count((S == 2) & (c("sex") == "F"))),
    Probe("coh", "how many non users of statin?", count(S == 2)),
    Probe("coh", "how many patients not on placebo?", count(c("treatment") != "placebo")),
    Probe("coh", "how many patients excluding women?", count(c("sex") != "F")),
    Probe("coh", "how many patients other than men?", count(c("sex") != "M")),
    Probe("coh", "how many patients outside the west?", count(c("site") != "West")),
    Probe("coh", "how many patients except those in the west?", count(c("site") != "West")),
    Probe("coh", "how many patients without a normal dexa result?", count(c(DEXA) != 1)),
    Probe("coh", "how many patients in the north or west?", count(c("site").is_in(["North", "West"]))),
    Probe(
        "coh",
        "how many women in the north or west on drug?",
        count((c("sex") == "F") & c("site").is_in(["North", "West"]) & (c("treatment") == "drug")),
    ),
    Probe("coh", "how many patients on drug or placebo?", count(c("treatment").is_in(["drug", "placebo"]))),
    # The measure ends where a limit starts; value inflections.
    Probe("coh", "average ldl within the south site", stat("ldl", "mean", c("site") == "South"), stat="mean"),
    # Ranges and negated comparisons.
    Probe("coh", "how many patients between 50 and 65?", count(c("age").is_between(50, 65))),
    Probe("coh", "how many patients aged 40 to 60?", count(c("age").is_between(40, 60))),
    Probe("coh", "how many patients with bmi between 25 and 30?", count(c("bmi").is_between(25, 30))),
    Probe(
        "coh",
        "average age of patients older than 50 and younger than 70",
        stat("age", "mean", (c("age") > 50) & (c("age") < 70)),
        stat="mean",
    ),
    Probe("coh", "how many patients who are at least 65?", count(c("age") >= 65)),
    Probe("coh", "how many patients not over 65?", count(c("age") <= 65)),
    Probe(
        "coh",
        "how many patients in the north on statin not over 65?",
        count((c("site") == "North") & (S == 1) & (c("age") <= 65)),
    ),
    # Units and stems after a number.
    Probe("coh", "how many patients followed up over 365 days?", count(c("followup_days") > 365)),
    Probe("coh", "how many patients older than 65 years?", count(c("age") > 65)),
    # Rates: groups before "more likely", the rate after it.
    Probe("coh", "are men more likely to use statins than women?", by("sex", rate(S == 1)), group_by="sex"),
    Probe(
        "coh",
        "Are women more likely than men to have diabetes?",
        by("sex", rate(c("diabetes") == 1)),
        group_by="sex",
    ),
    Probe(
        "coh",
        "Are patients on placebo more likely to have the outcome?",
        by("treatment", rate(c("outcome") == 1)),
        group_by="treatment",
    ),
    Probe("coh", "are diabetics more likely to use statins?", by("diabetes", rate(S == 1)), group_by="diabetes"),
    Probe("coh", "are statin users more likely to be diabetic?", by(STATIN, rate(c("diabetes") == 1)), group_by=STATIN),
    # The reviewer's frame: labels with digits or letters, absence labels, units.
    Probe("own", "how many patients with type 2 diabetes?", count(c("diabetes_type") == "Type 2")),
    Probe(
        "own",
        "average hba1c in type 1 diabetes",
        stat("hba1c", "mean", c("diabetes_type") == "Type 1"),
        stat="mean",
    ),
    Probe("own", "how many patients on drug A?", count(c("arm") == "drug A")),
    Probe("own", "how many patients not on placebo?", count(c("arm") != "placebo")),
    Probe("own", "how many diabetic patients?", count(c("diabetes_type") != "None")),
    Probe("own", "how many patients without diabetes?", count(c("diabetes_type") == "None")),
    Probe("own", "how many patients on statins?", count(OWN_STATIN != "none")),
    Probe("own", "how many patients not on a statin?", count(OWN_STATIN == "none")),
    Probe("own", "how many patients over 100 kg?", count(c("Weight (kg)") > 100)),
    Probe("own", "how many patients with ldl above 160 mg/dl?", count(c("LDL mg/dL") > 160)),
    Probe(
        "own",
        "average sbp in hypertensive patients",
        stat("SBP mmHg", "mean", c("Hypertension") == "Yes"),
        stat="mean",
    ),
    Probe("own", "how many female smokers?", count((c("Smoker 1: Yes 2: No") == 1) & (c("Gender") == "Female"))),
    Probe(
        "own",
        "are smokers more likely to be readmitted?",
        by("Smoker 1: Yes 2: No", rate(READMITTED)),
        group_by="Smoker 1: Yes 2: No",
    ),
    Probe("own", "are men more likely to be readmitted?", by("Gender", rate(READMITTED)), group_by="Gender"),
    # A label digit binds the label, never a stored number of another column.
    Probe("creat", "how many patients with type 2 diabetes?", count(c("diabetes_type") == "Type 2")),
)

HELD: tuple[Probe, ...] = (
    # "type 2 diabetes" must never become Creatinine = 2.2.
    Probe("creat", "how many patients with creatinine 2.2 type diabetes?", None),
    Probe("creat", "how many patients with type 2.2 diabetes?", None),
    # A reversed rate: the rate is what follows "more likely".
    Probe("coh", "are statin users more likely to be women?", None),
    Probe("coh", "are statin users more likely to be women than men?", None),
    # An ambiguous number, list, or range pair.
    Probe("coh", "what is the average bmi in patients over 30?", None),
    Probe("coh", "median ldl over 70?", None),
    Probe("coh", "how many women and men?", None),
    Probe("coh", "how many patients in North and South sites?", None),
    Probe("coh", "how many patients under 40 or over 80?", None),
    Probe("coh", "how many patients not in the north or west?", None),
    # A negation before two items may cover both: "not (on placebo with diabetes)".
    Probe("coh", "how many patients not on placebo with diabetes?", None),
    Probe("own", "how many patients on a drug?", None),
    Probe("own", "what proportion of women are on statins?", None),
    # A grouping by a measurement (one group per exact LDL value) identifies patients.
    Probe("coh", "compare treatment between ldl groups", None),
    Probe("coh", "how many patients by ldl?", None),
    Probe("coh", "average age by bmi", None),
    Probe("coh", "compare ldl by age", None),
)
