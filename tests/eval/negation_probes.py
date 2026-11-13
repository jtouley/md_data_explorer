"""Adversarial regression set for the gate: negation, ranges, "or" lists, 0/1 columns, absence
labels, "per X" groupings and numeric-coded categories.

Every probe states its expected outcome: the exact limits the plan must apply (and the value
the app must then show, computed here with Polars), or a hold. The gate does not infer meaning,
so a hold is expected wherever the words need interpreting: any negation ("not", "without",
"excluding", "non", "never"), an "or" list, a 0/1 column with no codebook, a text column with
an absence label ("none", "Never"), a lone "per X" grouping, a reversed range, and a grouping
by many integer codes (which the user confirms). The target is 0 wrong.

Every question from both review rounds of #93 is here, and must hold or refuse. PLAN_PROBES
pins the wrong plans those reviews built: the gate must hold each one.

Frames: the committed synthetic cohort ("coh"), the reviewer's generated frame ("own",
adversarial_probes.own_frame), a seeded frame with integer-coded columns ("codes") and a
seeded frame with the second review's columns ("rev": statin with "none", smoking
Current/Former/Never, 0/1 outcome, repeated subject ids).
"""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypeAlias

import polars as pl

c = pl.col
STATIN = "Statin use 1: Yes 2: No"
S = c(STATIN)
SMOKER = "Smoker 1: Yes 2: No"
ALL = pl.lit(True)

Truth = Callable[[pl.DataFrame], Any]
Limit = tuple[str, str, Any]


@dataclass(frozen=True)
class ScopeProbe:
    dataset: str  # "coh", "own", "codes" or "rev"
    query: str
    limits: tuple[Limit, ...] | None  # None: must hold or refuse
    truth: Truth | None = None


def hold(dataset: str, query: str) -> ScopeProbe:
    return ScopeProbe(dataset, query, None)


def count(where: pl.Expr) -> Truth:
    return lambda frame: frame.filter(where).height


def mean(column: str, where: pl.Expr = ALL) -> Truth:
    return lambda frame: float(frame.filter(where).get_column(column).drop_nulls().mean())  # type: ignore[arg-type]


def share(event: pl.Expr, among: pl.Expr = ALL) -> Truth:
    def compute(frame: pl.DataFrame) -> float:
        cohort = frame.filter(among)
        return cohort.filter(event).height / cohort.height

    return compute


def codes_frame(size: int = 300) -> pl.DataFrame:
    """Seeded frame whose site and zip columns are integer codes with no header codebook."""
    rng = random.Random(13)
    return pl.DataFrame(
        {
            "patient_id": [f"K{i:04d}" for i in range(size)],
            "site_id": [rng.randint(1, 12) for _ in range(size)],
            "zip": [rng.choice([10001 + 7 * step for step in range(30)]) for _ in range(size)],
            "age": [rng.randint(30, 90) for _ in range(size)],
            "sex": [rng.choice(["F", "M"]) for _ in range(size)],
            "ldl": [round(rng.uniform(60, 200), 1) for _ in range(size)],
            "smoker": [rng.choice([0, 1]) for _ in range(size)],
        }
    )


def review_frame(size: int = 300) -> pl.DataFrame:
    """Seeded frame with the columns of the second review: absence labels, a 0/1 outcome with no
    codebook, smoking recorded as Current/Former/Never, and three rows per subject."""
    rng = random.Random(17)
    return pl.DataFrame(
        {
            "patient_id": [f"R{i:04d}" for i in range(size)],
            "subject_id": [f"S{i // 3:04d}" for i in range(size)],
            "age": [rng.randint(30, 90) for _ in range(size)],
            "sex": [rng.choice(["F", "M"]) for _ in range(size)],
            "treatment": [rng.choice(["placebo", "drug A"]) for _ in range(size)],
            "site": [rng.choice(["North", "South", "West"]) for _ in range(size)],
            "statin": [rng.choice(["atorvastatin", "simvastatin", "none"]) for _ in range(size)],
            "smoking": [rng.choice(["Current", "Former", "Never"]) for _ in range(size)],
            "outcome": [rng.choice([0, 1]) for _ in range(size)],
            "diabetes": [rng.choice([0, 1]) for _ in range(size)],
            "ldl": [round(rng.uniform(60, 200), 1) for _ in range(size)],
            "events": [rng.randint(0, 4) for _ in range(size)],
        }
    )


def _limits(*items: Limit) -> tuple[Limit, ...]:
    """Expected limits in the order headless.plan_limits writes them."""

    def norm(value: Any) -> Any:
        if isinstance(value, list | tuple):
            return tuple(sorted((norm(item) for item in value), key=repr))
        if isinstance(value, int | float) and not isinstance(value, bool):
            return float(value)
        return value

    return tuple(sorted(((op, column, norm(value)) for op, column, value in items), key=repr))


PROBES: tuple[ScopeProbe, ...] = (
    # ------------------------------------------------------------------
    # Negation x coordination ("and" / "or")
    # ------------------------------------------------------------------
    hold("own", "how many patients without diabetes and hypertension?"),
    hold("own", "how many patients without diabetes or hypertension?"),
    hold("coh", "how many patients excluding women on placebo?"),
    hold("coh", "how many patients excluding women and men?"),
    hold("coh", "how many patients without diabetes and on placebo?"),
    hold("coh", "how many patients without diabetes on placebo?"),
    hold("coh", "how many patients not on placebo with diabetes?"),
    hold("coh", "how many patients not on placebo or drug?"),
    hold("coh", "how many patients not in the north or west?"),
    hold("coh", "how many patients excluding the north and the west?"),
    hold("coh", "how many patients other than women on drug?"),
    hold("coh", "how many patients outside the west and the south?"),
    hold("coh", "how many patients are not in the west and on drug?"),
    # A negation right before its item still holds: no check reads it as "every other value".
    hold("own", "how many patients not on placebo and without hypertension?"),
    hold("coh", "how many non-diabetic women?"),
    hold("coh", "how many non-diabetic patients on placebo?"),
    hold("coh", "how many women without diabetes?"),
    hold("coh", "how many women not on placebo?"),
    hold("coh", "how many patients on placebo without diabetes?"),
    hold("coh", "how many statin non-users on placebo?"),
    hold("own", "how many smokers without hypertension?"),
    hold("coh", "how many patients without diabetes?"),
    # ------------------------------------------------------------------
    # Negation x comparison
    # ------------------------------------------------------------------
    hold("coh", "how many patients excluding those with ldl over 160?"),
    hold("coh", "how many patients without ldl over 160?"),
    hold("coh", "how many patients without diabetes over 65?"),
    hold("coh", "how many patients not on placebo over 65?"),
    hold("coh", "how many patients not between 50 and 65?"),
    hold("coh", "how many patients excluding those aged 40-60?"),
    hold("coh", "how many patients excluding women over 65?"),
    # "not over 65" is never read as "65 or under": every negated comparison holds.
    hold("coh", "how many patients exclude patients over 65?"),
    hold("coh", "how many patients not over 65?"),
    hold("coh", "how many patients not under 40?"),
    hold("coh", "how many patients excluding those over 65?"),
    hold("coh", "how many patients over 65 without diabetes?"),
    hold("coh", "how many women not over 65?"),
    hold("coh", "how many patients over 65 not on placebo?"),
    # ------------------------------------------------------------------
    # Ranges: hyphenated, reversed, open-ended
    # ------------------------------------------------------------------
    ScopeProbe(
        "coh",
        "how many patients aged 40-60?",
        _limits(("ge", "age", 40), ("le", "age", 60)),
        count(c("age").is_between(40, 60)),
    ),
    ScopeProbe(
        "coh",
        "how many patients aged 40 - 60?",
        _limits(("ge", "age", 40), ("le", "age", 60)),
        count(c("age").is_between(40, 60)),
    ),
    ScopeProbe(
        "coh",
        "how many patients aged 40 to 60?",
        _limits(("ge", "age", 40), ("le", "age", 60)),
        count(c("age").is_between(40, 60)),
    ),
    ScopeProbe(
        "coh",
        "how many patients between 50 and 65?",
        _limits(("ge", "age", 50), ("le", "age", 65)),
        count(c("age").is_between(50, 65)),
    ),
    ScopeProbe(
        "coh",
        "how many patients with bmi between 25 and 30?",
        _limits(("ge", "bmi", 25), ("le", "bmi", 30)),
        count(c("bmi").is_between(25, 30)),
    ),
    ScopeProbe(
        "coh",
        "how many patients aged 40-60 on placebo?",
        _limits(("ge", "age", 40), ("le", "age", 60), ("eq", "treatment", "placebo")),
        count(c("age").is_between(40, 60) & (c("treatment") == "placebo")),
    ),
    hold("coh", "how many patients between 65 and 50?"),
    hold("coh", "how many patients aged 60-40?"),
    hold("coh", "how many patients aged 60 to 40?"),
    hold("coh", "how many patients with bmi between 30 and 25?"),
    hold("coh", "how many patients under 40 or over 80?"),
    ScopeProbe("coh", "how many patients 65 or older?", _limits(("ge", "age", 65)), count(c("age") >= 65)),
    ScopeProbe("coh", "how many patients 65+?", _limits(("ge", "age", 65)), count(c("age") >= 65)),
    ScopeProbe("coh", "how many patients who are at least 65?", _limits(("ge", "age", 65)), count(c("age") >= 65)),
    ScopeProbe("coh", "how many patients over 65?", _limits(("gt", "age", 65)), count(c("age") > 65)),
    ScopeProbe("coh", "how many patients under 40?", _limits(("lt", "age", 40)), count(c("age") < 40)),
    # ------------------------------------------------------------------
    # "or" lists in filters, shares and rates: never bound as a list
    # ------------------------------------------------------------------
    hold("coh", "how many patients in the north or west?"),
    hold("coh", "how many women in the north or south?"),
    hold("coh", "how many patients on drug or placebo?"),
    hold("coh", "what share of patients are in the north or west?"),
    hold("coh", "what proportion of women are in the north or west?"),
    hold("coh", "what percentage of patients are on drug or placebo?"),
    hold("own", "are smokers more likely to be in the north or west?"),
    hold("coh", "are women more likely to be on drug or placebo?"),
    hold("coh", "are patients in the north or west more likely to have diabetes?"),
    hold("coh", "how many patients not on drug or placebo?"),
    hold("coh", "how many patients in the north and south?"),
    # ------------------------------------------------------------------
    # Integer-coded categories: confirmed, never refused, never guessed
    # ------------------------------------------------------------------
    hold("codes", "count patients by site_id"),
    hold("codes", "how many patients by zip?"),
    hold("codes", "average ldl by site_id"),
    hold("codes", "compare ldl by site_id"),
    hold("codes", "how many patients at site 3?"),
    hold("codes", "how many patients by ldl?"),
    # "site_id" splits into "site" and "id"; the filter parser binds the number to patient_id,
    # and the unused word "site" holds that plan.
    hold("codes", "how many patients with site_id over 6?"),
    # A 0/1 column with no codebook is not read as yes/no.
    hold("codes", "how many smokers?"),
    # ------------------------------------------------------------------
    # Second review round: each ran a wrong answer without a hold
    # ------------------------------------------------------------------
    hold("rev", "how many patients not on statins?"),
    hold("own", "how many patients not on statins?"),
    hold("rev", "how many patients on statins?"),
    hold("rev", "average outcome by site"),
    hold("coh", "average outcome by site"),
    hold("rev", "how many patients who are smokers?"),
    hold("rev", "how many never smokers?"),
    hold("rev", "how many patients on placebo without outcome?"),
    hold("coh", "how many patients on placebo without outcome?"),
    hold("own", "drug A non-users"),
    hold("own", "how many drug A non-users?"),
    hold("rev", "how many events per subject?"),
    hold("rev", "how many patients per subject?"),
    hold("coh", "compare ldl in diabetics over 65 between women and men"),
    hold("rev", "compare ldl in diabetics over 65 between women and men"),
    # "diabetics over 65": the cutoff names no measure, never "diabetes > 65".
    hold("coh", "how many diabetics over 65?"),
)

# The wrong plans the review rounds built, as (dataset, question, plan). The gate must hold each:
# unused_words or unjustified_elements is not empty. Plans are built in the test.
PlanCase: TypeAlias = tuple[str, str, dict[str, Any]]
PLAN_PROBES: tuple[PlanCase, ...] = (
    ("rev", "how many patients not on statins?", {"steps": [("ne", "statin", "none")]}),
    ("rev", "how many patients not on statins?", {"filters": [("!=", "statin", "none")]}),
    ("rev", "average outcome by site", {"intent": "DESCRIBE", "metric": "outcome", "group_by": "site",
                                        "statistic": "mean", "steps": [("eq", "outcome", 1)]}),
    ("rev", "how many patients who are smokers?", {"steps": [("ne", "smoking", "Never")]}),
    ("rev", "how many patients who are smokers?", {"filters": [("!=", "smoking", "Never")]}),
    ("rev", "how many patients on placebo without outcome?",
     {"steps": [("ne", "treatment", "placebo"), ("eq", "outcome", 0)]}),
    ("rev", "how many patients on placebo without outcome?",
     {"filters": [("!=", "treatment", "placebo"), ("==", "outcome", 0)]}),
    ("rev", "how many women on placebo?", {"steps": [("ne", "sex", "F"), ("eq", "treatment", "placebo")]}),
    ("rev", "how many women on placebo?", {"filters": [("!=", "sex", "F"), ("==", "treatment", "placebo")]}),
    ("own", "how many drug A non-users?", {"steps": [("eq", "arm", "drug A")]}),
    ("rev", "how many events per subject?", {"group_by": "subject_id"}),
    ("own", "how many patients without diabetes and hypertension?",
     {"steps": [("eq", "diabetes_type", "None"), ("eq", "Hypertension", True)]}),
    ("coh", "how many patients excluding women on placebo?",
     {"filters": [("!=", "sex", "F"), ("==", "treatment", "placebo")]}),
    ("coh", "how many patients excluding those with ldl over 160?", {"steps": [("gt", "ldl", 160.0)]}),
    ("coh", "how many patients excluding those with ldl over 160?", {"steps": [("le", "ldl", 160.0)]}),
    ("coh", "how many patients not over 65?", {"steps": [("le", "age", 65.0)]}),
    ("coh", "how many patients between 65 and 50?", {"steps": [("ge", "age", 65.0), ("le", "age", 50.0)]}),
    ("coh", "how many patients between 65 and 50?", {"steps": [("ge", "age", 50.0), ("le", "age", 65.0)]}),
    ("coh", "what share of patients are in the north or west?",
     {"proportion": ("site", ["North", "West"])}),
    ("coh", "how many patients in the north or west?", {"steps": [("in", "site", ["North", "West"])]}),
    ("own", "are smokers more likely to be in the north or west?",
     {"steps": [("in", "Region", ["North", "West"])], "group_by": "Smoker 1: Yes 2: No"}),
)  # fmt: skip
