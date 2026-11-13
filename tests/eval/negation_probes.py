"""Adversarial regression set for the gate's second review: negation scope, ranges, "or" lists
and numeric-coded categories.

Every probe states its expected outcome: the exact limits the plan must apply (and the value
the app must then show, computed here with Polars), or a hold. A hold is expected wherever the
words leave the meaning open: a negation before two coordinated items, a negation that may or
may not reach a comparison, a reversed range, an "or" list after a share verb or "more likely",
and a grouping by many integer codes (which the user confirms). The target is 0 wrong.

Frames: the committed synthetic cohort ("coh"), the reviewer's generated frame ("own",
adversarial_probes.own_frame) and a seeded frame with integer-coded columns ("codes").
"""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

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
    dataset: str  # "coh", "own" or "codes"
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
    # Two negations, each right before its own item: both apply, nothing is in doubt. The CSV
    # reader stores Yes/No as booleans, so "without hypertension" is Hypertension = False.
    ScopeProbe(
        "own",
        "how many patients not on placebo and without hypertension?",
        _limits(("ne", "arm", "placebo"), ("eq", "Hypertension", False)),
        count((c("arm") != "placebo") & (c("Hypertension") == "No")),
    ),
    ScopeProbe(
        "coh",
        "how many non-diabetic women?",
        _limits(("eq", "diabetes", 0), ("eq", "sex", "F")),
        count((c("diabetes") == 0) & (c("sex") == "F")),
    ),  # fmt: skip
    ScopeProbe(
        "coh",
        "how many non-diabetic patients on placebo?",
        _limits(("eq", "diabetes", 0), ("eq", "treatment", "placebo")),
        count((c("diabetes") == 0) & (c("treatment") == "placebo")),
    ),
    ScopeProbe(
        "coh",
        "how many women without diabetes?",
        _limits(("eq", "diabetes", 0), ("eq", "sex", "F")),
        count((c("diabetes") == 0) & (c("sex") == "F")),
    ),
    ScopeProbe(
        "coh",
        "how many women not on placebo?",
        _limits(("ne", "treatment", "placebo"), ("eq", "sex", "F")),
        count((c("treatment") != "placebo") & (c("sex") == "F")),
    ),
    ScopeProbe(
        "coh",
        "how many patients on placebo without diabetes?",
        _limits(("eq", "treatment", "placebo"), ("eq", "diabetes", 0)),
        count((c("treatment") == "placebo") & (c("diabetes") == 0)),
    ),
    ScopeProbe(
        "coh",
        "how many statin non-users on placebo?",
        _limits(("eq", STATIN, 2), ("eq", "treatment", "placebo")),
        count((S == 2) & (c("treatment") == "placebo")),
    ),
    ScopeProbe(
        "own",
        "how many smokers without hypertension?",
        _limits(("eq", SMOKER, 1), ("eq", "Hypertension", False)),
        count((c(SMOKER) == 1) & (c("Hypertension") == "No")),
    ),
    ScopeProbe(
        "coh",
        "how many patients without diabetes?",
        _limits(("eq", "diabetes", 0)),
        count(c("diabetes") == 0),
    ),
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
    # The peel and the gate agree on "age <= 65", but the filter parser also reads "exclude
    # patients" as patient_id != "patients": that invalid plan is refused, never offered.
    hold("coh", "how many patients exclude patients over 65?"),
    ScopeProbe("coh", "how many patients not over 65?", _limits(("le", "age", 65)), count(c("age") <= 65)),
    ScopeProbe("coh", "how many patients not under 40?", _limits(("ge", "age", 40)), count(c("age") >= 40)),
    ScopeProbe("coh", "how many patients excluding those over 65?", _limits(("le", "age", 65)), count(c("age") <= 65)),
    ScopeProbe(
        "coh",
        "how many patients over 65 without diabetes?",
        _limits(("gt", "age", 65), ("eq", "diabetes", 0)),
        count((c("age") > 65) & (c("diabetes") == 0)),
    ),
    ScopeProbe(
        "coh",
        "how many women not over 65?",
        _limits(("eq", "sex", "F"), ("le", "age", 65)),
        count((c("sex") == "F") & (c("age") <= 65)),
    ),
    ScopeProbe(
        "coh",
        "how many patients over 65 not on placebo?",
        _limits(("gt", "age", 65), ("ne", "treatment", "placebo")),
        count((c("age") > 65) & (c("treatment") != "placebo")),
    ),
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
    # "or" lists in filters, shares and rates
    # ------------------------------------------------------------------
    ScopeProbe(
        "coh",
        "how many patients in the north or west?",
        _limits(("in", "site", ["North", "West"])),
        count(c("site").is_in(["North", "West"])),
    ),
    ScopeProbe(
        "coh",
        "how many women in the north or south?",
        _limits(("in", "site", ["North", "South"]), ("eq", "sex", "F")),
        count(c("site").is_in(["North", "South"]) & (c("sex") == "F")),
    ),
    ScopeProbe(
        "coh",
        "how many patients on drug or placebo?",
        _limits(("in", "treatment", ["drug", "placebo"])),
        count(c("treatment").is_in(["drug", "placebo"])),
    ),
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
    ScopeProbe("codes", "how many smokers?", _limits(("eq", "smoker", 1)), count(c("smoker") == 1)),
)
