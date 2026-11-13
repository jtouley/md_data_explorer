"""Reviewer probe questions on the synthetic cohort, with independently computed truth.

Phrasings outside the golden set, collected by an independent review of PR #60 (run on the
base branch and on #60). Each probe either has a truth computed here with Polars, written
from the question's plain meaning and not from any production definition, or is HOLD: no
single correct answer exists (ambiguous, or the data has no such column), so any success is
wrong. For a probe with a truth the system may answer it correctly or hold/refuse it; it
must never succeed with a different value.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import polars as pl

c = pl.col
WOMEN = c("sex") == "F"
MEN = c("sex") == "M"
DRUG = c("treatment") == "drug"
PLACEBO = c("treatment") == "placebo"
STATIN = c("Statin use 1: Yes 2: No") == 1
NO_STATIN = c("Statin use 1: Yes 2: No") == 2
DEXA = "DEXA result 1: Normal 2: Osteopenia 3: Osteoporosis"
ALL = pl.lit(True)


@dataclass(frozen=True)
class Probe:
    query: str
    kind: str  # count, stat, proportion, rates, count_by, mean_by, or hold
    truth: Callable[[pl.DataFrame], Any] | None = None
    stat: str | None = None
    group_by: str | None = None


def _count(where: pl.Expr) -> Callable[[pl.DataFrame], Any]:
    return lambda frame: frame.filter(where).height


def _stat(column: str, how: str, where: pl.Expr = ALL) -> Callable[[pl.DataFrame], Any]:
    def compute(frame: pl.DataFrame) -> float:
        series = frame.filter(where).get_column(column).drop_nulls()
        return float(getattr(series, how)())

    return compute


def _share(event: pl.Expr, among: pl.Expr = ALL) -> Callable[[pl.DataFrame], Any]:
    def compute(frame: pl.DataFrame) -> float:
        cohort = frame.filter(among)
        return cohort.filter(event).height / cohort.height

    return compute


def _rates(group: str, event: pl.Expr) -> Callable[[pl.DataFrame], Any]:
    def compute(frame: pl.DataFrame) -> dict[str, float]:
        return {
            str(key[0]): rows.filter(event).height / rows.height
            for key, rows in frame.group_by(group, maintain_order=True)
        }

    return compute


def _count_by(group: str, where: pl.Expr = ALL) -> Callable[[pl.DataFrame], Any]:
    def compute(frame: pl.DataFrame) -> dict[str, int]:
        return {str(key[0]): rows.height for key, rows in frame.filter(where).group_by(group, maintain_order=True)}

    return compute


def _mean_by(column: str, group: str, where: pl.Expr = ALL) -> Callable[[pl.DataFrame], Any]:
    def compute(frame: pl.DataFrame) -> dict[str, float]:
        return {
            str(key[0]): float(rows.get_column(column).mean())  # type: ignore[arg-type]
            for key, rows in frame.filter(where).group_by(group, maintain_order=True)
        }

    return compute


def count(query: str, where: pl.Expr) -> Probe:
    return Probe(query, "count", _count(where))


def stat(query: str, column: str, how: str, where: pl.Expr = ALL) -> Probe:
    name = {"mean": "average", "median": "median", "max": "max"}[how]
    return Probe(query, "stat", _stat(column, how, where), stat=name)


def share(query: str, event: pl.Expr, among: pl.Expr = ALL) -> Probe:
    return Probe(query, "proportion", _share(event, among))


def hold(query: str) -> Probe:
    return Probe(query, "hold")


PROBES: tuple[Probe, ...] = (
    # q1: counts, single limits, negation, or/except, shares
    count("how many patients are not on statin?", NO_STATIN),
    count("how many patients not on statins?", NO_STATIN),
    count("how many patients excluding women?", MEN),
    count("how many patients without diabetes?", c("diabetes") == 0),
    count("how many women over 65 on drug?", WOMEN & (c("age") > 65) & DRUG),
    count("how many patients over 65?", c("age") > 65),
    count("how many patients older than 65?", c("age") > 65),
    count("how many patients aged 65 or more?", c("age") >= 65),
    count("how many patients 65+?", c("age") >= 65),
    count("how many patients under 50?", c("age") < 50),
    count("how many males?", MEN),
    count("how many male patients?", MEN),
    count("how many females?", WOMEN),
    count("how many female patients?", WOMEN),
    count("how many men?", MEN),
    count("how many patients at the North site?", c("site") == "North"),
    count("how many patients in the south?", c("site") == "South"),
    count("how many patients on placebo?", PLACEBO),
    count("how many patients not on placebo?", DRUG),
    count("how many patients on drug?", DRUG),
    count("how many patients are statin users?", STATIN),
    count("how many patients with osteoporosis?", c(DEXA) == 3),
    count("how many patients with osteopenia?", c(DEXA) == 2),
    count("how many patients have normal DEXA?", c(DEXA) == 1),
    count("how many patients with no event?", c("event") == 0),
    count("how many patients had an event?", c("event") == 1),
    count("how many patients?", ALL),
    count("how many patients had the outcome?", c("outcome") == 1),
    count("how many patients with bmi over 30?", c("bmi") > 30),
    count("how many patients with ldl above 130?", c("ldl") > 130),
    stat("what is the average ldl in women?", "ldl", "mean", WOMEN),
    stat("what is the average ldl for patients over 65?", "ldl", "mean", c("age") > 65),
    stat("average ldl in men on drug?", "ldl", "mean", MEN & DRUG),
    stat("what is the average age of patients not on statin?", "age", "mean", NO_STATIN),
    stat("what is the average age of women not on drug?", "age", "mean", WOMEN & PLACEBO),
    count("how many people are there?", ALL),
    hold("what is the mortality in women?"),
    hold("show me women"),
    hold("list women on drug"),
    count("how many women are not on drug?", WOMEN & PLACEBO),
    count("how many patients other than men?", WOMEN),
    count("how many patients except those in the west?", c("site") != "West"),
    count("how many patients outside the west?", c("site") != "West"),
    share("what proportion of women are on statin?", STATIN, WOMEN),
    share("what proportion of patients are on statin?", STATIN),
    share("what share of patients had an event?", c("event") == 1),
    Probe("are women more likely to have an event?", "rates", _rates("sex", c("event") == 1), group_by="sex"),
    count("how many patients followed up over 365 days?", c("followup_days") > 365),
    count("how many patients with followup over 365 days?", c("followup_days") > 365),
    count("how many patients over 65 with ldl over 130?", (c("age") > 65) & (c("ldl") > 130)),
    # q2: stats under limits, ranges, shares, comparisons, or/and between values
    stat("what is the average ldl in patients younger than 65?", "ldl", "mean", c("age") < 65),
    stat("what is the average ldl in patients under 65?", "ldl", "mean", c("age") < 65),
    stat("what is the average ldl in patients aged 65 or more?", "ldl", "mean", c("age") >= 65),
    stat("what is the average ldl in patients 65+?", "ldl", "mean", c("age") >= 65),
    stat("what is the average ldl in patients 65 and older?", "ldl", "mean", c("age") >= 65),
    stat("what is the average ldl excluding placebo?", "ldl", "mean", DRUG),
    stat("what is the average ldl in patients not on placebo?", "ldl", "mean", DRUG),
    stat("what is the average ldl outside the north?", "ldl", "mean", c("site") != "North"),
    stat("what is the average ldl in patients not from the north?", "ldl", "mean", c("site") != "North"),
    stat("what is the average ldl in non-statin users?", "ldl", "mean", NO_STATIN),
    stat("what is the average ldl in patients who are not women?", "ldl", "mean", MEN),
    stat("what is the average ldl in non-diabetic patients?", "ldl", "mean", c("diabetes") == 0),
    stat("what is the average ldl in diabetic patients?", "ldl", "mean", c("diabetes") == 1),
    stat("what is the average ldl in patients with diabetes?", "ldl", "mean", c("diabetes") == 1),
    stat("what is the average ldl?", "ldl", "mean"),
    stat("what is the average hdl in patients with ldl over 130?", "hdl", "mean", c("ldl") > 130),
    stat("what is the average age in patients with ldl over 130?", "age", "mean", c("ldl") > 130),
    hold("what is the average bmi in patients over 30?"),
    stat("what is the average ldl in patients older than 65 who are women?", "ldl", "mean", (c("age") > 65) & WOMEN),
    share("what proportion of patients are not on statin?", NO_STATIN),
    share("what proportion of men are not on statin?", NO_STATIN, MEN),
    share("what proportion of patients are on drug?", DRUG),
    share("what proportion of women are on drug?", DRUG, WOMEN),
    Probe("compare ldl by sex", "mean_by", _mean_by("ldl", "sex"), group_by="sex"),
    Probe("compare ldl between drug and placebo", "mean_by", _mean_by("ldl", "treatment"), group_by="treatment"),
    Probe("compare ldl by treatment in women", "mean_by", _mean_by("ldl", "treatment", WOMEN), group_by="treatment"),
    Probe(
        "compare ldl by sex excluding the west",
        "mean_by",
        _mean_by("ldl", "sex", c("site") != "West"),
        group_by="sex",
    ),
    Probe("how many patients by site?", "count_by", _count_by("site"), group_by="site"),
    Probe("how many women by site?", "count_by", _count_by("site", WOMEN), group_by="site"),
    Probe("how many patients by site not on drug?", "count_by", _count_by("site", PLACEBO), group_by="site"),
    Probe("what is the average ldl by site in women?", "mean_by", _mean_by("ldl", "site", WOMEN), group_by="site"),
    hold("tell me about the data"),
    hold("what is the death rate?"),
    hold("how many deaths?"),
    hold("how many patients died?"),
    stat("what is the median age of men?", "age", "median", MEN),
    stat("what is the maximum ldl in the west?", "ldl", "max", c("site") == "West"),
    stat("what is the average ldl in westerners?", "ldl", "mean", c("site") == "West"),
    stat("average ldl in northern patients?", "ldl", "mean", c("site") == "North"),
    count("how many patients in the northern site?", c("site") == "North"),
    count("how many patients in the North or South?", c("site").is_in(["North", "South"])),
    count("how many patients in North and South sites?", c("site").is_in(["North", "South"])),
    count("how many women or men?", WOMEN | MEN),
    hold("how many women and men?"),
    hold("what is the average ldl for women and men?"),
    count("how many patients on drug or placebo?", DRUG | PLACEBO),
    count("how many drug patients over 65 not on statin?", DRUG & (c("age") > 65) & NO_STATIN),
    # q3: wrong column, missing columns, compound limits, inclusive ranges
    hold("what is the average ldl over 130?"),
    hold("what is the average weight?"),
    hold("what is the average weight in women?"),
    hold("how many patients have a weight recorded?"),
    hold("what is the average cholesterol?"),
    hold("what is the average ldl in smokers?"),
    hold("what is the average ldl in women who smoke?"),
    hold("how many women smoke?"),
    count("how many patients in the north on statin not over 65?", (c("site") == "North") & STATIN & (c("age") <= 65)),
    Probe(
        "compare hdl by sex for patients over 65",
        "mean_by",
        _mean_by("hdl", "sex", c("age") > 65),
        group_by="sex",
    ),
    Probe("compare ldl by sex in the west", "mean_by", _mean_by("ldl", "sex", c("site") == "West"), group_by="sex"),
    stat("what is the average ldl in patients aged over 65?", "ldl", "mean", c("age") > 65),
    count("how many patients who are at least 65?", c("age") >= 65),
    stat("what is the average ldl in patients between 50 and 65?", "ldl", "mean", c("age").is_between(50, 65)),
    count("how many patients between 50 and 65?", c("age").is_between(50, 65)),
    count("how many statin users are women?", STATIN & WOMEN),
    count("how many women are statin users?", STATIN & WOMEN),
    count("how many women are not statin users?", NO_STATIN & WOMEN),
    stat("what is the average ldl in women not on statin?", "ldl", "mean", WOMEN & NO_STATIN),
    hold("what is the average ldl in non-smoking women?"),
)
