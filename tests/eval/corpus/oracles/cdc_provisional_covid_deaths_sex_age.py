"""Ground truth for tests/eval/corpus/questions/cdc_provisional_covid_deaths_sex_age.yaml (Polars).

Written from the NCHS dataset description (data.cdc.gov 9bhg-hcku), not from the app's code.
Grain: Group x Year x Month x State x Sex x Age Group. Every filter names the Group, State, Sex and
Age Group it reads, so stacked grains and summary rows never mix. A blank count is a suppressed
1-9: a sum that would need it is None (unknown), never a sum that treats it as zero.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from clinical_analytics.eval.corpus import oracle_kit as kit
from clinical_analytics.eval.corpus.score import OracleAnswer

T = "deaths"
COVID = "COVID-19 Deaths"
TOTAL = "Total Deaths"
FLU = "Influenza Deaths"
PNEUMONIA = "Pneumonia Deaths"
PIC = "Pneumonia, Influenza, or COVID-19 Deaths"
US = pl.col("State") == "United States"
ALL_SEXES = pl.col("Sex") == "All Sexes"
ALL_AGES = pl.col("Age Group") == "All Ages"
BY_YEAR = pl.col("Group") == "By Year"
BY_MONTH = pl.col("Group") == "By Month"
BY_TOTAL = pl.col("Group") == "By Total"
AGE_65_PLUS = ("65-74 years", "75-84 years", "85 years and over")


def _year(year: int) -> pl.Expr:
    return pl.col("Year") == str(year)


def _sum_known(column: str) -> pl.Expr:
    """Sum of a count column, or null when any part is suppressed (unknown, not zero)."""
    values = kit.number(column)
    return pl.when(values.null_count() > 0).then(None).otherwise(values.sum())


def _ratio(numerator: pl.Expr, denominator: pl.Expr) -> pl.Expr:
    return numerator.first() / denominator.first()


_MALE = kit.number(COVID).filter(pl.col("Sex") == "Male")
_FEMALE = kit.number(COVID).filter(pl.col("Sex") == "Female")

ORACLES: dict[str, kit.Oracle] = {
    "pcd-flu-year-001": kit.lookup_by(T, "Year", FLU, BY_YEAR & US & ALL_SEXES & ALL_AGES, "deaths"),
    "pcd-age-2022-002": kit.lookup_by(
        T, "Age Group", COVID, BY_YEAR & _year(2022) & US & ALL_SEXES & ~ALL_AGES, "deaths"
    ),
    "pcd-sex-2022-003": kit.lookup_by(T, "Sex", COVID, BY_YEAR & _year(2022) & US & ~ALL_SEXES & ALL_AGES, "deaths"),
    "pcd-sex-2021-004": kit.lookup_by(T, "Sex", COVID, BY_YEAR & _year(2021) & US & ~ALL_SEXES & ALL_AGES, "deaths"),
    "pcd-rank-65-005": kit.derived_by(
        T,
        "State",
        _sum_known(PNEUMONIA),
        BY_YEAR & _year(2021) & ~US & ALL_SEXES & pl.col("Age Group").is_in(AGE_65_PLUS),
        "deaths",
        top=10,
        rows=len(AGE_65_PLUS),
    ),
    "pcd-share-age-006": kit.derived_by(
        T,
        "Age Group",
        _ratio(kit.number(COVID), kit.number(TOTAL)),
        BY_YEAR & _year(2020) & US & ALL_SEXES & ~ALL_AGES,
        "proportion",
        rows=1,
    ),
    "pcd-ratio-007": kit.derived_by(
        T,
        "Age Group",
        _MALE.first() / _FEMALE.first(),
        BY_TOTAL & US & pl.col("Sex").is_in(["Male", "Female"]) & ~ALL_AGES,
        "ratio",
        rows=2,
    ),
    "pcd-month-2022-008": kit.lookup_by(T, "Month", FLU, BY_MONTH & _year(2022) & US & ALL_SEXES & ALL_AGES, "deaths"),
    "pcd-85-share-009": kit.derived_by(
        T,
        "Year",
        _ratio(kit.number(PIC), kit.number(TOTAL)),
        BY_YEAR & US & ALL_SEXES & (pl.col("Age Group") == "85 years and over"),
        "proportion",
        rows=1,
    ),
    "pcd-total-010": kit.lookup_values(T, {"deaths": COVID}, BY_TOTAL & US & ALL_SEXES & ALL_AGES, "deaths"),
    "pcd-suppressed-011": kit.derived_values(
        T, {"suppressed": kit.number(COVID).null_count().cast(pl.Float64)}, pl.col("State") == "Wyoming", "cells"
    ),
    "pcd-howsure-012": kit.held(T, "provisional counts with suppression, no confidence intervals"),
    "pcd-per100k-013": kit.held(T, "the file has no population denominator"),
    "pcd-pooled-014": kit.held(T, "stacked grains and summary rows: a column sum counts each death several times"),
    "pcd-before2020-015": kit.held(T, "the data start in 2020; there is no pre-2020 level"),
}


def answers(csv_path: Path, oracles: dict[str, kit.Oracle] | None = None) -> dict[str, OracleAnswer]:
    return kit.answers({T: kit.scan_text(csv_path)}, oracles or ORACLES)
