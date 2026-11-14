"""Ground truth for tests/eval/corpus/questions/cdc_respnet_rates.yaml, in Polars only.

Written from the RESP-NET dataset description (data.cdc.gov kvib-3txy), not from the app's code.
A season's cumulative rate is the cumulative rate at its last reported week (the series is
cumulative within a season): the oracle takes that row, after checking there is one row per week
for the dimensions it fixes. Every filter names the network, site, age, sex, race, data type,
estimate type and rate type it reads, so observed and age-adjusted rates, rates and percentages,
and summary rows never mix.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from clinical_analytics.eval.corpus import oracle_kit as kit
from clinical_analytics.eval.corpus.score import OracleAnswer

T = "rates"
EST = "Estimate"
SITE_TOTAL = pl.col("State") == "Overall"
ALL_SEX = pl.col("Sex") == "All"
ALL_RACE = pl.col("Race") == "All"
OVERALL_AGE = pl.col("Age Category") == "Overall"
WEEKLY_ROWS = (pl.col("Date Type") == "Week Ending Date") & (pl.col("Estimate Type") == "Rate per 100,000")
OBSERVED = pl.col("Rate Type") == "Observed"
CUMULATIVE = WEEKLY_ROWS & (pl.col("Data Type") == "Cumulative Rate") & OBSERVED
WEEKLY = WEEKLY_ROWS & (pl.col("Data Type") == "Weekly Rate") & OBSERVED
SEASON_END = kit.number(EST).sort_by("Date").last()
SITES = ~pl.col("State").is_in(["Overall", "COVID-NET", "FluSurv-NET", "RSV-NET"])


def _net(name: str) -> pl.Expr:
    return pl.col("Surveillance Network") == name


def _age(label: str) -> pl.Expr:
    return pl.col("Age Category") == label


def _season(label: str) -> pl.Expr:
    return pl.col("Season") == label


def _season_end_by(group: str, where: pl.Expr) -> kit.Oracle:
    return kit.derived_by(T, group, SEASON_END, CUMULATIVE & where, "rate per 100,000", unique_by="Date")


_FLU_65 = _net("FluSurv-NET") & SITE_TOTAL & _age("65+ yr") & ALL_SEX & ALL_RACE

ORACLES: dict[str, kit.Oracle] = {
    "rsp-flu65-001": _season_end_by("Season", _FLU_65),
    "rsp-rsv65-002": _season_end_by("Season", _net("RSV-NET") & SITE_TOTAL & _age("65+ yr") & ALL_SEX & ALL_RACE),
    "rsp-age-rsv-003": _season_end_by(
        "Age Category", _net("RSV-NET") & SITE_TOTAL & ~OVERALL_AGE & ALL_SEX & ALL_RACE & _season("2024-25")
    ),
    "rsp-sex-004": _season_end_by(
        "Sex", _net("RSV-NET") & SITE_TOTAL & OVERALL_AGE & ~ALL_SEX & ALL_RACE & _season("2024-25")
    ),
    "rsp-rank-sites-005": _season_end_by(
        "State", _net("COVID-NET") & SITES & OVERALL_AGE & ALL_SEX & ALL_RACE & _season("2024-25")
    ),
    "rsp-network-006": _season_end_by(
        "Surveillance Network",
        ~_net("Combined") & SITE_TOTAL & _age("75+ yr") & ALL_SEX & ALL_RACE & _season("2024-25"),
    ),
    "rsp-peak-007": kit.derived_by(
        T,
        "Season",
        kit.number(EST).max(),
        WEEKLY & _net("FluSurv-NET") & SITE_TOTAL & OVERALL_AGE & ALL_SEX & ALL_RACE,
        "rate per 100,000",
        unique_by="Date",
    ),
    "rsp-before-after-008": _season_end_by("Season", _FLU_65 & pl.col("Season").is_in(["2018-19", "2022-23"])),
    "rsp-under5-009": _season_end_by("Season", _net("Combined") & SITE_TOTAL & _age("0-4 yr") & ALL_SEX & ALL_RACE),
    "rsp-icu-010": kit.lookup_by(
        T,
        "Season",
        EST,
        _net("FluSurv-NET")
        & (pl.col("Date Type") == "Season")
        & (pl.col("Data Type") == "Admitted to Intensive Care Unit")
        & (pl.col("Estimate Type") == "Percent")
        & _age("Overall")
        & ALL_SEX
        & ALL_RACE,
        "percent",
    ),
    "rsp-rate-types-011": kit.derived_by(T, "Rate Type", pl.len().cast(pl.Float64), pl.lit(True), "rows"),
    "rsp-howsure-012": kit.held(T, "RESP-NET publishes no confidence intervals"),
    "rsp-pooled-013": kit.held(T, "rows mix weekly and cumulative rates, percentages, and overlapping strata"),
    "rsp-agesex-014": kit.held(T, "sex is published only for all ages: there is no 65+ by sex row"),
}


def answers(csv_path: Path, oracles: dict[str, kit.Oracle] | None = None) -> dict[str, OracleAnswer]:
    return kit.answers({T: kit.scan_text(csv_path)}, oracles or ORACLES)
