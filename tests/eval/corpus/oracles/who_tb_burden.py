"""Ground truth for tests/eval/corpus/questions/who_tb_burden.yaml, in Polars only.

Written from WHO's TB data dictionary (TB_data_dictionary.csv), not from the app's code. Every
column is read as text, so Namibia's iso2 "NA" stays a value, and converted with the kit. A rate
for a group of countries is the sum of their estimated cases over the sum of their population
(per 100,000), over countries with an estimate; uncertainty bounds are never added up.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from clinical_analytics.eval.corpus import oracle_kit as kit
from clinical_analytics.eval.corpus.score import OracleAnswer

COUNTRIES = "countries"
AGE_SEX = "age_sex"
FILES = {COUNTRIES: "TB_burden_countries.csv", AGE_SEX: "TB_burden_age_sex.csv"}
PER_100K = "per 100,000"


def _iso3(*codes: str) -> pl.Expr:
    return pl.col("iso3").is_in(list(codes))


def _year(year: int) -> pl.Expr:
    return pl.col("year") == str(year)


def _bounds(measure: str) -> tuple[str, str, str]:
    return measure, f"{measure}_lo", f"{measure}_hi"


def _rate(cases: str = "e_inc_num") -> pl.Expr:
    """Cases over population per 100,000, over the countries that have an estimate."""
    known = kit.number(cases).is_not_null()
    return kit.number(cases).filter(known).sum() / kit.number("e_pop_num").filter(known).sum() * 100_000


_MALE = kit.number("best").filter(pl.col("sex") == "m").first()
_FEMALE = kit.number("best").filter(pl.col("sex") == "f").first()
# No ratio when the female estimate is zero (it is undefined, not infinite).
_male_to_female = pl.when(_FEMALE > 0).then(_MALE / _FEMALE)
ALL_RISK = pl.col("risk_factor") == "all"
ZAF = _iso3("ZAF")

ORACLES: dict[str, kit.Oracle] = {
    "tb-global-001": kit.derived_by(COUNTRIES, "year", _rate(), pl.lit(True), PER_100K, unique_by="iso3"),
    "tb-region-002": kit.derived_by(COUNTRIES, "g_whoregion", _rate(), _year(2024), PER_100K, unique_by="iso3"),
    "tb-top10-003": kit.lookup_ci_by(COUNTRIES, "country", _bounds("e_inc_100k"), _year(2024), PER_100K, top=10),
    "tb-ind-idn-004": kit.lookup_ci_by(
        COUNTRIES, "country", _bounds("e_inc_100k"), _year(2024) & _iso3("IND", "IDN"), PER_100K
    ),
    "tb-mort-005": kit.lookup_ci_by(
        COUNTRIES, "country", _bounds("e_mort_100k"), _year(2024) & _iso3("IND", "IDN"), PER_100K
    ),
    "tb-zaf-sex-006": kit.lookup_by(
        AGE_SEX,
        "sex",
        "best",
        ZAF & ALL_RISK & (pl.col("age_group") == "all") & pl.col("sex").is_in(["f", "m"]),
        "cases",
    ),
    "tb-zaf-age-007": kit.lookup_by(
        AGE_SEX, "age_group", "best", ZAF & ALL_RISK & (pl.col("sex") == "a") & (pl.col("age_group") != "all"), "cases"
    ),
    "tb-pooled-008": kit.held(
        AGE_SEX, "the rows overlap: all ages with bands, sex a with f and m, risk factors with all"
    ),
    "tb-india-notif-009": kit.lookup_by(
        COUNTRIES, "year", "c_newinc_100k", _iso3("IND") & (_year(2019) | _year(2020)), PER_100K
    ),
    "tb-nigeria-cfr-010": kit.lookup_ci_by(COUNTRIES, "year", _bounds("cfr"), _iso3("NGA"), "proportion"),
    "tb-namibia-011": kit.lookup_values(COUNTRIES, {"rate": "e_inc_100k"}, _iso3("NAM") & _year(2024), PER_100K),
    "tb-hiv-afr-012": kit.derived_values(
        COUNTRIES,
        {
            "share": kit.number("e_inc_tbhiv_num").filter(kit.number("e_inc_num").is_not_null()).sum()
            / kit.number("e_inc_num").filter(kit.number("e_inc_tbhiv_num").is_not_null()).sum()
        },
        (pl.col("g_whoregion") == "AFR") & _year(2024),
        "proportion",
    ),
    "tb-global-howsure-013": kit.held(COUNTRIES, "country bounds cannot be added into a global interval"),
    "tb-mf-ratio-014": kit.derived_by(
        AGE_SEX,
        "country",
        _male_to_female,
        ALL_RISK & (pl.col("age_group") == "15plus") & pl.col("sex").is_in(["f", "m"]),
        "ratio",
        top=10,
        rows=2,
    ),
}


def answers(files_dir: Path, oracles: dict[str, kit.Oracle] | None = None) -> dict[str, OracleAnswer]:
    tables = {name: kit.scan_text(files_dir / file) for name, file in FILES.items()}
    return kit.answers(tables, oracles or ORACLES)
