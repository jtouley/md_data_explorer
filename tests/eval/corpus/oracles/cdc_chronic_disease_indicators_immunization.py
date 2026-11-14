"""Ground truth for cdc_chronic_disease_indicators_immunization.yaml, in Polars only.

Written from the CDI dataset description (data.cdc.gov hksd-2xuw), not from the app's code.
Grain: yearstart x locationabbr x questionid x datavaluetypeid x stratificationid1. Every filter
names the indicator, the value type (crude unless asked), the location and the stratum, so crude
and age-adjusted values, national and state rows, and overall and stratified rows never mix.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from clinical_analytics.eval.corpus import oracle_kit as kit
from clinical_analytics.eval.corpus.score import OracleAnswer

T = "cdi"
VALUE = "datavalue"
CI = (VALUE, "lowconfidencelimit", "highconfidencelimit")
FLU = pl.col("questionid") == "IMM01"
PNEUMO_65 = pl.col("questionid") == "IMM04"
CRUDE = pl.col("datavaluetypeid") == "CRDPREV"
US = pl.col("locationabbr") == "US"
OVERALL = pl.col("stratificationid1") == "OVR"
UNIT = "percent"


def _year(year: int) -> pl.Expr:
    return pl.col("yearstart") == str(year)


def _category(name: str) -> pl.Expr:
    return pl.col("stratificationcategoryid1") == name


def _below_national(tables: kit.Tables) -> kit.Query:
    """States whose 2023 crude adult flu CI upper bound is below the national crude estimate."""
    rows = FLU & CRUDE & OVERALL & _year(2023)

    def finish(df: pl.DataFrame) -> OracleAnswer:
        national = df.filter(pl.col("locationabbr") == "US")
        if national.height != 1 or national["estimate"][0] is None:
            raise kit.OracleError(f"{national.height} national rows for 2023 adult flu coverage")
        below = df.filter((pl.col("locationabbr") != "US") & (pl.col("high") < national["estimate"][0]))
        return OracleAnswer({str(r["locationdesc"]): r["estimate"] for r in below.iter_rows(named=True)}, UNIT)

    query = (
        tables[T]
        .filter(rows)
        .select("locationabbr", "locationdesc", kit.number(VALUE).alias("estimate"), kit.number(CI[2]).alias("high"))
    )
    return kit.Query(UNIT, query, finish)


ORACLES: dict[str, kit.Oracle] = {
    "cdi-trend-001": kit.lookup_by(T, "yearstart", VALUE, FLU & CRUDE & US & OVERALL, UNIT),
    "cdi-65-002": kit.lookup_by(
        T, "yearstart", VALUE, FLU & CRUDE & US & (pl.col("stratificationid1") == "AGE65P"), UNIT
    ),
    "cdi-howsure-003": kit.lookup_values(
        T, dict(zip(kit.CI_PARTS, CI, strict=True)), FLU & CRUDE & US & OVERALL & _year(2023), UNIT
    ),
    "cdi-rank-004": kit.lookup_by(
        T, "locationdesc", VALUE, PNEUMO_65 & CRUDE & ~US & OVERALL & _year(2023), UNIT, top=10
    ),
    "cdi-rank-ci-005": kit.lookup_ci_by(
        T, "locationdesc", CI, PNEUMO_65 & CRUDE & ~US & OVERALL & _year(2023), UNIT, top=10
    ),
    "cdi-sex-006": kit.lookup_ci_by(T, "stratification1", CI, FLU & CRUDE & US & _category("SEX") & _year(2023), UNIT),
    "cdi-age-007": kit.lookup_by(T, "stratification1", VALUE, FLU & CRUDE & US & _category("AGE") & _year(2023), UNIT),
    "cdi-race-008": kit.lookup_ci_by(
        T, "stratification1", CI, FLU & CRUDE & US & _category("RACE") & _year(2023), UNIT, top=1, order="asc"
    ),
    "cdi-before-after-009": kit.lookup_ci_by(
        T, "yearstart", CI, FLU & CRUDE & US & OVERALL & (_year(2019) | _year(2021)), UNIT
    ),
    "cdi-crude-adj-010": kit.lookup_by(
        T, "datavaluetype", VALUE, FLU & (pl.col("locationabbr") == "CA") & OVERALL & _year(2023), UNIT
    ),
    "cdi-suppressed-011": kit.derived_values(
        T,
        {"suppressed": pl.len().cast(pl.Float64)},
        pl.col("datavaluefootnote").str.starts_with("Data suppressed").fill_null(False),
        "estimates",
    ),
    "cdi-texas-012": kit.lookup_ci_by(
        T, "yearstart", CI, PNEUMO_65 & CRUDE & (pl.col("locationabbr") == "TX") & OVERALL, UNIT
    ),
    "cdi-pooled-013": kit.held(T, "rows overlap: national with states, overall with strata, crude with age-adjusted"),
    "cdi-below-national-014": _below_national,
}


def answers(csv_path: Path, oracles: dict[str, kit.Oracle] | None = None) -> dict[str, OracleAnswer]:
    return kit.answers({T: kit.scan_text(csv_path)}, oracles or ORACLES)
