"""Ground truth for tests/eval/corpus/questions/cdc_leading_causes_death.yaml, in Polars only.

Written from the NCHS dataset description (data.cdc.gov bi63-dtpu), not from the app's code.
Grain: Year x State x Cause Name. Every answer reads published rows; the kit raises when a filter
leaves more than one row per answer cell, so nothing is pooled by accident.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from clinical_analytics.eval.corpus import oracle_kit as kit
from clinical_analytics.eval.corpus.score import OracleAnswer

T = "deaths"
RATE = "Age-adjusted Death Rate"
DEATHS = "Deaths"
US = pl.col("State") == "United States"
STATES = pl.col("State") != "United States"
ALL_CAUSES = pl.col("Cause Name") == "All causes"


def _cause(name: str) -> pl.Expr:
    return pl.col("Cause Name") == name


def _year(year: int) -> pl.Expr:
    return pl.col("Year") == str(year)


FLU = _cause("Influenza and pneumonia")


def _share_2017(tables: kit.Tables) -> kit.Query:
    """Influenza-and-pneumonia deaths over all-cause deaths, both from the 2017 US rows."""

    def finish(df: pl.DataFrame) -> OracleAnswer:
        if df.height != 2:
            raise kit.OracleError(f"{df.height} rows for the 2017 US flu and all-cause deaths, expected 2")
        by_cause = dict(zip(df["Cause Name"], df["deaths"], strict=True))
        return OracleAnswer({"share": by_cause["Influenza and pneumonia"] / by_cause["All causes"]}, "proportion")

    query = (
        tables[T].filter(US & _year(2017) & (FLU | ALL_CAUSES)).select("Cause Name", kit.number(DEATHS).alias("deaths"))
    )
    return kit.Query("proportion", query, finish)


def _change_1999_2017(tables: kit.Tables) -> kit.Query:
    """Per cause, the 2017 minus the 1999 US age-adjusted rate; the largest rise."""
    rate = kit.number(RATE)
    change = rate.filter(_year(2017)).first() - rate.filter(_year(1999)).first()
    return kit.derived_by(
        T, "Cause Name", change, US & ~ALL_CAUSES & (_year(1999) | _year(2017)), "rate change", top=1, rows=2
    )(tables)


ORACLES: dict[str, kit.Oracle] = {
    "lcd-trend-001": kit.lookup_by(T, "Year", RATE, US & FLU, "deaths per 100,000"),
    "lcd-trend-alz-002": kit.lookup_by(T, "Year", RATE, US & _cause("Alzheimer's disease"), "deaths per 100,000"),
    "lcd-deaths-year-003": kit.lookup_by(T, "Year", DEATHS, US & FLU, "deaths"),
    "lcd-rank-states-004": kit.lookup_by(T, "State", RATE, STATES & FLU & _year(2017), "deaths per 100,000", top=10),
    "lcd-rank-lowest-005": kit.lookup_by(
        T, "State", RATE, STATES & _cause("Heart disease") & _year(2017), "deaths per 100,000", top=5, order="asc"
    ),
    "lcd-rank-causes-006": kit.lookup_by(
        T, "Cause Name", DEATHS, (pl.col("State") == "California") & _year(2010) & ~ALL_CAUSES, "deaths"
    ),
    "lcd-share-007": _share_2017,
    "lcd-total-008": kit.lookup_values(T, {"deaths": DEATHS}, US & ALL_CAUSES & _year(2017), "deaths"),
    "lcd-vs-national-009": kit.lookup_by(
        T, "State", RATE, pl.col("State").is_in(["Texas", "United States"]) & FLU & _year(2017), "deaths per 100,000"
    ),
    "lcd-peak-010": kit.lookup_by(T, "Year", RATE, US & _cause("Kidney disease"), "deaths per 100,000", top=1),
    "lcd-change-011": _change_1999_2017,
    "lcd-sex-012": kit.held(T, "the file has no sex column"),
    "lcd-howsure-013": kit.held(T, "the file publishes no confidence intervals or standard errors"),
    "lcd-pooled-014": kit.held(T, "rows overlap: United States with the states, All causes with each cause"),
    "lcd-after2020-015": kit.held(T, "the data end in 2017"),
}


def answers(csv_path: Path, oracles: dict[str, kit.Oracle] | None = None) -> dict[str, OracleAnswer]:
    return kit.answers({T: kit.scan_text(csv_path)}, oracles or ORACLES)
