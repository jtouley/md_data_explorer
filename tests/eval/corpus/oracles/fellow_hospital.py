"""Ground truth for the fellow case-study scenarios (questions/fellow_*.yaml), in Polars only.

Written from each extract's data dictionary (eval/corpus/synthetic_hospital.py, the
``*_DICTIONARY`` text), not from the generator's code or the app's. Every column is read as text.
The dictionaries say rows are visits and some rows are duplicates, so every count here is of
distinct keys: patients by ``patient_id``, admissions and tests by their encounter key.

Each dataset's oracle file (``fellow_cdiff.py`` and so on) loads this module by path and calls
``answers(csv_path, dataset)``. Questions that need aggregate mode (a public join) have no
oracle: the harness skips them.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple

import polars as pl

from clinical_analytics.eval.corpus.score import CellKey, OracleAnswer, OracleError

Z_95 = 1.959963984540054
PROTECTIVE_MIU_PER_ML = 120.0  # measles dictionary: evidence of immunity
MIU_PER_IU = 1000.0


class Query(NamedTuple):
    """One oracle: the lazy query, and how its collected frame becomes the answer."""

    unit: str
    frame: pl.LazyFrame
    finish: Callable[[pl.DataFrame], OracleAnswer]


Oracle = Callable[[pl.LazyFrame], Query]


def _age() -> pl.Expr:
    return pl.col("age_years").cast(pl.Int64, strict=True)


def _count_distinct(key: str, unit: str, where: pl.Expr) -> Oracle:
    """How many distinct ``key`` values meet ``where`` (patients or admissions)."""

    def build(frame: pl.LazyFrame) -> Query:
        def finish(df: pl.DataFrame) -> OracleAnswer:
            n = int(df.item())
            return OracleAnswer(n, unit, {("total",): n})

        return Query(unit, frame.filter(where).select(pl.col(key).n_unique()), finish)

    return build


def _count_distinct_by(key: str, group: str, where: pl.Expr) -> Oracle:
    """Distinct ``key`` values per ``group`` among rows meeting ``where``."""

    def build(frame: pl.LazyFrame) -> Query:
        def finish(df: pl.DataFrame) -> OracleAnswer:
            value = {str(row[group]): int(row["n"]) for row in df.iter_rows(named=True)}
            return OracleAnswer(value, "patients", {(k,): n for k, n in value.items()})

        query = frame.filter(where & pl.col(group).is_not_null()).group_by(group).agg(pl.col(key).n_unique().alias("n"))
        return Query("patients", query, finish)

    return build


def _rates_over(units: Callable[[pl.LazyFrame], pl.LazyFrame], group: str) -> Oracle:
    """Share with the event per ``group``, over the one-row-per-unit frame ``units`` builds
    (columns ``group`` and a boolean ``event``)."""

    def build(frame: pl.LazyFrame) -> Query:
        def finish(df: pl.DataFrame) -> OracleAnswer:
            cells: dict[CellKey, int] = {}
            value: dict[str, float] = {}
            for row in df.iter_rows(named=True):
                key, n, events = str(row[group]), int(row["n"]), int(row["events"])
                value[key] = events / n
                cells |= {(key,): n, (key, "events"): events, (key, "non_events"): n - events}
            return OracleAnswer(value, "proportion", cells)

        query = (
            units(frame)
            .filter(pl.col(group).is_not_null())
            .group_by(group)
            .agg(pl.len().alias("n"), pl.col("event").sum().alias("events"))
        )
        return Query("proportion", query, finish)

    return build


def _odds_ratio(units: Callable[[pl.LazyFrame], pl.LazyFrame], exposure: str, outcome: str) -> Oracle:
    """Sample odds ratio and Woolf 95% CI over one row per unit with boolean ``exposed`` and
    ``event``; equal to an unadjusted logistic regression."""

    def build(frame: pl.LazyFrame) -> Query:
        def finish(df: pl.DataFrame) -> OracleAnswer:
            table = {(bool(r["exposed"]), bool(r["event"])): int(r["n"]) for r in df.iter_rows(named=True)}
            a, b = table.get((True, True), 0), table.get((True, False), 0)
            c, d = table.get((False, True), 0), table.get((False, False), 0)
            cells: dict[CellKey, int] = {
                (f"{exposure}=yes", f"{outcome}=yes"): a,
                (f"{exposure}=yes", f"{outcome}=no"): b,
                (f"{exposure}=no", f"{outcome}=yes"): c,
                (f"{exposure}=no", f"{outcome}=no"): d,
            }
            if 0 in (a, b, c, d):
                return OracleAnswer(None, "odds ratio", cells, undefined="a cell of the 2x2 table is zero")
            log_or = math.log((a * d) / (b * c))
            se = math.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
            value = {
                "or": math.exp(log_or),
                "ci_low": math.exp(log_or - Z_95 * se),
                "ci_high": math.exp(log_or + Z_95 * se),
            }
            return OracleAnswer(value, "odds ratio", cells)

        return Query("odds ratio", units(frame).group_by("exposed", "event").agg(pl.len().alias("n")), finish)

    return build


def _listing(patients: Callable[[pl.LazyFrame], pl.LazyFrame]) -> Oracle:
    """A question that must be held: no value, only how many patients the listing would name
    (``patients`` builds one row per such patient)."""

    def build(frame: pl.LazyFrame) -> Query:
        def finish(df: pl.DataFrame) -> OracleAnswer:
            n = int(df.item())
            return OracleAnswer(None, "listing", {("total",): n})

        return Query("listing", patients(frame).select(pl.len()), finish)

    return build


def _patients_where(where: pl.Expr) -> Callable[[pl.LazyFrame], pl.LazyFrame]:
    return lambda frame: frame.filter(where).select("patient_id").unique()


def _median_by(value: pl.Expr, group: str, key: str) -> Oracle:
    """Median of ``value`` over distinct ``key`` rows per ``group``; cells are the patients
    with a value in each group."""
    unit = "median measles_igg"

    def build(frame: pl.LazyFrame) -> Query:
        def finish(df: pl.DataFrame) -> OracleAnswer:
            rows = list(df.iter_rows(named=True))
            values = {str(r[group]): None if r["value"] is None else float(r["value"]) for r in rows}
            return OracleAnswer(values, unit, {(str(r[group]),): int(r["patients"]) for r in rows})

        query = (
            frame.with_columns(value.alias("_value"))
            .filter(pl.col("_value").is_not_null())
            .unique(subset=[key])
            .group_by(group)
            .agg(pl.col("_value").median().alias("value"), pl.col("patient_id").n_unique().alias("patients"))
        )
        return Query(unit, query, finish)

    return build


# ---------------------------------------------------------------------------- C. diff

_OLDER = _age() >= 65
_POSITIVE = pl.col("cdiff_result") == "positive"


def _cdiff_tests_65(frame: pl.LazyFrame) -> pl.LazyFrame:
    """One row per test (encounter) in patients 65+, duplicates dropped."""
    return frame.filter(_OLDER).unique(subset=["encounter_id"])


CDIFF: dict[str, Oracle] = {
    "fellow_cdiff-cases-by-year-01": _count_distinct_by("patient_id", "test_year", _OLDER & _POSITIVE),
    "fellow_cdiff-positivity-by-year-02": _rates_over(
        lambda f: _cdiff_tests_65(f).select("test_year", _POSITIVE.alias("event")), "test_year"
    ),
    "fellow_cdiff-abx-odds-ratio-03": _odds_ratio(
        lambda f: _cdiff_tests_65(f).select(
            (pl.col("outpatient_abx_90d") == "yes").alias("exposed"), _POSITIVE.alias("event")
        ),
        "outpatient_abx_90d",
        "cdiff_result",
    ),
    # Patients with positive tests on two or more distinct encounters.
    "fellow_cdiff-repeat-positives-listing-04": _listing(
        lambda f: (
            f.filter(_POSITIVE)
            .group_by("patient_id")
            .agg(pl.col("encounter_id").n_unique().alias("n"))
            .filter(pl.col("n") >= 2)
        )
    ),
}


# ---------------------------------------------------------------------------- hepatitis B


def _hepb_patients(frame: pl.LazyFrame) -> pl.LazyFrame:
    """One row per patient: birthplace, ever screened, ever newly diagnosed."""
    return frame.group_by("patient_id").agg(
        pl.col("birth_region").first(),
        (pl.col("hbv_screen_done") == "yes").any().alias("screened"),
        (pl.col("new_hbv_diagnosis") == "yes").any().alias("new_dx"),
    )


_NEW_DX = pl.col("new_hbv_diagnosis") == "yes"

HEPB: dict[str, Oracle] = {
    "fellow_hepb-adults-seen-01": _count_distinct("patient_id", "patients", pl.lit(True)),
    "fellow_hepb-ever-screened-02": _count_distinct("patient_id", "patients", pl.col("hbv_screen_done") == "yes"),
    "fellow_hepb-new-diagnoses-03": _count_distinct("patient_id", "patients", _NEW_DX),
    "fellow_hepb-screened-by-birthplace-04": _rates_over(
        lambda f: _hepb_patients(f).select("birth_region", pl.col("screened").alias("event")), "birth_region"
    ),
    "fellow_hepb-new-diagnoses-by-birthplace-05": _count_distinct_by("patient_id", "birth_region", _NEW_DX),
    "fellow_hepb-reactive-listing-06": _listing(_patients_where(pl.col("hbsag_result") == "reactive")),
}


# ---------------------------------------------------------------------------- pneumonia

_SEVENTY = _age() >= 70


def _pneumonia_by_pathogen(vaccine: str) -> Callable[[pl.LazyFrame], pl.LazyFrame]:
    """One row per admission and pathogen in patients 70+, with documented vaccination as the event."""

    def units(frame: pl.LazyFrame) -> pl.LazyFrame:
        return (
            frame.filter(_SEVENTY)
            .unique(subset=["encounter_id", "pathogen"])
            .select("pathogen", (pl.col(vaccine) == "yes").alias("event"))
        )

    return units


PNEUMONIA: dict[str, Oracle] = {
    "fellow_pneumonia-admissions-70-01": _count_distinct("encounter_id", "visits", _SEVENTY),
    "fellow_pneumonia-pathogens-70-02": _count_distinct_by("encounter_id", "pathogen", _SEVENTY),
    "fellow_pneumonia-pneumococcal-vaccine-03": _rates_over(_pneumonia_by_pathogen("pneumococcal_vaccine"), "pathogen"),
    "fellow_pneumonia-flu-vaccine-04": _rates_over(_pneumonia_by_pathogen("flu_vaccine_this_season"), "pathogen"),
    "fellow_pneumonia-deaths-listing-05": _listing(
        _patients_where((pl.col("pathogen") == "S. pneumoniae") & (pl.col("died_in_hospital") == "yes"))
    ),
}


# ---------------------------------------------------------------------------- measles


def _igg_miu() -> pl.Expr:
    """Measles IgG in mIU/mL whatever the lab's unit; null when not drawn. A unit outside the
    dictionary's two makes the oracle fail (see ``_measles_units_known``)."""
    level = pl.col("measles_igg").cast(pl.Float64, strict=True)
    unit = pl.col("measles_igg_unit")
    return pl.when(unit == "IU/mL").then(level * MIU_PER_IU).when(unit == "mIU/mL").then(level).otherwise(None)


def _measles_patients(frame: pl.LazyFrame) -> pl.LazyFrame:
    """One row per patient: risk group, two documented doses, protective IgG on any check."""
    return frame.group_by("patient_id").agg(
        pl.col("risk_group").first(),
        (pl.col("mmr_doses_documented").cast(pl.Int64, strict=True) >= 2).any().alias("two_doses"),
        (_igg_miu() >= PROTECTIVE_MIU_PER_ML).fill_null(False).any().alias("protective"),
    )


def _with_unit_check(oracle: Oracle) -> Oracle:
    """Fail the oracle (graded ERROR) if the file has an IgG unit the dictionary does not name.
    The units are read on their own, so the check holds even when the answer has no rows."""

    def build(frame: pl.LazyFrame) -> Query:
        inner = oracle(frame)
        units = frame.select(pl.col("measles_igg_unit").drop_nulls().unique())

        def finish(df: pl.DataFrame) -> OracleAnswer:
            found = set(units.collect()["measles_igg_unit"].to_list())
            if extra := sorted(found - {"IU/mL", "mIU/mL"}):
                raise OracleError(f"measles_igg_unit has units {extra} outside the dictionary")
            return inner.finish(df)

        return Query(inner.unit, inner.frame, finish)

    return build


MEASLES: dict[str, Oracle] = {
    "fellow_measles-checked-by-group-01": _count_distinct_by("patient_id", "risk_group", pl.lit(True)),
    "fellow_measles-two-doses-by-group-02": _rates_over(
        lambda f: _measles_patients(f).select("risk_group", pl.col("two_doses").alias("event")), "risk_group"
    ),
    "fellow_measles-median-igg-by-group-03": _with_unit_check(_median_by(_igg_miu(), "risk_group", "visit_id")),
    "fellow_measles-no-immunity-by-group-04": _with_unit_check(
        _rates_over(
            lambda f: _measles_patients(f).select(
                "risk_group", (~pl.col("two_doses") & ~pl.col("protective")).alias("event")
            ),
            "risk_group",
        )
    ),
    # Healthcare workers without evidence of immunity.
    "fellow_measles-nonimmune-listing-05": _listing(
        lambda f: _measles_patients(f).filter(
            (pl.col("risk_group") == "Healthcare worker") & ~pl.col("two_doses") & ~pl.col("protective")
        )
    ),
}

ORACLES: dict[str, dict[str, Oracle]] = {
    "fellow_cdiff": CDIFF,
    "fellow_hepb": HEPB,
    "fellow_pneumonia": PNEUMONIA,
    "fellow_measles": MEASLES,
}


def scan(csv_path: Path) -> pl.LazyFrame:
    return pl.scan_csv(csv_path, infer_schema=False)


def answers(csv_path: Path, dataset: str) -> dict[str, OracleAnswer]:
    """Every question's answer for one dataset. Each query is collected on its own, so a value one
    question cannot read (an IgG of "<5", a non-numeric age) makes that question alone
    ``oracle_error`` (graded ERROR) and the others still get their answers."""
    frame = scan(csv_path)
    result: dict[str, OracleAnswer] = {}
    for question_id, oracle in ORACLES[dataset].items():
        query = oracle(frame)
        try:
            result[question_id] = query.finish(query.frame.collect())
        except (OracleError, pl.exceptions.PolarsError) as error:
            result[question_id] = OracleAnswer(None, query.unit, oracle_error=str(error))
    return result
