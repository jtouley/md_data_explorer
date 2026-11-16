"""Ground truth for tests/eval/corpus/questions/covid_ms.yaml, in Polars only.

Written from the PhysioNet data description (GDSI v1.0.1), not from the app's code. Every
column is read as text, as the file stores it ("yes"/"no", "female"/"male", "1"/"2"), so no
type inference stands between the file and the answer. Each answer carries the patients behind
each cell, so the scorer knows which cells must be hidden.

Every oracle is a lazy query plus a finishing step; ``answers`` collects all the queries in one
``pl.collect_all``, so the CSV is planned once. Rows whose group or outcome is empty are left out,
as the app leaves them out. A finishing step that finds the data does not support the answer
raises ``OracleError``; that question alone is graded ERROR, the run goes on.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple

import polars as pl

from clinical_analytics.contracts.config import MISSING_VALUE_TOKENS
from clinical_analytics.eval.corpus.score import CellKey, OracleAnswer, OracleError

HOSPITAL = "covid19_admission_hospital"
CONFIRMED = "covid19_confirmed_case"
ONSET = "year_onset"
YES = "yes"
YES_NO = ("yes", "no")
SEXES = ("female", "male")
Z_95 = 1.959963984540054


class Query(NamedTuple):
    """One oracle: the lazy query, and how its collected frame becomes the answer. (A NamedTuple,
    not a dataclass: the harness loads this file by path, outside ``sys.modules``.)"""

    unit: str
    frame: pl.LazyFrame
    finish: Callable[[pl.DataFrame], OracleAnswer]


Oracle = Callable[[pl.LazyFrame], Query]


def scan(csv_path: Path) -> pl.LazyFrame:
    return pl.scan_csv(csv_path, infer_schema=False)


def _present(*columns: str) -> pl.Expr:
    return pl.all_horizontal(pl.col(column).is_not_null() for column in columns)


def _total(frame: pl.LazyFrame) -> Query:
    def finish(df: pl.DataFrame) -> OracleAnswer:
        n = int(df.item())
        return OracleAnswer(n, "patients", {("total",): n})

    return Query("patients", frame.select(pl.len()), finish)


def _count_by(group: str, where: pl.Expr = pl.lit(True)) -> Oracle:
    def build(frame: pl.LazyFrame) -> Query:
        def finish(df: pl.DataFrame) -> OracleAnswer:
            value = {str(row[group]): int(row["n"]) for row in df.iter_rows(named=True)}
            return OracleAnswer(value, "patients", {(key,): n for key, n in value.items()})

        query = frame.filter(where & _present(group)).group_by(group).agg(pl.len().alias("n"))
        return Query("patients", query, finish)

    return build


def _count(where: pl.Expr) -> Oracle:
    def build(frame: pl.LazyFrame) -> Query:
        def finish(df: pl.DataFrame) -> OracleAnswer:
            n = int(df.item())
            return OracleAnswer(n, "patients", {("total",): n})

        return Query("patients", frame.filter(where).select(pl.len()), finish)

    return build


def _numeric(metric: str) -> tuple[pl.Expr, pl.Expr]:
    """The metric as numbers (missing-value markers are null), and a flag for any other text."""
    text = pl.col(metric).str.strip_chars()
    missing = text.is_null() | text.str.to_lowercase().is_in(list(MISSING_VALUE_TOKENS))
    number = pl.when(missing).then(pl.lit(None, dtype=pl.String)).otherwise(text).cast(pl.Float64, strict=False)
    return number, ~missing & number.is_null()


def _stat_by(stat: str, metric: str, group: str) -> Oracle:
    unit = f"{stat} {metric}"

    def build(frame: pl.LazyFrame) -> Query:
        number, unreadable = _numeric(metric)
        reducer = number.median() if stat == "median" else number.mean()

        def finish(df: pl.DataFrame) -> OracleAnswer:
            bad = int(df["unreadable"].sum())
            if bad:
                raise OracleError(f"{metric}: {bad} value(s) are neither numbers nor missing-value markers")
            rows = list(df.iter_rows(named=True))
            # A group with no values has no mean: None, which the scorer requires to be withheld.
            value = {str(row[group]): None if row["value"] is None else float(row["value"]) for row in rows}
            return OracleAnswer(value, unit, {(str(row[group]),): int(row["n"]) for row in rows})

        query = (
            frame.filter(_present(group))
            .group_by(group)
            .agg(reducer.alias("value"), number.is_not_null().sum().alias("n"), unreadable.sum().alias("unreadable"))
        )
        return Query(unit, query, finish)

    return build


def _rates(outcome: str, group: str) -> Oracle:
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
            frame.filter(_present(outcome, group))
            .group_by(group)
            .agg(pl.len().alias("n"), (pl.col(outcome) == YES).sum().alias("events"))
        )
        return Query("proportion", query, finish)

    return build


TwoByTwo = dict[tuple[str, str], int]


def _two_by_two(
    exposure: str,
    outcome: str,
    levels: tuple[tuple[str, str], tuple[str, str]],
    unit: str,
    from_table: Callable[[TwoByTwo], OracleAnswer],
) -> Oracle:
    """The exposure x outcome table over patients with both recorded, as the one input of the
    odds-ratio and chi-square oracles. A level outside ``levels`` raises OracleError: dropping
    it would silently test a different table than the app does."""
    exposure_levels, outcome_levels = levels

    def build(frame: pl.LazyFrame) -> Query:
        def finish(df: pl.DataFrame) -> OracleAnswer:
            found = {(str(r[exposure]), str(r[outcome])): int(r["n"]) for r in df.iter_rows(named=True)}
            wanted = {(e, o) for e in exposure_levels for o in outcome_levels}
            if extra := sorted(set(found) - wanted):
                raise OracleError(f"{exposure} x {outcome}: levels {extra} outside {levels}")
            return from_table({(e, o): found.get((e, o), 0) for e in exposure_levels for o in outcome_levels})

        query = frame.filter(_present(exposure, outcome)).group_by(exposure, outcome).agg(pl.len().alias("n"))
        return Query(unit, query, finish)

    return build


def _chi_square(exposure: str, outcome: str, exposure_levels: tuple[str, str]) -> Oracle:
    """Pearson chi-square on the 2x2 table, with and without Yates' correction (df=1)."""

    def from_table(table: TwoByTwo) -> OracleAnswer:
        cells: dict[CellKey, int] = dict(table)
        rows = sorted({e for e, _ in table})
        cols = sorted({o for _, o in table})
        total = sum(table.values())
        row_sum = {r: sum(table[(r, c)] for c in cols) for r in rows}
        col_sum = {c: sum(table[(r, c)] for r in rows) for c in cols}
        if 0 in row_sum.values() or 0 in col_sum.values():
            return OracleAnswer(None, "p", cells, undefined="a row or column of the 2x2 table is empty")
        plain = yates = 0.0
        for r in rows:
            for c in cols:
                expected = row_sum[r] * col_sum[c] / total
                deviation = abs(table[(r, c)] - expected)
                plain += deviation**2 / expected
                yates += max(deviation - 0.5, 0.0) ** 2 / expected
        # Survival function of chi-square with 1 df: erfc(sqrt(x / 2)).
        value = {"pearson": math.erfc(math.sqrt(plain / 2)), "yates": math.erfc(math.sqrt(yates / 2))}
        return OracleAnswer(value, "p", cells)

    return _two_by_two(exposure, outcome, (exposure_levels, YES_NO), "p", from_table)


def _odds_ratio(exposure: str, outcome: str) -> Oracle:
    """Sample odds ratio and Woolf (Wald) 95% CI; equal to an unadjusted logistic regression."""

    def from_table(table: TwoByTwo) -> OracleAnswer:
        cells: dict[CellKey, int] = {(f"{exposure}={e}", f"{outcome}={o}"): n for (e, o), n in table.items()}
        a, b, c, d = table[("yes", "yes")], table[("yes", "no")], table[("no", "yes")], table[("no", "no")]
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

    return _two_by_two(exposure, outcome, (YES_NO, YES_NO), "odds ratio", from_table)


def _held(frame: pl.LazyFrame) -> Query:
    def finish(df: pl.DataFrame) -> OracleAnswer:
        n = int(df.item())
        return OracleAnswer(None, "listing", {("total",): n})

    return Query("listing", frame.filter(pl.col("covid19_icu_stay") == YES).select(pl.len()), finish)


_FEMALE = pl.col("sex") == "female"
_CONFIRMED = pl.col(CONFIRMED) == YES

ORACLES: dict[str, Oracle] = {
    "covid_ms-total-001": _total,
    "covid_ms-sex-002": _count_by("sex"),
    "covid_ms-mstype-003": _count_by("ms_type2"),
    "covid_ms-hosp-004": _count_by(HOSPITAL),
    "covid_ms-dmt-005": _count_by("current_dmt"),
    "covid_ms-filter-006": _count(_FEMALE & (pl.col("ms_type2") == "relapsing_remitting") & _CONFIRMED),
    "covid_ms-mean-007": _stat_by("mean", ONSET, "ms_type2"),
    "covid_ms-median-008": _stat_by("median", ONSET, "sex"),
    "covid_ms-agerate-009": _rates(CONFIRMED, "age_in_cat"),
    "covid_ms-sexrate-010": _rates(HOSPITAL, "sex"),
    "covid_ms-chisq-011": _chi_square("sex", CONFIRMED, SEXES),
    "covid_ms-or-012": _odds_ratio("has_comorbidities", HOSPITAL),
    "covid_ms-confirmed-013": _count_by("ms_type2", _CONFIRMED),
    "covid_ms-followup-014": _count_by("ms_type2", _CONFIRMED & _FEMALE),
    "covid_ms-listing-015": _held,
}


def answers(csv_path: Path, oracles: dict[str, Oracle] | None = None) -> dict[str, OracleAnswer]:
    """Every question's answer, from one ``collect_all`` over one scan. A question whose data
    does not support an answer gets ``oracle_error`` instead of aborting the others."""
    frame = scan(csv_path)
    queries = {question_id: oracle(frame) for question_id, oracle in (oracles or ORACLES).items()}
    collected = pl.collect_all([query.frame for query in queries.values()])
    result: dict[str, OracleAnswer] = {}
    for (question_id, query), df in zip(queries.items(), collected, strict=True):
        try:
            result[question_id] = query.finish(df)
        except OracleError as error:
            result[question_id] = OracleAnswer(None, query.unit, oracle_error=str(error))
    return result
