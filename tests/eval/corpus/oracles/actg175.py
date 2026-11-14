"""Ground truth for tests/eval/corpus/questions/actg175.yaml (the held-out set), in Polars only.

Written from the UCI variable list for ACTG 175, not from the app's code. Every column is read as
text, as the file stores it (0/1 codes, "0".."3" for trt), and numbers are cast here, so no type
inference stands between the file and the answer. Each answer carries the patients behind each
cell, so the scorer knows which cells must be hidden. The file has no missing values; a row whose
group is empty would be left out, as the app leaves it out. Codes from the variable list: trt
0-3 (ZDV, ZDV+ddI, ZDV+Zal, ddI), cid 1 = event, symptom 1 = symptomatic, drugs 1 = history of
IV drug use, wtkg = weight in kg, time = days to event or censoring.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple

import polars as pl

from clinical_analytics.eval.corpus.score import CellKey, OracleAnswer, OracleError

EVENT = "1"


class Query(NamedTuple):
    """One oracle: the lazy query, and how its collected frame becomes the answer. (A NamedTuple,
    not a dataclass: the harness loads this file by path, outside ``sys.modules``.)"""

    unit: str
    frame: pl.LazyFrame
    finish: Callable[[pl.DataFrame], OracleAnswer]


Oracle = Callable[[pl.LazyFrame], Query]


def scan(csv_path: Path) -> pl.LazyFrame:
    return pl.scan_csv(csv_path, infer_schema=False)


def _number(column: str) -> pl.Expr:
    """The column as numbers; any text that is not a number raises (strict cast)."""
    return pl.col(column).str.strip_chars().cast(pl.Float64, strict=True)


def _count(where: pl.Expr) -> Oracle:
    def build(frame: pl.LazyFrame) -> Query:
        def finish(df: pl.DataFrame) -> OracleAnswer:
            n = int(df.item())
            return OracleAnswer(n, "patients", {("total",): n})

        return Query("patients", frame.filter(where).select(pl.len()), finish)

    return build


def _count_by(group: str) -> Oracle:
    def build(frame: pl.LazyFrame) -> Query:
        def finish(df: pl.DataFrame) -> OracleAnswer:
            value = {str(row[group]): int(row["n"]) for row in df.iter_rows(named=True)}
            return OracleAnswer(value, "patients", {(key,): n for key, n in value.items()})

        query = frame.filter(pl.col(group).is_not_null()).group_by(group).agg(pl.len().alias("n"))
        return Query("patients", query, finish)

    return build


def _stat_by(stat: str, metric: str, group: str) -> Oracle:
    unit = f"{stat} {metric}"

    def build(frame: pl.LazyFrame) -> Query:
        number = _number(metric)
        reducer = number.median() if stat == "median" else number.mean()

        def finish(df: pl.DataFrame) -> OracleAnswer:
            rows = list(df.iter_rows(named=True))
            if not rows:
                raise OracleError(f"{metric} by {group}: no groups")
            value = {str(row[group]): None if row["value"] is None else float(row["value"]) for row in rows}
            return OracleAnswer(value, unit, {(str(row[group]),): int(row["n"]) for row in rows})

        query = (
            frame.filter(pl.col(group).is_not_null())
            .group_by(group)
            .agg(reducer.alias("value"), number.is_not_null().sum().alias("n"))
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
            frame.filter(pl.col(outcome).is_not_null() & pl.col(group).is_not_null())
            .group_by(group)
            .agg(pl.len().alias("n"), (pl.col(outcome) == EVENT).sum().alias("events"))
        )
        return Query("proportion", query, finish)

    return build


def _held(frame: pl.LazyFrame) -> Query:
    def finish(df: pl.DataFrame) -> OracleAnswer:
        n = int(df.item())
        return OracleAnswer(None, "listing", {("total",): n})

    return Query("listing", frame.filter(pl.col("cid") == EVENT).select(pl.len()), finish)


ORACLES: dict[str, Oracle] = {
    "actg175-total-001": _count(pl.lit(True)),
    "actg175-arm-002": _count_by("trt"),
    "actg175-trt-003": _count_by("trt"),
    "actg175-cd4-004": _stat_by("mean", "cd40", "trt"),
    "actg175-age-005": _stat_by("median", "age", "gender"),
    "actg175-filter-006": _count((_number("age") > 40) & (_number("cd40") < 200)),
    "actg175-event-007": _rates("cid", "trt"),
    "actg175-weight-008": _stat_by("mean", "wtkg", "race"),
    "actg175-symptom-009": _count_by("symptom"),
    "actg175-time-010": _stat_by("median", "time", "trt"),
    "actg175-drugs-011": _count(pl.col("drugs") == "1"),
    "actg175-listing-012": _held,
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
