"""Ground truth for tests/eval/corpus/questions/covid_ms.yaml, in Polars only.

Written from the PhysioNet data description (GDSI v1.0.1), not from the app's code. Every
column is read as text, as the file stores it ("yes"/"no", "female"/"male", "1"/"2"), so no
type inference stands between the file and the answer. Each answer carries the patients behind
each cell, so the scorer knows which cells must be hidden.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path

import polars as pl

from clinical_analytics.eval.corpus.score import CellKey, OracleAnswer

HOSPITAL = "covid19_admission_hospital"
CONFIRMED = "covid19_confirmed_case"
ONSET = "year_onset"
YES = "yes"
Z_95 = 1.959963984540054

Oracle = Callable[[pl.LazyFrame], OracleAnswer]


def scan(csv_path: Path) -> pl.LazyFrame:
    return pl.scan_csv(csv_path, infer_schema=False)


def _total(frame: pl.LazyFrame) -> OracleAnswer:
    n = int(frame.select(pl.len()).collect().item())
    return OracleAnswer(n, "patients", {("total",): n})


def _count_by(group: str, where: pl.Expr = pl.lit(True)) -> Oracle:
    def compute(frame: pl.LazyFrame) -> OracleAnswer:
        counts = frame.filter(where).group_by(group).agg(pl.len().alias("n")).collect()
        value = {str(row[group]): int(row["n"]) for row in counts.iter_rows(named=True)}
        return OracleAnswer(value, "patients", {(key,): n for key, n in value.items()})

    return compute


def _count(where: pl.Expr) -> Oracle:
    def compute(frame: pl.LazyFrame) -> OracleAnswer:
        n = int(frame.filter(where).select(pl.len()).collect().item())
        return OracleAnswer(n, "patients", {("total",): n})

    return compute


def _stat_by(stat: str, metric: str, group: str) -> Oracle:
    def compute(frame: pl.LazyFrame) -> OracleAnswer:
        value_expr = pl.col(metric).cast(pl.Float64)
        reducer = value_expr.median() if stat == "median" else value_expr.mean()
        rows = (
            frame.group_by(group)
            .agg(reducer.alias("value"), value_expr.is_not_null().sum().alias("n"))
            .collect()
            .iter_rows(named=True)
        )
        rows = list(rows)
        value = {str(row[group]): float(row["value"]) for row in rows}
        return OracleAnswer(value, f"{stat} {metric}", {(str(row[group]),): int(row["n"]) for row in rows})

    return compute


def _rates(outcome: str, group: str) -> Oracle:
    def compute(frame: pl.LazyFrame) -> OracleAnswer:
        rows = list(
            frame.filter(pl.col(outcome).is_not_null())
            .group_by(group)
            .agg(pl.len().alias("n"), (pl.col(outcome) == YES).sum().alias("events"))
            .collect()
            .iter_rows(named=True)
        )
        cells: dict[CellKey, int] = {}
        value: dict[str, float] = {}
        for row in rows:
            key, n, events = str(row[group]), int(row["n"]), int(row["events"])
            value[key] = events / n
            cells |= {(key,): n, (key, "events"): events, (key, "non_events"): n - events}
        return OracleAnswer(value, "proportion", cells)

    return compute


def _two_by_two(frame: pl.LazyFrame, exposure: str, outcome: str) -> dict[tuple[str, str], int]:
    counts = frame.group_by(exposure, outcome).agg(pl.len().alias("n")).collect()
    table = {(str(r[exposure]), str(r[outcome])): int(r["n"]) for r in counts.iter_rows(named=True)}
    return {(e, o): table.get((e, o), 0) for e in ("yes", "no") for o in ("yes", "no")}


def _chi_square(exposure: str, outcome: str) -> Oracle:
    """Pearson chi-square on the 2x2 table, with and without Yates' correction (df=1)."""

    def compute(frame: pl.LazyFrame) -> OracleAnswer:
        counts = frame.group_by(exposure, outcome).agg(pl.len().alias("n")).collect()
        cells = {(str(r[exposure]), str(r[outcome])): int(r["n"]) for r in counts.iter_rows(named=True)}
        rows = sorted({k[0] for k in cells})
        cols = sorted({k[1] for k in cells})
        total = sum(cells.values())
        row_sum = {r: sum(cells.get((r, c), 0) for c in cols) for r in rows}
        col_sum = {c: sum(cells.get((r, c), 0) for r in rows) for c in cols}
        plain = yates = 0.0
        for r in rows:
            for c in cols:
                expected = row_sum[r] * col_sum[c] / total
                observed = cells.get((r, c), 0)
                plain += (observed - expected) ** 2 / expected
                yates += (max(abs(observed - expected) - 0.5, 0.0)) ** 2 / expected
        # Survival function of chi-square with 1 df: erfc(sqrt(x / 2)).
        value = {"pearson": math.erfc(math.sqrt(plain / 2)), "yates": math.erfc(math.sqrt(yates / 2))}
        return OracleAnswer(value, "p", {(r, c): n for (r, c), n in cells.items()})

    return compute


def _odds_ratio(exposure: str, outcome: str) -> Oracle:
    """Sample odds ratio and Woolf (Wald) 95% CI; equal to an unadjusted logistic regression."""

    def compute(frame: pl.LazyFrame) -> OracleAnswer:
        t = _two_by_two(frame, exposure, outcome)
        a, b, c, d = t[("yes", "yes")], t[("yes", "no")], t[("no", "yes")], t[("no", "no")]
        log_or = math.log((a * d) / (b * c))
        se = math.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
        value = {
            "or": math.exp(log_or),
            "ci_low": math.exp(log_or - Z_95 * se),
            "ci_high": math.exp(log_or + Z_95 * se),
        }
        return OracleAnswer(value, "odds ratio", {(f"{exposure}={e}", f"{outcome}={o}"): n for (e, o), n in t.items()})

    return compute


def _held(frame: pl.LazyFrame) -> OracleAnswer:
    n = int(frame.filter(pl.col("covid19_icu_stay") == YES).select(pl.len()).collect().item())
    return OracleAnswer(None, "listing", {("total",): n})


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
    "covid_ms-chisq-011": _chi_square("sex", CONFIRMED),
    "covid_ms-or-012": _odds_ratio("has_comorbidities", HOSPITAL),
    "covid_ms-confirmed-013": _count_by("ms_type2", _CONFIRMED),
    "covid_ms-followup-014": _count_by("ms_type2", _CONFIRMED & _FEMALE),
    "covid_ms-listing-015": _held,
}


def answers(csv_path: Path) -> dict[str, OracleAnswer]:
    frame = scan(csv_path)
    return {question_id: oracle(frame) for question_id, oracle in ORACLES.items()}
