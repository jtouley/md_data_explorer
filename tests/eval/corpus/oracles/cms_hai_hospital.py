"""Ground truth for tests/eval/corpus/questions/cms_hai_hospital.yaml, in Polars only.

Written from the CMS Provider Data dictionary for Healthcare-Associated Infections, not from the
app's code. Both files are long: one row per facility (or state) x Measure ID, value in Score. A
C. diff SIR and its CI are three rows (HAI_6_SIR, HAI_6_CILOWER, HAI_6_CIUPPER); the oracle pivots
them per facility after checking there is one row of each. "Not Available" is no value.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from clinical_analytics.eval.corpus import oracle_kit as kit
from clinical_analytics.eval.corpus.score import OracleAnswer

HOSPITAL = "hospital"
STATE = "state"
FILES = {HOSPITAL: "Healthcare_Associated_Infections-Hospital.csv", STATE: "Healthcare_Associated_Infections-State.csv"}
SCORE = "Score"
CDIFF = "HAI_6"
CLABSI = "HAI_1"
UNIT = "SIR"


def _measure(measure_id: str) -> pl.Expr:
    return pl.col("Measure ID") == measure_id


def _part(measure_id: str) -> pl.Expr:
    return kit.number(SCORE).filter(_measure(measure_id)).first()


def _per_key(
    tables: kit.Tables, table: str, key: str, measures: dict[str, str], where: pl.Expr = pl.lit(True)
) -> pl.LazyFrame:
    """One row per ``key`` with each measure's Score as a number; ``rows`` counts the long rows."""
    return (
        tables[table]
        .filter(where & pl.col("Measure ID").is_in(list(measures.values())))
        .group_by(key)
        .agg(pl.len().alias("rows"), *(_part(m).alias(name) for name, m in measures.items()))
    )


_CI = {"estimate": f"{CDIFF}_SIR", "ci_low": f"{CDIFF}_CILOWER", "ci_high": f"{CDIFF}_CIUPPER"}


def _ci_by(table: str, key: str, where: pl.Expr = pl.lit(True), top: int | None = None) -> kit.Oracle:
    def build(tables: kit.Tables) -> kit.Query:
        def finish(df: pl.DataFrame) -> OracleAnswer:
            if df.height and set(df["rows"].to_list()) != {len(_CI)}:
                raise kit.OracleError(f"{key}: {sorted(set(df['rows'].to_list()))} rows per key, expected {len(_CI)}")
            found: dict[str, object] = {}
            for row in df.iter_rows(named=True):
                parts = {name: row[name] for name in _CI}
                found[str(row[key])] = None if any(v is None for v in parts.values()) else parts
            if top is not None:
                ranked = sorted((kv for kv in found.items() if kv[1] is not None), key=lambda kv: -kv[1]["estimate"])  # type: ignore[index]
                found = dict(ranked[:top])
            return OracleAnswer(found, UNIT)

        return kit.Query(UNIT, _per_key(tables, table, key, _CI, where), finish)

    return build


def _hospital_values(values: dict[str, pl.Expr], measures: dict[str, str], unit: str) -> kit.Oracle:
    """Aggregates over hospitals, from the pivoted per-facility frame."""

    def build(tables: kit.Tables) -> kit.Query:
        def finish(df: pl.DataFrame) -> OracleAnswer:
            row = df.row(0, named=True)
            if int(row["max_rows"]) > len(measures):
                raise kit.OracleError("a facility has two rows for one measure")
            if any(row[name] is None for name in values):
                raise kit.OracleError(f"no value for {[n for n in values if row[n] is None]}")
            return OracleAnswer({name: float(row[name]) for name in values}, unit)

        frame = _per_key(tables, HOSPITAL, "Facility ID", measures).select(
            pl.col("rows").max().alias("max_rows"), *(expr.alias(name) for name, expr in values.items())
        )
        return kit.Query(unit, frame, finish)

    return build


def _pooled_state_sir(tables: kit.Tables) -> kit.Query:
    """Sum of observed over sum of predicted C. diff cases per state, over hospitals with both."""

    def finish(df: pl.DataFrame) -> OracleAnswer:
        if df.height and int(df["max_rows"].max()) > 2:  # type: ignore[arg-type]
            raise kit.OracleError("a facility has two rows for one measure")
        ranked = df.filter(pl.col("sir").is_not_null()).sort("sir", descending=True).head(10)
        return OracleAnswer({str(r["State"]): float(r["sir"]) for r in ranked.iter_rows(named=True)}, UNIT)

    measures = {"observed": f"{CDIFF}_NUMERATOR", "predicted": f"{CDIFF}_ELIGCASES"}
    per_facility = (
        tables[HOSPITAL]
        .filter(pl.col("Measure ID").is_in(list(measures.values())))
        .group_by("Facility ID", "State")
        .agg(pl.len().alias("rows"), *(_part(m).alias(name) for name, m in measures.items()))
        .filter(pl.col("observed").is_not_null() & pl.col("predicted").is_not_null())
    )
    query = per_facility.group_by("State").agg(
        (pl.col("observed").sum() / pl.col("predicted").sum()).alias("sir"), pl.col("rows").max().alias("max_rows")
    )
    return kit.Query(UNIT, query, finish)


def _ri_state_ci(tables: kit.Tables) -> kit.Query:
    def finish(df: pl.DataFrame) -> OracleAnswer:
        if df.height != 1 or int(df["rows"][0]) != len(_CI):
            raise kit.OracleError(f"Rhode Island: {df.height} state rows")
        row = df.row(0, named=True)
        if any(row[name] is None for name in _CI):
            raise kit.OracleError("Rhode Island's state SIR or CI is not available")
        return OracleAnswer({name: float(row[name]) for name in _CI}, UNIT)

    return kit.Query(UNIT, _per_key(tables, STATE, "State", _CI, pl.col("State") == "RI"), finish)


_CDIFF_SIR = _measure(f"{CDIFF}_SIR")

ORACLES: dict[str, kit.Oracle] = {
    "hai-rank-states-001": kit.lookup_by(STATE, "State", SCORE, _CDIFF_SIR, UNIT, top=10),
    "hai-rank-ci-002": _ci_by(STATE, "State", top=10),
    "hai-above1-003": _hospital_values(
        {"hospitals": (pl.col("ci_low") > 1).sum().cast(pl.Float64)}, {"ci_low": f"{CDIFF}_CILOWER"}, "hospitals"
    ),
    "hai-na-004": _hospital_values(
        {"hospitals": pl.col("sir").is_null().sum().cast(pl.Float64)}, {"sir": f"{CDIFF}_SIR"}, "hospitals"
    ),
    "hai-median-005": _hospital_values(
        {
            "median": pl.col("sir").median(),
            "q1": pl.col("sir").quantile(0.25, interpolation="linear"),
            "q3": pl.col("sir").quantile(0.75, interpolation="linear"),
        },
        {"sir": f"{CDIFF}_SIR"},
        UNIT,
    ),
    "hai-benchmark-006": kit.derived_by(
        HOSPITAL, "Compared to National", pl.len().cast(pl.Float64), _CDIFF_SIR, "hospitals", unique_by="Facility ID"
    ),
    "hai-ri-007": _ci_by(HOSPITAL, "Facility Name", pl.col("State") == "RI"),
    "hai-pooled-sir-008": _pooled_state_sir,
    "hai-corr-009": _hospital_values(
        {"r": pl.corr("cdiff", "clabsi")}, {"cdiff": f"{CDIFF}_SIR", "clabsi": f"{CLABSI}_SIR"}, "correlation"
    ),
    "hai-age-010": kit.held(HOSPITAL, "facility-level data: no patient age or sex"),
    "hai-trend-011": kit.held(HOSPITAL, "one reporting period (2025)"),
    "hai-pooled-012": kit.held(
        HOSPITAL, "the C. diff rows mix the SIR with its CI bounds, case counts and patient days"
    ),
    "hai-ri-state-013": _ri_state_ci,
    "hai-before2020-014": kit.held(HOSPITAL, "one reporting period (2025)"),
}


def answers(files_dir: Path, oracles: dict[str, kit.Oracle] | None = None) -> dict[str, OracleAnswer]:
    tables = {name: kit.scan_text(files_dir / file) for name, file in FILES.items()}
    return kit.answers(tables, oracles or ORACLES)
