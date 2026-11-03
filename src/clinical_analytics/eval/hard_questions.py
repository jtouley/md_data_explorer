"""Hard and provider-style questions with DuckDB SQL oracles.

The SQL is the ground truth. The parser never sees it. Cohort cutoffs use the
same order as execution: demographic filters, then a median or percent_rank
window, then outcome flags.

Provider questions follow task shapes reported in clinician LLM studies:
short searcher queries, subgroup research counts, risk and likelihood,
workflow management, and comparison questions.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import duckdb
import polars as pl

from clinical_analytics.eval.catalog import dataset_path

CITATIONS = {
    "searcher": (
        "npj Digital Medicine physician-chatbot typology: searchers ask short queries. "
        "https://www.nature.com/articles/s41746-025-02184-y"
    ),
    "research": (
        "JMIR preprint 76941: 60% of 106,942 physician LLM queries were medical research. "
        "https://preprints.jmir.org/preprint/76941"
    ),
    "risk": (
        "JAMA Network Open LLM diagnostic-aid trial: clinicians ask likelihood and risk. "
        "https://jamanetwork.com/journals/jamanetworkopen/fullarticle/2825395"
    ),
    "workflow": ("Rao et al., JMIR 2023;25:e48659: management questions across the clinical workflow."),
    "comparison": (
        "BMC Medical Informatics and Decision Making: treatment-uncertainty questions "
        "are often comparisons and are poorly PICO-structured."
    ),
    "triage": ("medRxiv 2023.11.24.23298844: vignette questions on triage, risk, and treatment."),
    "tasks": ("PMC11023712: clinicians rate differential, treatment, and risk questions as appropriate LLM tasks."),
}


def _q(
    dataset_id: str,
    family: str,
    number: int,
    query: str,
    sql: str | PolarsOracle,
    kind: str,
    *,
    stat: str | None = None,
    group_by: str | None = None,
    citation: str | None = None,
) -> dict[str, Any]:
    return {
        "id": f"{dataset_id}-{family}-{number:03d}",
        "dataset_id": dataset_id,
        "family": family,
        "query": query,
        "sql": sql.strip() if isinstance(sql, str) else None,
        "compute": None if isinstance(sql, str) else sql,
        "kind": kind,
        "stat": stat,
        "group_by": group_by,
        "citation": citation,
    }


def _ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _count(table: str, *preds: str) -> str:
    where = " AND ".join(pred for pred in preds if pred) or "TRUE"
    return f"SELECT count(*) FROM {_ident(table)} WHERE {where}"


def _stat(table: str, fn: str, column: str, *preds: str) -> str:
    where = " AND ".join(pred for pred in preds if pred) or "TRUE"
    return f"SELECT {fn}({_ident(column)}) FROM {_ident(table)} WHERE {where}"


def _proportion(table: str, column: str, literal: str, *preds: str) -> str:
    where = " AND ".join(pred for pred in preds if pred) or "TRUE"
    return (
        f"SELECT avg(CASE WHEN {_ident(column)} = {literal} THEN 1.0 ELSE 0.0 END) FROM {_ident(table)} WHERE {where}"
    )


def _rates(table: str, group: str, column: str, literal: str, *preds: str) -> str:
    where = " AND ".join(pred for pred in preds if pred) or "TRUE"
    return (
        f"SELECT {_ident(group)}, avg(CASE WHEN {_ident(column)} = {literal} THEN 1.0 ELSE 0.0 END) "
        f"FROM {_ident(table)} WHERE {where} GROUP BY 1"
    )


def _mean_by(table: str, group: str, column: str, *preds: str) -> str:
    where = " AND ".join(pred for pred in preds if pred) or "TRUE"
    return f"SELECT {_ident(group)}, avg({_ident(column)}) FROM {_ident(table)} WHERE {where} GROUP BY 1"


def _crosstab(table: str, group: str, column: str) -> str:
    return f"SELECT {_ident(group)}, {_ident(column)}, count(*) FROM {_ident(table)} GROUP BY 1, 2"


def _corr(table: str, left: str, right: str) -> str:
    return f"SELECT corr({_ident(left)}, {_ident(right)}) FROM {_ident(table)}"


# Cutoff oracles are computed in Polars, not SQL: production applies these cutoffs with
# DuckDB SQL, and an oracle that copies that SQL would agree with it by construction.
PolarsOracle = Callable[[pl.DataFrame], Any]


def _decile_avg(column: str, where: pl.Expr = pl.lit(True)) -> PolarsOracle:
    """Mean of the rows at or above the 90th percentile (linear interpolation) of the cohort."""

    def compute(frame: pl.DataFrame) -> float:
        cohort = frame.filter(where)
        cut = cohort.get_column(column).quantile(0.9, interpolation="linear")
        return float(cohort.filter(pl.col(column) >= cut).get_column(column).mean())  # type: ignore[arg-type]

    return compute


def _median_count(median_column: str, outcome: pl.Expr, where: pl.Expr = pl.lit(True)) -> PolarsOracle:
    """Rows strictly above the cohort median that also meet the outcome."""

    def compute(frame: pl.DataFrame) -> int:
        cohort = frame.filter(where)
        cut = cohort.get_column(median_column).median()
        return cohort.filter((pl.col(median_column) > cut) & outcome).height

    return compute


def _finish(dataset_id: str, family: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    queries = [row["query"] for row in rows]
    if len(set(queries)) != len(queries):
        raise ValueError(f"{dataset_id} {family} has duplicate questions")
    numbered: list[dict[str, Any]] = []
    for number, row in enumerate(rows, start=1):
        numbered.append(
            _q(
                dataset_id,
                family,
                number,
                row["query"],
                row["sql"],
                row["kind"],
                stat=row.get("stat"),
                group_by=row.get("group_by"),
                citation=row.get("citation"),
            )
        )
    return numbered


def _row(
    query: str,
    sql: str | PolarsOracle,
    kind: str,
    *,
    stat: str | None = None,
    group_by: str | None = None,
    citation: str | None = None,
) -> dict[str, Any]:
    return {
        "query": query,
        "sql": sql,
        "kind": kind,
        "stat": stat,
        "group_by": group_by,
        "citation": citation,
    }


def _cohort_analytical(table: str) -> list[dict[str, Any]]:
    """Cohort restrictions, cutoffs and rates on the synthetic cohort."""
    age, ldl, hdl, bmi = "age", "ldl", "hdl", "bmi"
    women, men = f"{_ident('sex')} = 'F'", f"{_ident('sex')} = 'M'"
    drug, placebo = f"{_ident('treatment')} = 'drug'", f"{_ident('treatment')} = 'placebo'"
    diabetic = f"{_ident('diabetes')} = 1"
    over65 = f"{_ident(age)} > 65"
    outcome = f"{_ident('outcome')} = 1"
    rows = [
        _row("How many women are there?", _count(table, women), "count"),
        _row("How many men over 65?", _count(table, men, over65), "count"),
        _row("How many women over 65 have diabetes?", _count(table, women, over65, diabetic), "count"),
        _row("How many patients received placebo?", _count(table, placebo), "count"),
        _row("How many patients on the drug had the outcome?", _count(table, drug, outcome), "count"),
    ]
    for who, preds in (("for women", (women,)), ("for men", (men,)), ("for patients over 65", (over65,))):
        for phrase, column in (("ldl", ldl), ("bmi", bmi)):
            for stat_name, fn in (("average", "avg"), ("median", "median")):
                rows.append(
                    _row(
                        f"What is the {stat_name} {phrase} {who}?",
                        _stat(table, fn, column, *preds),
                        "stat",
                        stat=stat_name,
                    )
                )
    rows.extend(
        [
            _row("What is the average age in the oldest decile?", _decile_avg(age), "stat", stat="average"),
            _row(
                "Among women, what is the average ldl in the top decile of ldl?",
                _decile_avg(ldl, pl.col("sex") == "F"),
                "stat",
                stat="average",
            ),
            _row(
                "Among men older than the median age, how many had the outcome?",
                _median_count(age, pl.col("outcome") == 1, pl.col("sex") == "M"),
                "count",
            ),
            _row("What proportion of women have diabetes?", _proportion(table, "diabetes", "1", women), "proportion"),
            _row("What proportion of men had the outcome?", _proportion(table, "outcome", "1", men), "proportion"),
            _row(
                "Are women more likely than men to have diabetes?",
                _rates(table, "sex", "diabetes", "1"),
                "rates",
                group_by="sex",
            ),
            _row("Compare hdl by sex", _mean_by(table, "sex", hdl), "mean_by", group_by="sex"),
            _row("What is the correlation between age and bmi?", _corr(table, age, bmi), "corr"),
            # "male" is a substring of "female": the value binding must pick the right sex.
            _row("How many male patients?", _count(table, men), "count"),
            _row(
                "What is the average ldl for female patients?", _stat(table, "avg", ldl, women), "stat", stat="average"
            ),
        ]
    )
    return rows


def _cohort_provider(table: str) -> list[dict[str, Any]]:
    """Clinician-shaped questions (short searches, subgroup counts, risk, comparison)."""
    age, ldl = "age", "ldl"
    women, men = f"{_ident('sex')} = 'F'", f"{_ident('sex')} = 'M'"
    drug, placebo = f"{_ident('treatment')} = 'drug'", f"{_ident('treatment')} = 'placebo'"
    over70 = f"{_ident(age)} > 70"
    outcome = f"{_ident('outcome')} = 1"
    diabetic = f"{_ident('diabetes')} = 1"
    return [
        _row("ldl for women?", _stat(table, "avg", ldl, women), "stat", stat="average", citation=CITATIONS["searcher"]),
        _row(
            "median ldl over 70?",
            _stat(table, "median", ldl, over70),
            "stat",
            stat="median",
            citation=CITATIONS["searcher"],
        ),
        _row(
            "How many women over 70 are on the drug?",
            _count(table, women, over70, drug),
            "count",
            citation=CITATIONS["research"],
        ),
        _row(
            "How many men on placebo have diabetes?",
            _count(table, men, placebo, diabetic),
            "count",
            citation=CITATIONS["research"],
        ),
        _row(
            "Risk check: what proportion of patients over 70 had the outcome?",
            _proportion(table, "outcome", "1", over70),
            "proportion",
            citation=CITATIONS["risk"],
        ),
        _row(
            "Are patients on placebo more likely to have the outcome?",
            _rates(table, "treatment", "outcome", "1"),
            "rates",
            group_by="treatment",
            citation=CITATIONS["comparison"],
        ),
        _row(
            "Compare ldl by treatment",
            _mean_by(table, "treatment", ldl),
            "mean_by",
            group_by="treatment",
            citation=CITATIONS["comparison"],
        ),
        _row(
            "What is the average ldl for patients on the drug?",
            _stat(table, "avg", ldl, drug),
            "stat",
            stat="average",
            citation=CITATIONS["workflow"],
        ),
        _row(
            "How many patients over 70 had the outcome?",
            _count(table, over70, outcome),
            "count",
            citation=CITATIONS["triage"],
        ),
    ]


_BUILDERS = {
    "cohort": (_cohort_analytical, _cohort_provider),
}


def questions_for_dataset(dataset_id: str) -> list[dict[str, Any]]:
    """Analytical and provider questions for one dataset, each with a SQL oracle."""
    if dataset_id not in _BUILDERS:
        known = ", ".join(sorted(_BUILDERS))
        raise KeyError(f"No hard questions for '{dataset_id}'. Known: {known}")
    analytical, provider = _BUILDERS[dataset_id]
    return _finish(dataset_id, "analytical", analytical(dataset_id)) + _finish(
        dataset_id, "provider", provider(dataset_id)
    )


def all_hard_questions() -> list[dict[str, Any]]:
    questions: list[dict[str, Any]] = []
    for dataset_id in _BUILDERS:
        questions.extend(questions_for_dataset(dataset_id))
    return questions


def oracle_value(question: dict[str, Any], workspace: Path | None = None) -> Any:
    """Expected answer: the question's SQL in DuckDB, or its Polars oracle, over the dataset CSV."""
    dataset_id = question["dataset_id"]
    path = dataset_path(dataset_id, workspace)
    compute = question.get("compute")
    if compute is not None:
        value = compute(pl.read_csv(path))
        return None if value is None else (int(value) if question["kind"] == "count" else float(value))
    con = duckdb.connect()
    try:
        literal = "'" + str(path).replace("'", "''") + "'"
        con.execute(f"CREATE VIEW {_ident(dataset_id)} AS SELECT * FROM read_csv_auto({literal})")
        rows = con.execute(question["sql"]).fetchall()
    finally:
        con.close()
    kind = question["kind"]
    if kind in {"count", "stat", "proportion", "corr"}:
        value = rows[0][0]
        if value is None:
            return None
        if kind == "count":
            return int(value)
        return float(value)
    if kind in {"rates", "mean_by"}:
        mapped: dict[str, float | None] = {}
        for key, value in rows:
            label = "null" if key is None else str(key)
            mapped[label] = None if value is None else float(value)
        return mapped
    if kind == "crosstab":
        grouped: dict[str, dict[str, int]] = {}
        for key, level, count in rows:
            group_key = "null" if key is None else str(key)
            level_key = "null" if level is None else str(level)
            grouped.setdefault(group_key, {})[level_key] = int(count)
        return grouped
    raise ValueError(f"Unknown question kind {kind}")
