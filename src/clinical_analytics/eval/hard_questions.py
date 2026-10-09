"""Hard and provider-style questions with DuckDB SQL oracles.

The SQL is the ground truth. The parser never sees it. Cohort cutoffs use the
same order as execution: demographic filters, then a median or percent_rank
window, then outcome flags.

Provider questions follow task shapes reported in clinician LLM studies:
short searcher queries, subgroup research counts, risk and likelihood,
workflow management, and comparison questions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import duckdb

from clinical_analytics.eval.catalog import DATASET_TABLES, duckdb_path

PER_FAMILY = 50

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
    sql: str,
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
        "sql": sql.strip(),
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


def _decile_avg(table: str, column: str, *preds: str) -> str:
    where = " AND ".join(pred for pred in preds if pred) or "TRUE"
    col = _ident(column)
    return f"""
    WITH cohort AS (SELECT * FROM {_ident(table)} WHERE {where}),
    cut AS (SELECT quantile_cont({col}, 0.9) AS q FROM cohort)
    SELECT avg({col}) FROM cohort, cut WHERE {col} >= q
    """


def _median_count(table: str, median_column: str, outcome: str, *preds: str) -> str:
    where = " AND ".join(pred for pred in preds if pred) or "TRUE"
    col = _ident(median_column)
    return f"""
    WITH cohort AS (SELECT * FROM {_ident(table)} WHERE {where}),
    cut AS (SELECT median({col}) AS m FROM cohort),
    limited AS (SELECT cohort.* FROM cohort, cut WHERE cohort.{col} > cut.m)
    SELECT count(*) FROM limited WHERE {outcome}
    """


def _median_avg(table: str, column: str, *preds: str) -> str:
    where = " AND ".join(pred for pred in preds if pred) or "TRUE"
    col = _ident(column)
    return f"""
    WITH cohort AS (SELECT * FROM {_ident(table)} WHERE {where}),
    cut AS (SELECT median({col}) AS m FROM cohort),
    limited AS (SELECT cohort.* FROM cohort, cut WHERE cohort.{col} > cut.m)
    SELECT avg({col}) FROM limited
    """


def _finish(dataset_id: str, family: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(rows) != PER_FAMILY:
        raise ValueError(f"{dataset_id} {family} has {len(rows)} questions, expected {PER_FAMILY}")
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
    sql: str,
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


def _gdsi_analytical(table: str) -> list[dict[str, Any]]:
    icu = "covid19_icu_stay"
    confirmed = "covid19_confirmed_case"
    ventilation = "covid19_ventilation"
    comorbid = "has_comorbidities"
    recovered = "covid19_outcome_recovered"
    onset = "year_onset"
    duration = "duration_treatment_cat2"
    sex = "sex"
    ms = "ms_type2"
    flags = [
        ("stayed in the ICU", icu),
        ("were confirmed COVID cases", confirmed),
        ("needed ventilation", ventilation),
        ("have comorbidities", comorbid),
        ("recovered", recovered),
    ]
    rows: list[dict[str, Any]] = []
    for phrase, column in flags:
        rows.append(
            _row(
                f"How many patients {phrase}?",
                _count(table, f"{_ident(column)} = 'yes'"),
                "count",
            )
        )
    for who, token in (("women", "female"), ("men", "male")):
        for phrase, column in flags:
            rows.append(
                _row(
                    f"How many {who} {phrase}?",
                    _count(table, f"{_ident(sex)} = '{token}'", f"{_ident(column)} = 'yes'"),
                    "count",
                )
            )
    rows.extend(
        [
            _row(
                "What is the average year of onset for women?",
                _stat(table, "avg", onset, f"{_ident(sex)} = 'female'"),
                "stat",
                stat="average",
            ),
            _row(
                "What is the median year of onset for men?",
                _stat(table, "median", onset, f"{_ident(sex)} = 'male'"),
                "stat",
                stat="median",
            ),
            _row(
                "What is the average duration treatment for women?",
                _stat(table, "avg", duration, f"{_ident(sex)} = 'female'"),
                "stat",
                stat="average",
            ),
            _row(
                "What is the median duration treatment for men?",
                _stat(table, "median", duration, f"{_ident(sex)} = 'male'"),
                "stat",
                stat="median",
            ),
            _row(
                "Among women, what is the average year of onset in the top decile of year of onset?",
                _decile_avg(table, onset, f"{_ident(sex)} = 'female'"),
                "stat",
                stat="average",
            ),
            _row(
                "Among men, what is the average duration treatment in the top decile of duration treatment?",
                _decile_avg(table, duration, f"{_ident(sex)} = 'male'"),
                "stat",
                stat="average",
            ),
            _row(
                "Among women higher than the median year of onset, how many stayed in the ICU?",
                _median_count(table, onset, f"{_ident(icu)} = 'yes'", f"{_ident(sex)} = 'female'"),
                "count",
            ),
            _row(
                "Among men higher than the median duration treatment, how many were confirmed COVID cases?",
                _median_count(table, duration, f"{_ident(confirmed)} = 'yes'", f"{_ident(sex)} = 'male'"),
                "count",
            ),
            _row(
                "What proportion of women stayed in the ICU?",
                _proportion(table, icu, "'yes'", f"{_ident(sex)} = 'female'"),
                "proportion",
            ),
            _row(
                "What proportion of confirmed cases stayed in the ICU?",
                _proportion(table, icu, "'yes'", f"{_ident(confirmed)} = 'yes'"),
                "proportion",
            ),
            _row(
                "What proportion of women recovered?",
                _proportion(table, recovered, "'yes'", f"{_ident(sex)} = 'female'"),
                "proportion",
            ),
            _row(
                "Compare year of onset by sex",
                _mean_by(table, sex, onset),
                "mean_by",
                group_by=sex,
            ),
            _row("Compare duration treatment by sex", _mean_by(table, sex, duration), "mean_by", group_by=sex),
            _row(
                "Are women more likely than men to have needed ventilation?",
                _rates(table, sex, ventilation, "'yes'"),
                "rates",
                group_by=sex,
            ),
            _row(
                "What is the correlation between year of onset and duration treatment?",
                _corr(table, onset, duration),
                "corr",
            ),
            _row(
                "How many women with relapsing ms stayed in the ICU?",
                _count(
                    table,
                    f"{_ident(sex)} = 'female'",
                    f"{_ident(ms)} = 'relapsing_remitting'",
                    f"{_ident(icu)} = 'yes'",
                ),
                "count",
            ),
            _row(
                "How many men with progressive ms were confirmed COVID cases?",
                _count(
                    table,
                    f"{_ident(sex)} = 'male'",
                    f"{_ident(ms)} = 'progressive_MS'",
                    f"{_ident(confirmed)} = 'yes'",
                ),
                "count",
            ),
            _row(
                "What is the average year of onset for relapsing ms?",
                _stat(table, "avg", onset, f"{_ident(ms)} = 'relapsing_remitting'"),
                "stat",
                stat="average",
            ),
            _row(
                "What is the median duration treatment for progressive ms?",
                _stat(table, "median", duration, f"{_ident(ms)} = 'progressive_MS'"),
                "stat",
                stat="median",
            ),
            _row(
                "Among women with relapsing ms, what is the average year of onset in the top decile of year of onset?",
                _decile_avg(
                    table,
                    onset,
                    f"{_ident(sex)} = 'female'",
                    f"{_ident(ms)} = 'relapsing_remitting'",
                ),
                "stat",
                stat="average",
            ),
            _row(
                "What proportion of men were confirmed COVID cases?",
                _proportion(table, confirmed, "'yes'", f"{_ident(sex)} = 'male'"),
                "proportion",
            ),
            _row(
                "What share of women needed ventilation?",
                _proportion(table, ventilation, "'yes'", f"{_ident(sex)} = 'female'"),
                "proportion",
            ),
            _row(
                "What proportion of men recovered?",
                _proportion(table, recovered, "'yes'", f"{_ident(sex)} = 'male'"),
                "proportion",
            ),
            _row(
                "What share of women were confirmed COVID cases?",
                _proportion(table, confirmed, "'yes'", f"{_ident(sex)} = 'female'"),
                "proportion",
            ),
            _row(
                "How many women with progressive ms stayed in the ICU?",
                _count(
                    table,
                    f"{_ident(sex)} = 'female'",
                    f"{_ident(ms)} = 'progressive_MS'",
                    f"{_ident(icu)} = 'yes'",
                ),
                "count",
            ),
            _row(
                "How many men with relapsing ms were confirmed COVID cases?",
                _count(
                    table,
                    f"{_ident(sex)} = 'male'",
                    f"{_ident(ms)} = 'relapsing_remitting'",
                    f"{_ident(confirmed)} = 'yes'",
                ),
                "count",
            ),
            _row(
                "What is the median year of onset for women?",
                _stat(table, "median", onset, f"{_ident(sex)} = 'female'"),
                "stat",
                stat="median",
            ),
            _row(
                "What is the average year of onset for men?",
                _stat(table, "avg", onset, f"{_ident(sex)} = 'male'"),
                "stat",
                stat="average",
            ),
            _row(
                "What is the average duration treatment for men?",
                _stat(table, "avg", duration, f"{_ident(sex)} = 'male'"),
                "stat",
                stat="average",
            ),
            _row(
                "What is the median duration treatment for women?",
                _stat(table, "median", duration, f"{_ident(sex)} = 'female'"),
                "stat",
                stat="median",
            ),
            _row(
                "Among men, what is the average year of onset in the top decile of year of onset?",
                _decile_avg(table, onset, f"{_ident(sex)} = 'male'"),
                "stat",
                stat="average",
            ),
            _row(
                "Among women higher than the median duration treatment, how many were confirmed COVID cases?",
                _median_count(table, duration, f"{_ident(confirmed)} = 'yes'", f"{_ident(sex)} = 'female'"),
                "count",
            ),
            _row(
                "Are men more likely than women to have comorbidities?",
                _rates(table, sex, comorbid, "'yes'"),
                "rates",
                group_by=sex,
            ),
            _row(
                "What proportion of men needed ventilation?",
                _proportion(table, ventilation, "'yes'", f"{_ident(sex)} = 'male'"),
                "proportion",
            ),
            _row(
                "Among women higher than the median year of onset, how many needed ventilation?",
                _median_count(table, onset, f"{_ident(ventilation)} = 'yes'", f"{_ident(sex)} = 'female'"),
                "count",
            ),
        ]
    )
    return _finish("gdsi", "analytical", rows)


def _gdsi_provider(table: str) -> list[dict[str, Any]]:
    icu = "covid19_icu_stay"
    confirmed = "covid19_confirmed_case"
    ventilation = "covid19_ventilation"
    comorbid = "has_comorbidities"
    recovered = "covid19_outcome_recovered"
    onset = "year_onset"
    duration = "duration_treatment_cat2"
    sex = "sex"
    rows = [
        _row(
            "Average year of onset for women",
            _stat(table, "avg", onset, f"{_ident(sex)} = 'female'"),
            "stat",
            stat="average",
            citation=CITATIONS["searcher"],
        ),
        _row(
            "How many stayed in the ICU?",
            _count(table, f"{_ident(icu)} = 'yes'"),
            "count",
            citation=CITATIONS["searcher"],
        ),
        _row(
            "How many confirmed COVID cases?",
            _count(table, f"{_ident(confirmed)} = 'yes'"),
            "count",
            citation=CITATIONS["searcher"],
        ),
        _row(
            "Compare recovery by sex",
            _crosstab(table, sex, recovered),
            "crosstab",
            citation=CITATIONS["comparison"],
        ),
        _row(
            "Are men more likely to be admitted to the ICU?",
            _rates(table, sex, icu, "'yes'"),
            "rates",
            group_by=sex,
            citation=CITATIONS["risk"],
        ),
        _row(
            "Are men more likely than women to have recovered?",
            _rates(table, sex, recovered, "'yes'"),
            "rates",
            group_by=sex,
            citation=CITATIONS["risk"],
        ),
        _row(
            "For the research cohort, how many women were confirmed COVID cases?",
            _count(table, f"{_ident(sex)} = 'female'", f"{_ident(confirmed)} = 'yes'"),
            "count",
            citation=CITATIONS["research"],
        ),
        _row(
            "For the research cohort, how many men stayed in the ICU?",
            _count(table, f"{_ident(sex)} = 'male'", f"{_ident(icu)} = 'yes'"),
            "count",
            citation=CITATIONS["research"],
        ),
        _row(
            "For the research cohort, how many women needed ventilation?",
            _count(table, f"{_ident(sex)} = 'female'", f"{_ident(ventilation)} = 'yes'"),
            "count",
            citation=CITATIONS["research"],
        ),
        _row(
            "For the research cohort, how many men have comorbidities?",
            _count(table, f"{_ident(sex)} = 'male'", f"{_ident(comorbid)} = 'yes'"),
            "count",
            citation=CITATIONS["research"],
        ),
        _row(
            "What proportion of men stayed in the ICU?",
            _proportion(table, icu, "'yes'", f"{_ident(sex)} = 'male'"),
            "proportion",
            citation=CITATIONS["triage"],
        ),
        _row(
            "What proportion of women needed ventilation?",
            _proportion(table, ventilation, "'yes'", f"{_ident(sex)} = 'female'"),
            "proportion",
            citation=CITATIONS["triage"],
        ),
        _row(
            "What proportion of men have comorbidities?",
            _proportion(table, comorbid, "'yes'", f"{_ident(sex)} = 'male'"),
            "proportion",
            citation=CITATIONS["tasks"],
        ),
        _row(
            "What proportion of women were confirmed COVID cases?",
            _proportion(table, confirmed, "'yes'", f"{_ident(sex)} = 'female'"),
            "proportion",
            citation=CITATIONS["tasks"],
        ),
        _row(
            "Before deciding on monitoring, what is the median year of onset for women?",
            _stat(table, "median", onset, f"{_ident(sex)} = 'female'"),
            "stat",
            stat="median",
            citation=CITATIONS["workflow"],
        ),
        _row(
            "Before deciding on monitoring, what is the average duration treatment for men?",
            _stat(table, "avg", duration, f"{_ident(sex)} = 'male'"),
            "stat",
            stat="average",
            citation=CITATIONS["workflow"],
        ),
        _row(
            "Compare year of onset by sex for the panel",
            _mean_by(table, sex, onset),
            "mean_by",
            group_by=sex,
            citation=CITATIONS["comparison"],
        ),
    ]
    # The loop below fills the remaining provider slots with cited subgroup and window questions.
    extras = [
        (
            "Subgroup check: how many women recovered?",
            _count(table, f"{_ident(sex)} = 'female'", f"{_ident(recovered)} = 'yes'"),
            "count",
            None,
            None,
            "research",
        ),
        (
            "Subgroup check: how many men recovered?",
            _count(table, f"{_ident(sex)} = 'male'", f"{_ident(recovered)} = 'yes'"),
            "count",
            None,
            None,
            "research",
        ),
        (
            "Subgroup check: how many women have comorbidities?",
            _count(table, f"{_ident(sex)} = 'female'", f"{_ident(comorbid)} = 'yes'"),
            "count",
            None,
            None,
            "research",
        ),
        (
            "Risk check: what proportion of men recovered?",
            _proportion(table, recovered, "'yes'", f"{_ident(sex)} = 'male'"),
            "proportion",
            None,
            None,
            "risk",
        ),
        (
            "Risk check: what proportion of women have comorbidities?",
            _proportion(table, comorbid, "'yes'", f"{_ident(sex)} = 'female'"),
            "proportion",
            None,
            None,
            "risk",
        ),
        (
            "Management check: what is the average year of onset for men?",
            _stat(table, "avg", onset, f"{_ident(sex)} = 'male'"),
            "stat",
            "average",
            None,
            "workflow",
        ),
        (
            "Management check: what is the median duration treatment for women?",
            _stat(table, "median", duration, f"{_ident(sex)} = 'female'"),
            "stat",
            "median",
            None,
            "workflow",
        ),
        (
            "Comparison check: compare duration treatment by sex",
            _mean_by(table, sex, duration),
            "mean_by",
            None,
            sex,
            "comparison",
        ),
        (
            "Searcher check: how many men needed ventilation?",
            _count(table, f"{_ident(sex)} = 'male'", f"{_ident(ventilation)} = 'yes'"),
            "count",
            None,
            None,
            "searcher",
        ),
        (
            "Searcher check: how many women stayed in the ICU?",
            _count(table, f"{_ident(sex)} = 'female'", f"{_ident(icu)} = 'yes'"),
            "count",
            None,
            None,
            "searcher",
        ),
    ]
    # Repeat cited window and median-cutoff questions until the provider family is full.
    windows = [
        (
            "Among women in the top decile of year of onset, what is the average year of onset?",
            _decile_avg(table, onset, f"{_ident(sex)} = 'female'"),
            "stat",
            "average",
            "research",
        ),
        (
            "Among men in the top decile of duration treatment, what is the average duration treatment?",
            _decile_avg(table, duration, f"{_ident(sex)} = 'male'"),
            "stat",
            "average",
            "research",
        ),
        (
            "Among women higher than the median duration treatment, how many stayed in the ICU?",
            _median_count(table, duration, f"{_ident(icu)} = 'yes'", f"{_ident(sex)} = 'female'"),
            "count",
            None,
            "triage",
        ),
        (
            "Among men higher than the median year of onset, how many needed ventilation?",
            _median_count(table, onset, f"{_ident(ventilation)} = 'yes'", f"{_ident(sex)} = 'male'"),
            "count",
            None,
            "triage",
        ),
        (
            "Among women higher than the median year of onset, how many recovered?",
            _median_count(table, onset, f"{_ident(recovered)} = 'yes'", f"{_ident(sex)} = 'female'"),
            "count",
            None,
            "workflow",
        ),
    ]
    more = [
        (
            "Literature check: how many women with relapsing ms were confirmed COVID cases?",
            _count(
                table,
                f"{_ident(sex)} = 'female'",
                "\"ms_type2\" = 'relapsing_remitting'",
                f"{_ident(confirmed)} = 'yes'",
            ),
            "count",
            None,
            None,
            "research",
        ),
        (
            "Literature check: how many men with progressive ms stayed in the ICU?",
            _count(
                table,
                f"{_ident(sex)} = 'male'",
                "\"ms_type2\" = 'progressive_MS'",
                f"{_ident(icu)} = 'yes'",
            ),
            "count",
            None,
            None,
            "research",
        ),
        (
            "Likelihood check: are men more likely than women to have stayed in the ICU?",
            _rates(table, sex, icu, "'yes'"),
            "rates",
            None,
            sex,
            "risk",
        ),
        (
            "Likelihood check: are women more likely than men to have recovered?",
            _rates(table, sex, recovered, "'yes'"),
            "rates",
            None,
            sex,
            "risk",
        ),
        (
            "Workup check: what is the average year of onset for relapsing ms?",
            _stat(table, "avg", onset, "\"ms_type2\" = 'relapsing_remitting'"),
            "stat",
            "average",
            None,
            "workflow",
        ),
        (
            "Workup check: what is the median duration treatment for progressive ms?",
            _stat(table, "median", duration, "\"ms_type2\" = 'progressive_MS'"),
            "stat",
            "median",
            None,
            "workflow",
        ),
        (
            "Panel comparison: what proportion of confirmed cases needed ventilation?",
            _proportion(table, ventilation, "'yes'", f"{_ident(confirmed)} = 'yes'"),
            "proportion",
            None,
            None,
            "comparison",
        ),
        (
            "Panel comparison: what proportion of confirmed cases have comorbidities?",
            _proportion(table, comorbid, "'yes'", f"{_ident(confirmed)} = 'yes'"),
            "proportion",
            None,
            None,
            "comparison",
        ),
        (
            "Short query: how many men were confirmed COVID cases?",
            _count(table, f"{_ident(sex)} = 'male'", f"{_ident(confirmed)} = 'yes'"),
            "count",
            None,
            None,
            "searcher",
        ),
        (
            "Short query: how many women needed ventilation?",
            _count(table, f"{_ident(sex)} = 'female'", f"{_ident(ventilation)} = 'yes'"),
            "count",
            None,
            None,
            "searcher",
        ),
        (
            "Triage query: among women higher than the median duration treatment, how many needed ventilation?",
            _median_count(table, duration, f"{_ident(ventilation)} = 'yes'", f"{_ident(sex)} = 'female'"),
            "count",
            None,
            None,
            "triage",
        ),
        (
            "Triage query: among men higher than the median duration treatment, how many stayed in the ICU?",
            _median_count(table, duration, f"{_ident(icu)} = 'yes'", f"{_ident(sex)} = 'male'"),
            "count",
            None,
            None,
            "triage",
        ),
        (
            "Task query: what proportion of men have comorbidities and were confirmed COVID cases?",
            _proportion(
                table,
                confirmed,
                "'yes'",
                f"{_ident(sex)} = 'male'",
                f"{_ident(comorbid)} = 'yes'",
            ),
            "proportion",
            None,
            None,
            "tasks",
        ),
        (
            "Task query: among women in the top decile of duration treatment, what is the average duration treatment?",
            _decile_avg(table, duration, f"{_ident(sex)} = 'female'"),
            "stat",
            "average",
            None,
            "tasks",
        ),
        (
            "Research query: among men in the top decile of year of onset, what is the average year of onset?",
            _decile_avg(table, onset, f"{_ident(sex)} = 'male'"),
            "stat",
            "average",
            None,
            "research",
        ),
        (
            "Research query: what is the correlation between year of onset and duration treatment?",
            _corr(table, onset, duration),
            "corr",
            None,
            None,
            "research",
        ),
        (
            "Management query: how many women with relapsing ms needed ventilation?",
            _count(
                table,
                f"{_ident(sex)} = 'female'",
                "\"ms_type2\" = 'relapsing_remitting'",
                f"{_ident(ventilation)} = 'yes'",
            ),
            "count",
            None,
            None,
            "workflow",
        ),
        (
            "Management query: how many men with progressive ms recovered?",
            _count(
                table,
                f"{_ident(sex)} = 'male'",
                "\"ms_type2\" = 'progressive_MS'",
                f"{_ident(recovered)} = 'yes'",
            ),
            "count",
            None,
            None,
            "workflow",
        ),
    ]
    pool = extras + [(query, sql, kind, stat, None, cite) for query, sql, kind, stat, cite in windows]
    pool.extend(more)
    for query, sql, kind, stat, group, cite in pool:
        rows.append(_row(query, sql, kind, stat=stat, group_by=group, citation=CITATIONS[cite]))
    return _finish("gdsi", "provider", rows)


def _dexa_questions(table: str, family: str) -> list[dict[str, Any]]:
    sex, age, cd4 = "Gender", "Age", "CD4 Count"
    race = "Race"
    tscore = "DEXA Score          (T score)"
    female, male = f"{_ident(sex)} = 'Female'", f"{_ident(sex)} = 'Male'"
    black = f"{_ident(race)} = 'Black or African-American'"
    white = f"{_ident(race)} = 'White'"
    over65 = f"{_ident(age)} > 65"
    over50 = f"{_ident(age)} > 50"
    cite = family == "provider"
    rows = [
        _row("How many women?", _count(table, female), "count", citation=CITATIONS["searcher"] if cite else None),
        _row("How many men?", _count(table, male), "count", citation=CITATIONS["searcher"] if cite else None),
        _row(
            "How many black patients?", _count(table, black), "count", citation=CITATIONS["research"] if cite else None
        ),
        _row(
            "How many white patients?", _count(table, white), "count", citation=CITATIONS["research"] if cite else None
        ),
        _row(
            "How many patients over 65?", _count(table, over65), "count", citation=CITATIONS["triage"] if cite else None
        ),
        _row(
            "How many women over 65?",
            _count(table, female, over65),
            "count",
            citation=CITATIONS["triage"] if cite else None,
        ),
        _row(
            "How many men over 65?", _count(table, male, over65), "count", citation=CITATIONS["tasks"] if cite else None
        ),
        _row(
            "How many black women?",
            _count(table, female, black),
            "count",
            citation=CITATIONS["research"] if cite else None,
        ),
        _row(
            "How many white men?", _count(table, male, white), "count", citation=CITATIONS["research"] if cite else None
        ),
        _row(
            "How many patients over 50?",
            _count(table, over50),
            "count",
            citation=CITATIONS["workflow"] if cite else None,
        ),
    ]
    cohorts = [
        ("for women", (female,)),
        ("for men", (male,)),
        ("for patients over 65", (over65,)),
        ("for black patients", (black,)),
        ("for white patients", (white,)),
        ("for women over 65", (female, over65)),
        ("for men over 65", (male, over65)),
    ]
    metrics = [("age", age), ("cd4", cd4)]
    stats = [("average", "avg"), ("median", "median")]
    for who, preds in cohorts:
        for phrase, column in metrics:
            for stat_name, fn in stats:
                rows.append(
                    _row(
                        f"What is the {stat_name} {phrase} {who}?",
                        _stat(table, fn, column, *preds),
                        "stat",
                        stat=stat_name,
                        citation=CITATIONS["workflow"] if cite else None,
                    )
                )
    rows.extend(
        [
            _row(
                "What is the average age in the oldest decile?",
                _decile_avg(table, age),
                "stat",
                stat="average",
                citation=CITATIONS["research"] if cite else None,
            ),
            _row(
                "Among women, what is the average age in the oldest decile?",
                _decile_avg(table, age, female),
                "stat",
                stat="average",
                citation=CITATIONS["research"] if cite else None,
            ),
            _row(
                "What is the average cd4 in the top decile of cd4?",
                _decile_avg(table, cd4),
                "stat",
                stat="average",
                citation=CITATIONS["comparison"] if cite else None,
            ),
            _row(
                "Among women, what is the average cd4 in the top decile of cd4?",
                _decile_avg(table, cd4, female),
                "stat",
                stat="average",
                citation=CITATIONS["comparison"] if cite else None,
            ),
            _row(
                "How many women older than the cohort median age?",
                _median_count(table, age, "TRUE", female),
                "count",
                citation=CITATIONS["risk"] if cite else None,
            ),
            _row(
                "What is the average age for women higher than the median age?",
                _median_avg(table, age, female),
                "stat",
                stat="average",
                citation=CITATIONS["risk"] if cite else None,
            ),
            _row(
                "What is the correlation between age and cd4?",
                _corr(table, age, cd4),
                "corr",
                citation=CITATIONS["tasks"] if cite else None,
            ),
            _row(
                "Compare age by gender",
                _mean_by(table, sex, age),
                "mean_by",
                group_by=sex,
                citation=CITATIONS["comparison"] if cite else None,
            ),
            _row(
                "How many black men?",
                _count(table, male, black),
                "count",
                citation=CITATIONS["research"] if cite else None,
            ),
            _row(
                "How many white women?",
                _count(table, female, white),
                "count",
                citation=CITATIONS["research"] if cite else None,
            ),
            _row(
                "What is the median cd4 for black women?",
                _stat(table, "median", cd4, female, black),
                "stat",
                stat="median",
                citation=CITATIONS["workflow"] if cite else None,
            ),
            _row(
                "What is the average age for white men over 65?",
                _stat(table, "avg", age, male, white, over65),
                "stat",
                stat="average",
                citation=CITATIONS["triage"] if cite else None,
            ),
        ]
    )
    if family == "provider":
        for row in rows:
            row["query"] = f"Clinical question: {row['query']}"
        rows[0] = _row(
            "Average age of women",
            _stat(table, "avg", age, female),
            "stat",
            stat="average",
            citation=CITATIONS["searcher"],
        )
        rows.append(
            _row(
                "compare DEXA T score by gender",
                _crosstab(table, sex, tscore),
                "crosstab",
                citation=CITATIONS["comparison"],
            )
        )
        # The first slot was a count; keep the bank at 50 by dropping the duplicate count slot's neighbor.
        del rows[1]
    return _finish("dexa", family, rows)


def _statin_questions(table: str, family: str) -> list[dict[str, Any]]:
    sex, age = "Gender", "Age"
    ldl = "LDL mg/dL"
    cd4 = "Most Recent CD4 /uL"
    prescribed = "Statin Prescribed?              1: Yes    2: No"
    race = "Race"
    female, male = f"{_ident(sex)} = 'Female'", f"{_ident(sex)} = 'Male'"
    on_statin, off_statin = f"{_ident(prescribed)} = 1", f"{_ident(prescribed)} = 2"
    over65 = f"{_ident(age)} > 65"
    black = f"{_ident(race)} = 'Black or African-American'"
    white = f"{_ident(race)} = 'White'"
    cite = family == "provider"
    rows = [
        _row("How many patients were not prescribed a statin?", _count(table, off_statin), "count"),
        _row("How many patients were prescribed a statin?", _count(table, on_statin), "count"),
        _row("How many women were not prescribed a statin?", _count(table, female, off_statin), "count"),
        _row("How many men were prescribed a statin?", _count(table, male, on_statin), "count"),
        _row("How many patients over 65 were prescribed a statin?", _count(table, over65, on_statin), "count"),
        _row(
            "How many women over 65 were not prescribed a statin?",
            _count(table, female, over65, off_statin),
            "count",
        ),
        _row("How many black patients were prescribed a statin?", _count(table, black, on_statin), "count"),
        _row("How many white patients were not prescribed a statin?", _count(table, white, off_statin), "count"),
        _row("How many men over 65 were prescribed a statin?", _count(table, male, over65, on_statin), "count"),
        _row("How many black women were prescribed a statin?", _count(table, female, black, on_statin), "count"),
    ]
    for who, preds in (
        ("for women", (female,)),
        ("for men", (male,)),
        ("for patients on a statin", (on_statin,)),
        ("for patients not prescribed a statin", (off_statin,)),
        ("for patients over 65", (over65,)),
    ):
        for phrase, column in (("ldl", ldl), ("age", age)):
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
            _row("What is the average cd4 for women?", _stat(table, "avg", cd4, female), "stat", stat="average"),
            _row("What is the average cd4 for men?", _stat(table, "avg", cd4, male), "stat", stat="average"),
            _row(
                "What is the average cd4 for patients on a statin?",
                _stat(table, "avg", cd4, on_statin),
                "stat",
                stat="average",
            ),
            _row(
                "What is the average cd4 for patients not prescribed a statin?",
                _stat(table, "avg", cd4, off_statin),
                "stat",
                stat="average",
            ),
            _row(
                "Among patients on a statin, what is the average ldl in the top decile of ldl?",
                _decile_avg(table, ldl, on_statin),
                "stat",
                stat="average",
            ),
            _row(
                "What is the average age in the oldest decile?",
                _decile_avg(table, age),
                "stat",
                stat="average",
            ),
            _row(
                "Among women, what is the average age in the oldest decile?",
                _decile_avg(table, age, female),
                "stat",
                stat="average",
            ),
            _row(
                "Among patients higher than the median ldl, how many were prescribed a statin?",
                _median_count(table, ldl, on_statin),
                "count",
            ),
            _row(
                "Among women higher than the median age, how many were not prescribed a statin?",
                _median_count(table, age, off_statin, female),
                "count",
            ),
            _row("What is the correlation between age and ldl?", _corr(table, age, ldl), "corr"),
            _row("Compare ldl by gender", _mean_by(table, sex, ldl), "mean_by", group_by=sex),
            _row(
                "What proportion of women were prescribed a statin?",
                _proportion(table, prescribed, "1", female),
                "proportion",
            ),
            _row(
                "What proportion of patients over 65 were prescribed a statin?",
                _proportion(table, prescribed, "1", over65),
                "proportion",
            ),
            _row(
                "Are men more likely than women to have been prescribed a statin?",
                _rates(table, sex, prescribed, "1"),
                "rates",
                group_by=sex,
            ),
            _row(
                "What is the median ldl for black patients on a statin?",
                _stat(table, "median", ldl, black, on_statin),
                "stat",
                stat="median",
            ),
            _row(
                "What is the average ldl for men on a statin?",
                _stat(table, "avg", ldl, male, on_statin),
                "stat",
                stat="average",
            ),
            _row(
                "What is the median age for women not prescribed a statin?",
                _stat(table, "median", age, female, off_statin),
                "stat",
                stat="median",
            ),
            _row(
                "Among men on a statin, what is the average ldl in the top decile of ldl?",
                _decile_avg(table, ldl, male, on_statin),
                "stat",
                stat="average",
            ),
            _row(
                "What proportion of men were not prescribed a statin?",
                _proportion(table, prescribed, "2", male),
                "proportion",
            ),
            _row(
                "How many white women were prescribed a statin?",
                _count(table, female, white, on_statin),
                "count",
            ),
        ]
    )
    if cite:
        for row in rows:
            row["query"] = f"Clinical question: {row['query']}"
            row["citation"] = CITATIONS["workflow"]
        rows[0] = _row(
            "How many were not prescribed a statin?",
            _count(table, off_statin),
            "count",
            citation=CITATIONS["searcher"],
        )
        rows[1] = _row(
            "Average LDL for patients on a statin",
            _stat(table, "avg", ldl, on_statin),
            "stat",
            stat="average",
            citation=CITATIONS["workflow"],
        )
        rows[2] = _row(
            "Average age among women",
            _stat(table, "avg", age, female),
            "stat",
            stat="average",
            citation=CITATIONS["searcher"],
        )
    return _finish("statin", family, rows)


def _mimic_questions(table: str, family: str) -> list[dict[str, Any]]:
    sex, age, year, death = "gender", "anchor_age", "anchor_year", "dod"
    female, male = f"{_ident(sex)} = 'F'", f"{_ident(sex)} = 'M'"
    died = f"{_ident(death)} IS NOT NULL"
    over65 = f"{_ident(age)} > 65"
    over80 = f"{_ident(age)} > 80"
    over50 = f"{_ident(age)} > 50"
    cite = family == "provider"
    rows = [
        _row("How many patients have a dod?", _count(table, died), "count"),
        _row("How many women have a dod?", _count(table, female, died), "count"),
        _row("How many men have a dod?", _count(table, male, died), "count"),
        _row("How many patients over 65?", _count(table, over65), "count"),
        _row("How many women over 65?", _count(table, female, over65), "count"),
        _row("How many men over 65?", _count(table, male, over65), "count"),
        _row("How many patients over 80 have a dod?", _count(table, over80, died), "count"),
        _row("How many women over 80 have a dod?", _count(table, female, over80, died), "count"),
        _row("How many men over 50?", _count(table, male, over50), "count"),
        _row("How many women over 50 have a dod?", _count(table, female, over50, died), "count"),
    ]
    for who, preds in (
        ("for women", (female,)),
        ("for men", (male,)),
        ("for patients who have a dod", (died,)),
        ("for patients over 65", (over65,)),
        ("for women who have a dod", (female, died)),
    ):
        for stat_name, fn in (("average", "avg"), ("median", "median")):
            rows.append(
                _row(
                    f"What is the {stat_name} age {who}?",
                    _stat(table, fn, age, *preds),
                    "stat",
                    stat=stat_name,
                )
            )
    rows.extend(
        [
            _row(
                "What is the average anchor year for women?",
                _stat(table, "avg", year, female),
                "stat",
                stat="average",
            ),
            _row(
                "What is the median anchor year for men?",
                _stat(table, "median", year, male),
                "stat",
                stat="median",
            ),
            _row(
                "What is the average anchor year for patients who have a dod?",
                _stat(table, "avg", year, died),
                "stat",
                stat="average",
            ),
            _row(
                "What is the average age in the oldest decile?",
                _decile_avg(table, age),
                "stat",
                stat="average",
            ),
            _row(
                "Among women, what is the average age in the oldest decile?",
                _decile_avg(table, age, female),
                "stat",
                stat="average",
            ),
            _row(
                "Among men, what is the average anchor year in the top decile of anchor year?",
                _decile_avg(table, year, male),
                "stat",
                stat="average",
            ),
            _row(
                "Among women higher than the median age, how many have a dod?",
                _median_count(table, age, died, female),
                "count",
            ),
            _row(
                "Among men higher than the median age, how many have a dod?",
                _median_count(table, age, died, male),
                "count",
            ),
            _row(
                "What is the correlation between anchor age and anchor year?",
                _corr(table, age, year),
                "corr",
            ),
            _row("Compare age by gender", _mean_by(table, sex, age), "mean_by", group_by=sex),
            _row(
                "What is the average age for women over 65?",
                _stat(table, "avg", age, female, over65),
                "stat",
                stat="average",
            ),
            _row(
                "What is the median anchor year for women who have a dod?",
                _stat(table, "median", year, female, died),
                "stat",
                stat="median",
            ),
            _row(
                "What is the average age for men over 80?",
                _stat(table, "avg", age, male, over80),
                "stat",
                stat="average",
            ),
            _row(
                "How many men over 65 have a dod?",
                _count(table, male, over65, died),
                "count",
            ),
            _row(
                "What is the median age for women over 65?",
                _stat(table, "median", age, female, over65),
                "stat",
                stat="median",
            ),
            _row(
                "What is the average anchor year for men over 65?",
                _stat(table, "avg", year, male, over65),
                "stat",
                stat="average",
            ),
            _row(
                "What is the median age for patients over 80?",
                _stat(table, "median", age, over80),
                "stat",
                stat="median",
            ),
            _row(
                "Among patients higher than the median age, how many have a dod?",
                _median_count(table, age, died),
                "count",
            ),
            _row(
                "What is the average age for men who have a dod?",
                _stat(table, "avg", age, male, died),
                "stat",
                stat="average",
            ),
            _row(
                "What is the median anchor year for patients over 65?",
                _stat(table, "median", year, over65),
                "stat",
                stat="median",
            ),
            _row(
                "Among women, what is the average anchor year in the top decile of anchor year?",
                _decile_avg(table, year, female),
                "stat",
                stat="average",
            ),
            _row(
                "How many women over 65 have a dod?",
                _count(table, female, over65, died),
                "count",
            ),
            _row(
                "What is the average age for men over 65?",
                _stat(table, "avg", age, male, over65),
                "stat",
                stat="average",
            ),
            _row(
                "Compare anchor year by gender",
                _mean_by(table, sex, year),
                "mean_by",
                group_by=sex,
            ),
            _row(
                "What is the median age for men who have a dod?",
                _stat(table, "median", age, male, died),
                "stat",
                stat="median",
            ),
            _row(
                "How many patients over 50 have a dod?",
                _count(table, over50, died),
                "count",
            ),
            _row(
                "What is the average anchor year for women over 65?",
                _stat(table, "avg", year, female, over65),
                "stat",
                stat="average",
            ),
            _row(
                "Among men higher than the median anchor year, how many have a dod?",
                _median_count(table, year, died, male),
                "count",
            ),
            _row(
                "What is the median age in the oldest decile?",
                _decile_avg(table, age).replace("avg(", "median(", 1),
                "stat",
                stat="median",
            ),
            _row(
                "How many men over 80 have a dod?",
                _count(table, male, over80, died),
                "count",
            ),
        ]
    )
    if cite:
        for row in rows:
            row["query"] = f"Clinical question: {row['query']}"
            row["citation"] = CITATIONS["research"]
        rows[0] = _row(
            "How many MIMIC patients have a dod?",
            _count(table, died),
            "count",
            citation=CITATIONS["searcher"],
        )
        rows[1] = _row(
            "Average age of women",
            _stat(table, "avg", age, female),
            "stat",
            stat="average",
            citation=CITATIONS["searcher"],
        )
        rows[2] = _row(
            "Median age for men over 65",
            _stat(table, "median", age, male, over65),
            "stat",
            stat="median",
            citation=CITATIONS["triage"],
        )
    return _finish("mimic_patients", family, rows)


def questions_for_dataset(dataset_id: str) -> list[dict[str, Any]]:
    """50 analytical and 50 provider questions for one dataset."""
    if dataset_id not in DATASET_TABLES:
        known = ", ".join(sorted(DATASET_TABLES))
        raise KeyError(f"Unknown dataset '{dataset_id}'. Known: {known}")
    table = DATASET_TABLES[dataset_id]
    if dataset_id == "gdsi":
        return _gdsi_analytical(table) + _gdsi_provider(table)
    if dataset_id == "dexa":
        return _dexa_questions(table, "analytical") + _dexa_questions(table, "provider")
    if dataset_id == "statin":
        return _statin_questions(table, "analytical") + _statin_questions(table, "provider")
    return _mimic_questions(table, "analytical") + _mimic_questions(table, "provider")


def all_hard_questions() -> list[dict[str, Any]]:
    questions: list[dict[str, Any]] = []
    for dataset_id in DATASET_TABLES:
        questions.extend(questions_for_dataset(dataset_id))
    return questions


def oracle_value(question: dict[str, Any], workspace: Path | None = None) -> Any:
    """Run the question's SQL against DuckDB. This is the expected answer."""
    path = duckdb_path(workspace)
    con = duckdb.connect(str(path), read_only=True)
    try:
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
