# SPDX-License-Identifier: MIT
"""Golden samples of the model-written query path: recorded model replies replayed through the
real compile-and-execute layer on synthetic tables (tests/eval/model_query_golden.yaml).

No model runs: each sample's replies are what the local model answered, in order. The values of
the visit-level samples are recomputed here by independent Polars oracles.
"""

from __future__ import annotations

import datetime as dt
import math
from pathlib import Path
from typing import Any

import polars as pl
import pytest
import yaml

from clinical_analytics.core.result_presentation import present_analysis
from clinical_analytics.model_query.engine import answer_question
from clinical_analytics.model_query.examples import load_examples
from clinical_analytics.model_query.plan import TABULATIONS, compile_plan, parse_reply
from clinical_analytics.sql.sandbox import SandboxConnection
from tests.model_query import tables

GOLDEN = yaml.safe_load((Path(__file__).parent / "model_query_golden.yaml").read_text(encoding="utf-8"))["samples"]
TABLES = {
    "patients": (tables.patients, None, "patient_id"),
    "visits": (tables.visits, tables.VISITS_DICTIONARY, "patient_id"),
    "clinic": (tables.clinic_visits, tables.CLINIC_DICTIONARY, "pt_id"),
}
Z_95 = 1.959963984540054


def _older_tests() -> pl.DataFrame:
    """One row per test (encounter) in patients 65 and older, duplicate rows dropped."""
    return tables.visits().filter(pl.col("age_years") >= 65).unique(subset=["encounter_id"])


def _positive_patients_65_by_year() -> dict[str, Any]:
    frame = tables.visits().filter((pl.col("age_years") >= 65) & (pl.col("result") == "positive"))
    counted = frame.group_by("visit_year").agg(pl.col("patient_id").n_unique().alias("n"))
    return {f"visit_year={year}": n for year, n in counted.iter_rows()}


def _positive_share_of_tests_65_by_year() -> dict[str, Any]:
    shares = _older_tests().group_by("visit_year").agg((pl.col("result") == "positive").mean().alias("rate"))
    return {f"visit_year={year}": rate for year, rate in shares.iter_rows()}


def _odds_ratio_positive_abx_65() -> dict[str, Any]:
    """Sample odds ratio with a Woolf interval: equal to the unadjusted logistic regression."""
    cells = _older_tests().group_by("abx", "result").agg(pl.len().alias("n"))
    table = {(abx, result): n for abx, result, n in cells.iter_rows()}
    a, b = table["yes", "positive"], table["yes", "negative"]
    c, d = table["no", "positive"], table["no", "negative"]
    log_or, se = math.log(a * d / (b * c)), math.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
    return {
        "odds ratio": math.exp(log_or),
        "95% CI low": math.exp(log_or - Z_95 * se),
        "95% CI high": math.exp(log_or + Z_95 * se),
    }


def _median_titer_miu_by_region() -> dict[str, Any]:
    miu = pl.when(pl.col("titer_unit") == "IU/mL").then(pl.col("titer") * 1000).otherwise(pl.col("titer"))
    frame = tables.visits().unique().with_columns(miu.alias("miu")).filter(pl.col("miu").is_not_null())
    return dict(frame.group_by("region").agg(pl.col("miu").median()).iter_rows())


ORACLES = {
    "positive_patients_65_by_year": _positive_patients_65_by_year,
    "positive_share_of_tests_65_by_year": _positive_share_of_tests_65_by_year,
    "odds_ratio_positive_abx_65": _odds_ratio_positive_abx_65,
    "median_titer_miu_by_region": _median_titer_miu_by_region,
}


def _shown_groups(table: dict[str, Any]) -> dict[str, Any]:
    """{first column's label: the first value column's value} of a presented table."""
    label, value = table["columns"][0], table["columns"][1]
    if "odds ratio" in table["columns"]:
        return {name: table["rows"][0][name] for name in ("odds ratio", "95% CI low", "95% CI high")}
    return {str(row[label]): row[value] for row in table["rows"]}


@pytest.mark.parametrize("sample", GOLDEN, ids=[s["id"] for s in GOLDEN])
def test_model_query_golden_recordedReply_givesExpectedReadingSqlAndValue(sample, make_query_table):
    # Arrange: the synthetic table, and the recorded replies in the order the model gave them
    build, dictionary, id_column = TABLES[sample["table"]]
    catalog, context = make_query_table(build(), doc_context=dictionary, id_column=id_column)
    replies = iter(sample["replies"])
    expect = sample["expect"]

    # Act
    answer = answer_question(
        sample["question"],
        context,
        model=lambda _system, _user, _schema: next(replies),
        open_sandbox=lambda: SandboxConnection(catalog),
        fmt="plan",
        examples=[],
    )

    # Assert
    assert answer.attempts == len(sample["replies"])
    if "status" in expect:
        assert answer.status == expect["status"]
        assert expect["reason"] in (answer.reason or "")
        assert len(answer.choices) == expect.get("choices", 0)
        return
    assert answer.status == "answered", answer.reason
    assert answer.reading == sample["reading"]
    shown_sql = " ".join((answer.sql or "").split())  # pretty-printed over several lines
    for fragment in sample["sql"]:
        assert fragment in shown_sql, f"{fragment!r} not in the shown SQL"
    presented = present_analysis(answer.analysis)
    if "analysis_type" in expect:
        assert answer.analysis is not None and answer.analysis["type"] == expect["analysis_type"]
    if "claim" in expect:
        assert (presented["claim"] or "").startswith(expect["claim"])
    for name, value in expect.get("analysis", {}).items():
        assert answer.analysis is not None and answer.analysis[name] == pytest.approx(value, abs=5e-5), name
    wanted = ORACLES[expect["oracle"]]() if "oracle" in expect else expect.get("groups")
    if wanted is not None:
        assert presented["table"] is not None
        shown = _shown_groups(presented["table"])
        assert set(shown) == {str(key) for key in wanted}
        for key, value in wanted.items():
            assert shown[str(key)] == pytest.approx(value, abs=5e-5), key


def test_model_query_examples_everyPlan_compilesOnATableWithItsColumns(make_query_table):
    """The few-shot pool's plans are valid replies: a broken example would teach the model a
    reply the app itself rejects. Each tabulation plan is compiled on a table holding the
    columns its made-up table names (values and types are not checked here)."""
    # Arrange
    frames = {
        "stroke": pl.DataFrame(
            {
                "patient_id": ["a", "b", "c"], "sex": ["F", "M", "F"], "age": [70, 85, 60],
                "stroke_type": ["TIA", "ischemic", "hemorrhagic"], "diabetes": ["No", "Yes", "No"],
                "smoker": ["current", "never", "former"], "nihss": [4.0, 9.0, 2.0],
                "thrombolysis": ["No", "Yes", "No"], "died_30d": ["No", "Yes", "No"], "onset_year": [2020, 2021, 2022],
            }
        ),
        "dialysis": pl.DataFrame(
            {
                "patient_id": ["a", "a", "b"], "session_id": ["s1", "s2", "s3"], "session_year": [2024, 2024, 2023],
                "age_years": [77, 77, 50], "sex": ["female", "female", "male"],
                "access_type": ["catheter", "graft", "fistula"], "infection": ["no", "yes", "no"],
                "hospitalized": ["no", "yes", "no"], "kt_v": [1.2, 1.4, 1.3],
            }
        ),
        "emergency": pl.DataFrame(
            {
                "patient_id": ["a", "a"], "encounter_id": ["e1", "e2"], "arrival_year": [2024, 2024],
                "age_years": [70, 70], "triage_level": [1, 2], "admitted": ["no", "yes"],
                "sepsis_flag": ["no", "yes"], "lactate": [2.0, 40.0], "lactate_unit": ["mmol/L", "mg/dL"],
            }
        ),
        "lipid": pl.DataFrame(
            {
                "patient_id": ["a", "a", "b"], "visit_no": [1, 2, 1],
                "visit_date": [dt.date(2024, 1, 1), dt.date(2024, 3, 1), dt.date(2024, 2, 1)],
                "ldl": [130.0, 95.0, 88.0], "statin": ["no", "yes", "no"], "age": [55, 55, 72],
            }
        ),
        "vaccine": pl.DataFrame(
            {
                "person_id": ["a", "b", "c"], "age_group": [0, 1, 2], "region": ["East", "North", "South"],
                "doses": [0, 3, 1], "antibody_titer": [5.0, 80.0, 20.0], "pregnant": ["no", "yes", "no"],
            }
        ),
    }  # fmt: skip
    contexts = {
        name: make_query_table(frame, id_column="person_id" if name == "vaccine" else "patient_id")[1]
        for name, frame in frames.items()
    }
    by_table = {" ".join(text.split()): name for name, text in _pool_tables().items()}
    examples = load_examples()

    # Act
    compiled = [
        compile_plan(parse_reply(example.reply("plan")), example.question, contexts[by_table[example.table]])
        for example in examples
        if example.plan["answer"] in TABULATIONS
    ]

    # Assert
    assert len(examples) >= 25
    assert all(c.sql.startswith("SELECT") for c in compiled)


def _pool_tables() -> dict[str, str]:
    from clinical_analytics.core.resources import config_file
    from clinical_analytics.model_query.examples import EXAMPLES_FILE

    pool: dict[str, Any] = yaml.safe_load(config_file(EXAMPLES_FILE).read_text(encoding="utf-8"))
    tables_of: dict[str, str] = pool["tables"]
    return tables_of
