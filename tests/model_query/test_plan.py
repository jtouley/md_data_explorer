# SPDX-License-Identifier: MIT
"""The plan compiler: each plan's SQL, run in the sandbox, against an independent Polars oracle."""

from __future__ import annotations

from typing import Any

import polars as pl
import pytest

from clinical_analytics.model_query.plan import Plan, PlanError, compile_plan, parse_plan, reply_schema
from clinical_analytics.sql.sandbox import SandboxConnection
from tests.model_query import tables


def _plan(**fields: Any) -> Plan:
    return parse_plan(tables.reply(**fields))


_cond = tables.cond


@pytest.fixture
def clinic(make_query_table):
    return make_query_table(tables.clinic(), doc_context=tables.CLINIC_DICTIONARY)


def _run(table: tuple[Any, Any], plan: Plan) -> pl.DataFrame:
    catalog, context = table
    with SandboxConnection(catalog) as sandbox:
        return sandbox.run_model_sql(compile_plan(plan, context).sql)


def _last_visits() -> pl.DataFrame:
    return tables.clinic().sort("visit_seq").group_by("pt_ref").last()


def _first_visits() -> pl.DataFrame:
    return tables.clinic().sort("visit_seq").group_by("pt_ref").first()


@pytest.mark.parametrize(
    ("which", "expected"),
    [
        pytest.param("any", lambda: tables.clinic().filter(pl.col("smoker") == "Y")["pt_ref"].n_unique(), id="any"),
        pytest.param("last", lambda: _last_visits().filter(pl.col("smoker") == "Y").height, id="last"),
        pytest.param("first", lambda: _first_visits().filter(pl.col("smoker") == "Y").height, id="first"),
        pytest.param(
            "every",
            lambda: (
                tables.clinic().group_by("pt_ref").agg((pl.col("smoker") == "Y").all()).filter(pl.col("smoker")).height
            ),
            id="every",
        ),
    ],
)
def test_compile_plan_patientCountByVisit_matchesOracle(clinic, which, expected):
    # Arrange
    plan = _plan(filters=[_cond("smoker", "is", values=["yes"])], which_visit=which)

    # Act
    result = _run(clinic, plan)

    # Assert
    assert result.item(0, "count") == expected()


def test_compile_plan_groupByLatestCategory_countsEachPatientOnce(clinic):
    # Arrange: patients by their latest treatment arm (the arm changes between some visits)
    plan = _plan(group_by="arm", which_visit="last")
    labels = {1: "ACE inhibitor", 2: "calcium blocker", 3: "diuretic"}
    oracle = _last_visits().group_by("arm").len().sort("arm")

    # Act
    result = _run(clinic, plan)

    # Assert: one row per arm code, each patient counted once
    assert result["count"].to_list() == oracle["len"].to_list()
    assert result["count"].sum() == tables.CLINIC_PATIENTS
    assert set(labels) == set(result["grp"].to_list())


def test_compile_plan_visitCount_countsRecords(clinic):
    # Arrange
    plan = _plan(count_of="clinic visits", filters=[_cond("systolic_mmhg", ">=", 140)])

    # Act
    result = _run(clinic, plan)

    # Assert
    assert result.item(0, "count") == tables.clinic().filter(pl.col("systolic_mmhg") >= 140).height


@pytest.mark.parametrize("which", ["any", "every", "last"])
def test_compile_plan_shareOfPatients_matchesOracle(clinic, which):
    # Arrange
    plan = _plan(answer="share", numerator=[_cond("smoker", "is", values=["yes"])], which_visit=which)
    frame = tables.clinic()
    if which == "last":
        per = _last_visits().select("pt_ref", (pl.col("smoker") == "Y").alias("e"))
    else:
        agg = (pl.col("smoker") == "Y").any() if which == "any" else (pl.col("smoker") == "Y").all()
        per = frame.group_by("pt_ref").agg(agg.alias("e"))

    # Act
    result = _run(clinic, plan)

    # Assert
    assert result.item(0, "events") == per["e"].sum()
    assert result.item(0, "n") == per.height


@pytest.mark.parametrize(
    ("which", "oracle"),
    [
        pytest.param(
            "first",
            lambda f: f.sort("visit_seq").group_by("pt_ref").first()["systolic_mmhg"].mean(),
            id="first",
        ),
        pytest.param(
            "change",
            lambda f: (
                f.sort("visit_seq")
                .group_by("pt_ref")
                .agg(
                    pl.col("systolic_mmhg").drop_nulls().last() - pl.col("systolic_mmhg").first(),
                    pl.col("systolic_mmhg").count().alias("n"),
                )
                .filter(pl.col("n") >= 2)["systolic_mmhg"]
                .mean()
            ),
            id="change",
        ),
        pytest.param("any", lambda f: f["systolic_mmhg"].mean(), id="pooled"),
    ],
)
def test_compile_plan_meanPerPatientValue_matchesOracle(clinic, which, oracle):
    # Arrange
    plan = _plan(answer="mean", measure="systolic_mmhg", which_visit=which)

    # Act
    result = _run(clinic, plan)

    # Assert
    assert result.item(0, "value") == pytest.approx(oracle(tables.clinic()))


def test_compile_plan_numberCutPoints_groupsIntoBands(clinic):
    # Arrange
    plan = _plan(count_of="clinic visits", group_by="age_at_visit", cut_points=[50, 65])
    ages = tables.clinic()["age_at_visit"]

    # Act
    result = _run(clinic, plan)

    # Assert
    counts = dict(zip(result["grp"].to_list(), result["count"].to_list(), strict=True))
    assert counts == {
        "under 50": int((ages < 50).sum()),
        "50 to under 65": int(((ages >= 50) & (ages < 65)).sum()),
        "65 or more": int((ages >= 65).sum()),
    }


def test_compile_plan_labelValue_isStoredCode(clinic):
    # Arrange: the reply names the dictionary label; the stored value is the code
    plan = _plan(count_of="clinic visits", filters=[_cond("arm", "is not", values=["diuretic"])])

    # Act
    result = _run(clinic, plan)

    # Assert
    assert result.item(0, "count") == tables.clinic().filter(pl.col("arm") != 3).height


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        pytest.param({"filters": [_cond("arm", "is", values=["placebo"])]}, "not a value of arm", id="unknown_value"),
        pytest.param({"filters": [_cond("pt_ref", "is", values=["H001"])]}, "not the ID", id="patient_key"),
        pytest.param({"answer": "mean", "measure": "sex"}, "not a number column", id="mean_of_category"),
        pytest.param({"answer": "share"}, "numerator", id="share_without_numerator"),
        pytest.param({"group_by": "arm", "which_visit": "every"}, "not to a breakdown", id="every_breakdown"),
        pytest.param({"answer": "list_patients"}, "does not list", id="listing"),
    ],
)
def test_compile_plan_unsupported_raisesPlanErrorNamingOnlySupported(clinic, fields, message):
    # Arrange
    _catalog, context = clinic

    # Act / Assert
    with pytest.raises(PlanError, match=message):
        compile_plan(_plan(**fields), context)


def test_reply_schema_enumsComeFromContract(clinic):
    # Arrange
    _catalog, context = clinic

    # Act
    schema = reply_schema(context)

    # Assert: labels, not codes; the patient key is never a slot; both units offered
    branches = schema["properties"]["filters"]["items"]["anyOf"]
    arm = next(b for b in branches if b["properties"]["column"]["const"] == "arm")
    assert arm["properties"]["values"]["items"]["enum"] == ["ACE inhibitor", "calcium blocker", "diuretic"]
    assert "pt_ref" not in schema["properties"]["group_by"]["anyOf"][0]["enum"]
    assert schema["properties"]["count_of"]["enum"] == ["patients", "clinic visits"]


@pytest.mark.parametrize("payload", [None, [], {"answer": "explain"}, {"answer": "count", "cut_points": [1] * 9}])
def test_parse_plan_invalid_raisesPlanError(payload):
    # Act / Assert
    with pytest.raises(PlanError):
        parse_plan(payload)
