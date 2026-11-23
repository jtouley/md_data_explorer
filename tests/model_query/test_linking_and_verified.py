# SPDX-License-Identifier: MIT
"""The annotated schema, value linking before the plan call, and verified plans."""

from __future__ import annotations

import json

import polars as pl
import pytest

from clinical_analytics.model_query import linking, verified
from clinical_analytics.model_query.context import column_line, context_text
from tests.model_query import tables

_cond = tables.cond


@pytest.fixture
def shots(make_query_table):
    return make_query_table(tables.shots(), doc_context=tables.SHOTS_DICTIONARY)


def test_column_line_annotatesRoleUnitAndTopValues(make_query_table):
    # Arrange: one extract is far more common than the others
    frame = tables.shots().with_columns(
        pl.when(pl.col("shot_round") > 0).then(pl.lit(2)).otherwise(pl.col("extract")).alias("extract")
    )
    frame = pl.concat([frame, tables.shots().head(3)])
    _catalog, context = make_query_table(frame, doc_context=tables.SHOTS_DICTIONARY)

    # Act
    text = context_text(context)

    # Assert: roles, units, and the most frequent value listed first
    assert "holder_ref [patient key]" in text
    assert "ige_kul [measure, kU/L]" in text
    assert "shot_round [time]" in text
    assert column_line(context.column("extract"), context).startswith("extract [category]: values 'dust mite'")


def test_link_values_exactAndNearMisses(shots):
    # Arrange
    _catalog, context = shots

    # Act
    exact = linking.link_values("How many patients got dust mite shots?", context)
    near = linking.link_values("How many patients got dust mitt shots?", context)

    # Assert
    assert [(x.label, x.column, x.exact) for x in exact] == [("dust mite", "extract", True)]
    assert [(x.said, x.label, x.exact) for x in near] == [("dust mitt", "dust mite", False)]


def test_answer_question_valueHints_areInThePrompt(shots, make_model_ask):
    # Act
    _answer, prompts = make_model_ask(shots)(
        "How many patients got dust mite shots at any shot visit?",
        tables.reply(filters=[_cond("extract", "is", values=["dust mite"])]),
    )

    # Assert
    assert "'dust mite' is a value of extract." in prompts[0]


def test_answer_question_nearMissUnresolved_isAskedWithOptions(shots, make_model_ask):
    # Act: the plan does not use the value the near miss points to
    answer, _prompts = make_model_ask(shots)("How many patients got dust mitt shots?", tables.reply())

    # Assert: asked, each option the question with the value said exactly
    assert answer.status == "clarify" and answer.analysis is None
    assert answer.choices == (("dust mite", "How many patients got dust mite shots?"),)


def test_mask_question_keepsNoValueOrNumber():
    # Arrange
    plan = tables.reply(filters=[_cond("extract", "is", values=["dust mite"]), _cond("years_at_shot", ">", 60)])

    # Act
    masked = verified.mask("How many patients over 60 on dust mite by years_at_shot?", plan)

    # Assert
    assert masked == "How many patients over <number> on <value> by <column>?"


def test_similar_plans_sameReadingOnly_neverAcrossUnits(shots, tmp_path):
    # Arrange: one confirmed question about patients
    _catalog, context = shots
    plan = tables.reply(filters=[_cond("extract", "is", values=["dust mite"])])
    verified.save(tmp_path, "ds1", "v1", "How many patients got dust mite shots?", plan)

    # Act
    patients = verified.similar(tmp_path, "ds1", "How many patients got grass pollen shots?", context)
    visits = verified.similar(tmp_path, "ds1", "How many shot visits gave grass pollen?", context)

    # Assert: retrieved for the same reading; never for a question about visits
    assert [p for _m, p in patients] == [plan]
    assert visits == []
    stored = json.loads((tmp_path / "model_query" / "verified" / "ds1.json").read_text())
    assert "dust mite" not in stored[0]["masked"]


def test_answer_question_verifiedPlan_isAPriorTurnNeverTheAnswer(shots):
    # Arrange
    from clinical_analytics.model_query.engine import answer_question
    from clinical_analytics.sql.sandbox import SandboxConnection

    catalog, context = shots
    prior = [
        ("How many patients got <value> shots?", tables.reply(filters=[_cond("extract", "is", values=["dust mite"])]))
    ]
    prompts: list[str] = []

    def model(_s, user, _schema):
        prompts.append(user)
        return tables.reply(filters=[_cond("extract", "is", values=["grass pollen"])])

    # Act
    answer = answer_question(
        "How many patients got grass pollen shots at any shot visit?", context, model=model,
        open_sandbox=lambda: SandboxConnection(catalog), examples=[], verified=prior,
    )  # fmt: skip

    # Assert: shown as a worked example; the model still writes the plan, which is checked and run
    assert "Questions on your table whose replies the user confirmed" in prompts[0]
    assert answer.status == "answered" and answer.reply["filters"][0]["values"] == ["grass pollen"]
