# SPDX-License-Identifier: MIT
"""The plan compiler: the model's reply checked against the table and written as SQL."""

from __future__ import annotations

import pytest

from clinical_analytics.model_query.plan import (
    LISTING_REASON,
    ModelReply,
    PlanError,
    compile_plan,
    compile_sql,
    compile_test,
    parse_reply,
    reply_schema,
    resolve_unit,
)
from clinical_analytics.model_query.units import UnitError
from tests.model_query import tables


def _flat(sql: str) -> str:
    """The shown SQL on one line (it is pretty-printed over several)."""
    return " ".join(sql.split())


def _reply(answer: str, **fields: str | None) -> ModelReply:
    return ModelReply(answer=answer, **fields)


@pytest.fixture
def patients(make_query_table):
    return make_query_table(tables.patients())[1]


@pytest.fixture
def visits(make_query_table):
    return make_query_table(tables.visits(), doc_context=tables.VISITS_DICTIONARY)[1]


@pytest.mark.parametrize(
    ("question", "key", "plural"),
    [
        ("How many patients tested positive?", "patient_id", "patients"),
        ("How many tests were positive?", "encounter_id", "tests"),
        ("What share of admissions were in men?", "encounter_id", "admissions"),
        ("What share were positive?", "patient_id", "patients"),
        ("How many rows are in the file?", None, "rows"),
    ],
)
def test_resolve_unit_visitRows_countsWhatTheQuestionNames(visits, question, key, plural):
    # Act
    unit = resolve_unit(question, visits)

    # Assert
    assert (unit.key, unit.plural) == (key, plural)


def test_resolve_unit_oneRowPerPatient_countsRowsAsPatients(patients):
    # Act
    unit = resolve_unit("How many patients were admitted?", patients)

    # Assert: every row is a patient, so nothing is counted distinct
    assert (unit.key, unit.word) == (None, "patients")


def test_resolve_unit_eventsOnOneRowPerPatient_isAskedNotCountedAsPatients(patients):
    # Act / Assert: admissions cannot be told apart on a table of one row per patient
    with pytest.raises(UnitError, match="Each row of this table is one patient"):
        resolve_unit("How many admissions were there?", patients)


@pytest.mark.parametrize(
    ("where", "expected"),
    [
        ("sex = 'Female'", "\"sex\" = 'female'"),  # case
        ("sex = 'F'", "\"sex\" = 'female'"),  # an abbreviation of one value
        ('sex = "male"', "\"sex\" = 'male'"),  # a double-quoted value is text, not a column
        ("admitted = TRUE", "\"admitted\" = 'Yes'"),  # true/false for a yes/no column
        ("admitted = 1", "\"admitted\" = 'Yes'"),
        ("age_cat = '2'", '"age_cat" = 2'),  # a number column compared with text
        ("admitted", "\"admitted\" = 'Yes'"),  # a yes/no column on its own
        ("sex IN ('F')", "\"sex\" IN ('female')"),
    ],
)
def test_compile_plan_valueSpelledDifferently_isRepairedToTheStoredValue(patients, where, expected):
    # Act
    compiled = compile_plan(_reply("count", where=where), "How many?", patients)

    # Assert
    assert expected in compiled.sql


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        ({"where": "sex = 'unknown'"}, "Column sex has no value 'unknown'"),
        ({"where": "gender = 'male'"}, "There is no column named gender"),
        ({"where": "COUNT(*) > 3"}, "cannot use COUNT"),
        ({"where": "onset_year"}, "must be a condition"),
        ({"number": "dx"}, "To count the rows with one value of dx"),
        ({"where": "sex = 'male' AND (SELECT 1) = 1"}, "subquery"),
    ],
)
def test_compile_plan_countThatCannotRun_saysWhatToFix(patients, fields, message):
    # Act / Assert
    with pytest.raises(PlanError, match=message.replace("(", r"\(").replace(")", r"\)").replace("*", r"\*")):
        compile_plan(_reply("count", **fields), "How many?", patients)


def test_compile_plan_conditionsInOtherFields_areAllApplied(patients):
    # Arrange: the model put conditions in numerator and number instead of where
    reply = _reply("count", where="sex = 'male'", numerator="comorbid = 'Yes'", number="comorbid = 'Yes'")

    # Act
    compiled = compile_plan(reply, "How many men have a comorbidity?", patients)

    # Assert: nothing it wrote is dropped, and a repeated term is written once
    assert compiled.reading == "Counted patients where sex = 'male' AND comorbid = 'Yes'."


def test_compile_plan_whereListsEveryValue_becomesTheGrouping(patients):
    # Act
    compiled = compile_plan(_reply("count", where="sex IN ('female', 'male') AND dx = 'other'"), "q", patients)

    # Assert: the term that keeps every row is dropped; its column is the breakdown
    assert compiled.group == "sex"
    assert compiled.reading == "Counted patients where dx = 'other', by sex."


def test_compile_plan_wherePinsTheGroup_isFedBack(patients):
    # Act / Assert
    with pytest.raises(PlanError, match="where limits sex, but group_by sex asks for each"):
        compile_plan(_reply("median", number="onset_year", where="sex = 'male'", group_by="sex"), "q", patients)


@pytest.mark.parametrize(
    "fields",
    [
        {"numerator": "dx"},  # the column whose categories are meant
        {"numerator": "dx = 'other'", "group_by": "dx"},  # an event on the grouping itself
        {"group_by": "dx"},  # no event at all
    ],
)
def test_compile_plan_shareOfEachCategory_isTheDistribution(patients, fields):
    # Act
    compiled = compile_plan(_reply("share", **fields), "What share of patients have each dx?", patients)

    # Assert
    assert (compiled.kind, compiled.group) == ("count", "dx")


def test_compile_plan_shareWithoutEventOrGroup_isFedBack(patients):
    # Act / Assert
    with pytest.raises(PlanError, match="A share needs numerator"):
        compile_plan(_reply("share"), "What share?", patients)


def test_compile_plan_shareOnVisitRows_rollsEachPatientUpAndChecksVariation(visits):
    # Act
    ever = compile_plan(_reply("share", numerator="screened = 'yes'"), "What share was ever screened?", visits)
    always = compile_plan(_reply("share", numerator="screened = 'yes'"), "What share was always screened?", visits)
    unsaid = compile_plan(_reply("share", numerator="screened = 'yes'"), "What share was screened?", visits)

    # Assert: the roll-up the question names is used and not asked about; an unnamed one is checked
    assert ("BOOL_OR" in ever.sql, ever.rollup, ever.varies_sql) == (True, "any", None)
    assert ("BOOL_AND" in always.sql, always.rollup, always.varies_sql) == (True, "all", None)
    assert unsaid.varies_sql is not None and "HAVING" in unsaid.varies_sql


def test_compile_plan_statisticOnVisitRows_dropsDuplicateRowsAndSaysRows(visits):
    # Act
    compiled = compile_plan(
        _reply("mean", number="age_years", group_by="region"), "Mean age at each visit, by region?", visits
    )

    # Assert
    assert "SELECT DISTINCT *" in _flat(compiled.sql)
    assert compiled.reading == "Mean of age_years, by region, over visits (rows), not one value per patient."
    assert compiled.per_patient_varies_sql is None


@pytest.mark.parametrize(
    "fields",
    [
        {"number": "titer"},
        {"number": "age_years", "where": "titer >= 120"},
    ],
)
def test_compile_plan_numberInTwoUnitsUsedUnconverted_isFedBack(visits, fields):
    # Act / Assert
    with pytest.raises(PlanError, match="titer is recorded in different units"):
        compile_plan(_reply("median", **fields), "Median?", visits)


def test_compile_plan_unitConversionWritten_passesTheUnitCheck(visits):
    # Arrange
    number = "CASE WHEN titer_unit = 'IU/mL' THEN titer * 1000 ELSE titer END"

    # Act
    compiled = compile_plan(_reply("median", number=number), "Median titer in mIU/mL?", visits)

    # Assert: the table labels the value by its column
    assert compiled.value_text == "titer (converted)"


@pytest.mark.parametrize(
    "fields",
    [{"group_by": "patient_id"}, {"where": "admitted = 'Yes'", "group_by": "PATIENT_ID"}],
)
def test_compile_plan_groupedByPatientId_isRefusedAsAListing(patients, fields):
    # Act / Assert
    with pytest.raises(PlanError, match="list individual patients"):
        compile_plan(_reply("count", **fields), "Who was admitted?", patients)


def test_compile_test_outcomeGivenAsCondition_readsItAsYesNoOutcome(patients):
    # Arrange: the model named the column and, separately, the value that is the outcome
    reply = _reply("compare_groups", outcome="dx", numerator="dx = 'relapsing'", group_by="sex")

    # Act
    call = compile_test(reply, patients)

    # Assert
    assert call.outcome == "dx_relapsing"
    assert call.reading == "Compared dx = 'relapsing' between the groups of sex."
    assert "CASE WHEN" in _flat(call.sql)


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        ({"outcome": "confirmed"}, "needs group_by"),
        ({"group_by": "sex"}, "needs outcome"),
        ({"outcome": "sex", "group_by": "sex"}, "outcome and group_by are both sex"),
        ({"outcome": "confirmed", "group_by": "sex, dx"}, "group_by"),
    ],
)
def test_compile_test_incompleteCall_saysWhatIsMissing(patients, fields, message):
    # Act / Assert
    with pytest.raises(PlanError, match=message):
        compile_test(_reply("odds_ratio", **fields), patients)


def test_compile_test_visitRows_readsDistinctRowsWithThePatientKey(visits):
    # Act
    call = compile_test(_reply("odds_ratio", outcome="result", group_by="abx", where="age_years >= 65"), visits)

    # Assert
    assert call.entity_key == "patient_id"
    assert "SELECT DISTINCT *" in _flat(call.sql) and '"patient_id"' in call.sql


@pytest.mark.parametrize(
    ("sql", "message"),
    [
        ("SELECT * FROM unified", "list individual patients"),
        ("SELECT patient_id, COUNT(*) FROM unified GROUP BY patient_id", "list individual patients"),
        ("SELECT sex FROM unified", "list individual patients"),
        ("SELECT COUNT(*) FROM other_table", "Read only the table unified"),
        ("DELETE FROM unified", "exactly one SELECT"),
        ("SELECT COUNT(*) FROM unified; SELECT 1", "exactly one SELECT"),
        ("SELECT COUNT(*) FROM unified WHERE result = 'positive'", "Rows are visits, not patients"),
        ("SELECT MEDIAN(titer) AS median FROM unified", "recorded in different units"),
        ("SELECT COUNT(DISTINCT patient_id) FROM unified WHERE result = 'maybe'", "has no value 'maybe'"),
    ],
)
def test_compile_sql_queryThatMustNotRun_isRefusedWithTheReason(visits, sql, message):
    # Act / Assert
    with pytest.raises(PlanError, match=message):
        compile_sql(_reply("sql", sql=sql), "How many patients tested positive?", visits)


def test_compile_sql_countDistinctPatients_isReadAsAGroupedCount(visits):
    # Arrange
    sql = "SELECT visit_year, COUNT(DISTINCT patient_id) AS n FROM unified WHERE result = 'Positive' GROUP BY 1"

    # Act
    compiled = compile_sql(_reply("sql", sql=sql, thought="Patients positive per year."), "How many patients?", visits)

    # Assert: the value's case is repaired, and the result is presented as a count per group
    assert (compiled.kind, compiled.group, compiled.reading) == ("count", "visit_year", "Patients positive per year.")
    assert "'positive'" in compiled.sql


def test_parse_reply_nullWrittenAsText_isRead(patients):
    # Act
    reply = parse_reply({"answer": "count", "where": "null", "group_by": " ", "extra": 1})

    # Assert
    assert (reply.where, reply.group_by) == (None, None)


@pytest.mark.parametrize("payload", [None, [], "count", {"where": "x"}])
def test_parse_reply_notAReplyObject_raisesPlanError(payload):
    # Act / Assert
    with pytest.raises(PlanError):
        parse_reply(payload)


@pytest.mark.parametrize(("fmt", "first", "has_sql"), [("plan", "count", False), ("sql", "sql", True)])
def test_reply_schema_format_listsItsAnswersAndFields(fmt, first, has_sql):
    # Act
    schema = reply_schema(fmt)

    # Assert
    assert schema["properties"]["answer"]["enum"][0] == first
    assert ("sql" in schema["properties"]) is has_sql
    assert schema["required"] == list(schema["properties"])
    assert LISTING_REASON.startswith("This would list")


def test_compile_plan_whereRepeatsTheNumerator_isNotALimit(patients):
    # Arrange: the model wrote the event as a limit too, which would make every share 100%
    reply = _reply("share", numerator="confirmed = 'yes'", where="age_cat = 2 AND confirmed = 'yes'", group_by="sex")

    # Act
    compiled = compile_plan(reply, "What share of the older ones had a confirmed infection, by sex?", patients)

    # Assert
    assert compiled.reading == (
        "Share of patients with confirmed = 'Yes' where age_cat = 2, by sex, among those with it recorded."
    )


def test_compile_plan_whereMixesAndWithOr_isFedBack(patients):
    # Act / Assert
    with pytest.raises(PlanError, match="mixes AND and OR without parentheses"):
        compile_plan(_reply("count", where="age_cat = 2 AND dx = 'other' OR sex = 'male'"), "How many?", patients)


def test_compile_plan_shareWithTheEventWrittenAsTheLimit_countsItAmongEveryone(patients):
    # Arrange: no numerator; the counted condition is in where
    reply = _reply("share", where="confirmed = 'yes'", group_by="sex")

    # Act
    compiled = compile_plan(reply, "What share of each sex had a confirmed infection?", patients)

    # Assert
    assert compiled.kind == "share"
    assert compiled.reading == "Share of patients with confirmed = 'Yes', by sex, among those with it recorded."


@pytest.mark.parametrize(
    "fields", [{"where": "sex = 'female' OR sex = 'male'"}, {"numerator": "sex IN ('female', 'male')"}]
)
def test_compile_plan_shareNamingEveryValue_isTheDistribution(patients, fields):
    # Act: "what share are women or men" names every value of sex
    compiled = compile_plan(_reply("share", **fields), "What share of patients are women or men?", patients)

    # Assert: a count per value (the app adds the percentages), not a share that is always 100%
    assert (compiled.kind, compiled.group_column) == ("count", "sex")


def test_compile_plan_rangeOnTheGroupColumn_isALimitNotAnError(visits):
    # Act: "each year since 2022" limits the years shown, it does not pin one
    compiled = compile_plan(
        _reply("count", where="visit_year >= 2022", group_by="visit_year"),
        "How many visits each year since 2022?",
        visits,
    )

    # Assert
    assert (compiled.group_column, [_flat(c.sql()) for c in compiled.conditions]) == (
        "visit_year",
        ['"visit_year" >= 2022'],
    )
