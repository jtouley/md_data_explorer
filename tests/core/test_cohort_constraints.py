"""Cohort binding must never widen or redirect a question.

Pure-function tests on peel_cohort / attach_cohort, plus the semantic layer's fail-closed
handling of cohort steps it cannot apply.
"""

from __future__ import annotations

from typing import Any

import pytest

from clinical_analytics.core.cohort_constraints import PeelResult, attach_cohort, peel_cohort
from clinical_analytics.core.query_plan import CohortStep, CohortStepError, FilterSpec, QueryPlan

_COLUMNS = ["patient_id", "age", "gender", "treatment", "ldl", "Statin use 1: Yes 2: No"]
_VALUES: dict[str, list[Any]] = {
    "patient_id": ["a", "b"],
    "age": [40, 55, 70],
    "gender": ["Female", "Male"],
    "treatment": ["drug", "placebo"],
    "ldl": [100.0, 150.0, 175.0],
    "Statin use 1: Yes 2: No": [1, 2],
}


def _peel(query: str) -> PeelResult:
    return peel_cohort(query, _COLUMNS, lambda column: _VALUES[column])


@pytest.mark.parametrize(
    ("query", "expected_value"),
    [
        pytest.param("How many male patients?", "Male", id="male_is_not_female"),
        pytest.param("How many female patients?", "Female", id="female"),
        pytest.param("average ldl for male patients", "Male", id="male_in_stat_question"),
        pytest.param("how many women?", "Female", id="inflection_women"),
        pytest.param("how many men?", "Male", id="inflection_men"),
    ],
)
def test_peelCohort_sexWord_bindsTheNamedValue(query: str, expected_value: str) -> None:
    # Arrange / Act
    peeled = _peel(query)

    # Assert
    assert peeled.steps == [CohortStep("eq", "gender", expected_value)]


@pytest.mark.parametrize(
    ("query", "metric", "group"),
    [
        pytest.param("compare ldl by treatment", "ldl", "treatment", id="named_group_not_sex"),
        pytest.param("compare age by treatment", "age", "treatment", id="metric_before_by"),
        pytest.param("compare ldl by gender", "ldl", "gender", id="sex_group_when_named"),
        pytest.param("compare ldl between treatment", "ldl", "treatment", id="between"),
    ],
)
def test_peelCohort_compareQuestion_locksNamedMetricAndGroup(query: str, metric: str, group: str) -> None:
    # Arrange / Act
    peeled = _peel(query)

    # Assert
    assert (peeled.locked_metric, peeled.locked_group) == (metric, group)


def test_peelCohort_compareWithoutGroup_doesNotDefaultToSex() -> None:
    # Arrange / Act
    peeled = _peel("compare ldl")

    # Assert
    assert peeled.locked_group is None


@pytest.mark.parametrize(
    "query",
    [
        pytest.param("how many patients by statin use 1 yes 2 no?", id="count_by_coded_yes_no"),
        pytest.param("what is the average ldl by statin use 1 yes 2 no?", id="mean_by_coded_yes_no"),
    ],
)
def test_peelCohort_groupColumn_isNotAlsoFiltered(query: str) -> None:
    # Arrange / Act
    peeled = _peel(query)

    # Assert
    assert [step for step in peeled.steps if step.column == "Statin use 1: Yes 2: No"] == []


def test_peelCohort_statPhraseBeforeBy_locksMetricNotGroup() -> None:
    # Arrange / Act
    peeled = _peel("what is the average ldl by treatment?")

    # Assert
    assert peeled.locked_metric == "ldl"


@pytest.mark.parametrize(
    "intent",
    [pytest.param(None, id="unparsed"), pytest.param("low", id="low_confidence")],
)
def test_attachCohort_nothingPeeled_doesNotFabricateConfidentCount(intent: str | None) -> None:
    # Arrange
    from clinical_analytics.core.nl_query_engine import QueryIntent

    parsed = None if intent is None else QueryIntent(intent_type="DESCRIBE", confidence=0.3)
    peeled = PeelResult(residual="tell me about the weather?")

    # Act
    attached = attach_cohort(parsed, peeled)

    # Assert
    assert attached is parsed
    assert attached is None or attached.confidence < 0.9


def test_attachCohort_peeledSteps_buildCountWithSteps() -> None:
    # Arrange
    peeled = PeelResult(residual="how many patients?", steps=[CohortStep("eq", "gender", "Male")])

    # Act
    attached = attach_cohort(None, peeled)

    # Assert
    assert attached.intent_type == "COUNT"
    assert attached.cohort_steps == [CohortStep("eq", "gender", "Male")]


def test_attachCohort_unparsedResidual_buildsIntentAtZeroConfidence() -> None:
    # Arrange: confidence is never raised; an intent built for an unparsed residual is held
    peeled = PeelResult(residual="how many patients?", steps=[CohortStep("eq", "gender", "Male")])

    # Act
    attached = attach_cohort(None, peeled)

    # Assert
    assert attached.confidence == 0.0


@pytest.mark.parametrize("confidence", [pytest.param(0.3, id="weak"), pytest.param(0.9, id="pattern")])
def test_attachCohort_parsedIntent_keepsItsConfidence(confidence: float) -> None:
    # Arrange
    from clinical_analytics.core.nl_query_engine import QueryIntent

    parsed = QueryIntent(intent_type="COUNT", confidence=confidence)
    peeled = PeelResult(residual="how many patients?", steps=[CohortStep("eq", "gender", "Male")])

    # Act
    attached = attach_cohort(parsed, peeled)

    # Assert
    assert attached.confidence == confidence
    assert attached.cohort_steps == [CohortStep("eq", "gender", "Male")]


@pytest.mark.parametrize(
    ("values", "query", "expected"),
    [
        pytest.param(
            ["North", "North East", "South"], "how many patients in North East?", "North East", id="most_specific"
        ),
        pytest.param(["North", "North East", "South"], "how many patients in the north?", "North", id="shorter_exact"),
        pytest.param(
            ["Type 1", "Type 2", "None"], "how many patients with type 2 diabetes?", "Type 2", id="digits_decide"
        ),
        pytest.param(
            ["Type 1", "Type 2", "None"], "how many patients with type diabetes?", None, id="tie_binds_nothing"
        ),
        # "count" prefix-matches "county" in every value: a longer value must not win the tie.
        pytest.param(
            ["Colorado", "Adams County", "Las Animas County", "El Paso County"],
            "count by sex",
            None,
            id="count_is_not_the_longest_county",
        ),
        pytest.param(
            ["Colorado", "Adams County", "Las Animas County", "El Paso County"],
            "average rate by county",
            None,
            id="county_names_no_one_county",
        ),
        pytest.param(
            ["Colorado", "Adams County", "Las Animas County", "El Paso County"],
            "count in las animas county",
            "Las Animas County",
            id="named_county_still_binds",
        ),
    ],
)
def test_peelCohort_storedValues_bindMostSpecificAndNeverATie(
    values: list[Any], query: str, expected: str | None
) -> None:
    # Arrange
    columns = ["patient_id", "label"]

    # Act
    peeled = peel_cohort(query, columns, lambda column: values if column == "label" else ["p1"])

    # Assert
    bound = [step.value for step in peeled.steps if step.column == "label"]
    assert bound == ([] if expected is None else [expected])


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        pytest.param("how many patients with ldl over 130?", [CohortStep("gt", "ldl", 130.0)], id="named_measure"),
        pytest.param("how many patients over 65?", [CohortStep("gt", "age", 65.0)], id="subject_means_age"),
        # A measure named before "patients over N": N inside that measure's observed range
        # could be either, so nothing binds; outside it, N can only be an age.
        pytest.param("what is the average ldl in patients over 120?", [], id="in_measure_range_unbound"),
        pytest.param(
            "what is the average ldl in patients over 30?", [CohortStep("gt", "age", 30.0)], id="out_of_range_is_age"
        ),
        pytest.param("how many patients aged 65 or older?", [], id="ge_is_not_gt"),
    ],
)
def test_peelCohort_overNumber_bindsItsOwnerOrNothing(query: str, expected: list[CohortStep]) -> None:
    # Arrange / Act
    peeled = _peel(query)

    # Assert
    assert [step for step in peeled.steps if step.op == "gt"] == expected


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        pytest.param("how many patients aged 65 or older?", CohortStep("ge", "age", 65.0), id="or_older"),
        pytest.param("how many patients 65+?", CohortStep("ge", "age", 65.0), id="plus"),
        pytest.param("how many patients at least 65?", CohortStep("ge", "age", 65.0), id="at_least"),
        pytest.param("how many patients with ldl under 120?", CohortStep("lt", "ldl", 120.0), id="under_named"),
        pytest.param("how many patients 65 or younger?", CohortStep("le", "age", 65.0), id="or_younger"),
    ],
)
def test_peelCohort_comparisonForms_bindOperatorOnNamedColumn(query: str, expected: CohortStep) -> None:
    # Arrange / Act
    peeled = _peel(query)

    # Assert
    assert peeled.steps == [expected]


@pytest.mark.parametrize(
    "columns",
    [
        pytest.param(["patient_id", "Age (years)"], id="age_years_label"),
        pytest.param(["patient_id", "age_years"], id="snake"),
    ],
)
def test_peelCohort_ageByMeaning_bindsAgeColumn(columns: list[str]) -> None:
    # Arrange
    values: dict[str, list[Any]] = {columns[0]: ["a"], columns[1]: [30, 50, 70]}

    # Act
    peeled = peel_cohort("how many patients over 65?", columns, lambda column: values[column])

    # Assert
    assert peeled.steps == [CohortStep("gt", columns[1], 65.0)]


@pytest.mark.parametrize(
    ("query", "column", "value"),
    [
        pytest.param("how many patients taking statins?", "Statin use 1: Yes 2: No", 1, id="plural_column"),
        pytest.param("how many females?", "gender", "Female", id="plural_value"),
    ],
)
def test_peelCohort_plurals_bindLikeSingular(query: str, column: str, value: Any) -> None:
    # Arrange / Act
    peeled = _peel(query)

    # Assert
    assert peeled.steps == [CohortStep("eq", column, value)]


def test_peelCohort_codedHeaderLabel_bindsItsCode() -> None:
    # Arrange
    dexa = "DEXA result 1: Normal 2: Osteopenia 3: Osteoporosis"
    values: dict[str, list[Any]] = {"patient_id": ["a"], dexa: [1, 2, 3]}

    # Act
    peeled = peel_cohort("how many patients with osteopenia?", ["patient_id", dexa], lambda column: values[column])

    # Assert
    assert peeled.steps == [CohortStep("eq", dexa, 2)]


def test_peelCohort_eachSite_locksGrouping() -> None:
    # Arrange
    values: dict[str, list[Any]] = {"patient_id": ["a"], "site": ["North", "South"]}

    # Act
    peeled = peel_cohort("how many patients in each site?", ["patient_id", "site"], lambda column: values[column])

    # Assert
    assert peeled.locked_group == "site"


def test_attachCohort_correlation_setsPredictorVariables() -> None:
    # Arrange
    peeled = _peel("what is the correlation between age and ldl?")

    # Act
    attached = attach_cohort(None, peeled)

    # Assert
    assert attached.intent_type == "CORRELATIONS"
    assert sorted(attached.predictor_variables) == ["age", "ldl"]


@pytest.mark.parametrize(
    "query",
    [
        pytest.param("remove male patients", id="remove"),
        pytest.param("exclude the placebo group", id="exclude"),
        pytest.param("how many patients without placebo?", id="without"),
        pytest.param("don't want female patients", id="dont_want"),
    ],
)
def test_peelCohort_negatedValue_isNeverBoundAsEquality(query: str) -> None:
    # Arrange / Act
    peeled = _peel(query)

    # Assert
    assert [step for step in peeled.steps if step.op == "eq"] == []


@pytest.mark.parametrize(
    ("query", "expected_code"),
    [
        pytest.param("how many patients on statin?", 1, id="affirmative_yes_code"),
        pytest.param("how many patients not on statin?", 2, id="not_no_code"),
        pytest.param("how many patients excluding statin users?", 2, id="excluding_no_code"),
    ],
)
def test_peelCohort_codedYesNoColumn_polarityFollowsNegation(query: str, expected_code: int) -> None:
    # Arrange / Act
    peeled = _peel(query)

    # Assert
    assert peeled.steps == [CohortStep("eq", "Statin use 1: Yes 2: No", expected_code)]


@pytest.mark.parametrize(
    "query",
    [
        pytest.param("remove control group", id="refinement"),
        pytest.param("how many patients", id="plain_count"),
        pytest.param("tell me about the weather", id="unrelated"),
    ],
)
def test_peelCohort_nothingPeeled_residualIsTheQuestionAsAsked(query: str) -> None:
    # Arrange / Act
    peeled = _peel(query)

    # Assert
    assert peeled.residual == query


# An op the semantic layer does not know: it must refuse, never skip.
_UNSUPPORTED_STEP = CohortStep("ge_decile", "ldl", quantile=0.9)  # type: ignore[arg-type]
_LAYER_DATA = {"patient_id": [1, 2, 3, 4], "age": [40, 50, 60, 70], "ldl": [100.0, 120.0, 140.0, 160.0]}


@pytest.mark.parametrize(
    ("plan_kwargs", "reason"),
    [
        pytest.param(
            {"cohort_steps": [CohortStep("eq", "missing_col", 1)]}, "not in this dataset", id="missing_column"
        ),
        pytest.param({"cohort_steps": [_UNSUPPORTED_STEP]}, "not supported", id="unsupported_op"),
        pytest.param({"filters": [FilterSpec("missing_col", "==", 1)]}, "missing_col", id="missing_filter"),
    ],
)
def test_executeQueryPlan_unappliableRestriction_refusesWithReason(
    make_semantic_layer, plan_kwargs: dict[str, Any], reason: str
) -> None:
    # Arrange
    layer = make_semantic_layer(data=_LAYER_DATA)
    plan = QueryPlan(intent="COUNT", confidence=0.95, **plan_kwargs)

    # Act
    result = layer.execute_query_plan(plan)

    # Assert
    assert result["success"] is False
    assert reason in result["failure_reason"]


@pytest.mark.parametrize(
    "step",
    [
        pytest.param(CohortStep("eq", "missing_col", 1), id="missing_column"),
        pytest.param(_UNSUPPORTED_STEP, id="unsupported_op"),
    ],
)
def test_applyCohortStep_unappliableStep_raisesInsteadOfSkipping(make_semantic_layer, step: CohortStep) -> None:
    # Arrange
    layer = make_semantic_layer(data=_LAYER_DATA)

    # Act / Assert
    with pytest.raises(CohortStepError):
        layer._apply_cohort_step(layer.get_base_view(), step)


def test_executeQueryPlan_supportedStep_stillAnswers(make_semantic_layer) -> None:
    # Arrange
    layer = make_semantic_layer(data=_LAYER_DATA)
    plan = QueryPlan(intent="COUNT", confidence=0.95, cohort_steps=[CohortStep("gt", "age", 55)])

    # Act
    result = layer.execute_query_plan(plan)

    # Assert
    assert result["success"] is True


def test_peelCohort_columnNameInsideStoredValue_isNotAValueFilter() -> None:
    # Arrange
    columns = ["patient_id", "Current Regimen"]
    values: dict[str, list[Any]] = {"patient_id": ["a"], "Current Regimen": ["regimen_1", "regimen_2"]}

    # Act
    peeled = peel_cohort("what was the most common Current Regimen", columns, lambda column: values[column])

    # Assert
    assert peeled.steps == []
    assert peeled.residual == "what was the most common Current Regimen"


_SHARE_COLUMNS = ["patient_id", "sex", "treatment", "site", "Statin use 1: Yes 2: No"]
_SHARE_VALUES: dict[str, list[Any]] = {
    "patient_id": ["a", "b", "c"],
    "sex": ["F", "M"],
    "treatment": ["drug", "placebo"],
    "site": ["North", "South", "West"],
    "Statin use 1: Yes 2: No": [1, 2],
}
_STATIN = "Statin use 1: Yes 2: No"


@pytest.mark.parametrize(
    ("query", "limits", "numerator"),
    [
        pytest.param(
            "what proportion of statin users are women?",
            [CohortStep("eq", _STATIN, 1)],
            ("sex", "F"),
            id="statin_users_are_women",
        ),
        pytest.param(
            "what proportion of women are on statin?",
            [CohortStep("eq", "sex", "F")],
            (_STATIN, 1),
            id="women_on_statin",
        ),
        pytest.param(
            "what share of women are not on statin?",
            [CohortStep("eq", "sex", "F")],
            (_STATIN, 2),
            id="negated_numerator",
        ),
        pytest.param("what proportion of patients are on drug?", [], ("treatment", "drug"), id="value_numerator"),
        pytest.param(
            "what percentage of patients in the north use statins?",
            [CohortStep("eq", "site", "North")],
            (_STATIN, 1),
            id="denominator_with_limit",
        ),
    ],
)
def test_peelCohort_shareQuestion_denominatorBeforeVerbNumeratorAfter(
    query: str, limits: list[CohortStep], numerator: tuple[str, Any]
) -> None:
    # Arrange / Act
    peeled = peel_cohort(query, _SHARE_COLUMNS, lambda column: _SHARE_VALUES[column])

    # Assert
    assert peeled.steps == limits
    assert (peeled.proportion_column, peeled.proportion_value) == numerator


@pytest.mark.parametrize(
    "query",
    [
        pytest.param("what is the statin proportion for women?", id="no_of_split"),
        pytest.param("what proportion are women on drug?", id="no_denominator"),
        pytest.param("what proportion of patients are women on drug?", id="two_numerators"),
    ],
)
def test_peelCohort_shareQuestionWithoutClearDirection_setsNoProportion(query: str) -> None:
    # Arrange / Act
    peeled = peel_cohort(query, _SHARE_COLUMNS, lambda column: _SHARE_VALUES[column])

    # Assert
    assert peeled.proportion_column is None


@pytest.mark.parametrize(
    ("gaps", "expected"),
    [
        (
            ["treatment", "site", "diabetes", "diabete", "diabete", "diabetes", "site", "treatment"],
            "('treatment', 'site', 'diabetes')",
        ),
        # Unrelated words that share a prefix are both kept: only inflections collapse.
        (["men", "menopause"], "('men', 'menopause')"),
        (["age", "agent"], "('age', 'agent')"),
        (["match", "matches"], "('matches')"),
    ],
)
def test_executeQueryPlan_unappliedWords_listedOnceEach(
    make_semantic_layer, monkeypatch, gaps: list[str], expected: str
) -> None:
    # Arrange: the gap check reports each word (and a stem of one) more than once
    from clinical_analytics.core import cohort_constraints

    monkeypatch.setattr(cohort_constraints, "unapplied_constraints", lambda *_args: gaps)
    layer = make_semantic_layer(data=_LAYER_DATA)
    plan = QueryPlan(intent="COUNT", confidence=0.95)

    # Act
    result = layer.execute_query_plan(plan, query_text="how many by site and diabetes and treatment")

    # Assert
    assert result["success"] is False
    assert expected in result["failure_reason"]


# --- value words: whole words only, grammar never, ambiguity holds (aggregate triage #4) ------

_GEO_COLUMNS = ["location_name", "state", "sex", "mx"]
_GEO_VALUES: dict[str, list[Any]] = {
    "location_name": ["Colorado", "Las Animas County", "Adams County", "El Paso County", "Count Lake"],
    "state": ["New Mexico", "North Dakota", "South Dakota"],
    "sex": ["Male", "Female", "Total"],
    "mx": [101.5, 220.25, 349.75],
}


def _geo_hits(query: str) -> list[tuple[str, Any]]:
    from clinical_analytics.core.cohort_constraints import _value_hits

    hits = _value_hits(query.lower(), _GEO_COLUMNS, lambda column: _GEO_VALUES[column])
    return [(column, value) for column, value, _at in hits]


@pytest.mark.parametrize(
    "query",
    [
        pytest.param("count by sex", id="count_isNotCounty"),
        pytest.param("count", id="count_alone"),
        pytest.param("number of patients per state", id="number_per"),
        pytest.param("how many patients by state?", id="how_many_by"),
        pytest.param("total count", id="total_isNotSexTotal"),
        pytest.param("average mx by county", id="county_sharedByCounties"),
        pytest.param("how many patients in a county?", id="county_noLongestWins"),
        pytest.param("how many in the counties?", id="counties_plural_shared"),
        pytest.param("how many patients in dakota?", id="dakota_sharedByStates"),
        pytest.param("mean mx per year", id="mean_per"),
        pytest.param("", id="empty"),
    ],
)
def test_valueHits_grammarOrSharedWord_bindsNoValue(query: str) -> None:
    # Arrange: query above

    # Act
    hits = _geo_hits(query)

    # Assert
    assert hits == []


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        pytest.param("how many patients in las animas county?", [("location_name", "Las Animas County")], id="full"),
        pytest.param("how many patients in adams county?", [("location_name", "Adams County")], id="two_words"),
        pytest.param("how many patients in las animas?", [("location_name", "Las Animas County")], id="partial"),
        pytest.param("how many patients in count lake?", [("location_name", "Count Lake")], id="count_lake"),
        pytest.param("how many in north dakota?", [("state", "North Dakota")], id="more_words_named"),
        pytest.param("average mx in new mexico", [("state", "New Mexico")], id="state"),
        pytest.param("count of women in colorado", [("location_name", "Colorado"), ("sex", "Female")], id="two"),
    ],
)
def test_valueHits_wholeWordsNamed_bindsThatValue(query: str, expected: list[tuple[str, Any]]) -> None:
    # Arrange: query above

    # Act
    hits = _geo_hits(query)

    # Assert
    assert sorted(hits) == sorted(expected)


@pytest.mark.parametrize(
    ("word", "part", "expected"),
    [
        pytest.param("county", "county", True, id="same"),
        pytest.param("statins", "statin", True, id="plural"),
        pytest.param("counties", "county", True, id="plural_ies_sharedStem"),
        pytest.param("diabetic", "diabetes", True, id="sharedStem_bothGoOn"),
        pytest.param("count", "county", False, id="prefix_ofValueWord"),
        pytest.param("county", "count", False, id="valueWord_isPrefix"),
        pytest.param("northern", "north", False, id="northern_isNotNorth"),
        pytest.param("diabet", "diabetes", False, id="truncated"),
        pytest.param("lake", "lakewood", False, id="lake_isNotLakewood"),
    ],
)
def test_namesWord_pair_matchesWholeWordsOnly(word: str, part: str, expected: bool) -> None:
    # Arrange
    from clinical_analytics.core.cohort_constraints import _names_word

    # Act
    named = _names_word(word, part)

    # Assert
    assert named is expected


@pytest.mark.parametrize("word", ["count", "number", "how", "many", "total", "by", "per", "each", "average", "mean"])
def test_questionFormWords_grammarWord_isInSharedVocabulary(word: str) -> None:
    # Arrange
    from clinical_analytics.core.question_vocabulary import QUESTION_FORM_WORDS

    # Act
    present = word in QUESTION_FORM_WORDS

    # Assert
    assert present
