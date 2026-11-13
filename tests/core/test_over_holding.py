"""Over-holding fixes that cannot invert or widen a meaning, and the dangerous neighbours that
must still hold (label digit vs stored number, reversed rate, dropped filter, statin non-users,
negations, "or" lists, 0/1 columns and absence labels, which the gate never interprets)."""

from __future__ import annotations

from typing import Any

import polars as pl
import polars.testing
import pytest

from clinical_analytics.core.cohort_constraints import peel_cohort, yes_no_values
from clinical_analytics.core.query_plan import CohortStep, CohortStepError, QueryPlan, grouped_statistic
from clinical_analytics.core.question_vocabulary import NEGATION_WORDS, NO_LIMIT_WORDS
from clinical_analytics.core.word_accounting import (
    LOGIC_WORDS,
    explain_hold,
    measurement_grouping_reason,
    number_owner,
    tokenize,
    unjustified_elements,
    unused_words,
)

STATIN = "Statin use 1: Yes 2: No"
_COLUMNS = [
    "patient_id",
    "age",
    "sex",
    "treatment",
    "site",
    "ldl",
    "bmi",
    "diabetes",
    "diabetes_type",
    "Creatinine",
    "Weight (kg)",
    "followup_days",
    STATIN,
]
_VALUES: dict[str, list[Any]] = {
    "patient_id": ["a", "b", "c"],
    "age": [35, 50, 65, 84],
    "sex": ["F", "M"],
    "treatment": ["drug", "placebo"],
    "site": ["North", "South", "West"],
    "ldl": [80.5, 130.0, 175.7],
    "bmi": [18.3, 27.0, 37.5],
    "diabetes": [0, 1],
    "diabetes_type": ["Type 1", "Type 2", "None"],
    "Creatinine": [0.8, 1.4, 2.2, 3.1],
    "Weight (kg)": [45.0, 80.0, 140.0],
    "followup_days": [4, 400, 1458],
    "statin": ["atorvastatin", "simvastatin", "none"],
    STATIN: [1, 2],
}


def _values(column: str) -> list[Any]:
    return _VALUES[column]


def _steps(query: str) -> list[CohortStep]:
    return peel_cohort(query, _COLUMNS, _values).steps


# ---------------------------------------------------------------------------
# Shared vocabulary
# ---------------------------------------------------------------------------


def test_noLimitWords_vocabulary_sharesNoWordWithLogicOrNegation() -> None:
    # Arrange
    limiting = LOGIC_WORDS | NEGATION_WORDS | {"only", "currently", "or", "than"}

    # Act
    overlap = NO_LIMIT_WORDS & limiting

    # Assert
    assert overlap == set()


@pytest.mark.parametrize(
    "question",
    [
        pytest.param("can you tell me how many patients there are in the dataset?", id="politeness_and_dataset"),
        pytest.param("how many patients do we have in our data?", id="pronouns"),
        pytest.param("could you count my patients please", id="please"),
    ],
)
def test_unusedWords_fillerOnlyQuestion_everyFillerWordUsed(question: str) -> None:
    # Arrange
    plan = QueryPlan(intent="COUNT", confidence=0.9)

    # Act
    unused = unused_words(question, plan, _COLUMNS, _values)

    # Assert
    assert set(unused) <= {"tell"}


# ---------------------------------------------------------------------------
# Yes/no and absence coding
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("column", "values", "expected"),
    [
        pytest.param("diabetes", [0, 1], (None, None), id="indicator_without_codebook_is_not_yes_no"),
        pytest.param("diabetes", [0.0, 1.0, None], (None, None), id="indicator_float_with_null"),
        pytest.param("grade", [0, 1, 2], (None, None), id="three_codes_not_indicator"),
        pytest.param("diabetes", [1], (None, None), id="one_code_not_indicator"),
        pytest.param("diabetes", [], (None, None), id="empty"),
        pytest.param("flag 0: Yes 1: No", [0, 1], (0, 1), id="codebook_says_which"),
        pytest.param("smoker", ["yes", "no"], ("yes", "no"), id="text_yes_no"),
    ],
)
def test_yesNoValues_storedValues_readsTheCoding(column: str, values: list[Any], expected: tuple[Any, Any]) -> None:
    # Arrange: column and values above

    # Act
    coding = yes_no_values(column, values)

    # Assert
    assert coding == expected


# ---------------------------------------------------------------------------
# Peel: the dangerous neighbours of each fix
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        pytest.param(
            "how many patients with type 2 diabetes?",
            [CohortStep("eq", "diabetes_type", "Type 2")],
            id="label_digit_binds_label",
        ),
        pytest.param("how many patients with type 2.2 diabetes?", None, id="unknown_label_never_creatinine"),
        pytest.param("how many patients with creatinine 2.2?", [], id="bare_number_binds_nothing"),
    ],
)
def test_peelCohort_labelDigit_neverBindsAStoredNumber(query: str, expected: list[CohortStep] | None) -> None:
    # Arrange: query above (None: only the Creatinine property is checked; the gate holds "2.2")

    # Act
    steps = _steps(query)

    # Assert
    assert all(step.column != "Creatinine" for step in steps)
    assert expected is None or steps == expected


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        pytest.param("average age of statin non-users", [], id="postfix_non_users_unbound"),
        pytest.param("average age of statin users", [CohortStep("eq", STATIN, 1)], id="users"),
        pytest.param("how many non users of statin?", [CohortStep("eq", STATIN, 2)], id="stated_no_code"),
        pytest.param("how many statin non-users are women?", [], id="non_users_and_women_unbound"),
        pytest.param("how many patients not on placebo?", [], id="not_value"),
        pytest.param("how many patients excluding women?", [], id="excluding"),
        pytest.param("how many patients outside the west?", [], id="outside"),
        pytest.param("how many patients not on placebo with diabetes?", [], id="negation_before_two_items"),
        pytest.param("how many patients with no diabetes?", [], id="no_indicator_without_codebook"),
        pytest.param("how many drug a non-users?", [], id="multi_word_value_non_users"),
    ],
)
def test_peelCohort_negation_bindsOnlyAStatedNoCode(query: str, expected: list[CohortStep]) -> None:
    # Arrange: query above

    # Act
    steps = _steps(query)

    # Assert
    assert sorted(steps, key=repr) == sorted(expected, key=repr)


@pytest.mark.parametrize(
    "query",
    [
        pytest.param("how many patients in the north or west?", id="or_list"),
        pytest.param("how many patients in the north and south?", id="and_list"),
        pytest.param("how many patients not in the north or west?", id="negated_list"),
    ],
)
def test_peelCohort_valuesJoined_neverBindsAList(query: str) -> None:
    # Arrange: query above

    # Act
    steps = _steps(query)

    # Assert
    assert steps == []


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        pytest.param(
            "how many patients between 50 and 65?",
            [CohortStep("ge", "age", 50.0), CohortStep("le", "age", 65.0)],
            id="between",
        ),
        pytest.param(
            "how many patients aged 40 to 60?",
            [CohortStep("ge", "age", 40.0), CohortStep("le", "age", 60.0)],
            id="aged_to",
        ),
        pytest.param("how many patients not over 65?", [], id="negated_over_never_inverted"),
        pytest.param("how many patients over 100 kg?", [CohortStep("gt", "Weight (kg)", 100.0)], id="unit_names"),
        pytest.param("how many patients with ldl over 100 kg?", [], id="unit_conflicts_with_name"),
        pytest.param(
            "how many patients under 40 or over 80?",
            [CohortStep("lt", "age", 40.0), CohortStep("gt", "age", 80.0)],
            id="or_ranges_peeled_but_held_by_gate",
        ),
    ],
)
def test_peelCohort_comparison_bindsTheNamedRange(query: str, expected: list[CohortStep]) -> None:
    # Arrange: query above

    # Act
    steps = _steps(query)

    # Assert
    assert steps == expected


@pytest.mark.parametrize(
    ("query", "rate", "group"),
    [
        pytest.param("are men more likely to use statins than women?", STATIN, "sex", id="sex_before"),
        # A 0/1 column with no codebook is not yes/no: it is neither the groups nor the rate.
        pytest.param("are diabetics more likely to use statins?", STATIN, None, id="indicator_before"),
        pytest.param("are statin users more likely to be diabetic?", None, None, id="indicator_rate"),
        pytest.param("are patients on placebo more likely to be diabetic?", None, None, id="value_group"),
        pytest.param("are statin users more likely to be women?", None, None, id="sex_rate_unbound"),
    ],
)
def test_peelCohort_moreLikely_groupsBeforeRateAfter(query: str, rate: str | None, group: str | None) -> None:
    # Arrange: query above

    # Act
    peeled = peel_cohort(query, _COLUMNS, _values)

    # Assert
    assert (peeled.rate_column, peeled.locked_group) == (rate, group)


@pytest.mark.parametrize(
    "query",
    [
        pytest.param("compare ldl between women and men", id="every_value"),
        pytest.param("compare ldl between the north and the south", id="some_values"),
    ],
)
def test_peelCohort_compareBetweenValues_guessesNoGroup(query: str) -> None:
    # Arrange: query above

    # Act
    peeled = peel_cohort(query, _COLUMNS, _values)

    # Assert
    assert peeled.locked_group is None
    assert peeled.steps == []


@pytest.mark.parametrize(
    "query",
    [
        pytest.param("how many patients on statins?", id="on_statins"),
        pytest.param("how many patients not on a statin?", id="not_on"),
        pytest.param("which statin was most prescribed?", id="asking_about_values"),
    ],
)
def test_peelCohort_absenceLabel_neverReadAsYesNo(query: str) -> None:
    # Arrange: a dataset whose statin column records the drug, or "none"
    columns = ["patient_id", "age", "statin"]

    # Act
    steps = peel_cohort(query, columns, _values).steps

    # Assert
    assert steps == []


@pytest.mark.parametrize(
    "query",
    [
        pytest.param("how many events per subject?", id="per"),
        pytest.param("how many patients in each site?", id="each"),
    ],
)
def test_attachCohort_loneEachOrPerGrouping_isNotStructure(query: str) -> None:
    # Arrange
    from clinical_analytics.core.cohort_constraints import attach_cohort

    peeled = peel_cohort(
        query, [*_COLUMNS, "subject_id"], lambda column: ["s1", "s2"] if column == "subject_id" else _values(column)
    )

    # Act
    attached = attach_cohort(None, peeled)

    # Assert
    assert peeled.steps == []
    assert attached is None


# ---------------------------------------------------------------------------
# Gate
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("question", "plan", "expected"),
    [
        pytest.param(
            "are statin users more likely to be women?",
            QueryPlan(intent="COUNT", group_by="sex", rate_column=STATIN, rate_value=1, confidence=0.9),
            ["more likely", "women"],
            id="reversed_rate_held",
        ),
        pytest.param(
            "are men more likely to use statins than women?",
            QueryPlan(intent="COUNT", group_by="sex", rate_column=STATIN, rate_value=1, confidence=0.9),
            [],
            id="rate_in_question_order_used",
        ),
        pytest.param(
            "how many patients not on placebo with diabetes?",
            QueryPlan(intent="COUNT", cohort_steps=[CohortStep("ne", "treatment", "placebo")], confidence=0.9),
            ["not", "placebo", "diabetes"],
            id="dropped_filter_held",
        ),
        pytest.param(
            "average age of statin non-users",
            QueryPlan(
                intent="DESCRIBE",
                metric="age",
                statistic="mean",
                cohort_steps=[CohortStep("eq", STATIN, 1)],
                confidence=0.9,
            ),
            ["non"],
            id="non_users_as_users_held",
        ),
        pytest.param(
            "average age of patients older than 50 and younger than 70",
            QueryPlan(
                intent="DESCRIBE",
                metric="age",
                statistic="mean",
                cohort_steps=[CohortStep("gt", "age", 50.0), CohortStep("lt", "age", 70.0)],
                confidence=0.9,
            ),
            [],
            id="age_words_say_age",
        ),
        pytest.param(
            "median ldl over 70?",
            QueryPlan(
                intent="DESCRIBE",
                metric="ldl",
                statistic="median",
                cohort_steps=[CohortStep("gt", "ldl", 70.0)],
                confidence=0.9,
            ),
            ["over", "70"],
            id="summarised_measure_cutoff_still_ambiguous",
        ),
        pytest.param(
            "compare median ldl by sex",
            QueryPlan(intent="COMPARE_GROUPS", metric="ldl", group_by="sex", statistic="median", confidence=0.9),
            [],
            id="grouped_median_used",
        ),
        pytest.param(
            "how many patients under 40 or over 80?",
            QueryPlan(
                intent="COUNT",
                cohort_steps=[CohortStep("lt", "age", 40.0), CohortStep("gt", "age", 80.0)],
                confidence=0.9,
            ),
            ["or"],
            id="or_of_ranges_held",
        ),
    ],
)
def test_unusedWords_looserGate_stillHoldsTheDangerousPlans(
    question: str, plan: QueryPlan, expected: list[str]
) -> None:
    # Arrange: question and plan above

    # Act
    unused = unused_words(question, plan, _COLUMNS, _values)

    # Assert
    assert unused == expected


@pytest.mark.parametrize(
    ("question", "steps"),
    [
        pytest.param("how many patients on statins?", [CohortStep("ne", "statin", "none")], id="absence_label"),
        pytest.param("how many patients not on statins?", [CohortStep("ne", "statin", "none")], id="negated_absence"),
        pytest.param("how many patients not on placebo?", [CohortStep("ne", "treatment", "placebo")], id="negated"),
        pytest.param("how many patients on placebo?", [CohortStep("ne", "treatment", "drug")], id="no_negation"),
        pytest.param(
            "how many patients in the north or west?",
            [CohortStep("in", "site", ["North", "West"])],
            id="or_list",
        ),
    ],
)
def test_unjustifiedElements_exclusionOrListStep_neverJustified(question: str, steps: list[CohortStep]) -> None:
    # Arrange
    plan = QueryPlan(intent="COUNT", cohort_steps=steps, confidence=0.9)

    # Act
    unasked = unjustified_elements(question, plan, _COLUMNS, _values)

    # Assert
    assert unasked != []


@pytest.mark.parametrize(
    ("question", "plan", "expected"),
    [
        pytest.param(
            "what is the average bmi in patients over 30?",
            QueryPlan(intent="DESCRIBE", metric="bmi", statistic="mean", confidence=0.9),
            "Say which measure 30 applies to, for example 'age over 30' or 'bmi over 30'.",
            id="ambiguous_number",
        ),
        pytest.param(
            "compare bmi by sex for patients between 20 and 30",
            QueryPlan(intent="COMPARE_GROUPS", metric="bmi", group_by="sex", confidence=0.9),
            "Say which measure 20 to 30 applies to, for example 'age between 20 and 30' or 'bmi between 20 and 30'.",
            id="ambiguous_range",
        ),
        pytest.param(
            "how many women and men?",
            QueryPlan(intent="COUNT", confidence=0.9),
            "Say 'women or men' to count them together, or 'by sex' to see each group.",
            id="values_joined_by_and",
        ),
        pytest.param(
            "what is the mortality in women?",
            QueryPlan(intent="DESCRIBE", cohort_steps=[CohortStep("eq", "sex", "F")], confidence=0.9),
            "I couldn't match 'mortality' to a column or value in this dataset; rephrase with a column name from "
            "your data, or leave it out.",
            id="unknown_word",
        ),
        pytest.param(
            "ldl for women?",
            QueryPlan(intent="DESCRIBE", cohort_steps=[CohortStep("eq", "sex", "F")], confidence=0.9),
            "Say which statistic you want, for example 'average ldl' or 'median ldl'.",
            id="statistic_missing",
        ),
        pytest.param(
            "how many patients under 40 or over 80?",
            QueryPlan(
                intent="COUNT",
                cohort_steps=[CohortStep("lt", "age", 40.0), CohortStep("gt", "age", 80.0)],
                confidence=0.9,
            ),
            "I can't keep patients who meet either of two ranges; ask about each range in its own question.",
            id="ranges_joined_by_or",
        ),
        pytest.param("how many patients?", QueryPlan(intent="COUNT", confidence=0.9), None, id="nothing_to_explain"),
    ],
)
def test_explainHold_cause_isOneSentenceSayingWhatToChange(
    question: str, plan: QueryPlan, expected: str | None
) -> None:
    # Arrange: question and plan above

    # Act
    sentence = explain_hold(question, plan, _COLUMNS, _values)

    # Assert
    assert sentence == expected


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        pytest.param("how many patients over 100 kg", "Weight (kg)", id="unit_after_number"),
        pytest.param("how many patients over 365 days", "followup_days", id="days"),
        pytest.param("how many patients with ldl over 100 kg", None, id="unit_conflicts"),
        pytest.param("how many patients older than 65 years", "age", id="age_unit"),
    ],
)
def test_numberOwner_unitAfterNumber_namesItsColumn(question: str, expected: str | None) -> None:
    # Arrange
    tokens = tokenize(question)
    at = next(index for index, token in enumerate(tokens) if token.isdigit())
    numeric = {"age", "ldl", "bmi", "Weight (kg)", "followup_days"}

    # Act
    owner = number_owner(tokens, at, _COLUMNS, numeric)

    # Assert
    assert owner == expected


# ---------------------------------------------------------------------------
# Plan and execution
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("plan", "expected"),
    [
        pytest.param(QueryPlan(intent="DESCRIBE", metric="ldl", group_by="sex"), "mean", id="describe_default"),
        pytest.param(
            QueryPlan(intent="DESCRIBE", metric="ldl", group_by="sex", statistic="median"), "median", id="describe"
        ),
        pytest.param(
            QueryPlan(intent="COMPARE_GROUPS", metric="ldl", group_by="sex", statistic="median"),
            "median",
            id="compare_median",
        ),
        pytest.param(QueryPlan(intent="COMPARE_GROUPS", metric="ldl", group_by="sex"), None, id="compare_means_test"),
        pytest.param(
            QueryPlan(intent="COMPARE_GROUPS", metric="ldl", group_by="sex", statistic="mean"),
            None,
            id="compare_mean_test",
        ),
        pytest.param(QueryPlan(intent="DESCRIBE", metric="ldl"), None, id="ungrouped"),
        pytest.param(QueryPlan(intent="COUNT", group_by="sex"), None, id="count"),
    ],
)
def test_groupedStatistic_plan_isTheStatisticComputedPerGroup(plan: QueryPlan, expected: str | None) -> None:
    # Arrange: plan above

    # Act
    statistic = grouped_statistic(plan)

    # Assert
    assert statistic == expected


@pytest.mark.parametrize(
    "step",
    [
        pytest.param(CohortStep("ne", "treatment", "placebo"), id="ne"),  # type: ignore[arg-type]
        pytest.param(CohortStep("in", "site", ["North", "West"]), id="in"),  # type: ignore[arg-type]
    ],
)
def test_executeQueryPlan_exclusionOrListStep_isRefused(make_semantic_layer, step: CohortStep) -> None:
    # Arrange
    semantic = make_semantic_layer(
        dataset_name="ne_in_steps",
        data={
            "patient_id": ["P1", "P2", "P3"],
            "treatment": ["drug", "placebo", "drug"],
            "site": ["North", "West", "South"],
        },
    )
    plan = QueryPlan(intent="COUNT", cohort_steps=[step], confidence=0.95)

    # Act / Assert
    with pytest.raises(CohortStepError):
        semantic._filtered_view(plan)


def test_executeQueryPlan_compareMedianByGroup_computesMedians(make_semantic_layer) -> None:
    # Arrange
    data = {"patient_id": ["P1", "P2", "P3", "P4", "P5"], "sex": ["F", "F", "F", "M", "M"], "ldl": [1, 2, 10, 4, 6]}
    semantic = make_semantic_layer(dataset_name="compare_median", data=data)
    plan = QueryPlan(intent="COMPARE_GROUPS", metric="ldl", group_by="sex", statistic="median", confidence=0.95)
    expected = pl.DataFrame(data).group_by("sex").agg(pl.col("ldl").median().alias("median")).sort("sex")

    # Act
    result = semantic.execute_query_plan(plan)
    frame = result["result"]
    frame = frame if isinstance(frame, pl.DataFrame) else pl.from_pandas(frame)  # pandas: ibis returns it

    # Assert
    polars.testing.assert_frame_equal(frame.select("sex", "median"), expected, check_dtypes=False)


def _many_values(column: str) -> list[Any]:
    if column == "ldl":
        return [80.5 + index for index in range(40)]
    return _VALUES[column]


@pytest.mark.parametrize(
    ("plan", "expected"),
    [
        pytest.param(
            QueryPlan(intent="COMPARE_GROUPS", metric="treatment", group_by="ldl"),
            "ldl is a measurement, not a group; ask about a range of it such as 'ldl over 100', or ask "
            "'compare ldl by treatment'.",
            id="crosstab_by_measurement",
        ),
        pytest.param(QueryPlan(intent="COUNT", group_by="ldl"), "measurement", id="count_by_measurement"),
        pytest.param(QueryPlan(intent="COUNT", group_by="site"), None, id="labels"),
        pytest.param(QueryPlan(intent="COUNT", group_by="diabetes"), None, id="indicator"),
        pytest.param(QueryPlan(intent="COUNT", group_by="age"), None, id="few_numeric_levels"),
        pytest.param(QueryPlan(intent="COUNT"), None, id="no_grouping"),
        pytest.param(QueryPlan(intent="CORRELATIONS", metric="age", group_by="ldl"), None, id="correlation_pair"),
    ],
)
def test_measurementGroupingReason_plan_refusesOneGroupPerValue(plan: QueryPlan, expected: str | None) -> None:
    # Arrange: ldl holds 40 distinct values, age 4

    # Act
    reason = measurement_grouping_reason(plan, _COLUMNS, _many_values)

    # Assert
    if expected is None or expected == "measurement":
        assert (reason is None) == (expected is None)
        assert expected is None or "is a measurement, not a group" in (reason or "")
    else:
        assert reason == expected
