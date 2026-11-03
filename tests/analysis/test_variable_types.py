"""Classification, code decoding and event-level choice (analysis/variable_types.py).

Test name follows: test_unit_scenario_expectedBehavior
"""

import polars as pl
import pytest

from clinical_analytics.analysis.variable_types import (
    binary_event_level,
    classify_variable,
    code_labels,
    display_level,
    short_name,
)


@pytest.mark.parametrize(
    ("header", "expected"),
    [
        (
            "DEXA result 1: Normal 2: Osteopenia 3: Osteoporosis",
            {"1": "Normal", "2": "Osteopenia", "3": "Osteoporosis"},
        ),
        ("DEXA 1: Normal, 2: Osteopenia, 3: Osteoporosis", {"1": "Normal", "2": "Osteopenia", "3": "Osteoporosis"}),
        ("Smoker (1=yes, 2=no)", {"1": "yes", "2": "no"}),
        ("Smoker 1 = Yes / 2 = No", {"1": "Yes", "2": "No"}),
        ("Relapse within 1 year? Yes: 1 No: 0", {"1": "Yes", "0": "No"}),
        ("IL-6: pg/mL", {}),
        ("Week 4: weight kg", {}),
        ("ldl", {}),
        ("", {}),
    ],
    ids=[
        "dexa",
        "comma_separated",
        "parenthesised",
        "slash_separated",
        "label_first",
        "unit",
        "time_point",
        "plain",
        "empty",
    ],
)
def test_code_labels_header_decodesOnlyRealCodeLists(header, expected):
    # Arrange: header from parametrize

    # Act
    labels = code_labels(header)

    # Assert
    assert labels == expected


@pytest.mark.parametrize(
    ("series", "expected"),
    [
        (pl.Series("IL-6: pg/mL", [1.2, 3.4, 5.6, 7.8]), "continuous"),
        (pl.Series("flag", [0, 1, 2, 1, 0]), "categorical"),
        (pl.Series("grade", [1, 2, 3, 2]), "categorical"),
        (pl.Series("score", [0, 1, 2, 3, 4, 5]), "continuous"),
        (pl.Series("diabetes", [0, 1, 1, 0], dtype=pl.Int8), "binary"),
        (pl.Series("diabetes", [0.0, 1.0, None]), "binary"),
        (pl.Series("smoker", [True, False]), "binary"),
        (pl.Series("lab", [1234.5, 2345.6, 1234.5]), "continuous"),
        (pl.Series("DEXA 1: Normal 2: Osteopenia 3: Osteoporosis", [1, 2, 3, 1]), "categorical"),
        (pl.Series("sex", ["F", "M", None]), "binary"),
    ],
    ids=[
        "unit_header",
        "zero_one_two",
        "one_two_three",
        "score",
        "int8_flag",
        "float_flag",
        "bool",
        "two_measures",
        "coded",
        "text",
    ],
)
def test_classify_variable_byMeaning_returnsKind(series, expected):
    # Arrange: series from parametrize

    # Act
    kind = classify_variable(series)

    # Assert
    assert kind == expected


@pytest.mark.parametrize(
    ("series", "expected"),
    [
        (pl.Series("Status 1: Dead 2: Alive", [1, 2, 2]), 1),
        (pl.Series("Status 1: Alive 2: Dead", [1, 2, 2]), 2),
        (pl.Series("Statin use 1: Yes 2: No", [1, 2]), 1),
        (pl.Series("Smoker (1=yes, 2=no)", [1, 2]), 1),
        (pl.Series("outcome", ["Survived", "Died"]), "Died"),
        (pl.Series("outcome", ["no", "yes"]), "yes"),
        (pl.Series("outcome", [0, 0, 0]), 1),
        (pl.Series("outcome", [0.0, 1.0]), 1.0),
        (pl.Series("outcome", [True, False]), True),
        (pl.Series("smoker", [1, 2, 1]), None),
        (pl.Series("arm", ["A", "B"]), "B"),
        (pl.Series("Status 0: Dead 1: Alive", [0, 1, 1]), 0),
        (pl.Series("Alive 1: Yes 2: No", [1, 2]), None),
        (pl.Series("survived", [0, 1]), None),
    ],
    ids=[
        "dead_first", "dead_second", "statin_yes", "parenthesised_yes", "text_died", "text_yes", "all_zero",
        "float", "bool", "bare_one_two_ambiguous", "unrecognised_text_last", "zero_one_labels_win",
        "non_event_column_name", "non_event_column_zero_one",
    ],
)  # fmt: skip
def test_binary_event_level_labels_picksEventByMeaning(series, expected):
    # Arrange: series from parametrize

    # Act
    level = binary_event_level(series)

    # Assert
    assert level == expected


@pytest.mark.parametrize(
    ("column", "level", "expected"),
    [
        ("DEXA result 1: Normal 2: Osteopenia", 2.0, "Osteopenia"),
        ("DEXA result 1: Normal 2: Osteopenia", "2", "Osteopenia"),
        ("MI", 1, "MI=1"),
        ("arm", "drug", "drug"),
    ],
    ids=["float_code", "text_code", "bare_code", "text_level"],
)
def test_display_level_code_decodes(column, level, expected):
    # Arrange: values from parametrize

    # Act
    label = display_level(column, level)

    # Assert
    assert label == expected


@pytest.mark.parametrize(
    ("column", "expected"),
    [
        ("DEXA result 1: Normal 2: Osteopenia", "DEXA result"),
        ("Smoker (1=yes, 2=no)", "Smoker"),
        ("IL-6: pg/mL", "IL-6: pg/mL"),
    ],
    ids=["coded", "parenthesised", "unit_kept"],
)
def test_short_name_header_stripsOnlyCodeList(column, expected):
    # Arrange: values from parametrize

    # Act
    name = short_name(column)

    # Assert
    assert name == expected


@pytest.mark.parametrize(
    ("header", "expected"),
    [
        ("Age at visit 1: years, visit 2: years", {}),
        ("Age at visit 1: years visit 2: years", {}),
        ("Weight 1: kg 2: kg", {}),
        ("LDL 1: mg/dL 2: mg/dL", {}),
        ("1: Yes, 2: No", {"1": "Yes", "2": "No"}),
        ("0: male 1: female", {"0": "male", "1": "female"}),
        ("Sex 1: M 2: F", {"1": "M", "2": "F"}),
        ("Arm 1: Treatment A 2: Treatment B", {"1": "Treatment A", "2": "Treatment B"}),
    ],
    ids=[
        "visit_units_comma", "visit_units", "unit_kg", "unit_with_slash",
        "yes_no_comma", "male_female", "single_letters", "multi_word_labels",
    ],
)  # fmt: skip
def test_code_labels_unitOrRunOnLabels_rejectsListButKeepsRealCodebooks(header, expected):
    # Arrange: header from parametrize

    # Act
    labels = code_labels(header)

    # Assert
    assert labels == expected


def test_classify_variable_visitUnitsHeader_staysContinuous():
    # Arrange: ages under a header that mentions "visit 1: years, visit 2: years"
    series = pl.Series("Age at visit 1: years, visit 2: years", [21.5, 44.0, 63.2, 30.1, 55.0])

    # Act
    kind = classify_variable(series)

    # Assert
    assert kind == "continuous"


@pytest.mark.parametrize(
    ("series", "expected"),
    [
        (pl.Series("smoker", ["1", "2", "1"]), None),
        (pl.Series("smoker", ["2", "1", None]), None),
        (pl.Series("smoker", [" 1", "2 "]), None),
        (pl.Series("died", ["0", "1", "0"]), "1"),
        (pl.Series("died", ["0", "0"]), "1"),
        (pl.Series("Smoker (1=yes, 2=no)", ["1", "2"]), "1"),
        (pl.Series("Smoker 1: No 2: Yes", ["1", "2"]), "2"),
        (pl.Series("outcome", ["no", "yes"]), "yes"),
    ],
    ids=[
        "text_one_two", "text_one_two_null", "text_padded", "text_zero_one", "text_zero_only",
        "text_labelled_yes_first", "text_labelled_yes_second", "words_unchanged",
    ],
)  # fmt: skip
def test_binary_event_level_integerLikeText_followsNumericRule(series, expected):
    # Arrange: series from parametrize

    # Act
    level = binary_event_level(series)

    # Assert
    assert level == expected


@pytest.mark.parametrize(
    ("series", "expected"),
    [
        (pl.Series("censor", [0, 1, 1]), None),
        (pl.Series("CNSR", [0, 1, 1]), None),
        (pl.Series("Censor_flag", [0.0, 1.0]), None),
        (pl.Series("cnsr", ["0", "1"]), None),
        (pl.Series("CNSR 1: Yes 0: No", [0, 1]), None),
        (pl.Series("CNSR 0: event 1: censored", [0, 1]), 0),
        (pl.Series("censored 1: died 2: censored", [1, 2]), 1),
    ],
    ids=[
        "censor", "cnsr_upper", "censor_token_float", "cnsr_text", "cnsr_yes_no_labels",
        "cnsr_states_event", "censored_states_event",
    ],
)  # fmt: skip
def test_binary_event_level_censorColumnName_refusesUnlessHeaderStatesEvent(series, expected):
    # Arrange: series from parametrize; CDISC CNSR=1 means censored

    # Act
    level = binary_event_level(series)

    # Assert
    assert level == expected


@pytest.mark.parametrize(
    ("series", "expected"),
    [
        (pl.Series("n_stage", [0, 1]), 1),
        (pl.Series("no_relapse", [0, 1]), None),
        (pl.Series("No event", [0, 1]), None),
    ],
    ids=["n_token_is_not_no", "no_token_still_refused", "no_word_still_refused"],
)
def test_binary_event_level_nTokenInName_doesNotRefuseFlag(series, expected):
    # Arrange: series from parametrize

    # Act
    level = binary_event_level(series)

    # Assert
    assert level == expected
