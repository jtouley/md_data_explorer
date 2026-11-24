# SPDX-License-Identifier: MIT
"""Edges of the data dictionary reader (``contracts.dictionary``), found by mutation testing.

Each test pins a comparison, a boundary or a rule that changes which codes are missing values,
which are labels, which range or unit a column has, or which column the dictionary names as the
patient key. Invented dictionaries only: a wound clinic, a dental chart, a physiotherapy gym, a
lactation clinic, a sleep diary, a hand-therapy splint room.
"""

from __future__ import annotations

import pytest

from clinical_analytics.contracts.dictionary import (
    free_missing_pairs,
    parse_dictionary,
    parse_pairs,
    split_missing,
)


# --------------------------------------------------------------------------- split_missing
@pytest.mark.parametrize(
    ("pairs", "stated", "expected"),
    [
        pytest.param([("NA", "not recorded")], None, ({}, {}), id="code_not_a_number"),
        pytest.param([("0", "not recorded")], (0.0, 40.0), ({}, {}), id="code_at_low_edge"),
        pytest.param([("20", "not recorded")], (0.0, 40.0), ({}, {}), id="code_inside_range"),
        pytest.param([("40", "not recorded")], (0.0, 40.0), ({}, {}), id="code_at_high_edge"),
        pytest.param([("41", "not recorded")], (0.0, 40.0), ({}, {"41": "not recorded"}), id="code_above_range"),
        pytest.param([("12", "not recorded")], None, ({}, {}), id="no_range_not_a_usual_code"),
        pytest.param([("999", "severe slough")], None, ({}, {}), id="usual_code_label_not_missing"),
        pytest.param([("999", "not recorded")], (0.0, 1000.0), ({}, {}), id="usual_code_inside_stated_range"),
        pytest.param(
            [("999", "not recorded"), ("888", "not recorded")],
            (0.0, 1000.0),
            ({"999": "not recorded", "888": "not recorded"}, {}),
            id="two_notes_inside_stated_range_kept_as_list",
        ),
    ],
)
def test_splitMissing_codeAgainstStatedRange_missingOnlyOutsideIt(
    pairs: list[tuple[str, str]], stated: tuple[float, float] | None, expected: tuple[dict[str, str], dict[str, str]]
) -> None:
    # Act
    result = split_missing(pairs, stated)

    # Assert
    assert result == expected


# --------------------------------------------------------------------------- parse_pairs
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        pytest.param(" 1 = yes", [("1", "yes")], id="leading_space_is_a_line_start"),
        pytest.param("A 1 = yes", [], id="lone_number_after_a_word"),
        pytest.param("X 1 = yes", [], id="lone_number_after_a_capital"),
        pytest.param("See note...1 = lowest", [], id="code_glued_to_a_full_stop"),
        pytest.param("Y = yes 2: other", [("Y", "yes"), ("2", "other")], id="number_continues_a_letter_list"),
        pytest.param("1: Yes 2: No", [("1", "Yes"), ("2", "No")], id="colon_list"),
        pytest.param("Smoking 0=never 1=former", [("0", "never"), ("1", "former")], id="two_numbers_after_a_word"),
        pytest.param("(1 = left, 2 = right) and more", [("1", "left"), ("2", "right")], id="list_ends_at_its_paren"),
        pytest.param("blank = not taken, 1 = done, 2 = not done", [("1", "done"), ("2", "not done")], id="blank"),
        pytest.param("Blank = not taken, 1 = done, 2 = not done", [("1", "done"), ("2", "not done")], id="Blank"),
        pytest.param("1 = , 2 = no, 3 = yes", [("2", "no"), ("3", "yes")], id="empty_label_skipped_not_the_rest"),
    ],
)
def test_parsePairs_listStartsAndEnds_codesAndLabels(text: str, expected: list[tuple[str, str]]) -> None:
    # Act / Assert
    assert parse_pairs(text) == expected


def test_freeMissingPairs_twoCodesInOneSentence_bothFoundOnce() -> None:
    # Act
    found = free_missing_pairs("999 indicates not recorded; 888 indicates not recorded; 999 means not recorded")

    # Assert
    assert found == [("999", "not recorded"), ("888", "not recorded")]


# --------------------------------------------------------------------------- an entry's own facts
def test_parseDictionary_freeNoteInsideEntry_notedNotApplied() -> None:
    # Arrange
    text = "depth  Wound depth, mm. A value of 999 indicates the reading was not taken.\n"

    # Act
    entry = parse_dictionary(text, ["depth"]).entry("depth")

    # Assert
    assert entry is not None and set(entry.noted_missing) == {"999"} and entry.missing == {}


def test_parseDictionary_noteRepeatsAListedCode_notNotedTwice() -> None:
    # Arrange
    text = "depth  Wound depth. 999 = not recorded, 888 = not recorded. 999 indicates not recorded.\n"

    # Act
    entry = parse_dictionary(text, ["depth"]).entry("depth")

    # Assert
    assert entry is not None and entry.noted_missing == {} and set(entry.missing) == {"999", "888"}


@pytest.mark.parametrize(
    ("line", "description"),
    [
        pytest.param("smk  Smoker. Y = yes, N = no", "Smoker.", id="capital_codes"),
        pytest.param("smk  Smoker. y = yes, n = no", "Smoker.", id="small_codes"),
    ],
)
def test_parseDictionary_codeList_descriptionEndsBeforeIt(line: str, description: str) -> None:
    # Act
    entry = parse_dictionary(line + "\n", ["smk"]).entry("smk")

    # Assert
    assert entry is not None and entry.description == description


def test_parseDictionary_dateColumn_noRangeNoUnit() -> None:
    # Arrange: a date written with a format and years that read like a range
    text = "fitted  Clinic date (2020-2022), YYYY-MM-DD, in days\n"

    # Act
    entry = parse_dictionary(text, ["fitted"]).entry("fitted")

    # Assert
    assert entry is not None and entry.date_format is not None
    assert (entry.stated_range, entry.unit) == (None, None)


def test_parseDictionary_noteInsideOwnRange_neverMissing() -> None:
    # Act
    entry = parse_dictionary("grip  Grip (0-1000). 999 = sheet not returned\n", ["grip"]).entry("grip")

    # Assert
    assert entry is not None and entry.missing == {} and entry.stated_range == (0.0, 1000.0)


def test_parseDictionary_equalBounds_noStatedRange() -> None:
    # Act
    entry = parse_dictionary("teeth  Teeth charted (5-5)\n", ["teeth"]).entry("teeth")

    # Assert
    assert entry is not None and entry.stated_range is None


def test_parseDictionary_tabIndentedContinuation_partOfTheEntry() -> None:
    # Arrange
    text = "depth  Wound depth, mm\n\t999 = not recorded, 998 = not recorded\n"

    # Act
    dictionary = parse_dictionary(text, ["depth"])

    # Assert
    assert set(dictionary.entry("depth").missing) == {"999", "998"}
    assert dictionary.shared_missing == {}


@pytest.mark.parametrize(
    "text",
    [
        pytest.param("  an indented line before any entry\ndepth  Wound depth\n", id="indented_first_line"),
        pytest.param("General notes\n  an indented aside\ndepth  Wound depth\n", id="indented_after_prose"),
    ],
)
def test_parseDictionary_indentedLineOutsideAnEntry_readAsProse(text: str) -> None:
    # Act
    dictionary = parse_dictionary(text, ["depth"])

    # Assert
    assert dictionary.entry("depth").description == "Wound depth"


def test_parseDictionary_columnNamesWithSpaces_longestNameFirst() -> None:
    # Arrange
    columns = ["Splint Type", "Splint Type Detail", "Xfer", "Splint"]
    text = (
        "Splint Type Detail: Material and padding\n"
        "Splint Type\tThe kind of splint. 1 = resting, 2 = dynamic\n"
        "Splint Type\n"
        "Xfer  Transferred from another clinic\n"
    )

    # Act
    dictionary = parse_dictionary(text, columns)

    # Assert
    assert dictionary.entry("Splint Type Detail").description == "Material and padding"
    assert dictionary.entry("Splint Type").labels == {"1": "resting", "2": "dynamic"}
    assert dictionary.entry("Xfer").description == "Transferred from another clinic"
    assert dictionary.entry("Splint") is None


# --------------------------------------------------------------------------- prose
def test_parseDictionary_proseNoteThatIsNoMissingLabel_neverShared() -> None:
    # Act
    dictionary = parse_dictionary("Coded: 999 = severe reaction.\n", ["wheal"])

    # Assert
    assert dictionary.shared_missing == {}


def test_parseDictionary_noteInSecondSentence_shared() -> None:
    # Act
    dictionary = parse_dictionary("Scores are recorded daily. 999 = not recorded.\n", ["pain"])

    # Assert
    assert set(dictionary.shared_missing) == {"999"}


def test_parseDictionary_sentenceNamesAThreeLetterColumn_notedForIt() -> None:
    # Act
    dictionary = parse_dictionary("In sbp, 999 = not recorded.\n", ["sbp", "dbp_seated"])

    # Assert
    assert set(dictionary.noted_missing) == {"sbp"} and dictionary.shared_missing == {}


def test_parseDictionary_keyStatement_columnNamedOnceWhateverItsCase() -> None:
    # Arrange
    text = "In this file kin_ref identifies the patient. The patient identifier is kin_ref.\n"

    # Act
    dictionary = parse_dictionary(text, ["Kin_Ref", "visit"])

    # Assert
    assert dictionary.key_statements == ("Kin_Ref",)


def test_parseDictionary_keyStatementForNoColumn_nothing() -> None:
    # Act / Assert
    assert parse_dictionary("Ward identifies the patient.\n", ["kin_ref"]).key_statements == ()


@pytest.mark.parametrize(
    ("text", "one_per_person", "repeats"),
    [
        pytest.param("One row per patient.\n", True, False, id="one_per_patient"),
        pytest.param("Mothers appear more than once.\n", False, True, id="repeats"),
        pytest.param("Lactation clinic feeds.\n", False, False, id="neither"),
    ],
)
def test_parseDictionary_proseAboutRows_flags(text: str, one_per_person: bool, repeats: bool) -> None:
    # Act
    dictionary = parse_dictionary(text, ["feed"])

    # Assert
    assert (dictionary.one_per_person, dictionary.repeats_stated) == (one_per_person, repeats)


def test_parseDictionary_grainStatement_quotedWhole() -> None:
    # Act
    dictionary = parse_dictionary("One row per gym session.\n", ["reps"])

    # Assert
    assert (dictionary.grain_noun, dictionary.grain_text) == ("gym session", "One row per gym session")


# --------------------------------------------------------------------------- units
@pytest.mark.parametrize(
    ("description", "unit"),
    [
        pytest.param("Urea, mmol/L", "mmol/L", id="ratio"),
        pytest.param("Reading per area, AU/m²", "AU/m²", id="squared_atom"),
        pytest.param("Dose, µg/dose", "µg/dose", id="micro_sign"),
        pytest.param("Dose, μg/dose", "μg/dose", id="greek_mu"),
        pytest.param("Platelets per bag, x10^11/bag", "x10^11/bag", id="power_of_ten"),
        pytest.param("Answer Yes/No; mmol/L", "mmol/L", id="label_pair_then_unit"),
        pytest.param("Activity, U", "U", id="single_capital"),
        pytest.param("Volume, mL", "mL", id="mixed_case"),
    ],
)
def test_parseDictionary_unitWritten_readAsWritten(description: str, unit: str) -> None:
    # Act
    entry = parse_dictionary(f"lab  {description}\n", ["lab"]).entry("lab")

    # Assert
    assert entry is not None and entry.unit == unit


# --------------------------------------------------------------------------- codebook tables
def _codebook(label: str, *rows: str) -> str:
    return "\n".join(["Variable Name: lab", f"SAS Label: {label}", "Code or Value\tValue Description", *rows]) + "\n"


def test_codebook_rangeWithExplicitAndOtherCodes_onlyEdgeCodesCensored() -> None:
    # Arrange
    text = _codebook(
        "Hours of sleep", "1 to 64\tRange of Values", "0\t0 hours or less", "77\tRefused", "55\tWrong form"
    )

    # Act
    entry = parse_dictionary(text, ["lab"]).entry("lab")

    # Assert: the edge code below the range is censored; Refused is missing; the rest neither
    assert entry is not None and entry.censored == {"0": "0 hours or less"}
    assert entry.missing == {"77": "Refused"}


@pytest.mark.parametrize(
    ("rows", "labels"),
    [
        pytest.param(("A\tLeft", "B\tRight"), {"A": "Left", "B": "Right"}, id="capital_letters"),
        pytest.param(("a\tLeft", "b\tRight"), {"a": "Left", "b": "Right"}, id="small_letters"),
        pytest.param(("1\tYes",), {}, id="one_code_is_no_list"),
    ],
)
def test_codebook_codedItem_labelsOnlyForAList(rows: tuple[str, ...], labels: dict[str, str]) -> None:
    # Act
    entry = parse_dictionary(_codebook("Side splinted", *rows), ["lab"]).entry("lab")

    # Assert
    assert entry is not None and entry.labels == labels


@pytest.mark.parametrize(
    ("label", "rows", "unit"),
    [
        pytest.param("Minutes waited, minutes", (".\tMissing",), "minutes", id="no_codes_unit_read"),
        pytest.param("Dose band, mg", ("1\tLow", "2\tHigh"), None, id="coded_never_a_unit"),
    ],
)
def test_codebook_unit_onlyOnAMeasure(label: str, rows: tuple[str, ...], unit: str | None) -> None:
    # Act
    entry = parse_dictionary(_codebook(label, *rows), ["lab"]).entry("lab")

    # Assert
    assert entry is not None and entry.unit == unit


def test_codebook_codeBelowTheLowEdge_censored() -> None:
    # Arrange: "0 = 0 visits or fewer" beside 1 to 30 names the edge just below the range
    text = _codebook("Speech therapy visits", "1 to 30\tRange of Values", "0\t0 visits or fewer", "99\tDon't know")

    # Act
    entry = parse_dictionary(text, ["lab"]).entry("lab")

    # Assert
    assert entry is not None and entry.censored == {"0": "0 visits or fewer"}
    assert (entry.column, entry.text, entry.stated_range) == ("lab", "Speech therapy visits", (1.0, 30.0))


def test_codebook_variableWithoutLabel_tableStillRead() -> None:
    # Arrange: a header with no label line, straight into its table
    text = "Variable Name: lab\nCode or Value\tValue Description\n1\tLeft\n2\tRight\n"

    # Act
    entry = parse_dictionary(text, ["lab"]).entry("lab")

    # Assert
    assert entry is not None and (entry.column, entry.text) == ("lab", "")
    assert entry.labels == {"1": "Left", "2": "Right"}


def test_codebook_labelLineAfterATable_startsTheNextVariable() -> None:
    # Arrange: "Label:" before its variable line, after the previous variable's table
    text = (
        "Variable Name: lab\nSAS Label: Side fitted\nCode or Value\tValue Description\n1\tLeft\n2\tRight\n"
        "SAS Label: Speech sessions attended, in weeks\nVariable Name: wks\n"
    )

    # Act
    dictionary = parse_dictionary(text, ["lab", "wks"])

    # Assert
    assert dictionary.entry("lab").labels == {"1": "Left", "2": "Right"}
    assert dictionary.entry("wks").description == "Speech sessions attended, in weeks"


def test_codebook_headerWithNoLabelNoTable_noEntry() -> None:
    # Act / Assert
    assert parse_dictionary("Variable Name: lab\n", ["lab"]).entry("lab") is None


def test_codebook_codedItemAfterAnEntryLine_codebookLabelsWin() -> None:
    # Arrange: an entry line and a codebook table for the same column
    text = "lab  Side fitted\n\n" + _codebook("Side fitted", "1\tLeft", "2\tRight")

    # Act
    entry = parse_dictionary(text, ["lab"]).entry("lab")

    # Assert
    assert entry is not None and entry.labels == {"1": "Left", "2": "Right"}


@pytest.mark.parametrize(
    ("label", "rows", "direction"),
    [
        pytest.param("Stammer score, higher is worse", ("0 to 40\tRange of Values",), "lower_is_better", id="measure"),
        pytest.param("Fluency band, higher is better", ("1\tLow", "2\tHigh"), "higher_is_better", id="coded"),
    ],
)
def test_codebook_directionStated_readOnEitherKind(label: str, rows: tuple[str, ...], direction: str) -> None:
    # Act
    entry = parse_dictionary(_codebook(label, *rows), ["lab"]).entry("lab")

    # Assert
    assert entry is not None and entry.direction == direction


@pytest.mark.parametrize(
    ("line", "direction"),
    [
        pytest.param("pain  Pain score, lower is better", "lower_is_better", id="lower_better"),
        pytest.param("pain  Pain score, higher scores indicate worse", "lower_is_better", id="higher_worse"),
        pytest.param("pain  Grip score, higher is better", "higher_is_better", id="higher_better"),
        pytest.param("pain  Pain score, lower values are worse", "higher_is_better", id="lower_worse"),
    ],
)
def test_parseDictionary_directionStated_readAsStated(line: str, direction: str) -> None:
    # Act
    entry = parse_dictionary(line + "\n", ["pain"]).entry("pain")

    # Assert
    assert entry is not None and entry.direction == direction


# --------------------------------------------------------------------------- nesting and trimming
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        pytest.param(
            "(1 = arm (sling (left)) only, 2 = none)", [("1", "arm (sling (left)) only"), ("2", "none")], id="nested"
        ),
        pytest.param("1 = sling) or cast, 2 = none", [("1", "sling) or cast"), ("2", "none")], id="stray_close"),
        pytest.param("1 = splint type X.), 2 = none", [("1", "splint type X"), ("2", "none")], id="trailing_close"),
        pytest.param("1 = grade X, 2 = none", [("1", "grade X"), ("2", "none")], id="label_ends_in_capital_x"),
    ],
)
def test_parsePairs_parentheses_labelKeptWhole(text: str, expected: list[tuple[str, str]]) -> None:
    # Act / Assert
    assert parse_pairs(text) == expected


def test_parseDictionary_rangeAroundAParenthesis_read() -> None:
    # Act
    entry = parse_dictionary("grip  Grip strength 0 (kg) to 50\n", ["grip"]).entry("grip")

    # Assert
    assert entry is not None and entry.stated_range == (0.0, 50.0)


# --------------------------------------------------------------------------- Markdown tables
def test_markdownTable_rowsAsEntries_cellsJoinedAndLineBreaksKept() -> None:
    # Arrange: an orthotics workshop's table, code in backticks, a <BR> inside a cell
    text = (
        "| Column | Description | Notes |\n"
        "|---|---|---|\n"
        "| `depth` | Insole depth, mm<BR>999 = not recorded, 998 = not recorded | |\n"
        "| fit | Fit check | weekly |\n"
        "|Xray|Foot film taken, 1 = yes, 2 = no|\n"
    )

    # Act
    dictionary = parse_dictionary(text, ["depth", "fit", "Xray"])

    # Assert
    depth = dictionary.entry("depth")
    assert depth is not None and depth.description == "Insole depth, mm"
    assert set(depth.missing) == {"999", "998"}
    assert dictionary.entry("fit").text == "Fit check; weekly"
    assert dictionary.entry("Xray").labels == {"1": "yes", "2": "no"}


def test_markdownTable_rowWithEmptyFirstCell_notAContinuation() -> None:
    # Arrange
    text = "| depth | Insole depth |\n|  | 999 = not recorded, 998 = not recorded |\n"

    # Act
    entry = parse_dictionary(text, ["depth"]).entry("depth")

    # Assert
    assert entry is not None and entry.missing == {}


def test_markdownTable_rowWithEmptySecondCell_stillAnEntry() -> None:
    # Act
    entry = parse_dictionary("| depth |  | Insole depth |\n", ["depth"]).entry("depth")

    # Assert
    assert entry is not None and entry.description == "Insole depth"


def test_parseDictionary_pipesInsideAnEntryLine_notATable() -> None:
    # Act
    entry = parse_dictionary("depth  Insole depth | mm | 0-40 | daily\n", ["depth"]).entry("depth")

    # Assert
    assert entry is not None and entry.stated_range == (0.0, 40.0)


# --------------------------------------------------------------------------- grain nouns
@pytest.mark.parametrize(
    ("text", "noun"),
    [
        pytest.param("One row per left hand splint fitting.\n", "left hand splint fitting", id="four_words"),
        pytest.param("One row per treated wound.\n", "treated wound", id="participle_first_word"),
        pytest.param("One row per X-ray session -\n", "X-ray session", id="trailing_dash"),
    ],
)
def test_parseDictionary_grainNoun_wholePhrase(text: str, noun: str) -> None:
    # Act / Assert
    assert parse_dictionary(text, ["reps"]).grain_noun == noun
