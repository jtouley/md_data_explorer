"""Data dictionary into the contract: descriptions, units and codebooks, never a count."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest
import yaml
from open_data_contract_standard.model import OpenDataContractStandard

from clinical_analytics.contracts import config
from clinical_analytics.contracts.dictionary import (
    ColumnDoc,
    documented_unit,
    inline_codebook,
    read_dictionary,
    read_glossary,
    strip_statistics,
)
from clinical_analytics.contracts.quality import RANGE_METRIC, compile_contract, run_quality
from clinical_analytics.contracts.service import check_upload, renamed_id_source
from clinical_analytics.contracts.store import contract_path, read_contract
from clinical_analytics.core.metadata_patch import ResolvedColumnMetadata, ResolvedDatasetMetadata

# A synthetic dictionary in the layout of a published one (one entry per line, codes on the lines
# below it, observed shares in parentheses). Every name and number here is made up.
DICTIONARY = """
Data Description

patient_id: A unique identifier for the participant.

age_band: Age in categories. Missing values for this field: 0.00%.

0: if the age is below 40.

1: if the age is 40 or more.

statin: Statin use at baseline. “Yes” (0.35%) if taking a statin and “No” (99.65%) otherwise.

weight: weight (kg) at baseline

arm: treatment arm (0 = placebo; 1 = low dose, 2 = high dose)

smoking_cat: Smoking status. The unique values are “never” (61.2%) and “ever” (38.8%).

“never”: if the person never smoked.

“ever”: if the person smoked at some time.

smoking_ca: Smoking status in an earlier release.

0: not a code of smoking_cat.

Usage Notes

ref: see the paper
"""
COLUMNS = ["patient_id", "age_band", "statin", "weight", "arm", "smoking_cat"]


def _props(contract: OpenDataContractStandard) -> dict[str, dict]:
    doc = yaml.safe_load(contract.to_yaml())
    return {p["physicalName"]: p for p in doc["schema"][0]["properties"]}


def _custom(prop: dict) -> dict:
    return {c["property"]: c["value"] for c in prop.get("customProperties", [])}


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("“Yes” (0.35%) if taking it and “No” (99.65%) otherwise.", "“Yes” if taking it and “No” otherwise."),
        ("Age in categories. Missing values for this field: 0.00%.", "Age in categories."),
        ("Weight at baseline. 4 patients had no scale.", "Weight at baseline."),
        ("Cohort of n = 7 sites. Weight at baseline.", "Weight at baseline."),
        ("1141 people took part.", "1141 people took part."),  # a total, not a small count
        ("“Yes” (<1%) if so.", "“Yes” if so."),  # a share, whatever its comparison
        ("Under 1% of patients had it. Measured once.", "Measured once."),  # "of patients": a share
        ("Yes (n = 4) if so.", "Yes if so."),
        ("", ""),
    ],
)
def test_stripStatistics_sharesAndSmallCounts_removed(text: str, expected: str):
    # Arrange / Act
    result = strip_statistics(text)

    # Assert
    assert result == expected


@pytest.mark.parametrize(
    "text",
    [
        "EF category (0 = EF < 40%, 1 = EF >= 40%)",
        "Glycated haemoglobin; target HbA1c below 7% for most adults. Measured at baseline.",
        "Prediabetes is HbA1c 5.7–6.4%.",
        "Body fat between 25% and 30%.",
        "SpO2 ≥ 94% on room air.",
        "Ejection fraction at most 35%.",
    ],
)
def test_stripStatistics_clinicalThreshold_kept(text: str):
    # Arrange / Act
    result = strip_statistics(text)

    # Assert
    assert result == text


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            "treatment arm (0 = placebo; 1 = low dose, 2 = high dose)",
            {"0": "placebo", "1": "low dose", "2": "high dose"},
        ),
        ("hemophilia (0=no, 1=yes)", {"0": "no", "1": "yes"}),
        ("EF category (0 = EF < 40%, 1 = EF >= 40%)", {"0": "EF < 40%", "1": "EF >= 40%"}),
        ("one code only (1=yes)", {}),
        ("no codes at all", {}),
    ],
)
def test_inlineCodebook_codesInDescription_readWhenTwoOrMore(text: str, expected: dict[str, str]):
    # Arrange / Act
    result = inline_codebook(text)

    # Assert
    assert result == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [("weight (kg) at baseline", "kg"), ("age (yrs)", "yrs"), ("status (yes/no)", None), ("weight", None)],
)
def test_documentedUnit_parenthesisedUnit_onlyKnownUnits(text: str, expected: str | None):
    # Arrange / Act
    result = documented_unit(text)

    # Assert
    assert result == expected


def test_readDictionary_publishedLayout_descriptionsUnitsAndCodebooks():
    # Arrange / Act
    docs = read_dictionary(DICTIONARY, COLUMNS)

    # Assert
    assert set(docs) == set(COLUMNS)
    assert docs["age_band"] == ColumnDoc(
        description="Age in categories.",
        codebook={"0": "if the age is below 40", "1": "if the age is 40 or more"},
    )
    assert docs["weight"].unit == "kg"
    assert docs["arm"].codebook == {"0": "placebo", "1": "low dose", "2": "high dose"}
    assert docs["smoking_cat"].codebook == {
        "never": "if the person never smoked",
        "ever": "if the person smoked at some time",
    }  # the next variable's entry ("smoking_ca") ends the codebook


def test_readDictionary_anyText_neverHoldsAPercentage():
    # Arrange / Act
    docs = read_dictionary(DICTIONARY, COLUMNS)

    # Assert
    texts = [d.description or "" for d in docs.values()] + [v for d in docs.values() for v in d.codebook.values()]
    assert not any("%" in t for t in texts)
    assert docs["statin"].description == "Statin use at baseline. “Yes” if taking a statin and “No” otherwise."


@pytest.mark.parametrize(
    ("doc_text", "expected"),
    [
        # a numbered codebook does not go on with another variable's entry
        (
            "sex: Patient sex\n1: male\n2: female\nethnicity: Self-reported ethnicity\n1: hispanic\n2: other",
            {"1": "male", "2": "female"},
        ),
        # an entry heading numbered codes of its own ends a codebook of words
        (
            "sex: Patient sex\nmale: assigned male\nfemale: assigned female\nrace: Race\n1: white\n2: other",
            {"male": "assigned male", "female": "assigned female"},
        ),
        # an entry heading quoted codes ends it too
        ("sex: Patient sex\n1: male\n2: female\nsmoking: Smoking\n“ever”: smoked", {"1": "male", "2": "female"}),
        # a codebook of capital letters does not go on with a lower-case entry
        (
            "sex: Patient sex\nM: male\nF: female\nsmoking: Smoking status\nN: never\nC: current",
            {"M": "male", "F": "female"},
        ),
        # an entry heading capital-letter codes ends a codebook of words
        (
            "sex: Patient sex\nmale: assigned male\nfemale: assigned female\nsmoking: Smoking\nN: never\nC: current",
            {"male": "assigned male", "female": "assigned female"},
        ),
        # words all through stay one codebook
        (
            "sex: Patient sex\nmale: assigned male\nfemale: assigned female\nunknown: not recorded",
            {"male": "assigned male", "female": "assigned female", "unknown": "not recorded"},
        ),
    ],
)
def test_readDictionary_entryOfAnotherVariable_endsTheCodebook(doc_text: str, expected: dict[str, str]):
    # Arrange / Act
    docs = read_dictionary(doc_text, ["sex", "age"])

    # Assert
    assert docs["sex"].codebook == expected


def test_readGlossary_definitions_statisticsStripped():
    # Arrange
    text = "EDSS: Expanded Disability Status Scale\nICU: Intensive care unit (0.35%)\nRR: (4.2%)"

    # Act
    glossary = read_glossary(text)

    # Assert
    assert glossary == {"EDSS": "Expanded Disability Status Scale", "ICU": "Intensive care unit"}


@pytest.mark.parametrize(
    ("doc_text", "columns"),
    [(None, COLUMNS), ("", COLUMNS), (DICTIONARY, []), (DICTIONARY, ["Age band (years)"])],
)
def test_readDictionary_nothingMatchesExactly_returnsEmpty(doc_text: str | None, columns: list[str]):
    # Arrange / Act
    docs = read_dictionary(doc_text, columns)

    # Assert
    assert docs == {}


def test_buildContract_dictionaryDocs_inContractWithoutAFinding(make_contract):
    # Arrange
    df = pl.DataFrame(
        {"patient_id": ["a", "b"], "age_band": [0, 1], "weight": [70.5, 81.0], "arm": [0, 1], "flag": [True, False]}
    )
    docs = read_dictionary(DICTIONARY, df.columns) | {"flag": ColumnDoc(codebook={"1": "yes", "0": "no"})}

    # Act
    contract = make_contract(df, dictionary=docs)
    report = run_quality(df.lazy(), contract)

    # Assert
    props = _props(contract)
    assert props["age_band"]["description"] == "Age in categories."
    assert props["age_band"]["enum"] == [
        {"value": 0, "label": "if the age is below 40"},
        {"value": 1, "label": "if the age is 40 or more"},
    ]
    assert _custom(props["age_band"])[config.DICTIONARY_CODES_PROPERTY] == [0, 1]
    assert not any(q["metric"] == "invalidValues" for q in props["age_band"]["quality"])
    assert _custom(props["weight"])["unit"] == "kg"
    assert _custom(props["weight"])["unitSource"] == config.DICTIONARY_SOURCE
    assert "enum" not in props["flag"]  # a boolean column takes no codes
    assert report.failed == 0


@pytest.mark.parametrize(
    ("values", "dtype"),
    [
        (["Yes", "No"], pl.String),  # documented "0=no, 1=yes"; the column holds the words
        ([0, 7], pl.Int64),  # 7 is not documented
        ([None, None], pl.Int64),  # nothing to check the codes against
    ],
)
def test_buildContract_dictionaryCodesTheDataDoesNotUse_leftOut(make_contract, values: list, dtype: pl.DataType):
    # Arrange
    df = pl.DataFrame({"patient_id": ["a", "b"], "smoker": pl.Series(values, dtype=dtype)})
    docs = {"smoker": ColumnDoc(description="Ever smoked at baseline", codebook={"0": "no", "1": "yes"})}

    # Act
    contract = make_contract(df, dictionary=docs)

    # Assert
    smoker = _props(contract)["smoker"]
    assert "enum" not in smoker
    assert config.DICTIONARY_CODES_PROPERTY not in _custom(smoker)
    assert smoker["description"] == "Ever smoked at baseline"  # the description still holds


@pytest.mark.parametrize("lowest", [0, 1])
def test_buildContract_columnCodedOnlyByDictionary_noRangeCheck(make_contract, lowest: int):
    # Arrange: "glucose" has a plausibility range, but this column holds category codes
    df = pl.DataFrame({"patient_id": [f"p{i}" for i in range(20)], "glucose": [lowest, lowest + 1] * 10})
    docs = {"glucose": ColumnDoc(codebook={str(lowest): "normal", str(lowest + 1): "high"})}

    # Act
    contract = make_contract(df, dictionary=docs)
    report = run_quality(df.lazy(), contract)

    # Assert
    glucose = _props(contract)["glucose"]
    assert [e["value"] for e in glucose["enum"]] == [lowest, lowest + 1]
    assert "logicalTypeOptions" not in glucose
    assert "assumedUnit" not in _custom(glucose)
    assert not any(c.metric == RANGE_METRIC for c in compile_contract(contract, df.schema))
    assert report.failed == 0


def test_buildContract_headerCodeWrittenAsFloatInDictionary_oneEnumEntry(make_contract):
    # Arrange: the header codes 1 and 2; the dictionary writes them "1.0" and "2.0", and adds 3
    column = "Smoker 1: never 2: former"
    df = pl.DataFrame({"patient_id": ["a", "b", "c"], column: [1.0, 2.0, 3.0]})
    docs = {column: ColumnDoc(codebook={"1.0": "yes", "2.0": "no", "3": "current"})}

    # Act
    contract = make_contract(df, dictionary=docs)

    # Assert
    prop = _props(contract)[column]
    assert prop["enum"] == [
        {"value": 1, "label": "never"},
        {"value": 2, "label": "former"},
        {"value": 3, "label": "current"},
    ]
    assert _custom(prop)[config.DICTIONARY_CODES_PROPERTY] == [3]
    check = next(q for q in prop["quality"] if q["metric"] == "invalidValues")
    assert check["arguments"]["validValues"] == [1, 2, 3]


def test_buildContract_glossaryAndDictionaryRead_onTheContract(make_contract):
    # Arrange
    df = pl.DataFrame({"patient_id": ["a", "b"]})

    # Act
    contract = make_contract(df, glossary={"EDSS": "Expanded Disability Status Scale"})

    # Assert
    custom = {c.property: c.value for c in contract.customProperties or []}
    assert custom[config.GLOSSARY_PROPERTY] == {"EDSS": "Expanded Disability Status Scale"}
    assert custom[config.DICTIONARY_READ_PROPERTY] is True


def test_buildContract_acceptedPatchAndDictionary_patchWins(make_contract):
    # Arrange
    df = pl.DataFrame({"patient_id": ["a", "b"], "weight": [70.5, 81.0]})
    resolved = ResolvedDatasetMetadata(
        columns={"weight": ResolvedColumnMetadata(name="weight", description="Body weight", unit="lb")}
    )

    # Act
    contract = make_contract(df, resolved=resolved, dictionary=read_dictionary(DICTIONARY, df.columns))

    # Assert
    weight = _props(contract)["weight"]
    assert weight["description"] == "Body weight"
    assert _custom(weight)["unit"] == "lb"
    assert "unitSource" not in _custom(weight)


def test_buildContract_documentedUnitAndRangeCheck_detailKeepsTheBoundsUnit(make_contract):
    # Arrange: the dictionary names a unit; the age bound has its own (years)
    df = pl.DataFrame({"patient_id": ["a", "b"], "age": [40, 150]})

    # Act
    contract = make_contract(df, dictionary={"age": ColumnDoc(unit="yrs")})
    checks = compile_contract(contract, df.schema)

    # Assert
    detail = next(c.detail for c in checks if c.metric == RANGE_METRIC)
    assert detail == "0–122 years"


def test_checkUpload_docContext_contractCarriesDictionary(tmp_path: Path):
    # Arrange: the upload renamed its identifier "subject" to patient_id
    df = pl.DataFrame({"patient_id": [f"s{i}" for i in range(20)], "age_band": [0, 1] * 10})
    metadata = {
        "dataset_name": "cohort",
        "inferred_schema": {"column_mapping": {"patient_id": "patient_id"}},
        "synthetic_id_metadata": {
            "patient_id": {"patient_id_source": "single_column", "patient_id_columns": ["subject"]}
        },
        "doc_context": "subject: The participant's study number.\n"
        + DICTIONARY.replace("patient_id: A unique identifier for the participant.", ""),
    }

    # Act
    check_upload(tmp_path, df, upload_id="u1", dataset_version="v1", metadata=metadata)

    # Assert
    props = _props(read_contract(contract_path(tmp_path, "u1", "v1")))
    assert props["patient_id"]["description"] == "The participant's study number."
    assert props["age_band"]["description"] == "Age in categories."


@pytest.mark.parametrize(
    ("metadata", "expected"),
    [
        (
            {
                "synthetic_id_metadata": {
                    "patient_id": {"patient_id_source": "single_column", "patient_id_columns": ["x"]}
                }
            },
            "x",
        ),
        (
            {
                "synthetic_id_metadata": {
                    "patient_id": {"patient_id_source": "composite", "patient_id_columns": ["a", "b"]}
                }
            },
            None,
        ),
        ({"synthetic_id_metadata": {"patient_id": None}}, None),
        ({"synthetic_id_metadata": None}, None),
        ({}, None),
    ],
)
def test_renamedIdSource_uploadMetadata_singleRenamedColumnOnly(metadata: dict, expected: str | None):
    # Arrange / Act
    result = renamed_id_source(metadata)

    # Assert
    assert result == expected
