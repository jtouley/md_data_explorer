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
        ("", ""),
    ],
)
def test_stripStatistics_sharesAndSmallCounts_removed(text: str, expected: str):
    # Arrange / Act
    result = strip_statistics(text)

    # Assert
    assert result == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            "treatment arm (0 = placebo; 1 = low dose, 2 = high dose)",
            {"0": "placebo", "1": "low dose", "2": "high dose"},
        ),
        ("hemophilia (0=no, 1=yes)", {"0": "no", "1": "yes"}),
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
    ("doc_text", "columns"),
    [(None, COLUMNS), ("", COLUMNS), (DICTIONARY, []), (DICTIONARY, ["Age band (years)"])],
)
def test_readDictionary_nothingMatchesExactly_returnsEmpty(doc_text: str | None, columns: list[str]):
    # Arrange / Act
    docs = read_dictionary(doc_text, columns)

    # Assert
    assert docs == {}


def test_buildContract_dictionaryDocs_inContractButNotChecked(make_contract):
    # Arrange
    df = pl.DataFrame(
        {"patient_id": ["a", "b"], "age_band": [0, 7], "weight": [70.5, 81.0], "arm": [0, 1], "flag": [True, False]}
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
    assert not any(q["metric"] == "invalidValues" for q in props["age_band"]["quality"])  # code 7: no finding
    assert _custom(props["weight"])["unit"] == "kg"
    assert _custom(props["weight"])["unitSource"] == config.DICTIONARY_SOURCE
    assert "enum" not in props["flag"]  # a boolean column takes no codes
    assert report.failed == 0


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
