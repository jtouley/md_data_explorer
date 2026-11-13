"""ODCS contract builder: schema conformance, codebooks, plausibility bounds, no observed values."""

from __future__ import annotations

import importlib.util
import json
from datetime import UTC, datetime
from pathlib import Path

import jsonschema
import polars as pl
import pytest
import yaml
from open_data_contract_standard.model import OpenDataContractStandard

from clinical_analytics.contracts import config
from clinical_analytics.contracts.builder import (
    find_plausibility_range,
    logical_type,
    unique_names,
)
from clinical_analytics.core.metadata_patch import ResolvedColumnMetadata, ResolvedDatasetMetadata, SemanticType
from tests.contracts.conftest import DEXA, STATIN

_ROOT = Path(__file__).resolve().parents[2]
_SYNTHETIC_CONTRACT = _ROOT / "tests" / "fixtures" / "synthetic" / "contracts" / "cohort.odcs.yaml"
_spec = importlib.util.spec_from_file_location(
    "make_synthetic_contract", _ROOT / "scripts" / "make_synthetic_contract.py"
)
make_synthetic_contract = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(make_synthetic_contract)


def _props(contract: OpenDataContractStandard) -> dict[str, dict]:
    doc = yaml.safe_load(contract.to_yaml())
    return {p["physicalName"]: p for p in doc["schema"][0]["properties"]}


def test_syntheticContract_committedFile_matchesBuilderOutput():
    # Arrange
    committed = _SYNTHETIC_CONTRACT.read_text(encoding="utf-8")

    # Act
    regenerated = make_synthetic_contract.synthetic_contract_yaml()

    # Assert: run scripts/make_synthetic_contract.py if the builder changed on purpose.
    assert committed == regenerated


def _dataset_version(contract_yaml: str) -> str:
    props = yaml.safe_load(contract_yaml)["customProperties"]
    return next(p["value"] for p in props if p["property"] == "datasetVersion")


def test_syntheticContract_polarsWritesDifferentMetadata_datasetVersionUnchanged(monkeypatch):
    # Arrange: simulate a Polars release that writes different file metadata and loads the
    # cohort with another physical layout. The old Parquet-byte hash moved on exactly this.
    committed_version = _dataset_version(_SYNTHETIC_CONTRACT.read_text(encoding="utf-8"))
    real_write_parquet = pl.DataFrame.write_parquet
    real_read_csv = pl.read_csv

    def write_parquet_other_release(self, file, **kwargs):
        return real_write_parquet(self, file, **{**kwargs, "metadata": {"created_by": "Polars (python) 99.0.0"}})

    def read_csv_other_layout(source, **kwargs):
        frame = real_read_csv(source, **kwargs)
        return pl.concat([frame.head(1), frame.tail(frame.height - 1)], rechunk=False).with_columns(
            pl.col("site").cast(pl.Categorical)
        )

    monkeypatch.setattr(pl.DataFrame, "write_parquet", write_parquet_other_release)
    monkeypatch.setattr(pl, "read_csv", read_csv_other_layout)

    # Act
    regenerated_version = _dataset_version(make_synthetic_contract.synthetic_contract_yaml())

    # Assert
    assert regenerated_version == committed_version


def test_syntheticContract_bundledOdcsJsonSchema_validates():
    # Arrange
    schema = json.loads(OpenDataContractStandard.json_schema())
    document = yaml.safe_load(_SYNTHETIC_CONTRACT.read_text(encoding="utf-8"))

    # Act / Assert (raises on any violation)
    jsonschema.validate(document, schema)
    assert "synthetic" in document["tags"]


def test_syntheticContract_datacontractCliLint_passes():
    # Arrange: datacontract-cli is a dev (CI-only) dependency; lint is offline (bundled schema).
    data_contract = pytest.importorskip("datacontract.data_contract")

    # Act
    run = data_contract.DataContract(data_contract_file=str(_SYNTHETIC_CONTRACT)).lint()

    # Assert
    assert run.has_passed(), [c.reason for c in run.checks]


@pytest.mark.parametrize(
    ("dtype", "expected"),
    [
        (pl.Boolean(), "boolean"),
        (pl.Int8(), "integer"),
        (pl.UInt64(), "integer"),
        (pl.Float32(), "number"),
        (pl.Decimal(10, 2), "number"),
        (pl.Date(), "date"),
        (pl.Datetime("us"), "timestamp"),
        (pl.Time(), "time"),
        (pl.List(pl.Int64), "array"),
        (pl.Struct({"a": pl.Int64}), "object"),
        (pl.String(), "string"),
        (pl.Null(), "string"),
    ],
    ids=lambda v: str(v),
)
def test_logicalType_polarsDtype_mapsToOdcsEnum(dtype, expected):
    # Act / Assert
    assert logical_type(dtype) == expected


@pytest.mark.parametrize(
    ("columns", "expected"),
    [
        (["Age", "age"], ["age", "age_2"]),
        ([DEXA], ["dexa_result"]),
        (["???", "!!"], ["column_1", "column_2"]),
        (["HbA1c (%)"], ["hba1c"]),
        ([], []),
    ],
    ids=["collision", "coded_header", "no_ascii", "punctuation", "empty"],
)
def test_uniqueNames_physicalColumns_stableSlugs(columns, expected):
    # Act / Assert
    assert list(unique_names(columns).values()) == expected


@pytest.mark.parametrize(
    ("column", "expected"),
    [("age", "age"), ("Age (years)", "age"), ("AGE_YEARS", "age"), ("BMI", "bmi"), ("ldl", None), ("page", None)],
)
def test_findPlausibilityRange_columnName_matchesByNormalizedShortName(column, expected):
    # Act
    found = find_plausibility_range(column)

    # Assert
    assert (found.variable if found else None) == expected


def test_buildContract_codedHeader_enumAndInvalidRuleFromHeaderOnly(make_contract):
    # Arrange: values include 9 (undocumented); the contract must not learn it from the data.
    df = pl.DataFrame({"patient_id": ["A", "B", "C"], DEXA: [1, 3, 9]})

    # Act
    prop = _props(make_contract(df))[DEXA]

    # Assert
    assert [e["value"] for e in prop["enum"]] == [1, 2, 3]
    assert [e["label"] for e in prop["enum"]] == ["Normal", "Osteopenia", "Osteoporosis"]
    rule = next(q for q in prop["quality"] if q["metric"] == "invalidValues")
    assert rule["arguments"]["validValues"] == [1, 2, 3]
    assert prop["businessName"] == "DEXA result"


def test_buildContract_textColumn_observedLevelsNotWritten(make_contract):
    # Arrange
    df = pl.DataFrame({"patient_id": ["A", "B"], "diagnosis": ["zebrafinchitis", "flu"]})

    # Act
    text = make_contract(df).to_yaml()

    # Assert
    assert "zebrafinchitis" not in text
    assert "enum" not in _props(make_contract(df))["diagnosis"]


def test_buildContract_idColumn_requiredUniqueRestrictedWithoutMissingRule(make_contract):
    # Arrange
    df = pl.DataFrame({"patient_id": ["A", "B"], "age": [40, 50]})

    # Act
    prop = _props(make_contract(df))["patient_id"]

    # Assert
    assert (prop["required"], prop["unique"], prop["primaryKey"], prop["classification"]) == (
        True,
        True,
        True,
        "restricted",
    )
    assert [q["id"] for q in prop["quality"]] == ["patient_id_not_null", "patient_id_no_duplicates"]


@pytest.mark.parametrize(
    ("values", "has_range"),
    [([40, 50], True), ([40.5, 50.0], True), (["40", "50"], False)],
    ids=["int", "float", "text"],
)
def test_buildContract_ageColumn_rangeOnlyWhenNumeric(make_contract, values, has_range):
    # Arrange
    df = pl.DataFrame({"patient_id": ["A", "B"], "age": values})

    # Act
    prop = _props(make_contract(df))["age"]

    # Assert
    assert ("logicalTypeOptions" in prop) is has_range
    custom = {c["property"]: c["value"] for c in prop.get("customProperties", [])}
    if has_range:
        assert prop["logicalTypeOptions"] == {"minimum": 0, "maximum": 122}
        assert config.PLAUSIBILITY_RANGE_PROPERTY not in custom
    else:
        # ODCS allows no minimum/maximum on a string; the bound is kept so the check reports not_run.
        assert custom[config.PLAUSIBILITY_RANGE_PROPERTY] == {"minimum": 0, "maximum": 122}


def test_buildContract_resolvedMetadata_patchesFoldedIn(make_contract):
    # Arrange
    df = pl.DataFrame({"patient_id": ["A", "B"], "smk": [0, 1], STATIN: [1, 2]})
    resolved = ResolvedDatasetMetadata(
        columns={
            "smk": ResolvedColumnMetadata(
                name="smk",
                label="Current smoker",
                description="Smoking at baseline",
                unit=None,
                aliases=["smoker"],
                codebook={"0": "No", "1": "Yes", "n/a": "unknown"},
                semantic_type=SemanticType.CLINICAL,
            ),
            STATIN: ResolvedColumnMetadata(name=STATIN, codebook={"9": "Unknown"}, is_phi=True, unit="flag"),
        }
    )

    # Act
    contract = make_contract(df, resolved=resolved, accepted_patch_ids=("p1", "p2"))
    props = _props(contract)

    # Assert
    smk = props["smk"]
    assert (smk["businessName"], smk["description"]) == ("Current smoker", "Smoking at baseline")
    assert smk["synonyms"] == [{"synonym": "smoker"}]
    assert [e["value"] for e in smk["enum"]] == [0, 1]  # 'n/a' cannot be an Int64 code
    assert {"property": "clinicalSemanticType", "value": "clinical"} in smk["customProperties"]
    statin = props[STATIN]
    assert [e["value"] for e in statin["enum"]] == [1, 2, 9]  # header codes plus patched code
    assert statin["classification"] == "restricted"
    assert {"property": "unit", "value": "flag"} in statin["customProperties"]
    custom = {c.property: c.value for c in contract.customProperties}
    assert custom["metadataPatchIds"] == ["p1", "p2"]


def test_buildContract_unknownIdColumn_raisesValueError(make_contract):
    # Arrange
    df = pl.DataFrame({"a": [1]})

    # Act / Assert
    with pytest.raises(ValueError, match="Identifier column 'patient_id' not found"):
        make_contract(df)


def test_buildContract_noIdColumnAndNaiveTimestamp_tableRulesOnlyAndUtc(make_contract):
    # Arrange
    df = pl.DataFrame({"x": [None, None]}, schema={"x": pl.Null})

    # Act
    contract = make_contract(df, id_column=None, created_at=datetime(2026, 1, 2, 3, 4, tzinfo=UTC))

    # Assert
    assert contract.contractCreatedTs == "2026-01-02T03:04:00Z"
    prop = _props(contract)["x"]
    assert "customProperties" not in prop  # no variable kind for an all-null column
    assert [q["metric"] for q in prop["quality"]] == ["nullValues"]
    assert contract.tags is None
    table_rules = {q.id: (q.metric, q.arguments) for q in contract.schema_[0].quality}
    assert table_rules[config.ID_NOT_NULL_RULE_ID] == ("nullValues", {"properties": []})
    assert table_rules[config.ID_UNIQUE_RULE_ID] == ("duplicateValues", {"properties": []})


def test_buildContract_noIdAndTextAge_bundledOdcsJsonSchemaValidates(make_contract):
    # Arrange: the not-run paths add table rules with empty properties and a custom range.
    df = pl.DataFrame({"age": ["40", "unknown"], "bmi": [float("nan"), 25.0]})

    # Act
    document = yaml.safe_load(make_contract(df, id_column=None).to_yaml())

    # Assert (raises on any violation)
    jsonschema.validate(document, json.loads(OpenDataContractStandard.json_schema()))
