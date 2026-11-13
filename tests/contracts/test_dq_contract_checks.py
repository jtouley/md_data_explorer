"""Contract checks added for the independent DQ review (B1, H4, M1, M2, M5, L1)."""

from __future__ import annotations

from typing import Any

import polars as pl
import pytest
import yaml

from clinical_analytics.contracts import config
from clinical_analytics.contracts.builder import find_plausibility_range, implausible_values, range_for
from clinical_analytics.contracts.quality import run_quality
from clinical_analytics.contracts.service import analysis_columns_from_metadata

N = 200
IDS = [f"P{i:03d}" for i in range(N)]


def _props(contract: Any) -> dict[str, dict[str, Any]]:
    doc = yaml.safe_load(contract.to_yaml())
    return {p["physicalName"]: p for p in doc["schema"][0]["properties"]}


def _checks(report: Any) -> dict[str, Any]:
    return {c.id: c for c in report.checks}


def test_buildContract_repeatedIds_doesNotAssertUnique(make_contract: Any) -> None:
    # Arrange: 200 rows, 170 IDs (visit-level rows)
    df = pl.DataFrame({"patient_id": IDS[:170] + IDS[:30], "age": [40] * N})

    # Act
    contract = make_contract(df)

    # Assert
    prop = _props(contract)["patient_id"]
    assert prop["unique"] is False
    assert "primaryKey" not in prop
    granularity = yaml.safe_load(contract.to_yaml())["schema"][0]["dataGranularityDescription"]
    assert granularity.startswith("Several rows per patient_id")


def test_runQuality_repeatedIds_exactDuplicateCheckRuns(make_contract: Any) -> None:
    # Arrange
    df = pl.DataFrame({"patient_id": IDS[:170] + IDS[:30], "age": [40] * N})

    # Act
    report = run_quality(df.lazy(), make_contract(df))

    # Assert
    check = _checks(report)["patient_id_no_duplicates"]
    assert (check.count, check.status) == (30, "failed")


def test_runQuality_contractExpectsNumberDataIsText_typeCheckFails(make_contract: Any) -> None:
    # Arrange: the contract was built on integer ages; the data checked now holds them as text
    built_on = pl.DataFrame({"patient_id": IDS, "age": [40] * N})
    checked = pl.DataFrame({"patient_id": IDS, "age": ["40"] * N})

    # Act: version v2 checked against v1's contract
    report = run_quality(checked.lazy(), make_contract(built_on), data_version="v2")

    # Assert
    check = _checks(report)["age_type"]
    assert check.status == "failed"
    assert check.message == "age: stored as string, but the contract expects integer"


@pytest.mark.parametrize(
    ("limits", "count", "status"),
    [
        pytest.param(0, 0, "passed", id="none"),
        pytest.param(3, config.COUNT_HIDDEN, config.HIDDEN_STATUS, id="small_hidden"),
        pytest.param(30, 30, "warning", id="shown"),
    ],
)
def test_runQuality_valuesRecordedAsLimit_countedNotSilent(
    make_contract: Any, limits: int, count: Any, status: str
) -> None:
    # Arrange: on main the upload summary for 30 "<5" values was empty
    crp = ["<5"] * limits + [str(20 + i % 7) for i in range(N - limits)]
    df = pl.DataFrame({"patient_id": IDS, "crp": crp})

    # Act
    report = run_quality(df.lazy(), make_contract(df))

    # Assert
    check = _checks(report)["crp_censored"]
    assert (check.count, check.status) == (count, status)


def test_runQuality_missingTokens_countedAsMissingLikeTheAnalysis(make_contract: Any) -> None:
    # Arrange: "N/A" and "." are missing for the analysis; on main the report said 0 missing
    crp = ["N/A"] * 15 + ["."] * 15 + [str(20 + i % 7) for i in range(N - 30)]
    df = pl.DataFrame({"patient_id": IDS, "crp": crp})

    # Act
    report = run_quality(df.lazy(), make_contract(df))

    # Assert
    assert _checks(report)["crp_missing"].count == 30


def test_runQuality_categorySpellings_warnedAtUpload(make_contract: Any) -> None:
    # Arrange
    df = pl.DataFrame({"patient_id": IDS, "sex": ["Male"] * 60 + ["male"] * 50 + ["M"] * 40 + ["Female"] * 50})

    # Act
    report = run_quality(df.lazy(), make_contract(df))

    # Assert
    check = _checks(report)["sex_spellings"]
    assert check.status == "warning"
    assert "(for example M, Male, male)" in check.message
    assert check.message in report.summary


def test_runQuality_spellingOfFewRows_notNamed(make_contract: Any) -> None:
    # Arrange: "m" is held by 3 rows; its label is not shown
    df = pl.DataFrame({"patient_id": IDS, "sex": ["Male"] * 97 + ["m"] * 3 + ["Female"] * 100})

    # Act
    check = _checks(run_quality(df.lazy(), make_contract(df)))["sex_spellings"]

    # Assert
    assert check.status == "warning"
    assert "for example" not in check.message  # only one spelling has 11+ rows: none is named


@pytest.mark.parametrize(
    ("analysis", "missing", "status"),
    [
        pytest.param((), 100, "passed", id="optional_column_half_missing_not_flagged"),
        pytest.param((), N, "warning", id="optional_column_all_missing_flagged"),
        pytest.param(("ldl",), 12, "warning", id="analysis_variable_above_5pct_flagged"),
        pytest.param(("ldl",), 0, "passed", id="analysis_variable_complete"),
    ],
)
def test_runQuality_missingness_flaggedOnlyForAnalysisVariables(
    make_contract: Any, analysis: tuple[str, ...], missing: int, status: str
) -> None:
    # Arrange
    ldl = pl.Series("ldl", [None] * missing + [100.0] * (N - missing), dtype=pl.Float64)
    df = pl.DataFrame({"patient_id": IDS, "ldl": ldl})

    # Act
    check = _checks(run_quality(df.lazy(), make_contract(df, analysis_columns=analysis)))["ldl_missing"]

    # Assert
    assert check.status == status


def test_buildContract_measurementWithoutRange_markedNoRangeKnown(make_contract: Any) -> None:
    # Arrange
    df = pl.DataFrame({"patient_id": IDS, "ferritin": [float(i) for i in range(N)]})

    # Act
    prop = _props(make_contract(df))["ferritin"]

    # Assert
    custom = {c["property"]: c["value"] for c in prop["customProperties"]}
    assert custom[config.PLAUSIBILITY_RANGE_PROPERTY] == config.NO_RANGE_KNOWN


@pytest.mark.parametrize(
    "column",
    [pytest.param(r.variable, id=r.variable) for r in config.PLAUSIBILITY_RANGES],
)
def test_plausibilityRanges_everyRange_hasUnitAndSource(column: str) -> None:
    # Arrange
    bounds = next(r for r in config.PLAUSIBILITY_RANGES if r.variable == column)

    # Act / Assert
    assert bounds.unit and bounds.source
    assert bounds.maximum is None or bounds.maximum > bounds.minimum


# --- Review round (PR #96) --------------------------------------------------------------------


def test_runQuality_contractBuiltFromSameVersion_typeCheckNotRunNeverPassed(make_contract: Any) -> None:
    # Arrange: on the PR head every type check "passed" on the data the contract was read from
    df = pl.DataFrame({"patient_id": IDS, "age": [40] * N, "crp": ["<5"] * 10 + ["12"] * (N - 10)})

    # Act
    report = run_quality(df.lazy(), make_contract(df))

    # Assert
    types = [c for c in report.checks if c.metric == "logicalType"]
    assert types and {c.status for c in types} == {"not_run"}
    assert "built from this version" in types[0].message


@pytest.mark.parametrize(
    ("values", "missing"),
    [
        pytest.param(["None"] * 140 + ["Diabetes"] * 60, 0, id="categorical_none_is_a_level"),
        pytest.param(["-"] * 30 + ["Smoker"] * 170, 0, id="categorical_dash_is_a_level"),
        pytest.param(["#N/A"] * 20 + [str(10 + i % 9) for i in range(180)], 20, id="numeric_text_hash_na_missing"),
        pytest.param(["None"] * 20 + [str(10 + i % 9) for i in range(180)], 20, id="numeric_text_none_missing"),
    ],
)
def test_runQuality_missingTokens_onlyForNumericText(make_contract: Any, values: list[str], missing: int) -> None:
    # Arrange
    df = pl.DataFrame({"patient_id": IDS, "x": values})

    # Act
    check = _checks(run_quality(df.lazy(), make_contract(df, analysis_columns=("x",))))["x_missing"]

    # Assert
    assert check.count == missing


def test_missingValueTokens_includeHashNa() -> None:
    # Act / Assert: the same tuple as PR #95's constant
    assert config.MISSING_VALUE_TOKENS[-1] == "#n/a"
    assert len(set(config.MISSING_VALUE_TOKENS)) == len(config.MISSING_VALUE_TOKENS)


def test_runQuality_restrictedColumn_spellingsNotScannedOrEchoed(make_contract: Any) -> None:
    # Arrange: a clinician name column marked restricted (PHI)
    df = pl.DataFrame({"patient_id": IDS, "clinician": ["Dr Smith"] * 100 + ["dr smith"] * 100})
    contract = make_contract(df)
    prop = next(p for p in contract.schema_[0].properties if p.name == "clinician")
    prop.classification, prop.tags = "restricted", ["phi"]

    # Act
    report = run_quality(df.lazy(), contract)

    # Assert
    assert "clinician_spellings" not in _checks(report)
    assert not any("Smith" in c.message for c in report.checks)


def test_runQuality_tinyTableIdColumn_notScannedAsCategory(make_contract: Any) -> None:
    # Arrange: 12 rows, so the builder states neither unique nor primaryKey for the ID
    df = pl.DataFrame({"patient_id": ["M"] * 6 + ["m"] * 6, "age": [40] * 12})

    # Act
    report = run_quality(df.lazy(), make_contract(df))

    # Assert
    assert "patient_id_spellings" not in _checks(report)


@pytest.mark.parametrize(
    ("metadata", "expected"),
    [
        pytest.param(
            {"variable_mapping": {"outcome": "died", "time_variables": {"time_zero": "admit_date", "event": "dod"}}},
            ("died", "admit_date", "dod"),
            id="time_variables_dict_flattened",
        ),
        pytest.param({"variable_mapping": {"predictors": ["age", ["bmi"]]}}, ("age", "bmi"), id="nested_list"),
        pytest.param({"variable_mapping": {"patient_id": "pid"}}, (), id="id_not_analysis"),
        pytest.param({}, (), id="empty"),
    ],
)
def test_analysisColumnsFromMetadata_nestedValues_flattened(
    metadata: dict[str, Any], expected: tuple[str, ...]
) -> None:
    # Act / Assert
    assert analysis_columns_from_metadata(metadata) == expected


@pytest.mark.parametrize(
    ("column", "values", "expected"),
    [
        pytest.param("temperature", [37.0] * 199 + [98.6], "10–47 °C", id="celsius_by_median_flags_fahrenheit_value"),
        pytest.param("temperature", [98.6] * 199 + [20.0], "50–117 °F", id="fahrenheit_by_median_flags_20"),
    ],
)
def test_implausibleValues_temperature_unitFromMedian(column: str, values: list[float], expected: str) -> None:
    # Act
    found = implausible_values(pl.Series(column, values))

    # Assert
    assert found is not None and (found.count, found.bounds) == (1, expected)


def test_rangeFor_medianUnknown_saysWideBound() -> None:
    # Act
    bounds = range_for("temperature", None)

    # Assert
    assert bounds is not None and bounds.unit == "°C or °F (unit unknown, wide bound)"


@pytest.mark.parametrize(
    ("column", "expected"),
    [
        pytest.param("na", None, id="bare_na_is_not_sodium"),
        pytest.param("k", None, id="bare_k_is_not_potassium"),
        pytest.param("tc", None, id="treatment_code_is_not_cholesterol"),
        pytest.param("Na (mmol/L)", "sodium", id="na_with_unit_is_sodium"),
        pytest.param("K mmol/L", "potassium", id="k_with_unit_is_potassium"),
        pytest.param("sodium", "sodium", id="full_name"),
    ],
)
def test_findPlausibilityRange_shortAlias_needsUnitHint(column: str, expected: str | None) -> None:
    # Act
    found = find_plausibility_range(column)

    # Assert
    assert (found.variable if found else None) == expected


# --- Second review round (PR #96) ----------------------------------------------------------


@pytest.mark.parametrize(
    ("column", "expected"),
    [
        pytest.param("Glucose (mg/dL)", "glucose", id="glucose_mgdl"),
        pytest.param("LDL (mg/dL)", "ldl", id="ldl_mgdl"),
        pytest.param("creatinine umol/L", "creatinine", id="creatinine_umoll"),
        pytest.param("Glucose (mmol/L)", "glucose", id="glucose_mmoll"),
        pytest.param("glucose", "glucose", id="plain_alias_no_unit"),
        pytest.param("Glucose (notes)", None, id="non_unit_suffix_no_range"),
    ],
)
def test_findPlausibilityRange_plainAliasWithUnitSuffix_getsRange(column: str, expected: str | None) -> None:
    # Act
    found = find_plausibility_range(column)

    # Assert
    assert (found.variable if found else None) == expected


@pytest.mark.parametrize("name", ["count", "len", "__n"])
def test_runQuality_textColumnNamedLikeCountField_noCrash(make_contract: Any, name: str) -> None:
    # Arrange: value_counts() names its count field "count"; a column of that name collided
    df = pl.DataFrame({"patient_id": IDS, name: ["Male"] * 100 + ["male"] * 60 + ["Female"] * 40})

    # Act
    report = run_quality(df.lazy(), make_contract(df))

    # Assert
    assert _checks(report)[f"{name.strip('_') or 'column_2'}{config.LEVEL_VARIANTS_RULE_SUFFIX}"].status == "warning"


def test_smallLevelLabels_columnNamedCount_noCrash() -> None:
    # Arrange
    from clinical_analytics.eval.headless import small_level_labels

    frame = pl.DataFrame({"count": ["a"] * 3 + ["b"] * 20})

    # Act
    labels = small_level_labels(frame, ["count"], threshold=11)

    # Assert
    assert labels == {"a"}


def test_implausibleValues_distinctValuesWithWeights_shareWeightedByRows() -> None:
    # Arrange: 30 distinct readings held by 100 rows each, one of them -5, and 2 words held by
    # one row each: 30/32 distinct values are numbers (< 95%), but 3000/3002 rows are
    values = pl.Series("glucose", [str(v) for v in range(70, 99)] + ["-5", "pending", "see notes"])
    weights = pl.Series([100] * 30 + [1, 1])

    # Act
    found = implausible_values(values, weights)

    # Assert
    assert found is not None
    assert (found.count, found.present) == (100, 3000)


def test_checkUpload_typeChangedBetweenVersions_checkedAgainstPreviousContract(tmp_path: Any) -> None:
    # Arrange: version v1 stores age as integers; v2 (an overwrite of the same upload) as text
    from clinical_analytics.contracts.service import check_upload

    v1 = pl.DataFrame({"patient_id": IDS, "age": [40 + i % 30 for i in range(N)]})
    v2 = pl.DataFrame({"patient_id": IDS, "age": [str(40 + i % 30) for i in range(N)]})
    base: dict[str, Any] = {
        "dataset_name": "cohort",
        "inferred_schema": {"column_mapping": {"patient_id": "patient_id"}},
    }
    first_meta = {**base, "version_history": [{"version": "v1", "is_active": True}]}
    second_meta = {
        **base,
        "version_history": [{"version": "v1", "is_active": False}, {"version": "v2", "is_active": True}],
    }

    # Act
    first = check_upload(tmp_path, v1, upload_id="u1", dataset_version="v1", metadata=first_meta)
    second = check_upload(tmp_path, v2, upload_id="u1", dataset_version="v2", metadata=second_meta)

    # Assert
    assert _checks(first)[f"table{config.TYPE_RULE_SUFFIX}"].status == "not_run"
    age = _checks(second)[f"age{config.TYPE_RULE_SUFFIX}"]
    assert (age.status, age.message) == ("failed", "age: stored as string, but the contract expects integer")
    assert _checks(second)[f"patient_id{config.TYPE_RULE_SUFFIX}"].status == "passed"


def test_checkUpload_previousContractMissing_typeCheckNotRun(tmp_path: Any) -> None:
    # Arrange: a legacy previous version without a contract
    from clinical_analytics.contracts.service import check_upload

    df = pl.DataFrame({"patient_id": IDS, "age": [40] * N})
    meta = {
        "dataset_name": "cohort",
        "inferred_schema": {"column_mapping": {"patient_id": "patient_id"}},
        "version_history": [{"version": "v0", "is_active": False}, {"version": "v1", "is_active": True}],
    }

    # Act
    report = check_upload(tmp_path, df, upload_id="u1", dataset_version="v1", metadata=meta)

    # Assert
    assert _checks(report)[f"table{config.TYPE_RULE_SUFFIX}"].status == "not_run"


@pytest.mark.parametrize(
    ("rows", "tiny"),
    [
        pytest.param(0, False, id="empty"),
        pytest.param(1, True, id="one"),
        pytest.param(config.MAX_ROWS_NO_COUNTS, True, id="boundary"),
        pytest.param(config.MAX_ROWS_NO_COUNTS + 1, False, id="above"),
    ],
)
def test_isTinyTable_boundaries(rows: int, tiny: bool) -> None:
    # Act / Assert
    assert config.is_tiny_table(rows) is tiny


def test_tinyTableRule_builderAndReportUseOneRule(make_contract: Any, monkeypatch: Any) -> None:
    # Arrange: change the one rule; both the contract's grain and the report follow it
    df = pl.DataFrame({"patient_id": IDS, "age": [40] * N})
    monkeypatch.setattr(config, "is_tiny_table", lambda rows: True)

    # Act
    contract = make_contract(df)
    report = run_quality(df.lazy(), contract)

    # Assert
    granularity = yaml.safe_load(contract.to_yaml())["schema"][0]["dataGranularityDescription"]
    assert granularity == "One row per record as uploaded; too few rows to state the grain"
    assert report.summary[0] == f"{N} rows: {config.TOO_FEW_ROWS_NOTE}"
