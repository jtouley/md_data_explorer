# SPDX-License-Identifier: MIT
"""Patient ID detection never builds an ID from a measurement, never renames a source column,
and numbers the rows when nothing identifies a patient (answers then count rows).

Regression for the aggregate-data triage (#3): a unique float rate ("mx") became patient_id and
disappeared from the dataset, and answers said "n=6720 patients".
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from clinical_analytics.core.identifier_rules import (
    ROW_NUMBER_COLUMN,
    Interval,
    can_identify_patients,
    find_intervals,
    patient_id_is_row_key,
)
from clinical_analytics.core.schema_inference import SchemaInferenceEngine
from clinical_analytics.ui.components.variable_detector import VariableTypeDetector


@pytest.mark.parametrize(
    "series",
    [
        pytest.param(pl.Series("score", [0.5 + i for i in range(100)]), id="unique_float"),
        pytest.param(pl.Series("code", [float(i) for i in range(100)]), id="integral_float"),
        pytest.param(pl.Series("ref", [f"{i}.25" for i in range(100)]), id="fractional_numeric_text"),
        pytest.param(pl.Series("mx", list(range(100))), id="rate_named_integer"),
        pytest.param(pl.Series("upper", list(range(100))), id="bound_named_integer"),
        pytest.param(pl.Series("Death rate", list(range(100))), id="rate_token_in_name"),
        pytest.param(pl.Series("flag", [True, False] * 50), id="boolean"),
    ],
)
def test_canIdentifyPatients_measurementOrFloat_rejected(series: pl.Series) -> None:
    # Act / Assert
    assert can_identify_patients(series) is False


@pytest.mark.parametrize(
    "series",
    [
        pytest.param(pl.Series("record", list(range(100))), id="integer"),
        pytest.param(pl.Series("mrn_code", [f"P-{i:04d}" for i in range(100)]), id="text_code"),
        pytest.param(pl.Series("ref", [str(i) for i in range(100)]), id="whole_numeric_text"),
        pytest.param(pl.Series("ref", [None] * 100, dtype=pl.String), id="all_null_text"),
    ],
)
def test_canIdentifyPatients_integerOrText_accepted(series: pl.Series) -> None:
    # Act / Assert
    assert can_identify_patients(series) is True


@pytest.mark.parametrize(
    "options",
    [
        pytest.param({}, id="unique_float_mx"),
        pytest.param({"rounded": True}, id="rounded_integer_mx"),
        pytest.param({"unique_upper": True}, id="unique_float_upper"),
    ],
)
def test_ensurePatientId_populationEstimates_rowNumberNotPatientId(
    make_population_estimates: Callable[..., pl.DataFrame], options: dict[str, Any]
) -> None:
    # Arrange
    df = make_population_estimates(**options)

    # Act
    result, metadata = VariableTypeDetector.ensure_patient_id(df)

    # Assert: every source column kept unchanged, no patient_id, rows numbered 1..n
    assert metadata["patient_id_source"] == "row_index"
    assert metadata["patient_id_columns"] is None
    assert "patient_id" not in result.columns
    assert_frame_equal(result.drop(ROW_NUMBER_COLUMN), df)
    assert result[ROW_NUMBER_COLUMN].to_list() == list(range(1, df.height + 1))


def test_ensurePatientId_uniqueTextColumn_copiedNotRenamed() -> None:
    # Arrange
    df = pl.DataFrame({"record_code": [f"R{i:03d}" for i in range(50)], "sbp": [120.5 + i for i in range(50)]})

    # Act
    result, metadata = VariableTypeDetector.ensure_patient_id(df)

    # Assert: the source column stays queryable beside the copy
    assert metadata["patient_id_source"] == "single_column"
    assert metadata["patient_id_columns"] == ["record_code"]
    assert result["patient_id"].to_list() == df["record_code"].to_list()
    assert_frame_equal(result.drop("patient_id"), df)


def test_ensurePatientId_rowNumberNameTaken_usesFreeName() -> None:
    # Arrange: a non-unique column already called row_number
    df = pl.DataFrame({"row_number": [1, 1, 2, 2], "site": ["A", "B", "A", "B"]})

    # Act
    result, metadata = VariableTypeDetector.ensure_patient_id(df)

    # Assert
    assert metadata["row_number_column"] == f"_{ROW_NUMBER_COLUMN}"
    assert result["row_number"].to_list() == [1, 1, 2, 2]
    assert result[f"_{ROW_NUMBER_COLUMN}"].to_list() == [1, 2, 3, 4]


@pytest.mark.parametrize(
    "df",
    [
        pytest.param(pl.DataFrame({"a": [], "b": []}, schema={"a": pl.String, "b": pl.Float64}), id="empty"),
        pytest.param(
            pl.DataFrame({"a": [None, None], "b": [1.5, 2.5]}, schema={"a": pl.String, "b": pl.Float64}), id="nulls"
        ),
    ],
)
def test_ensurePatientId_emptyOrNullFrame_rowNumberOnly(df: pl.DataFrame) -> None:
    # Act
    result, metadata = VariableTypeDetector.ensure_patient_id(df)

    # Assert
    assert metadata["patient_id_source"] == "row_index"
    assert result.height == df.height
    assert "patient_id" not in result.columns


def test_findCompositeIdentifierCandidates_floatMakesUnique_noKey(
    make_population_estimates: Callable[..., pl.DataFrame],
) -> None:
    # Arrange: only (location, year, mx) is near-unique; mx is a rate and year has >20 values
    df = make_population_estimates(rounded=True)

    # Act
    candidates = VariableTypeDetector.find_composite_identifier_candidates(df)

    # Assert
    assert candidates is None


def test_findCompositeIdentifierCandidates_nearlyUnique_rejected() -> None:
    # Arrange: 20 rows, 19 distinct (site, arm) pairs: over 95% unique but one pair repeats
    sites = [f"S{i % 10}" for i in range(20)]
    arms = [f"A{i // 10}" for i in range(20)]
    arms[19] = arms[9]
    sites[19] = sites[9]
    df = pl.DataFrame({"site": sites, "arm": arms})

    # Act
    candidates = VariableTypeDetector.find_composite_identifier_candidates(df)

    # Assert
    assert candidates is None


def test_inferSchema_populationEstimates_noPatientIdColumn(
    make_population_estimates: Callable[..., pl.DataFrame],
) -> None:
    # Arrange: the stored frame after ensure_patient_id
    df, _ = VariableTypeDetector.ensure_patient_id(make_population_estimates(unique_upper=True))

    # Act
    schema = SchemaInferenceEngine().infer_schema(df)

    # Assert: neither the unique float "upper" nor the row number is the patient key
    assert schema.patient_id_column is None


@pytest.mark.parametrize(
    ("schema", "expected"),
    [
        pytest.param(
            {"mx": pl.Float64, "upper": pl.Float64, "lower": pl.Float64, "year": pl.Int64},
            [Interval("mx", "lower", "upper")],
            id="bare_bounds_one_float",
        ),
        pytest.param(
            {"val": pl.Float64, "Upper": pl.Float64, "Lower": pl.Float64, "mx": pl.Float64},
            [Interval("val", "Lower", "Upper")],
            id="bare_bounds_named_value",
        ),
        pytest.param(
            {"ldl": pl.Float64, "ldl_lower": pl.Float64, "ldl_upper": pl.Float64},
            [Interval("ldl", "ldl_lower", "ldl_upper")],
            id="suffix_bounds",
        ),
        pytest.param(
            {"a": pl.Float64, "b": pl.Float64, "upper": pl.Float64, "lower": pl.Float64},
            [],
            id="bare_bounds_ambiguous_value",
        ),
        pytest.param({"mx": pl.Float64, "upper": pl.Float64}, [], id="one_bound_only"),
        pytest.param({"mx": pl.String, "upper": pl.Float64, "lower": pl.Float64}, [], id="value_not_numeric"),
        pytest.param({}, [], id="empty"),
    ],
)
def test_findIntervals_boundNames_valueAndBoundsFound(schema: dict[str, pl.DataType], expected: list[Interval]) -> None:
    # Act / Assert
    assert find_intervals(schema) == expected


def test_ensurePatientId_compositeKey_hashedAndSourcesKept(
    make_population_estimates: Callable[..., pl.DataFrame],
) -> None:
    # Arrange: three causes, so cause_name is categorical and the cell key is unique
    df = make_population_estimates(causes=3)

    # Act
    result, metadata = VariableTypeDetector.ensure_patient_id(df)

    # Assert
    assert metadata["patient_id_source"] == "composite"
    assert metadata["patient_id_columns"] == ["location_name", "cause_name", "sex", "year"]
    assert result["patient_id"].n_unique() == df.height
    assert_frame_equal(result.drop("patient_id"), df)


@pytest.mark.parametrize(
    ("patient_id_meta", "expected"),
    [
        pytest.param({"patient_id_source": "composite", "patient_id_columns": ["a", "b"]}, True, id="composite"),
        pytest.param({"patient_id_source": "row_index"}, True, id="row_index"),
        pytest.param(
            {"patient_id_source": "single_column", "patient_id_columns": ["FIPS"]}, True, id="copy_not_named_id"
        ),
        pytest.param(
            {"patient_id_source": "single_column", "patient_id_columns": ["MRN"]}, False, id="copy_of_named_id"
        ),
        pytest.param({"patient_id_source": "existing"}, False, id="existing"),
        pytest.param({}, False, id="empty"),
        pytest.param(None, False, id="no_metadata"),
    ],
)
def test_patientIdIsRowKey_bySource_rowsUnlessRealPatientId(patient_id_meta: Any, expected: bool) -> None:
    # Arrange
    metadata = {} if patient_id_meta is None else {"synthetic_id_metadata": {"patient_id": patient_id_meta}}

    # Act / Assert
    assert patient_id_is_row_key(metadata) is expected
