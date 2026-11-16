# SPDX-License-Identifier: MIT
"""
Value-level (given input / expected output) tests for the multi-table cohort grain.

Covers the dbt-practice review findings on core/multi_table_handler.py:
- H2: anchor uniqueness is judged exactly and the mart is one row per anchor key
- M3: "last" follows the time column, never file order; no min/max on id/code columns;
  tables left out of the cohort are reported with a reason
"""

from datetime import date, datetime

import duckdb
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from clinical_analytics.core.multi_table_handler import (
    AggregationPolicy,
    DataQualityError,
    MultiTableHandler,
    aggregate_fact_table,
)
from clinical_analytics.core.multi_table_handler import DataQualityError as mth_DataQualityError
from clinical_analytics.core.schema import DataQualityError as schema_DataQualityError

# ---------------------------------------------------------------------------
# Anchor uniqueness (H2)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("n_duplicates", "expected_fragment"),
    [
        (1, "some duplicate"),
        (10, "some duplicate"),
        (11, "11 duplicate"),
        (12, "12 duplicate"),
    ],
)
@pytest.mark.parametrize("with_fact_table", [False, True])
def test_build_unified_cohort_duplicateAnchorKeys_raisesDataQualityError(
    make_cohort_with_categorical, n_duplicates, expected_fragment, with_fact_table
):
    # Arrange: 300 unique patients plus n duplicated rows (ratio stays under the old 5% tolerance)
    cohort = make_cohort_with_categorical(n_patients=300)
    patients = pl.concat([cohort, cohort.head(n_duplicates)])
    tables = {"patients": patients}
    if with_fact_table:
        tables["visits"] = pl.DataFrame({"patient_id": ["P001", "P001", "P002"], "sbp": [120, 130, 140]})
    handler = MultiTableHandler(tables)

    # Act
    with pytest.raises(DataQualityError) as exc_info:
        handler.build_unified_cohort()

    # Assert
    message = str(exc_info.value)
    assert "patients" in message
    assert "patient_id" in message
    assert expected_fragment in message
    if n_duplicates < 11:
        assert f"{n_duplicates} duplicate" not in message
    handler.close()


def test_classify_tables_duplicateBeyondSampleWindow_marksNotUniqueExactly(make_cohort_with_categorical):
    # Arrange: the only duplicate sits after the first 10,000 rows
    cohort = make_cohort_with_categorical(n_patients=20_000)
    patients = pl.concat([cohort, cohort.head(1)])
    handler = MultiTableHandler({"patients": patients})

    # Act
    classification = handler.classify_tables()["patients"]

    # Assert
    assert classification.is_unique_on_grain is False
    assert classification.cardinality_ratio == pytest.approx(20_001 / 20_000)
    handler.close()


def test_build_unified_cohort_duplicateBeyondSampleWindow_raisesDataQualityError(make_cohort_with_categorical):
    # Arrange
    cohort = make_cohort_with_categorical(n_patients=20_000)
    patients = pl.concat([cohort, cohort.head(1)])
    handler = MultiTableHandler({"patients": patients})

    # Act / Assert
    with pytest.raises(DataQualityError, match="some duplicate"):
        handler.build_unified_cohort()
    handler.close()


def test_build_unified_cohort_uniqueAnchorNoFacts_returnsOneRowPerPatient(make_cohort_with_categorical):
    # Arrange
    patients = make_cohort_with_categorical(n_patients=3)
    handler = MultiTableHandler({"patients": patients})

    # Act
    result = handler.build_unified_cohort()

    # Assert
    assert_frame_equal(result, patients)
    handler.close()


# ---------------------------------------------------------------------------
# "Last" follows time order, not file order (M3)
# ---------------------------------------------------------------------------


def test_aggregate_fact_table_fileOrderDiffersFromTimeOrder_lastFollowsTime():
    # Arrange: P001's latest visit (March) comes first in the file
    visits = pl.DataFrame(
        {
            "patient_id": ["P001", "P001", "P002"],
            "visit_date": [date(2024, 3, 1), date(2024, 1, 1), date(2024, 2, 1)],
            "sbp": [150, 120, 130],
        }
    )
    expected = pl.DataFrame(
        {
            "patient_id": ["P001", "P002"],
            "count": pl.Series([2, 1], dtype=pl.UInt32),
            "visit_date_count_distinct": pl.Series([2, 1], dtype=pl.UInt32),
            "visit_date_min": [date(2024, 1, 1), date(2024, 2, 1)],
            "visit_date_max": [date(2024, 3, 1), date(2024, 2, 1)],
            "visit_date_last": [date(2024, 3, 1), date(2024, 2, 1)],
            "sbp_count_distinct": pl.Series([2, 1], dtype=pl.UInt32),
            "sbp_min": [120, 130],
            "sbp_max": [150, 130],
            "sbp_last": [150, 130],
        }
    )

    # Act
    result = (
        aggregate_fact_table(
            visits, grain_key="patient_id", time_column="visit_date", policy=AggregationPolicy(), table_name="visits"
        )
        .collect()
        .sort("patient_id")
    )

    # Assert
    assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    ("visit_dates", "sbp", "expected_last"),
    [
        pytest.param(
            [date(2024, 3, 1), None, date(2024, 1, 1)],
            [150, 999, 120],
            150,
            id="null_time_ranks_before_dated_rows",
        ),
        pytest.param(
            [date(2024, 1, 1), date(2024, 2, 1), date(2024, 3, 1)],
            [120, 130, 150],
            150,
            id="file_order_matches_time_order",
        ),
        pytest.param(
            [date(2024, 2, 1), date(2024, 3, 1), date(2024, 1, 1)],
            [130, 150, None],
            150,
            id="latest_in_middle_of_file_with_null_value_last_in_file",
        ),
    ],
)
def test_aggregate_fact_table_timeOrderingEdgeCases_lastIsLatestByTime(visit_dates, sbp, expected_last):
    # Arrange
    visits = pl.DataFrame(
        {
            "patient_id": ["P001"] * 3,
            "visit_date": pl.Series(visit_dates, dtype=pl.Date),
            "sbp": pl.Series(sbp, dtype=pl.Int64),
        }
    )
    expected = pl.DataFrame({"patient_id": ["P001"], "sbp_last": pl.Series([expected_last], dtype=pl.Int64)})

    # Act
    result = (
        aggregate_fact_table(
            visits, grain_key="patient_id", time_column="visit_date", policy=AggregationPolicy(), table_name="visits"
        )
        .select("patient_id", "sbp_last")
        .collect()
    )

    # Assert
    assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "time_column",
    [pytest.param(None, id="no_time_column"), pytest.param("visit_date", id="string_time_column")],
)
def test_aggregate_fact_table_noTemporalTimeColumn_emitsNoLast(time_column):
    # Arrange: a string date column cannot order rows reliably ("03/01" < "12/01" < "2024-...")
    visits = pl.DataFrame(
        {
            "patient_id": ["P001", "P001"],
            "visit_date": ["03/01/2024", "01/01/2024"],
            "sbp": [150, 120],
        }
    ).select(["patient_id", "sbp"] + (["visit_date"] if time_column else []))
    expected = pl.DataFrame(
        {
            "patient_id": ["P001"],
            "count": pl.Series([2], dtype=pl.UInt32),
            "sbp_count_distinct": pl.Series([2], dtype=pl.UInt32),
            "sbp_min": [120],
            "sbp_max": [150],
        }
    )

    # Act
    result = (
        aggregate_fact_table(
            visits, grain_key="patient_id", time_column=time_column, policy=AggregationPolicy(), table_name="visits"
        )
        .select(expected.columns)
        .collect()
    )
    all_columns = (
        aggregate_fact_table(
            visits, grain_key="patient_id", time_column=time_column, policy=AggregationPolicy(), table_name="visits"
        )
        .collect_schema()
        .names()
    )

    # Assert
    assert_frame_equal(result, expected)
    assert not [c for c in all_columns if c.endswith("_last")]


def test_aggregate_fact_table_emptyTable_returnsEmptyFrame():
    # Arrange
    visits = pl.DataFrame(
        {"patient_id": [], "visit_date": [], "sbp": []},
        schema={"patient_id": pl.Utf8, "visit_date": pl.Date, "sbp": pl.Int64},
    )

    # Act
    result = aggregate_fact_table(
        visits, grain_key="patient_id", time_column="visit_date", policy=AggregationPolicy(), table_name="visits"
    ).collect()

    # Assert
    assert result.height == 0
    assert "sbp_last" in result.columns


# ---------------------------------------------------------------------------
# No min/max on id/code columns (M3)
# ---------------------------------------------------------------------------


def test_aggregate_fact_table_numericIdAndCodeColumns_emitNoMinMax():
    # Arrange
    visits = pl.DataFrame(
        {
            "patient_id": ["P001", "P001", "P002"],
            "ward_code": [3, 1, 2],
            "device_id": [10, 30, 20],
            "itemid": [220045, 220046, 220045],
            "sbp": [150, 120, 130],
        }
    )
    expected = pl.DataFrame(
        {
            "patient_id": ["P001", "P002"],
            "count": pl.Series([2, 1], dtype=pl.UInt32),
            "ward_code_count_distinct": pl.Series([2, 1], dtype=pl.UInt32),
            "device_id_count_distinct": pl.Series([2, 1], dtype=pl.UInt32),
            "itemid_count_distinct": pl.Series([2, 1], dtype=pl.UInt32),
            "sbp_count_distinct": pl.Series([2, 1], dtype=pl.UInt32),
            "sbp_min": [120, 130],
            "sbp_max": [150, 130],
        }
    )

    # Act
    result = (
        aggregate_fact_table(
            visits, grain_key="patient_id", time_column=None, policy=AggregationPolicy(), table_name="visits"
        )
        .collect()
        .sort("patient_id")
    )

    # Assert
    assert_frame_equal(result, expected)


# ---------------------------------------------------------------------------
# Tables left out of the cohort are reported (M3)
# ---------------------------------------------------------------------------


def test_build_unified_cohort_with_report_factTableWithoutAnchorKey_reportsExclusion(make_cohort_with_categorical):
    # Arrange: an admission-keyed fact table (>10 MB so it classifies as fact) with no patient_id
    patients = make_cohort_with_categorical(n_patients=3)
    n_rows = 1_200
    labs = pl.DataFrame(
        {
            "hadm_id": [f"H{i % 600}" for i in range(n_rows)],
            "note": ["x" * 10_000] * n_rows,
        }
    )
    handler = MultiTableHandler({"patients": patients, "labs": labs})

    # Act
    result = handler.build_unified_cohort_with_report(anchor_table="patients")

    # Assert
    assert handler.classifications["labs"].classification == "fact"
    excluded = {t.table_name: t for t in result.excluded_tables}
    assert "labs" in excluded
    assert "patient_id" in excluded["labs"].reason
    assert_frame_equal(result.data, patients)
    assert handler.build_unified_cohort(anchor_table="patients").height == patients.height
    assert [t.table_name for t in handler.last_build_result.excluded_tables] == ["labs"]
    handler.close()


def test_build_unified_cohort_with_report_smallReferenceTable_reportsExclusion(make_cohort_with_categorical):
    # Arrange: a small repeated-key table classifies as reference and is not joined
    patients = make_cohort_with_categorical(n_patients=3)
    visits = pl.DataFrame({"patient_id": ["P001", "P001", "P002"], "sbp": [120, 130, 140]})
    handler = MultiTableHandler({"patients": patients, "visits": visits})

    # Act
    result = handler.build_unified_cohort_with_report()

    # Assert
    assert result.anchor_table == "patients"
    assert result.grain_key == "patient_id"
    assert [(t.table_name, t.classification) for t in result.excluded_tables] == [("visits", "reference")]
    assert result.excluded_tables[0].reason
    assert_frame_equal(result.data, patients)
    handler.close()


# ---------------------------------------------------------------------------
# PR #92 review findings
# ---------------------------------------------------------------------------


def _large_fact_table(patient_ids: list[str], **columns: list) -> pl.DataFrame:
    """A fact table over 10 MB (so it classifies as fact, not reference) keyed by patient_id."""
    n_rows = len(patient_ids)
    return pl.DataFrame({"patient_id": patient_ids, **columns, "note": ["x" * 10_000] * n_rows})


@pytest.mark.parametrize(
    ("n_null_keys", "expected_count_text"),
    [
        pytest.param(1, "some", id="one_null_key"),
        pytest.param(2, "some", id="two_null_keys"),
        pytest.param(10, "some", id="ten_null_keys_small_cell"),
        pytest.param(11, "11", id="eleven_null_keys_reported_exactly"),
    ],
)
def test_build_unified_cohort_with_report_nullAnchorKeys_excludesThemAndReportsCount(
    make_cohort_with_categorical, n_null_keys, expected_count_text
):
    # Arrange: 300 patients, the first n of them have no patient_id
    cohort = make_cohort_with_categorical(n_patients=300)
    patients = cohort.with_columns(
        pl.when(pl.int_range(pl.len()) < n_null_keys).then(None).otherwise(pl.col("patient_id")).alias("patient_id")
    )
    expected = patients.filter(pl.col("patient_id").is_not_null())
    handler = MultiTableHandler({"patients": patients})

    # Act
    result = handler.build_unified_cohort_with_report()

    # Assert
    assert_frame_equal(result.data, expected)
    assert result.null_key_rows == n_null_keys
    assert result.null_key_note is not None
    assert f"{expected_count_text} rows" in result.null_key_note
    assert "excluded from the cohort" in result.null_key_note
    assert handler.classifications["patients"].is_unique_on_grain is True
    handler.close()


def test_build_unified_cohort_with_report_noNullAnchorKeys_reportsNone(make_cohort_with_categorical):
    # Arrange
    patients = make_cohort_with_categorical(n_patients=3)
    handler = MultiTableHandler({"patients": patients})

    # Act
    result = handler.build_unified_cohort_with_report()

    # Assert
    assert_frame_equal(result.data, patients)
    assert result.null_key_rows == 0
    assert result.null_key_note is None
    handler.close()


def test_find_anchor_by_centrality_nearUniqueHigherScoringTable_choosesUniqueTable(make_cohort_with_categorical):
    # Arrange: 'demographics' outscores 'patients' (+10 for hadm_id) but repeats 3 patient_ids
    patients = make_cohort_with_categorical(n_patients=300)
    patient_ids = patients["patient_id"].to_list()
    demo_ids = patient_ids + patient_ids[:3]
    demographics = pl.DataFrame(
        {
            "patient_id": demo_ids,
            "hadm_id": [f"H{i}" for i in range(len(demo_ids))],
            "race": ["A"] * len(demo_ids),
        }
    )
    handler = MultiTableHandler({"patients": patients, "demographics": demographics})

    # Act
    result = handler.build_unified_cohort_with_report()

    # Assert
    assert handler.classifications["demographics"].classification == "dimension"
    assert handler.classifications["demographics"].is_unique_on_grain is False
    assert result.anchor_table == "patients"
    assert_frame_equal(result.data, patients)
    handler.close()


@pytest.mark.parametrize("join_type", ["outer", "OUTER", " full "])
def test_build_unified_cohort_outerJoin_rejectedUpFront(make_cohort_with_categorical, join_type):
    # Arrange: anchor {P001, P002}; a dimension with {P001, P003} would add a row with no anchor key
    patients = make_cohort_with_categorical(n_patients=2)
    handler = MultiTableHandler({"patients": patients})

    # Act / Assert
    with pytest.raises(ValueError, match="join_type") as exc_info:
        handler.build_unified_cohort(join_type=join_type)
    assert "one row per" in str(exc_info.value)
    handler.close()


def test_materialize_mart_outerJoin_rejectedUpFront(make_cohort_with_categorical, tmp_path):
    # Arrange
    handler = MultiTableHandler({"patients": make_cohort_with_categorical(n_patients=2)})

    # Act / Assert
    with pytest.raises(ValueError, match="join_type"):
        handler.materialize_mart(output_path=tmp_path, join_type="outer")
    assert not any(tmp_path.iterdir())
    handler.close()


@pytest.mark.parametrize(
    ("visit_dates", "expected_last_date"),
    [
        pytest.param(["2024-03-01", "2024-01-01", "2024-02-01"], date(2024, 3, 1), id="iso_date_strings"),
        pytest.param(
            ["2024-03-01 08:00:00", "2024-03-01 07:00:00", None],
            datetime(2024, 3, 1, 8, 0),
            id="iso_datetime_strings_with_null",
        ),
        pytest.param(
            ["2024-03-01T08:00:00", "2024-03-01T07:00:00", "2024-01-01T00:00:00"],
            datetime(2024, 3, 1, 8, 0),
            id="iso_t_separated_datetime_strings",
        ),
    ],
)
def test_aggregate_fact_table_isoStringTimeColumn_parsedAndLastFollowsTime(visit_dates, expected_last_date):
    # Arrange: the latest row is first in the file
    visits = pl.DataFrame({"patient_id": ["P001"] * 3, "visit_date": visit_dates, "sbp": [150, 120, 130]})
    expected = pl.DataFrame(
        {"patient_id": ["P001"], "visit_date_last": [expected_last_date], "sbp_last": pl.Series([150], dtype=pl.Int64)}
    )

    # Act
    result = (
        aggregate_fact_table(
            visits, grain_key="patient_id", time_column="visit_date", policy=AggregationPolicy(), table_name="visits"
        )
        .select("patient_id", "visit_date_last", "sbp_last")
        .collect()
    )

    # Assert
    assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "visit_dates",
    [
        pytest.param(["03/01/2024"] * 600 + ["01/01/2024"] * 600, id="ambiguous_day_month_order"),
        pytest.param(["2024-01-01"] * 600 + ["not a date"] * 600, id="unparseable_values"),
    ],
)
def test_build_unified_cohort_with_report_unparseableTimeColumn_recordsWhyLastIsMissing(
    make_cohort_with_categorical, visit_dates
):
    # Arrange
    patients = make_cohort_with_categorical(n_patients=3)
    patient_ids = ["P001", "P002"] * 600
    visits = _large_fact_table(patient_ids, visit_date=visit_dates, sbp=list(range(1_200)))
    handler = MultiTableHandler({"patients": patients, "visits": visits})

    # Act
    result = handler.build_unified_cohort_with_report(anchor_table="patients")

    # Assert
    assert handler.classifications["visits"].classification == "fact"
    assert handler.classifications["visits"].time_column_name == "visit_date"
    assert not [c for c in result.data.columns if c.endswith("_last")]
    assert "sbp_min" in result.data.columns
    assert list(result.omitted_features) == ["visits"]
    assert "visit_date" in result.omitted_features["visits"]
    assert "_last" in result.omitted_features["visits"]
    handler.close()


def test_build_unified_cohort_duplicatesFromFeatureTable_errorNamesFeatureTable(
    make_cohort_with_categorical, monkeypatch
):
    # Arrange: a feature table that is not one row per patient (aggregation bypassed)
    import clinical_analytics.core.multi_table_handler as mth

    patients = make_cohort_with_categorical(n_patients=3)
    visits = _large_fact_table(["P001", "P002"] * 600, sbp=list(range(1_200)))
    monkeypatch.setattr(mth, "aggregate_fact_table", lambda df, grain_key, **_: df.lazy().select(grain_key, "sbp"))
    handler = MultiTableHandler({"patients": patients, "visits": visits})

    # Act
    with pytest.raises(DataQualityError) as exc_info:
        handler.build_unified_cohort(anchor_table="patients")

    # Assert
    message = str(exc_info.value)
    assert "'visits'" in message
    assert "Table 'patients' has" not in message
    handler.close()


def test_materialize_mart_twoFactTables_matchesBuildUnifiedCohort(make_cohort_with_categorical, tmp_path):
    # Arrange: two fact tables both produce a 'count' column
    patients = make_cohort_with_categorical(n_patients=3)
    patient_ids = ["P001", "P002"] * 600
    labs = _large_fact_table(patient_ids, lab_value=list(range(1_200)))
    vitals = _large_fact_table(patient_ids, sbp=list(range(1_200)))
    handler = MultiTableHandler({"patients": patients, "labs": labs, "vitals": vitals})
    handler._exclusion_reasons["stale_table"] = "left over from an earlier run"
    expected = handler.build_unified_cohort(anchor_table="patients")

    # Act
    handler._exclusion_reasons["stale_table"] = "left over from an earlier run"
    metadata = handler.materialize_mart(output_path=tmp_path, anchor_table="patients")
    materialized = pl.read_parquet(metadata.parquet_path)

    # Assert
    assert_frame_equal(materialized, expected)
    assert metadata.row_count == patients.height
    assert "stale_table" not in handler._exclusion_reasons
    handler.close()


def test_materialize_mart_nullAnchorKeys_excludedLikeBuild(make_cohort_with_categorical, tmp_path):
    # Arrange
    cohort = make_cohort_with_categorical(n_patients=300)
    patients = cohort.with_columns(
        pl.when(pl.int_range(pl.len()) < 2).then(None).otherwise(pl.col("patient_id")).alias("patient_id")
    )
    handler = MultiTableHandler({"patients": patients})

    # Act
    metadata = handler.materialize_mart(output_path=tmp_path)

    # Assert
    assert_frame_equal(pl.read_parquet(metadata.parquet_path), patients.filter(pl.col("patient_id").is_not_null()))
    handler.close()


def test_aggregate_fact_table_decimalColumn_getsMinMax():
    # Arrange
    visits = pl.DataFrame(
        {
            "patient_id": ["P001", "P001", "P002"],
            "dose": pl.Series(["1.50", "2.25", "0.75"]).str.to_decimal(scale=2),
        }
    )
    expected = pl.DataFrame(
        {
            "patient_id": ["P001", "P002"],
            "count": pl.Series([2, 1], dtype=pl.UInt32),
            "dose_count_distinct": pl.Series([2, 1], dtype=pl.UInt32),
            "dose_min": visits["dose"].gather([0, 2]),
            "dose_max": visits["dose"].gather([1, 2]),
        }
    )

    # Act
    result = (
        aggregate_fact_table(
            visits, grain_key="patient_id", time_column=None, policy=AggregationPolicy(), table_name="visits"
        )
        .collect()
        .sort("patient_id")
    )

    # Assert
    assert_frame_equal(result, expected)


def test_handler_contextManager_closesConnectionWhenBuildRaises(make_cohort_with_categorical):
    # Arrange: duplicate anchor keys with no unique alternative
    cohort = make_cohort_with_categorical(n_patients=300)
    patients = pl.concat([cohort, cohort.head(1)])

    # Act
    with pytest.raises(DataQualityError):
        with MultiTableHandler({"patients": patients}) as handler:
            handler.build_unified_cohort()

    # Assert
    with pytest.raises(duckdb.ConnectionException):
        handler.conn.execute("SELECT 1")


def test_dataQualityError_isSchemaDomainException():
    # Arrange / Act / Assert
    assert mth_DataQualityError is schema_DataQualityError
