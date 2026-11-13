# SPDX-License-Identifier: MIT
"""
Value-level (given input / expected output) tests for the multi-table cohort grain.

Covers the dbt-practice review findings on core/multi_table_handler.py:
- H2: anchor uniqueness is judged exactly and the mart is one row per anchor key
- M3: "last" follows the time column, never file order; no min/max on id/code columns;
  tables left out of the cohort are reported with a reason
"""

from datetime import date

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from clinical_analytics.core.multi_table_handler import (
    AggregationPolicy,
    DataQualityError,
    MultiTableHandler,
    aggregate_fact_table,
)

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
