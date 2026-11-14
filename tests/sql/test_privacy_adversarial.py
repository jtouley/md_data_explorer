"""Adversarial suite: queries that try to show, or work out, a count of 1-10 patients.

Synthetic cohort (``tests/sql/conftest.py``): 300 patients, ward ICU = 3 patients, 10 patients
aged 18, score 3 held by 4 patients. Each attack is checked against the truth computed
independently with Polars, and against what a reader could subtract from other shown answers.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import polars as pl
import polars.testing
import pytest

from clinical_analytics.analysis.grouping import FOLD_LABELS
from clinical_analytics.contracts.declarations import LevelMapping
from clinical_analytics.core.result_presentation import SECONDARY_SUPPRESSED
from clinical_analytics.sql.catalog import SqlCatalog
from clinical_analytics.sql.errors import SqlRefusedError
from clinical_analytics.sql.privacy import NO_VALUE_LABEL, SqlAnswer
from tests.sql.conftest import N_PATIENTS, synthetic_cohort

SMALL = range(1, 11)
Ask = Callable[..., SqlAnswer]


def _numbers(answer: SqlAnswer, column: str) -> dict[str, Any]:
    """Shown values of ``column`` by group label ("" when ungrouped); hidden ones left out."""
    label_col = answer.columns[0] if len(answer.columns) > 1 and answer.columns[0] != column else None
    shown = {}
    for row in answer.rows:
        value = row[column]
        if value != SECONDARY_SUPPRESSED:
            shown[str(row[label_col]) if label_col else ""] = value
    return shown


def _labels(answer: SqlAnswer) -> set[str]:
    return {str(row[answer.columns[0]]) for row in answer.rows} - FOLD_LABELS


# ---------------------------------------------------------------------------------------------
# Small groups and their labels
# ---------------------------------------------------------------------------------------------


def test_privacy_smallGroup_labelNeverShownAndFolded(ask_sql: Ask) -> None:
    # Arrange
    sql = "SELECT ward, count(*) AS n, avg(age) AS mean_age FROM unified GROUP BY ward"

    # Act
    answer = ask_sql(sql)

    # Assert: ICU (3 patients) is not named, and no shown number covers it alone
    assert "ICU" not in _labels(answer)
    assert any(str(row["ward"]) in FOLD_LABELS for row in answer.rows)
    assert all(
        value == SECONDARY_SUPPRESSED
        for row in answer.rows
        if row["ward"] in FOLD_LABELS
        for value in list(row.values())[1:]
    )


def test_privacy_smallGroup_complementHiddenSoTotalCannotRecoverIt(ask_sql: Ask) -> None:
    # Arrange: the dataset total (300) is public, so the ward counts must not add up to 297
    sql = "SELECT ward, count(*) AS n FROM unified GROUP BY ward"

    # Act
    shown = _numbers(ask_sql(sql), "n")

    # Assert
    assert N_PATIENTS - sum(shown.values()) not in SMALL


@pytest.mark.parametrize("group_size", [1, 10, 11, 12])
def test_privacy_groupAtThreshold_shownOnlyFrom11(
    make_sql_catalog: Callable[..., SqlCatalog], ask_sql: Ask, group_size: int
) -> None:
    # Arrange: one group of group_size patients, two large ones, all sizes distinct
    df = synthetic_cohort().with_columns(
        pl.when(pl.int_range(pl.len()) < group_size)
        .then(pl.lit("rare"))
        .when(pl.int_range(pl.len()) < 120)
        .then(pl.lit("common"))
        .otherwise(pl.lit("usual"))
        .alias("ward")
    )
    catalog = make_sql_catalog(df)

    # Act
    answer = ask_sql("SELECT ward, count(*) AS n FROM unified GROUP BY ward", catalog)

    # Assert
    assert ("rare" in _labels(answer)) == (group_size >= 11)
    if group_size >= 11:
        assert _numbers(answer, "n")["rare"] == group_size


def test_privacy_groupByNearUniqueColumn_noSmallGroupNamed(ask_sql: Ask) -> None:
    # Arrange: grouping by a measurement with many values makes every group small
    sql = "SELECT ldl, count(*) AS n FROM unified GROUP BY ldl"
    truth = synthetic_cohort().group_by("ldl").len()

    # Act
    answer = ask_sql(sql)

    # Assert: every named group really holds at least 11 patients
    sizes = {NO_VALUE_LABEL if r["ldl"] is None else str(r["ldl"]): r["len"] for r in truth.to_dicts()}
    assert all(sizes.get(label, 0) >= 11 for label in _labels(answer))


def test_privacy_groupByAge_everyNamedAgeHasAtLeast11(ask_sql: Ask) -> None:
    # Arrange
    sql = "SELECT age // 10 * 10 AS decade, count(*) AS n FROM unified GROUP BY decade ORDER BY decade"
    bands = synthetic_cohort().group_by(pl.col("age") // 10 * 10).len()
    sizes = {str(r["age"]): r["len"] for r in bands.to_dicts()}

    # Act
    answer = ask_sql(sql)

    # Assert
    assert _labels(answer)
    assert all(sizes[label] >= 11 for label in _labels(answer))


# ---------------------------------------------------------------------------------------------
# Differencing: subtracting one shown number from another
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("narrow", "wide"),
    [
        ("SELECT count(*) AS n FROM unified WHERE age > 18", "SELECT count(*) AS n FROM unified"),
        ("SELECT count(*) AS n FROM unified WHERE ward <> 'ICU'", "SELECT count(*) AS n FROM unified"),
        (
            "SELECT sex, count(*) AS n FROM unified WHERE age > 18 GROUP BY sex",
            "SELECT sex, count(*) AS n FROM unified GROUP BY sex",
        ),
        (
            "SELECT sex, count_if(died = 1) AS n FROM unified WHERE ward <> 'ICU' GROUP BY sex",
            "SELECT sex, count_if(died = 1) AS n FROM unified GROUP BY sex",
        ),
        (
            "SELECT ward, count(*) AS n FROM unified WHERE age > 18 GROUP BY ward",
            "SELECT ward, count(*) AS n FROM unified GROUP BY ward",
        ),
    ],
)
def test_privacy_sameQueryWithoutWhere_differenceNeverSmall(ask_sql: Ask, narrow: str, wide: str) -> None:
    # Arrange: the attacker runs the query with and without its WHERE clause

    # Act
    with_where, without = _numbers(ask_sql(narrow), "n"), _numbers(ask_sql(wide), "n")

    # Assert
    for label in with_where.keys() & without.keys():
        assert without[label] - with_where[label] not in SMALL


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT count(*) AS a, count(*) FILTER (WHERE age > 18) AS b FROM unified",
        "SELECT count(*) AS a, count_if(ward <> 'ICU') AS b FROM unified",
        "SELECT count(*) AS a, count(ldl) AS b FROM unified WHERE ward <> 'ICU'",
        "SELECT sex, count(*) AS a, count_if(age > 18) AS b FROM unified GROUP BY sex",
    ],
)
def test_privacy_twoMeasuresInOneAnswer_differenceNeverSmall(ask_sql: Ask, sql: str) -> None:
    # Arrange: a FILTER or count_if that leaves out a few patients, next to the full count

    # Act
    answer = ask_sql(sql)

    # Assert
    a, b = _numbers(answer, "a"), _numbers(answer, "b")
    for label in a.keys() & b.keys():
        assert a[label] - b[label] not in SMALL


def test_privacy_arithmeticOverCounts_cannotIsolateSmallGroup(ask_sql: Ask) -> None:
    # Arrange: the difference computed inside the query
    sql = "SELECT count(*) - count_if(ward <> 'ICU') AS icu FROM unified"

    # Act
    answer = ask_sql(sql)

    # Assert
    assert answer.rows[0]["icu"] == SECONDARY_SUPPRESSED


def test_privacy_valueDifferenceViaFilter_cannotRecoverSmallGroupSum(ask_sql: Ask) -> None:
    # Arrange: sum of age over everyone, and over everyone outside ICU (3 patients)
    sql = "SELECT sum(age) AS s, sum(age) FILTER (WHERE ward <> 'ICU') AS s2 FROM unified"

    # Act
    row = ask_sql(sql).rows[0]

    # Assert: never both, or s - s2 is the three ICU patients' ages
    assert SECONDARY_SUPPRESSED in (row["s"], row["s2"])


def test_privacy_valueDifferenceViaWhere_cannotRecoverSmallGroupMean(ask_sql: Ask) -> None:
    # Arrange: the mean with and without the 10 patients aged 18
    narrow = ask_sql("SELECT sum(ldl) AS s FROM unified WHERE age > 18").rows[0]["s"]
    wide = ask_sql("SELECT sum(ldl) AS s FROM unified").rows[0]["s"]

    # Act / Assert: at most one of them is shown
    assert SECONDARY_SUPPRESSED in (narrow, wide)


def test_privacy_groupsVersusTotal_hiddenMassNeverSmall(ask_sql: Ask) -> None:
    # Arrange: deaths per ward, against total deaths (a query without GROUP BY)
    grouped = _numbers(ask_sql("SELECT ward, count_if(died = 1) AS n FROM unified GROUP BY ward"), "n")
    total = _numbers(ask_sql("SELECT count_if(died = 1) AS n FROM unified"), "n")

    # Act
    hidden_mass = total[""] - sum(grouped.values())

    # Assert
    assert hidden_mass not in SMALL


# ---------------------------------------------------------------------------------------------
# Moments, quantiles and coded columns
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT avg(score) AS m, var_samp(score) AS v FROM unified",
        "SELECT avg(score) AS m, stddev_samp(score) AS v, sum(score) AS s FROM unified",
        "SELECT sex, avg(score) AS m, stddev(score) AS v FROM unified GROUP BY sex",
        "SELECT median(score) AS m, quantile_cont(score, 0.9) AS v FROM unified",
    ],
)
def test_privacy_momentsOfCodedColumn_hiddenWhenAValueIsRare(ask_sql: Ask, sql: str) -> None:
    # Arrange: score has 4 values; value 3 is held by 4 patients. Mean and variance together
    # would give every value's count.

    # Act
    answer = ask_sql(sql)

    # Assert: no row shows both moments
    for row in answer.rows:
        assert SECONDARY_SUPPRESSED in (row["m"], row["v"])


def test_privacy_meanOfBinaryColumn_hiddenWhenOutcomeCountSmall(ask_sql: Ask) -> None:
    # Arrange: ICU excluded, deaths among the 10 aged-18 patients (2): the mean would give 2/10
    sql = "SELECT avg(died) AS rate FROM unified WHERE age = 18"

    # Act
    answer = ask_sql(sql)

    # Assert
    assert answer.rows[0]["rate"] == SECONDARY_SUPPRESSED


def test_privacy_meanOfContinuousColumn_shownForLargeGroups(ask_sql: Ask) -> None:
    # Arrange
    truth = synthetic_cohort().group_by("sex").agg(pl.col("age").mean()).sort("sex")

    # Act
    answer = ask_sql("SELECT sex, avg(age) AS m FROM unified GROUP BY sex")

    # Assert: correct where shown
    shown = _numbers(answer, "m")
    assert shown == pytest.approx({r["sex"]: round(r["age"], 4) for r in truth.to_dicts()})


def test_privacy_meanWithFewMissingValues_hidden(ask_sql: Ask) -> None:
    # Arrange: 9 women and 9 men have no LDL. The means of each sex and of everyone weight the
    # groups by how many have a value, so together they would give those counts.
    sql = "SELECT sex, avg(ldl) AS m FROM unified GROUP BY sex"

    # Act
    answer = ask_sql(sql)

    # Assert
    assert _numbers(answer, "m") == {}


# ---------------------------------------------------------------------------------------------
# Ranking, ordering and limits
# ---------------------------------------------------------------------------------------------


def test_privacy_orderAscendingWithLimit_neverPicksTheHiddenSmallestGroup(ask_sql: Ask) -> None:
    # Arrange: the smallest ward is ICU (3); ORDER BY n LIMIT 1 would point at it
    sql = "SELECT ward, count(*) AS n FROM unified GROUP BY ward ORDER BY n ASC LIMIT 1"

    # Act
    answer = ask_sql(sql)

    # Assert
    assert "ICU" not in _labels(answer)
    shown = [row for row in answer.rows if row["ward"] not in FOLD_LABELS]
    assert all(row["n"] != SECONDARY_SUPPRESSED for row in shown[:1]) or not shown


def test_privacy_orderByHiddenValue_hiddenRowsLastWhateverTheirValue(ask_sql: Ask) -> None:
    # Arrange: ward B is hidden as the complement of ICU, whatever its size
    sql = "SELECT ward, count(*) AS n FROM unified GROUP BY ward ORDER BY n DESC"

    # Act
    rows = ask_sql(sql).rows

    # Assert: shown values first, in order; hidden after them
    flags = [row["n"] == SECONDARY_SUPPRESSED for row in rows]
    assert flags == sorted(flags)


# ---------------------------------------------------------------------------------------------
# Listings and grain
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT * FROM unified",
        "SELECT patient_id, age FROM unified",
        "SELECT patient_id, count(*) FROM unified GROUP BY patient_id",
        "SELECT upper(patient_id) AS p, count(*) FROM unified GROUP BY p",
        "SELECT age, ldl FROM unified WHERE ward = 'ICU'",
        "SELECT DISTINCT age FROM unified",
    ],
)
def test_privacy_patientListing_refused(ask_sql: Ask, sql: str) -> None:
    # Arrange: the query (parametrized)

    # Act / Assert
    with pytest.raises(SqlRefusedError):
        ask_sql(sql)


def test_privacy_visitRows_refusedSoRowsAreNeverCountedAsPatients(
    make_sql_catalog: Callable[..., SqlCatalog], ask_sql: Ask
) -> None:
    # Arrange: every patient on two rows
    df = synthetic_cohort()
    catalog = make_sql_catalog(pl.concat([df, df]))

    # Act / Assert
    with pytest.raises(SqlRefusedError, match="visits"):
        ask_sql("SELECT count(*) AS n FROM unified", catalog)


@pytest.mark.parametrize("rows", [5, 12])
def test_privacy_tinyDataset_everyCountHidden(
    make_sql_catalog: Callable[..., SqlCatalog], ask_sql: Ask, rows: int
) -> None:
    # Arrange
    catalog = make_sql_catalog(synthetic_cohort(rows))

    # Act
    answer = ask_sql("SELECT sex, count(*) AS n FROM unified GROUP BY sex", catalog)

    # Assert
    assert _numbers(answer, "n") == {}


def test_privacy_tooManyGroups_refused(ask_sql: Ask) -> None:
    # Arrange: almost one group per patient, over the limit of groups that can be checked
    sql = "SELECT ldl * 1000 + age AS k, count(*) AS n FROM unified GROUP BY k"

    # Act / Assert
    with pytest.raises(SqlRefusedError, match="more than"):
        ask_sql(sql)


# ---------------------------------------------------------------------------------------------
# Edge cases: nulls, zeros, empty filters, merges; and correctness where shown
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("sql", "column", "expected"),
    [
        ("SELECT count_if(ward = 'none') AS n FROM unified", "n", 0),
        ("SELECT count(*) AS n FROM unified WHERE age > 1000", "n", 0),
        ("SELECT count(*) AS n FROM unified", "n", N_PATIENTS),
        ("SELECT count(DISTINCT patient_id) AS n FROM unified", "n", N_PATIENTS),
        ("SELECT avg(ldl) AS n FROM unified WHERE age > 1000", "n", None),
    ],
)
def test_privacy_zeroAndEmpty_shownAsTheyAre(ask_sql: Ask, sql: str, column: str, expected: Any) -> None:
    # Arrange: zeros reveal no patient, so they are shown

    # Act
    value = ask_sql(sql).rows[0][column]

    # Assert
    assert value == expected


def test_privacy_nullGroup_labelledAndChecked(make_sql_catalog: Callable[..., SqlCatalog], ask_sql: Ask) -> None:
    # Arrange: sex missing for 30 patients
    df = synthetic_cohort().with_columns(
        pl.when(pl.int_range(pl.len()) >= 270).then(None).otherwise(pl.col("sex")).alias("sex")
    )
    catalog = make_sql_catalog(df)

    # Act
    answer = ask_sql("SELECT sex, count(*) AS n FROM unified GROUP BY sex", catalog)

    # Assert
    assert _numbers(answer, "n") == {"F": 135, "M": 135, "(no value)": 30}


def test_privacy_counts_matchIndependentComputation(ask_sql: Ask) -> None:
    # Arrange
    truth = synthetic_cohort().group_by("sex").agg(pl.len().alias("n"), (pl.col("age") > 40).sum().alias("old"))

    # Act
    answer = ask_sql("SELECT sex, count(*) AS n, count_if(age > 40) AS old FROM unified GROUP BY sex")

    # Assert
    expected = pl.DataFrame([{"sex": r["sex"], "n": r["n"], "old": r["old"]} for r in answer.rows]).sort("sex")
    pl.testing.assert_frame_equal(expected, truth.sort("sex"), check_dtypes=False)


def test_privacy_whereAndFilter_sameAnswer(ask_sql: Ask) -> None:
    # Arrange
    with_where = ask_sql("SELECT sex, count(*) AS n FROM unified WHERE died = 1 GROUP BY sex")
    with_filter = ask_sql("SELECT sex, count(*) FILTER (WHERE died = 1) AS n FROM unified GROUP BY sex")

    # Act
    a, b = _numbers(with_where, "n"), _numbers(with_filter, "n")

    # Assert: both hide the men's extra death the same way, or show the same numbers
    for label in a.keys() & b.keys():
        assert a[label] == b[label]


def test_privacy_confirmedLevelMerge_appliedInTheView(
    make_sql_catalog: Callable[..., SqlCatalog], ask_sql: Ask
) -> None:
    # Arrange: "M" and "Male" confirmed as one level
    df = synthetic_cohort().with_columns(
        pl.when((pl.col("sex") == "M") & (pl.int_range(pl.len()) % 4 == 1))
        .then(pl.lit("Male"))
        .otherwise(pl.col("sex"))
        .alias("sex")
    )
    merge = LevelMapping(column="sex", spellings=("M", "Male"), canonical="M")
    catalog = make_sql_catalog(df, levels=(merge,))

    # Act
    answer = ask_sql("SELECT sex, count(*) AS n FROM unified GROUP BY sex", catalog)

    # Assert
    assert _numbers(answer, "n") == {"F": 150, "M": 150}


def test_privacy_unitIsRows_whenNoIdentifier(make_sql_catalog: Callable[..., SqlCatalog], ask_sql: Ask) -> None:
    # Arrange
    catalog = make_sql_catalog(synthetic_cohort().drop("patient_id"), id_column=None)

    # Act
    answer = ask_sql("SELECT count(*) AS n FROM unified", catalog)

    # Assert
    assert answer.unit == "rows"
    assert answer.rows[0]["n"] == N_PATIENTS


# ---------------------------------------------------------------------------------------------
# Errors as an oracle: a failure that depends on the data answers a yes/no question
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT count(*) AS n FROM unified WHERE CASE WHEN ward = 'ICU' THEN CAST(ward AS INT) ELSE 1 END = 1",
        "SELECT count(*) FILTER (WHERE CASE WHEN age = 18 THEN age + 9223372036854775807 ELSE 0 END = 0) AS n "
        "FROM unified",
        "SELECT CASE WHEN ward = 'ICU' THEN CAST(sex AS INT) ELSE 0 END AS k, count(*) AS n FROM unified GROUP BY k",
        "SELECT count_if(ward = 'ICU') * 4611686018427387904 AS n FROM unified",
        "SELECT round(count_if(ward = 'ICU') * 1e308 * 10, 2) AS n FROM unified",
        "SELECT sum(age) FILTER (WHERE CAST(CASE WHEN ward = 'ICU' THEN 'x' ELSE '1' END AS INT) = 1) AS n "
        "FROM unified",
    ],
)
@pytest.mark.parametrize("icu_patients", [0, 3])
def test_privacy_dataDependentError_neverFailsTheQuery(
    make_sql_catalog: Callable[..., SqlCatalog], ask_sql: Ask, sql: str, icu_patients: int
) -> None:
    # Arrange: with and without the patients the error would point at
    df = synthetic_cohort()
    if icu_patients == 0:
        df = df.with_columns(pl.when(pl.col("ward") == "ICU").then(pl.lit("A")).otherwise(pl.col("ward")).alias("ward"))
    catalog = make_sql_catalog(df)

    # Act: no exception, whichever dataset
    answer = ask_sql(sql, catalog)

    # Assert
    assert answer.rows


def test_privacy_castAroundSummary_refusedBecauseItCanFail(ask_sql: Ask) -> None:
    # Arrange: CAST(1e300 AS INT) fails, so a cast over a count could fail only for some counts

    # Act / Assert
    with pytest.raises(SqlRefusedError, match="around a summary"):
        ask_sql("SELECT CAST(count_if(ward = 'ICU') * 1e300 AS INT) AS n FROM unified")
