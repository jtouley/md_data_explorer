"""Governed SQL answers: every value as computed, small counts included (ADR015).

Synthetic cohort (``tests/sql/conftest.py``): 300 patients, ward ICU = 3 patients, 10 patients
aged 18, score 3 held by 4 patients. Each answer is checked against the truth computed
independently with Polars.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from datetime import date
from typing import Any

import polars as pl
import polars.testing
import pytest

from clinical_analytics.contracts.declarations import LevelMapping
from clinical_analytics.sql.answer import NO_VALUE_LABEL, PatientRowsUnconfirmedError, SqlAnswer, _ordered
from clinical_analytics.sql.catalog import SqlCatalog
from clinical_analytics.sql.errors import SqlRefusedError
from clinical_analytics.sql.validate import validate_sql
from tests.sql.conftest import ICU_PATIENTS, N_PATIENTS, RARE_SCORE_PATIENTS, YOUNG_PATIENTS, synthetic_cohort

Ask = Callable[..., SqlAnswer]


def _numbers(answer: SqlAnswer, column: str) -> dict[str, Any]:
    """Values of ``column`` by group label ("" when ungrouped)."""
    label_col = answer.columns[0] if len(answer.columns) > 1 and answer.columns[0] != column else None
    return {str(row[label_col]) if label_col else "": row[column] for row in answer.rows}


def _old_age_cohort(old_women: int, old_men: int, n: int = 300) -> pl.DataFrame:
    """Half women; ``old_women`` women and ``old_men`` men are over 65, everyone else is 50."""
    half = n // 2
    return pl.DataFrame(
        {
            "patient_id": [f"P{i:04d}" for i in range(n)],
            "sex": ["F" if i < half else "M" for i in range(n)],
            "age": [70 if i < old_women or half <= i < half + old_men else 50 for i in range(n)],
        }
    )


@pytest.mark.parametrize(
    ("sql", "label", "expected"),
    [
        ("SELECT ward, count(*) AS n FROM unified GROUP BY ward", "ICU", ICU_PATIENTS),
        ("SELECT age, count(*) AS n FROM unified WHERE age < 20 GROUP BY age", "18", YOUNG_PATIENTS),
        ("SELECT score, count(*) AS n FROM unified GROUP BY score", "3", RARE_SCORE_PATIENTS),
        ("SELECT count(*) AS n FROM unified WHERE ward = 'ICU'", "", ICU_PATIENTS),
        ("SELECT count_if(ward = 'ICU') AS n FROM unified", "", ICU_PATIENTS),
        ("SELECT count(*) AS n FROM unified WHERE age > 1000", "", 0),
    ],
)
def test_answer_smallCount_shownWithItsExactValue(ask_sql: Ask, sql: str, label: str, expected: int) -> None:
    # Arrange: a count of 0, or of 1-10 patients (parametrized)

    # Act
    answer = ask_sql(sql)

    # Assert
    assert _numbers(answer, "n")[label] == expected


def test_answer_filtersDifferencedInOneMeasure_showTheDifference(ask_sql: Ask) -> None:
    # Arrange: the ICU men, computed independently
    truth = synthetic_cohort().filter((pl.col("sex") == "M") & (pl.col("ward") == "ICU")).height
    sql = (
        "SELECT count(*) FILTER (WHERE sex = 'M') - count(*) FILTER (WHERE sex = 'M' AND ward <> 'ICU') AS d "
        "FROM unified"
    )

    # Act
    row = ask_sql(sql).rows[0]

    # Assert
    assert row["d"] == truth


@pytest.mark.parametrize("rows", [1, 5, 12])
def test_answer_tinyDataset_everyCountShown(
    make_sql_catalog: Callable[..., SqlCatalog], ask_sql: Ask, rows: int
) -> None:
    # Arrange
    df = synthetic_cohort(rows)
    catalog = make_sql_catalog(df)
    truth = dict(df.group_by("sex").len().iter_rows())

    # Act
    answer = ask_sql("SELECT sex, count(*) AS n FROM unified GROUP BY sex", catalog)

    # Assert
    assert _numbers(answer, "n") == truth


def test_answer_groupedWithWhere_groupsWithNoPatientLeftOut(ask_sql: Ask) -> None:
    # Arrange: every ICU patient is 18, so no ICU patient meets the WHERE
    sql = "SELECT ward, count(*) AS n FROM unified WHERE age > 18 GROUP BY ward"
    truth = dict(synthetic_cohort().filter(pl.col("age") > 18).group_by("ward").len().iter_rows())

    # Act
    answer = ask_sql(sql)

    # Assert
    assert _numbers(answer, "n") == truth
    assert "ICU" not in truth


def test_answer_stepByStepCount_matchesTheData(make_sql_catalog: Callable[..., SqlCatalog], ask_sql: Ask) -> None:
    # Arrange: 300 patients, 150 women, 3 of them over 65
    catalog = make_sql_catalog(_old_age_cohort(old_women=3, old_men=40))

    # Act
    answer = ask_sql("SELECT count(*) AS n FROM unified WHERE sex = 'F' AND age > 65", catalog)

    # Assert
    assert answer.rows[0]["n"] == 3
    assert 'Step by step: 300 patients → 150 with "sex" = \'F\' → 3 with "age" > 65.' in answer.notes


def test_answer_noWhere_noStepByStepCount(ask_sql: Ask) -> None:
    # Arrange / Act
    answer = ask_sql("SELECT count(*) AS n FROM unified")

    # Assert
    assert not any(note.startswith("Step by step") for note in answer.notes)


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT * FROM unified",
        "SELECT patient_id, age FROM unified",
        "SELECT patient_id, count(*) FROM unified GROUP BY patient_id",
        "SELECT DISTINCT age FROM unified",
    ],
)
def test_answer_patientListing_refused(ask_sql: Ask, sql: str) -> None:
    # Arrange: the query (parametrized)

    # Act / Assert
    with pytest.raises(SqlRefusedError):
        ask_sql(sql)


def test_answer_visitRows_refusedSoRowsAreNeverCountedAsPatients(
    make_sql_catalog: Callable[..., SqlCatalog], ask_sql: Ask
) -> None:
    # Arrange: every patient on two rows
    df = synthetic_cohort()
    catalog = make_sql_catalog(pl.concat([df, df]))

    # Act / Assert
    with pytest.raises(SqlRefusedError, match="visits"):
        ask_sql("SELECT count(*) AS n FROM unified", catalog)


def test_answer_tooManyGroups_refused(ask_sql: Ask) -> None:
    # Arrange: almost one group per patient, over the limit of groups shown as a table
    sql = "SELECT ldl * 1000 + age AS k, count(*) AS n FROM unified GROUP BY k"

    # Act / Assert
    with pytest.raises(SqlRefusedError, match="more than"):
        ask_sql(sql)


@pytest.mark.parametrize(
    ("sql", "column", "expected"),
    [
        ("SELECT count_if(ward = 'none') AS n FROM unified", "n", 0),
        ("SELECT count(*) AS n FROM unified", "n", N_PATIENTS),
        ("SELECT count(DISTINCT patient_id) AS n FROM unified", "n", N_PATIENTS),
        ("SELECT avg(ldl) AS n FROM unified WHERE age > 1000", "n", None),
    ],
)
def test_answer_zeroAndEmpty_shownAsTheyAre(ask_sql: Ask, sql: str, column: str, expected: Any) -> None:
    # Arrange: the query (parametrized)

    # Act
    value = ask_sql(sql).rows[0][column]

    # Assert
    assert value == expected


def test_answer_nullGroup_labelled(make_sql_catalog: Callable[..., SqlCatalog], ask_sql: Ask) -> None:
    # Arrange: sex missing for 30 patients
    df = synthetic_cohort().with_columns(
        pl.when(pl.int_range(pl.len()) >= 270).then(None).otherwise(pl.col("sex")).alias("sex")
    )
    catalog = make_sql_catalog(df)

    # Act
    answer = ask_sql("SELECT sex, count(*) AS n FROM unified GROUP BY sex", catalog)

    # Assert
    assert _numbers(answer, "n") == {"F": 135, "M": 135, NO_VALUE_LABEL: 30}


def test_answer_counts_matchIndependentComputation(ask_sql: Ask) -> None:
    # Arrange
    truth = synthetic_cohort().group_by("sex").agg(pl.len().alias("n"), (pl.col("age") > 40).sum().alias("old"))

    # Act
    answer = ask_sql("SELECT sex, count(*) AS n, count_if(age > 40) AS old FROM unified GROUP BY sex")

    # Assert
    shown = pl.DataFrame([{"sex": r["sex"], "n": r["n"], "old": r["old"]} for r in answer.rows]).sort("sex")
    pl.testing.assert_frame_equal(shown, truth.sort("sex"), check_dtypes=False)


def test_answer_whereAndFilter_sameAnswer(ask_sql: Ask) -> None:
    # Arrange
    with_where = ask_sql("SELECT sex, count(*) AS n FROM unified WHERE died = 1 GROUP BY sex")
    with_filter = ask_sql("SELECT sex, count(*) FILTER (WHERE died = 1) AS n FROM unified GROUP BY sex")

    # Act
    a, b = _numbers(with_where, "n"), _numbers(with_filter, "n")

    # Assert
    assert a == b


def test_answer_confirmedLevelMerge_appliedInTheView(make_sql_catalog: Callable[..., SqlCatalog], ask_sql: Ask) -> None:
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


def test_answer_severalLevelMergesOnOneColumn_everyOneApplied(
    make_sql_catalog: Callable[..., SqlCatalog], ask_sql: Ask
) -> None:
    # Arrange: f/female confirmed as F and m/male as M, two merges on one column
    spelled = {0: "F", 1: "M", 2: "female", 3: "male", 4: "f", 5: "m"}
    df = synthetic_cohort().with_columns(
        pl.int_range(pl.len()).mod(6).replace_strict(spelled, return_dtype=pl.String).alias("sex")
    )
    merges = (
        LevelMapping(column="sex", spellings=("f", "female"), canonical="F"),
        LevelMapping(column="sex", spellings=("m", "male"), canonical="M"),
    )
    catalog = make_sql_catalog(df, levels=merges)

    # Act
    answer = ask_sql("SELECT sex, count(*) AS n FROM unified GROUP BY sex", catalog)

    # Assert: the same groups as the question box, whichever merge was confirmed last
    assert _numbers(answer, "n") == {"F": 150, "M": 150}


@pytest.mark.parametrize("confirmed_rows", [None, N_PATIENTS - 1, N_PATIENTS + 1, 0])
def test_answer_noIdentifier_refusedUntilOneRowPerPatientConfirmedOnTheseRows(
    make_sql_catalog: Callable[..., SqlCatalog], ask_sql: Ask, confirmed_rows: int | None
) -> None:
    # Arrange: no ID column, so rows can't be shown to be patients; a confirmation made on
    # another row count (an earlier upload) does not count
    catalog = make_sql_catalog(synthetic_cohort().drop("patient_id"), id_column=None)

    # Act / Assert
    with pytest.raises(PatientRowsUnconfirmedError, match="no patient ID column") as refused:
        ask_sql("SELECT count(*) AS n FROM unified", catalog, confirmed_rows)
    assert refused.value.key is None
    assert refused.value.rows == N_PATIENTS


def test_answer_noIdentifier_confirmedOneRowPerPatient_answeredAsPatients(
    make_sql_catalog: Callable[..., SqlCatalog], ask_sql: Ask
) -> None:
    # Arrange
    catalog = make_sql_catalog(synthetic_cohort().drop("patient_id"), id_column=None)

    # Act
    answer = ask_sql("SELECT ward, count(*) AS n FROM unified GROUP BY ward", catalog, N_PATIENTS)

    # Assert: the small ward shown as it is, and the note says whose word the grain rests on
    assert _numbers(answer, "n")["ICU"] == ICU_PATIENTS
    assert any("as you confirmed" in note for note in answer.notes)


def test_answer_rowsWithoutId_refusedUntilConfirmed(make_sql_catalog: Callable[..., SqlCatalog], ask_sql: Ask) -> None:
    # Arrange: an ID column, but two rows have no ID (they could be the same patient twice)
    df = synthetic_cohort().with_columns(
        pl.when(pl.int_range(pl.len()) < 2).then(None).otherwise(pl.col("patient_id")).alias("patient_id")
    )
    catalog = make_sql_catalog(df)

    # Act / Assert
    with pytest.raises(PatientRowsUnconfirmedError, match="no patient_id") as refused:
        ask_sql("SELECT count(*) AS n FROM unified", catalog)
    assert refused.value.key == "patient_id"
    assert ask_sql("SELECT count(*) AS n FROM unified", catalog, N_PATIENTS).rows[0]["n"] == N_PATIENTS


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
def test_answer_dataDependentError_neverFailsTheQuery(
    make_sql_catalog: Callable[..., SqlCatalog], ask_sql: Ask, sql: str, icu_patients: int
) -> None:
    # Arrange: with and without the patients the error would point at
    df = synthetic_cohort()
    if icu_patients == 0:
        df = df.with_columns(pl.when(pl.col("ward") == "ICU").then(pl.lit("A")).otherwise(pl.col("ward")).alias("ward"))
    catalog = make_sql_catalog(df)

    # Act: no failure, whichever dataset
    answer = ask_sql(sql, catalog)

    # Assert
    assert answer.rows


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT CAST(count_if(ward = 'ICU') * 1e300 AS INT) AS n FROM unified",
        # Reported on #106: BIGINT overflow for 3 ICU patients with 4e18, none with 2e18
        "SELECT TRY_CAST(count(*) FILTER (WHERE ward = 'ICU') AS BIGINT) * TRY_CAST(4e18 AS BIGINT) AS n FROM unified",
        "SELECT TRY_CAST(count_if(ward = 'ICU') AS HUGEINT) * 4e37 AS n FROM unified",
    ],
)
def test_validate_castAroundSummary_refusedBecauseItCanFail(ask_sql: Ask, sql: str) -> None:
    # Arrange: a cast back to an integer type can overflow for some counts and not others

    # Act / Assert
    with pytest.raises(SqlRefusedError, match="around a summary"):
        ask_sql(sql)


@pytest.fixture
def phi_catalog(make_sql_catalog: Callable[..., SqlCatalog]) -> SqlCatalog:
    """The cohort plus PHI columns: a birth date, a zip code (number and text) and a birth date
    stored as text. The contract tags PHI from accepted metadata (none on a synthetic upload): marked here."""
    df = synthetic_cohort().with_columns(
        pl.int_range(pl.len()).alias("i"),
    )
    df = (
        df.with_columns(
            pl.Series("dob", [date(1940 + i % 50, 1 + i % 12, 1 + i % 28) for i in range(df.height)]),
            (10_000 + pl.col("i") // 5).alias("zip"),
            (10_000 + pl.col("i") // 5).cast(pl.String).alias("zip_text"),
        )
        .with_columns(pl.col("dob").cast(pl.String).alias("dob_text"))
        .drop("i")
    )
    base = make_sql_catalog(df)
    phi = {"dob", "zip", "zip_text", "dob_text"}
    return dataclasses.replace(base, columns=tuple(dataclasses.replace(c, phi=c.name in phi) for c in base.columns))


@pytest.mark.parametrize(
    "dimension",
    [
        # Reported: a zip turned into a date, and the exact birth date spread over years
        "year(CAST(CAST(zip AS VARCHAR) || '-01-01' AS DATE))",
        "year(DATE '0001-01-01' + 366 * datediff('day', DATE '1900-01-01', dob))",
        # Casts: to text and back, from a number, to a non-calendar type
        "year(CAST(CAST(dob AS VARCHAR) AS DATE))",
        "year(CAST(zip AS DATE))",
        "year(TRY_CAST(zip AS TIMESTAMP))",
        "month(CAST(CAST(dob AS VARCHAR) AS TIMESTAMP))",
        "CAST(dob AS VARCHAR)",
        "CAST(dob AS DATE)",
        # String building
        "year(CAST(zip_text || '-01-01' AS DATE))",
        "year(CAST(substring(dob_text, 1, 10) AS DATE))",
        "substring(dob_text, 1, 7)",
        "lower(zip_text)",
        "year(dob_text)",
        # Date arithmetic
        "year(dob + 1)",
        "year(dob - 365)",
        "month(dob + 30 * (zip % 12))",
        "datediff('day', DATE '1900-01-01', dob)",
        "datediff('year', dob, DATE '2020-01-01')",
        "date_trunc('month', DATE '2000-01-01' + datediff('day', DATE '1900-01-01', dob))",
        # Nested coarseners whose inner unit is fine
        "year(date_trunc('day', dob))",
        "month(date_trunc('week', dob))",
        "year(date_trunc('year', DATE '0001-01-01' + datediff('day', DATE '1900-01-01', dob)))",
        "extract(year FROM date_trunc('day', dob))",
        "extract(day FROM dob)",
        "extract(epoch FROM dob)",
        "date_trunc('day', dob)",
        "date_trunc('decade', dob + 1)",
        # CASE and other row-level choices
        "year(CASE WHEN zip = 10001 THEN DATE '2001-01-01' ELSE DATE '2002-01-01' END)",
        "CASE WHEN dob = DATE '1980-03-04' THEN 'a' ELSE 'b' END",
        "year(CASE WHEN true THEN dob END)",
        "year(coalesce(dob, DATE '2000-01-01'))",
        "year(greatest(dob, DATE '1900-01-01'))",
        "year(nullif(dob, DATE '1980-03-04'))",
        "CASE WHEN year(dob) > 1960 THEN zip ELSE 0 END",
        # Arithmetic on dates and combined parts
        "year(dob) * 10000 + datediff('day', date_trunc('year', dob), dob)",
        "month(dob) + 0 * zip",
        "year(dob) + zip",
        "zip",
        "zip / 10",
        "dob",
    ],
)
def test_validate_phiInGroupBy_refusedUnlessCoarsenedStraightFromTheColumn(
    phi_catalog: SqlCatalog, dimension: str
) -> None:
    # Arrange
    sql = f"SELECT {dimension} AS g, count(*) AS n FROM unified GROUP BY g"

    # Act / Assert
    with pytest.raises(SqlRefusedError, match="personal information"):
        validate_sql(sql, phi_catalog)


def test_answer_valueAggregates_numbersNotText(make_sql_catalog: Callable[..., SqlCatalog], ask_sql: Ask) -> None:
    # Arrange: sum() of an integer column is HUGEINT in DuckDB; a boolean has no avg there
    df = synthetic_cohort().with_columns((pl.col("died") == 1).alias("dead"))
    catalog = make_sql_catalog(df)
    expected_rate = df.get_column("dead").mean()

    # Act
    row = ask_sql(
        "SELECT sum(age) AS s, avg(age) AS a, avg(dead) AS rate, median(age) AS m, count(*) AS n FROM unified", catalog
    ).rows[0]

    # Assert
    assert row["s"] == df.get_column("age").sum()
    assert all(isinstance(row[k], float) for k in ("s", "a", "rate", "m"))
    assert isinstance(row["n"], int)
    assert row["rate"] == pytest.approx(expected_rate, abs=1e-4)  # shown values are rounded


@pytest.mark.parametrize("engine_order", [["B", "A", "C"], ["C", "B", "A"], ["A", "C", "B"]])
def test_answer_orderTies_brokenByGroupWhateverTheEngineOrder(
    make_sql_catalog: Callable[..., SqlCatalog], engine_order: list[str]
) -> None:
    # Arrange: three groups, two tied on n, arriving in the engine's order
    query = validate_sql(
        "SELECT ward, count(*) AS n FROM unified GROUP BY ward ORDER BY n DESC LIMIT 1", make_sql_catalog()
    )
    values: dict[str, Any] = {"A": 5, "B": 20, "C": 5}
    rows = [{"ward": ward, "n": values[ward]} for ward in engine_order]
    raw = {id(row): row["ward"] for row in rows}

    # Act
    ordered = _ordered(rows, query, raw)

    # Assert: by value, then ties by group, never by arrival
    assert [row["ward"] for row in ordered] == ["B", "A", "C"]
