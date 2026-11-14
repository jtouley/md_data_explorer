"""Static checks: the admitted shape passes, everything else is refused with a reason."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable

import polars as pl
import pytest

from clinical_analytics.sql import config
from clinical_analytics.sql.catalog import ColumnSpec, SqlCatalog
from clinical_analytics.sql.errors import SqlRefusedError
from clinical_analytics.sql.validate import (
    COMBINED_ROWS,
    HAVING_REFUSAL,
    LISTS_PATIENTS,
    MIN_MAX_REFUSAL,
    NOT_IN_GROUP,
    ONE_PATIENT,
    ONE_TABLE,
    validate_sql,
)
from tests.sql.conftest import synthetic_cohort


@pytest.fixture
def catalog(make_sql_catalog: Callable[..., SqlCatalog]) -> SqlCatalog:
    """The synthetic cohort's catalog, plus a PHI date column (the contract tags PHI from
    accepted metadata, which a synthetic upload has none of)."""
    base = make_sql_catalog()
    return dataclasses.replace(base, columns=(*base.columns, ColumnSpec("admit_date", "date", is_id=False, phi=True)))


@pytest.mark.parametrize(
    ("sql", "dimension", "measures"),
    [
        ("SELECT count(*) AS n FROM unified", None, ["n"]),
        ("select SEX, COUNT(*) from UNIFIED group by sex", "sex", ["COUNT(*)"]),
        ("SELECT u.sex AS s, avg(u.age) AS a FROM unified AS u GROUP BY s", "s", ["a"]),
        (
            "SELECT sex, count_if(died = 1) AS d, round(100.0 * count_if(died = 1) / count(*), 1) AS pct "
            "FROM unified GROUP BY 1",
            "sex",
            ["d", "pct"],
        ),
        (
            "SELECT CASE WHEN age < 50 THEN 'young' ELSE 'old' END AS band, median(ldl) AS m "
            "FROM unified GROUP BY band",
            "band",
            ["m"],
        ),
        ("SELECT year(admit_date) AS y, count(*) AS n FROM unified GROUP BY y ORDER BY y DESC LIMIT 5", "y", ["n"]),
        ("SELECT date_trunc('month', admit_date) AS m, count(*) AS n FROM unified GROUP BY 1", "m", ["n"]),
        ("SELECT month(admit_date) AS m, count(*) AS n FROM unified GROUP BY 1", "m", ["n"]),
        ("SELECT quarter(admit_date) AS q, count(*) AS n FROM unified GROUP BY q", "q", ["n"]),
        (
            "SELECT count(DISTINCT patient_id) AS p, quantile_cont(ldl, 0.25) AS q1, stddev_samp(age) AS sd "
            "FROM unified WHERE ward IN ('A', 'B') AND age BETWEEN 30 AND 70",
            None,
            ["p", "q1", "sd"],
        ),
        (
            "SELECT sum(died) FILTER (WHERE sex = 'F') AS f_deaths FROM unified WHERE admit_date IS NOT NULL",
            None,
            ["f_deaths"],
        ),
    ],
)
def test_validate_admittedShape_returnsDimensionAndMeasures(
    catalog: SqlCatalog, sql: str, dimension: str | None, measures: list[str]
) -> None:
    # Arrange: the query (parametrized)

    # Act
    query = validate_sql(sql, catalog)

    # Assert
    assert query.dimension_name == dimension
    assert [m.name for m in query.measures] == measures
    assert query.shown_sql.startswith("SELECT")


@pytest.mark.parametrize(
    ("sql", "reason"),
    [
        # Listings of individual patients
        ("SELECT * FROM unified", LISTS_PATIENTS),
        ("SELECT age FROM unified", LISTS_PATIENTS),
        ("SELECT unified.* FROM unified", LISTS_PATIENTS),
        ("SELECT DISTINCT ward FROM unified", "SELECT DISTINCT"),
        ("SELECT sex, age FROM unified GROUP BY sex", NOT_IN_GROUP),
        ("SELECT patient_id, count(*) FROM unified GROUP BY patient_id", ONE_PATIENT),
        ("SELECT count(*) FROM unified WHERE patient_id = 'P0001'", ONE_PATIENT),
        ("SELECT avg(age) FILTER (WHERE patient_id <> 'P0001') FROM unified", ONE_PATIENT),
        ("SELECT admit_date, count(*) FROM unified GROUP BY admit_date", "personal information"),
        ("SELECT date_trunc('day', admit_date) AS d, count(*) FROM unified GROUP BY d", "personal information"),
        ("SELECT avg(admit_date) FROM unified", "personal information"),
        # Other tables, joins, subqueries, set operations
        ("SELECT count(*) FROM other", "Only the table"),
        ("SELECT count(*) FROM main.unified", "Only the table"),
        ("SELECT count(*) FROM unified a JOIN unified b ON a.patient_id = b.patient_id", ONE_TABLE),
        ("WITH x AS (SELECT * FROM unified) SELECT count(*) FROM x", ONE_TABLE),
        ("SELECT (SELECT count(*) FROM unified) AS n FROM unified", ONE_TABLE),
        ("SELECT count(*) FROM unified WHERE age IN (SELECT age FROM unified)", ONE_TABLE),
        ("SELECT count(*) FROM (SELECT * FROM unified)", ONE_TABLE),
        ("SELECT count(*) FROM unified UNION SELECT count(*) FROM unified", COMBINED_ROWS),
        ("SELECT sex, count(*) FROM unified GROUP BY ROLLUP (sex)", COMBINED_ROWS),
        ("SELECT sex, count(*) FROM unified GROUP BY GROUPING SETS ((sex), ())", COMBINED_ROWS),
        ("SELECT sex, ward, count(*) FROM unified GROUP BY CUBE (sex, ward)", COMBINED_ROWS),
        ("SELECT sex, ward, count(*) FROM unified GROUP BY sex, ward", "one column at a time"),
        ("SELECT sex, count(*) FROM unified GROUP BY ALL", "GROUP BY ALL"),
        # Filtering or ranking groups before suppression
        ("SELECT ward, count(*) FROM unified GROUP BY ward HAVING count(*) > 10", HAVING_REFUSAL),
        (
            "SELECT ward, count(*) FROM unified GROUP BY ward QUALIFY rank() OVER (ORDER BY count(*)) = 1",
            HAVING_REFUSAL,
        ),
        ("SELECT ward, lag(count(*)) OVER (ORDER BY ward) FROM unified GROUP BY ward", HAVING_REFUSAL),
        ("SELECT count(*) OVER () FROM unified", HAVING_REFUSAL),
        # One patient's value, or list-valued aggregates
        ("SELECT min(age) FROM unified", MIN_MAX_REFUSAL),
        ("SELECT sex, max(ldl) FROM unified GROUP BY sex", MIN_MAX_REFUSAL),
        ("SELECT list(age) FROM unified", "not supported"),
        ("SELECT string_agg(patient_id, ',') FROM unified", "not supported"),
        ("SELECT array_agg(age) FROM unified", "not supported"),
        ("SELECT mode(ward) FROM unified", "not supported"),
        ("SELECT arg_max(patient_id, age) FROM unified", "not supported"),
        ("SELECT any_value(patient_id) FROM unified", "not supported"),
        ("SELECT first(age) FROM unified", "not supported"),
        # Encoding a count inside a value aggregate
        ("SELECT avg(1000 * (age > 70)::INT + age / 1e6) FROM unified", "takes one column"),
        ("SELECT sum(CASE WHEN ward = 'ICU' THEN 1 ELSE 0 END) FROM unified", "takes one column"),
        ("SELECT sum(DISTINCT age) FROM unified", "takes one column"),
        ("SELECT count(DISTINCT ward) FROM unified", "patient ID only"),
        ("SELECT count(age, ldl) AS n FROM unified", "count takes one column"),
        ("SELECT count(age, ward, sex) AS n FROM unified", "count takes one column"),
        ("SELECT avg(ward) FROM unified", "not a number"),
        ("SELECT avg(count(*)) FROM unified", "summary of a summary"),
        ("SELECT quantile_cont(age, 0.99) FROM unified", "quantile_cont takes"),
        ("SELECT quantile_cont(age, [0.1, 0.2]) FROM unified", "not supported"),
        # Files, network, settings, functions outside the allowlist
        ("SELECT count(*) FROM read_parquet('/etc/passwd')", "Read from the table"),
        ("SELECT count(*) FROM read_csv_auto('x.csv')", "Read from the table"),
        ("SELECT count(*) FROM glob('/etc/*')", "Read from the table"),
        ("SELECT count(*), getenv('HOME') FROM unified", "not supported"),
        ("SELECT count(*) FROM unified WHERE current_setting('allowed_paths') IS NOT NULL", "not supported"),
        ("SELECT count(*) FROM unified WHERE random() < 0.5", "not supported"),
        ("SELECT count(*) FROM unified WHERE now() IS NOT NULL", "not supported"),
        ("SELECT count(*) FROM unified WHERE regexp_matches(ward, 'I.*')", "not supported"),
        # Not a single SELECT
        ("COPY unified TO '/tmp/out.csv'", "read-only"),
        ("ATTACH '/tmp/x.db' AS x", "read-only"),
        ("INSTALL httpfs", "read-only"),
        ("LOAD httpfs", "read-only"),
        ("SET enable_external_access = true", "read-only"),
        ("PRAGMA database_list", "read-only"),
        ("CREATE TABLE t AS SELECT 1", "read-only"),
        ("INSERT INTO unified VALUES (1)", "read-only"),
        ("DROP VIEW unified", "read-only"),
        ("EXPORT DATABASE '/tmp/x'", "read-only"),
        ("SELECT count(*) FROM unified; SELECT 1", "one SELECT"),
        ("SELECT count(*) FROM unified; COPY unified TO 'x.csv'", "one SELECT"),
        ("", "empty"),
        ("SELEC count(*) FROM unified", "not valid SQL"),
        # Names
        ("SELECT count(*) FROM unified WHERE nope > 1", "no column named nope"),
        ("SELECT count(*) AS __n FROM unified", "reserved"),
        ("SELECT count(*) AS n, count(*) AS N FROM unified", "same name"),
        ("SELECT count(*) FROM unified LIMIT 1 + 1", "whole number"),
        ("SELECT count(*) FROM unified ORDER BY age", "output columns"),
        ("SELECT ward, count(*) FROM unified GROUP BY sex", "GROUP BY column"),
    ],
)
def test_validate_outsideAdmittedShape_refusedWithReason(catalog: SqlCatalog, sql: str, reason: str) -> None:
    # Arrange: the query (parametrized)

    # Act
    with pytest.raises(SqlRefusedError) as refused:
        validate_sql(sql, catalog)

    # Assert
    assert reason in refused.value.reason


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT count(*) FROM unified WHERE " + " OR ".join(f"age = {i}" for i in range(200)),
        "SELECT count(*) FROM unified -- " + "x" * config.MAX_SQL_CHARS,
    ],
)
def test_validate_oversizedQuery_refused(catalog: SqlCatalog, sql: str) -> None:
    # Arrange: the query (parametrized)

    # Act / Assert
    with pytest.raises(SqlRefusedError, match="too long|longer than"):
        validate_sql(sql, catalog)


def test_validate_shownSql_isGeneratedFromTheCheckedTree(catalog: SqlCatalog) -> None:
    # Arrange: odd spacing, and a comment that tries to close itself in the generated SQL
    sql = "select   sex , count(*) as n /* note */ from unified -- */; COPY unified TO 'x.csv'\n group by sex"

    # Act
    query = validate_sql(sql, catalog)

    # Assert: what is shown (and run) is regenerated, never the user's text
    assert "note" not in query.shown_sql
    assert "COPY" not in query.shown_sql
    assert 'GROUP BY\n  "sex"' in query.shown_sql


@pytest.mark.parametrize(
    ("sql", "order"),
    [
        ("SELECT sex, count(*) AS age FROM unified GROUP BY sex ORDER BY age", [("age", False)]),
        ('SELECT sex, count(*) AS "age" FROM unified GROUP BY sex ORDER BY "age" DESC', [("age", True)]),
        ("SELECT ward AS sex, count(*) AS n FROM unified GROUP BY 1 ORDER BY sex, n", [("sex", False), ("n", False)]),
        ("SELECT sex, count(*) AS n FROM unified GROUP BY sex ORDER BY unified.sex", [("sex", False)]),
    ],
)
def test_validate_orderByNameOfAliasAndColumn_resolvesToTheOutputAlias(
    catalog: SqlCatalog, sql: str, order: list[tuple[str, bool]]
) -> None:
    # Arrange: an output alias with the name of a contract column (SQL reads ORDER BY as the alias)

    # Act
    query = validate_sql(sql, catalog)

    # Assert
    assert [(k.column, k.descending) for k in query.order] == order


def test_validate_valueAggregateOfBoolean_readsTrueAsOne(make_sql_catalog: Callable[..., SqlCatalog]) -> None:
    # Arrange
    df = synthetic_cohort().with_columns((pl.col("died") == 1).alias("died"))
    catalog = make_sql_catalog(df)

    # Act
    query = validate_sql("SELECT avg(died) AS rate FROM unified", catalog)

    # Assert: run as avg(CAST(died AS INTEGER)); the shown SQL is the user's
    assert "CAST" in query.aggregates[0].function.sql("duckdb")
    assert 'AVG("died")' in query.shown_sql
