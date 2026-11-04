"""Contract -> Polars DQ compiler: golden results on the synthetic cohort vs independent Polars."""

from __future__ import annotations

from typing import Any

import polars as pl
import pytest
from open_data_contract_standard.model import OpenDataContractStandard

from clinical_analytics.contracts import config
from clinical_analytics.contracts.errors import QualityCheckError
from clinical_analytics.contracts.quality import (
    ROWS_KEY,
    _passes,
    compile_contract,
    measure,
    run_quality,
)
from tests.contracts.conftest import DEXA, FIXED_TS, STATIN


def _by_id(report: Any) -> dict[str, Any]:
    return {c.id: c for c in report.checks}


def test_measure_cohortWithDefects_matchesIndependentPolarsCounts(cohort_with_defects, make_contract):
    # Arrange
    df = cohort_with_defects
    contract = make_contract(df)
    checks = compile_contract(contract, df.collect_schema())
    # Independent computation: group_by for duplicates, filters for the rest.
    expected_dupes = df.group_by("patient_id").len().filter(pl.col("len") > 1).select((pl.col("len") - 1).sum()).item()
    expected = {
        ROWS_KEY: df.height,
        "ldl_missing": df.filter(pl.col("ldl").is_null()).height,
        "patient_id_no_duplicates": expected_dupes,
        "patient_id_not_null": df.filter(pl.col("patient_id").is_null()).height,
        "dexa_result_invalid": df.filter(pl.col(DEXA).is_not_null() & ~pl.col(DEXA).is_in([1, 2, 3])).height,
        "statin_use_invalid": df.filter(pl.col(STATIN).is_not_null() & ~pl.col(STATIN).is_in([1, 2])).height,
        "age_range": df.filter((pl.col("age") < 0) | (pl.col("age") > 122)).height,
        "row_count_min": df.height,
    }

    # Act
    raw = measure(df.lazy(), checks)

    # Assert
    assert {k: raw[k] for k in expected} == expected
    assert expected == {
        ROWS_KEY: 403,
        "ldl_missing": 23,
        "patient_id_no_duplicates": 3,
        "patient_id_not_null": 0,
        "dexa_result_invalid": 13,
        "statin_use_invalid": 0,
        "age_range": 3,
        "row_count_min": 403,
    }


def test_runQuality_cohortWithDefects_statusesAndSuppressedMessages(cohort_with_defects, make_contract):
    # Arrange
    contract = make_contract(cohort_with_defects)

    # Act
    report = run_quality(cohort_with_defects.lazy(), contract)

    # Assert
    checks = _by_id(report)
    assert checks["patient_id_no_duplicates"].status == config.HIDDEN_STATUS  # hidden: fixed status
    assert checks["patient_id_no_duplicates"].count == config.COUNT_HIDDEN
    assert checks["patient_id_no_duplicates"].percent is None
    assert checks["dexa_result_invalid"].status == "warning"
    assert checks["dexa_result_invalid"].count == 13
    assert checks["dexa_result_invalid"].message == "DEXA result: 13 values are not in the codebook (1, 2, 3)"
    assert checks["age_range"].message == "age: some values outside 0–122 years; count hidden to protect privacy"
    assert checks["ldl_missing"].status == "warning"
    assert checks["ldl_missing"].message == "ldl: 5.7% missing"
    assert checks["statin_use_invalid"].status == "passed"
    assert (report.failed, report.review, report.not_run) == (0, 2, 0)
    assert "patient_id: some repeated values; count hidden to protect privacy" in report.summary


def test_runQuality_cleanSyntheticCohort_onlyLdlMissingnessFlagged(load_synthetic_dataset, make_contract):
    # Arrange: the committed cohort has 23/400 missing LDL (5.75%) and no other defect.
    df = load_synthetic_dataset("cohort")

    # Act
    report = run_quality(df.lazy(), make_contract(df))

    # Assert
    assert report.summary == ["ldl: 5.8% missing"]
    assert (report.failed, report.warnings) == (0, 1)
    assert report.row_count == 400


@pytest.mark.parametrize(
    ("ldl", "expected_count", "expected_percent", "expected_status"),
    [
        ([100.0] * 20, 0, 0.0, "passed"),
        ([None] * 1 + [100.0] * 19, config.COUNT_HIDDEN, None, config.HIDDEN_STATUS),
        ([None] * 10 + [100.0] * 10, config.COUNT_HIDDEN, None, config.HIDDEN_STATUS),
        ([None] * 11 + [100.0] * 9, config.COUNT_HIDDEN, None, config.HIDDEN_STATUS),
        ([None] * 11 + [100.0] * 11, 11, 50.0, "warning"),
        ([None] * 20, 20, 100.0, "warning"),
    ],
    ids=[
        "none_missing",
        "one_missing_5pct_boundary_passes",
        "boundary_10_suppressed",
        "count_11_complement_9_suppressed",
        "count_and_complement_11_shown",
        "all_null",
    ],
)
def test_runQuality_missingCounts_suppressedBelowMinCell(
    make_contract, ldl, expected_count, expected_percent, expected_status
):
    # Arrange
    df = pl.DataFrame({"patient_id": [f"P{i}" for i in range(len(ldl))], "ldl": pl.Series(ldl, dtype=pl.Float64)})

    # Act
    check = _by_id(run_quality(df.lazy(), make_contract(df)))["ldl_missing"]

    # Assert
    assert (check.count, check.percent, check.status) == (expected_count, expected_percent, expected_status)


@pytest.mark.parametrize(
    ("rows", "expected_status", "expected_row_count"),
    [
        (0, "warning", 0),
        (5, config.HIDDEN_STATUS, config.COUNT_HIDDEN),
        (10, config.HIDDEN_STATUS, config.COUNT_HIDDEN),
        (12, config.HIDDEN_STATUS, config.COUNT_HIDDEN),
        (13, "passed", 13),
    ],
    ids=["empty", "small", "boundary_10", "boundary_12_all_hidden", "boundary_13_shown"],
)
def test_runQuality_rowCount_warnsBelowSmallCellThreshold(make_contract, rows, expected_status, expected_row_count):
    # Arrange
    df = pl.DataFrame(
        {"patient_id": [f"P{i}" for i in range(rows)], "age": list(range(rows))},
        schema_overrides={
            "patient_id": pl.String,
            "age": pl.Int64,
        },
    )

    # Act
    report = run_quality(df.lazy(), make_contract(df))

    # Assert
    assert _by_id(report)["row_count_min"].status == expected_status
    assert report.row_count == expected_row_count


def test_measure_nullAndDuplicateIds_nullsNotCountedAsDuplicates(make_contract):
    # Arrange
    df = pl.DataFrame({"patient_id": ["A", "A", None, None, "B"], "age": [1, 2, 3, 4, 5]})
    checks = compile_contract(make_contract(df), df.collect_schema())

    # Act
    raw = measure(df.lazy(), checks)

    # Assert
    assert (raw["patient_id_not_null"], raw["patient_id_no_duplicates"]) == (2, 1)


def test_measure_floatCodedColumn_codebookComparedNumerically(make_contract):
    # Arrange: codes read as floats (e.g. Excel) still match integer codebook keys.
    df = pl.DataFrame({"patient_id": ["A", "B", "C"], STATIN: [1.0, 2.0, 3.0]})
    checks = compile_contract(make_contract(df), df.collect_schema())

    # Act
    raw = measure(df.lazy(), checks)

    # Assert
    assert raw["statin_use_invalid"] == 1


def test_measure_textCodedColumn_codebookComparedAsText(make_contract):
    # Arrange
    df = pl.DataFrame({"patient_id": ["A", "B", "C"], STATIN: ["1", "2", "x"]})
    checks = compile_contract(make_contract(df), df.collect_schema())

    # Act
    raw = measure(df.lazy(), checks)

    # Assert
    assert raw["statin_use_invalid"] == 1


def _contract_with(rules: list[dict[str, Any]], props: list[dict[str, Any]] | None = None) -> OpenDataContractStandard:
    return OpenDataContractStandard.model_validate(
        {
            "apiVersion": "v3.2.0",
            "kind": "DataContract",
            "id": "u1",
            "version": "1.0.0",
            "status": "active",
            "schema": [{"name": "t", "quality": rules, "properties": props or []}],
        }
    )


@pytest.mark.parametrize(
    ("rule", "reason"),
    [
        ({"id": "q", "type": "sql", "query": "SELECT 1", "mustBe": 0}, "'sql' rules are not run locally"),
        ({"id": "q", "type": "library", "metric": "missingValues", "mustBe": 0}, "not supported at table level"),
        (
            {
                "id": "q",
                "type": "library",
                "metric": "duplicateValues",
                "mustBe": 0,
                "arguments": {"properties": ["x"]},
            },
            "duplicate check names no known columns",
        ),
    ],
    ids=["sql", "missing_values", "duplicate_unknown_column"],
)
def test_runQuality_unsupportedRule_reportedNotRun(rule, reason):
    # Arrange
    df = pl.DataFrame({"a": [1, 2]})

    # Act
    report = run_quality(df.lazy(), _contract_with([rule]))

    # Assert
    assert report.checks[0].status == "not_run"
    assert reason in report.checks[0].message
    assert report.not_run == 1


@pytest.mark.parametrize(
    ("prop", "reason"),
    [
        ({"name": "gone", "quality": [{"id": "g", "metric": "nullValues", "mustBe": 0}]}, "column not in the data"),
        ({"name": "a", "quality": [{"id": "g", "metric": "invalidValues", "mustBe": 0}]}, "no validValues given"),
        (
            {
                "name": "a",
                "quality": [{"id": "g", "metric": "invalidValues", "mustBe": 0, "arguments": {"validValues": ["x"]}}],
            },
            "codebook values do not match the column type",
        ),
        ({"name": "s", "logicalTypeOptions": {"minimum": 0, "maximum": 1}}, "column is not numeric"),
    ],
    ids=["missing_column", "no_valid_values", "type_mismatch", "range_on_text"],
)
def test_runQuality_propertyRuleCannotRun_reportedNotRun(prop, reason):
    # Arrange
    df = pl.DataFrame({"a": [1, 2], "s": ["x", "y"]})

    # Act
    report = run_quality(df.lazy(), _contract_with([], [prop]))

    # Assert
    assert reason in report.checks[0].message


def test_runQuality_objectDuplicateOnColumnPair_countsRepeatedCombinations():
    # Arrange
    df = pl.DataFrame({"a": [1, 1, 1], "b": ["x", "x", "y"]})
    rule = {"id": "pair", "metric": "duplicateValues", "mustBe": 0, "arguments": {"properties": ["a", "b"]}}

    # Act
    report = run_quality(df.lazy(), _contract_with([rule], [{"name": "a"}, {"name": "b"}]))

    # Assert
    assert report.checks[0].count == config.COUNT_HIDDEN
    assert report.checks[0].status == config.HIDDEN_STATUS


@pytest.mark.parametrize(
    ("contract", "match"),
    [
        (
            _contract_with(
                [{"id": "x", "metric": "rowCount", "mustBe": 1}, {"id": "x", "metric": "rowCount", "mustBe": 1}]
            ),
            "not unique",
        ),
        (OpenDataContractStandard(apiVersion="v3.2.0", kind="DataContract", id="u", version="1"), "no schema object"),
    ],
    ids=["duplicate_ids", "no_schema"],
)
def test_compileContract_malformedContract_raisesQualityCheckError(contract, match):
    # Arrange
    schema = pl.Schema({"a": pl.Int64})

    # Act / Assert
    with pytest.raises(QualityCheckError, match=match):
        compile_contract(contract, schema)


@pytest.mark.parametrize(
    ("rule", "observed", "expected"),
    [
        ({"mustBe": 0}, 0, True),
        ({"mustNotBe": 0}, 0, False),
        ({"mustBeGreaterThan": 1}, 1, False),
        ({"mustBeGreaterOrEqualTo": 1}, 1, True),
        ({"mustBeLessThan": 1}, 1, False),
        ({"mustBeLessOrEqualTo": 5.0}, 5.0, True),
        ({"mustBeBetween": [1, 3]}, 3, True),
        ({"mustNotBeBetween": [1, 3]}, 2, False),
    ],
    ids=["eq", "ne", "gt", "ge", "lt", "le_boundary", "between_boundary", "not_between"],
)
def test_passes_odcsOperators_compareAsDocumented(rule, observed, expected):
    # Act / Assert
    assert _passes(rule, observed) is expected


def test_passes_ruleWithoutOperator_raisesQualityCheckError():
    # Act / Assert
    with pytest.raises(QualityCheckError, match="no comparison operator"):
        _passes({"id": "r"}, 1)


def test_runQuality_manyFindings_summaryCappedWithRemainder(make_contract):
    # Arrange: 20 rows (row count shown), one missing value in each of 12 columns -> 12 notable lines.
    data: dict[str, list[Any]] = {"patient_id": [f"P{i}" for i in range(20)]}
    data.update({f"lab{c}": [None if i == 0 else 1.0 for i in range(20)] for c in range(12)})
    df = pl.DataFrame(data)

    # Act
    report = run_quality(df.lazy(), make_contract(df))

    # Assert
    assert len(report.summary) == config.SUMMARY_MAX_LINES
    assert report.summary[-1] == f"…and {12 - config.SUMMARY_MAX_LINES + 1} more"


@pytest.mark.parametrize(
    ("n_missing", "expected_count", "expected_percent"),
    [
        (395, config.COUNT_HIDDEN, None),
        (390, config.COUNT_HIDDEN, None),
        (389, 389, 97.2),
        (400, 400, 100.0),
        (5, config.COUNT_HIDDEN, None),
        (11, 11, 2.8),
    ],
    ids=[
        "complement_5_hidden",
        "complement_10_boundary_hidden",
        "complement_11_shown",
        "complement_0_all_missing_shown",
        "count_small_hidden",
        "count_11_shown",
    ],
)
def test_runQuality_missingNearlyAll_complementSuppressed(make_contract, n_missing, expected_count, expected_percent):
    # Arrange: 400 rows; a visible row count would reveal rows - count when that is small.
    df = pl.DataFrame(
        {
            "patient_id": [f"P{i}" for i in range(400)],
            "rare_marker": pl.Series([None] * n_missing + [1.0] * (400 - n_missing), dtype=pl.Float64),
        }
    )

    # Act
    report = run_quality(df.lazy(), make_contract(df))

    # Assert
    check = _by_id(report)["rare_marker_missing"]
    assert (check.count, check.percent) == (expected_count, expected_percent)
    assert report.row_count == 400
    shown_text = " ".join([check.message, *report.summary])
    if expected_percent is None:
        assert "%" not in check.message
        assert str(400 - n_missing) + " " not in shown_text and str(n_missing) not in shown_text


def test_runQuality_nanBmi_countedMissingNotOutOfRange(make_contract):
    # Arrange: 30 rows, 12 NaN BMI and one 95 (above the 90 bound).
    bmi = [float("nan")] * 12 + [95.0] + [25.0] * 17
    df = pl.DataFrame({"patient_id": [f"P{i}" for i in range(30)], "bmi": bmi})

    # Act
    checks = _by_id(run_quality(df.lazy(), make_contract(df)))

    # Assert
    assert (checks["bmi_missing"].count, checks["bmi_missing"].percent) == (12, 40.0)
    assert (checks["bmi_range"].count, checks["bmi_range"].status) == (config.COUNT_HIDDEN, config.HIDDEN_STATUS)
    assert "90" in checks["bmi_range"].message


def test_runQuality_noIdColumn_idChecksReportedNotRun(make_contract):
    # Arrange
    df = pl.DataFrame({"patient_id": [f"P{i}" for i in range(20)], "x": list(range(20))})

    # Act
    report = run_quality(df.lazy(), make_contract(df, id_column=None))

    # Assert
    not_run = {c.id: c.message for c in report.checks if c.status == "not_run"}
    assert not_run == {
        config.ID_NOT_NULL_RULE_ID: f"Identifier: not checked ({config.NO_ID_COLUMN_REASON})",
        config.ID_UNIQUE_RULE_ID: f"Identifier: not checked ({config.NO_ID_COLUMN_REASON})",
    }
    assert f"Identifier: not checked ({config.NO_ID_COLUMN_REASON})" in report.summary


@pytest.mark.parametrize("column", ["age", "bmi"])
def test_runQuality_plausibilityColumnLoadedAsText_rangeReportedNotRun(make_contract, column):
    # Arrange
    df = pl.DataFrame({"patient_id": [f"P{i}" for i in range(20)], column: ["40"] * 19 + ["unknown"]})

    # Act
    checks = _by_id(run_quality(df.lazy(), make_contract(df)))

    # Assert
    assert checks[f"{column}_range"].status == "not_run"
    assert checks[f"{column}_range"].message.endswith("(column is not numeric)")


_HIDDEN = config.COUNT_HIDDEN


def _column_cohort(n_missing: int, n_bad: int, rows: int = 400) -> pl.DataFrame:
    """``rows`` patients; the Statin column and BMI each: missing, then bad (code 7 / BMI 500), then valid."""
    n_valid = rows - n_missing - n_bad
    return pl.DataFrame(
        {
            "patient_id": [f"P{i}" for i in range(rows)],
            STATIN: pl.Series([None] * n_missing + [7] * n_bad + [1] * n_valid, dtype=pl.Int64),
            "bmi": pl.Series([None] * n_missing + [500.0] * n_bad + [25.0] * n_valid, dtype=pl.Float64),
        }
    )


@pytest.mark.parametrize(
    ("n_missing", "n_bad", "expected_missing", "expected_bad"),
    [
        (380, 15, 380, _HIDDEN),
        (380, 12, 380, _HIDDEN),
        (380, 10, 380, config.COUNT_HIDDEN),
        (379, 11, 379, _HIDDEN),
        (378, 11, 378, 11),
        (380, 9, 380, config.COUNT_HIDDEN),
    ],
    ids=[
        "reviewer_statin_5_valid",
        "reviewer_bmi_8_in_range",
        "valid_10_and_bad_10",
        "valid_10_boundary_bad_hidden",
        "valid_11_boundary_all_shown",
        "bad_small_valid_11",
    ],
)
def test_runQuality_missingPlusBadOnOneColumn_validCountNotRecoverable(
    make_contract, n_missing, n_bad, expected_missing, expected_bad
):
    # Arrange: missing + bad + valid = 400 rows; a visible missing and bad count would give valid.
    df = _column_cohort(n_missing, n_bad)

    # Act
    report = run_quality(df.lazy(), make_contract(df))

    # Assert
    checks = _by_id(report)
    for bad_id in ("statin_use_invalid", "bmi_range"):
        assert (checks["statin_use_missing"].count, checks[bad_id].count) == (expected_missing, expected_bad)
        assert (checks[bad_id].percent is None) is (expected_bad != n_bad)
    text = report.model_dump_json()
    n_valid = 400 - n_missing - n_bad
    if expected_bad != n_bad:
        assert f" {n_bad} " not in " ".join(report.summary) and f'"count":{n_bad}' not in text.replace(" ", "")
    if n_valid < config.MIN_CELL_SIZE:
        assert isinstance(checks["statin_use_invalid"].count, str) or isinstance(
            checks["statin_use_missing"].count, str
        )


_LIMIT = config.MIN_CELL_SIZE


def _scheme_has_small_label() -> bool:
    """Whether the report scheme labels 1..10 counts differently from other hidden counts.

    The scheme is public: a reader who has seen any report knows it, so it is probed once here.
    """
    from clinical_analytics.contracts.builder import ContractInputs, build_contract

    probe = pl.DataFrame({"patient_id": [f"P{i}" for i in range(20)], "x": [None] + [1.0] * 19})
    contract = build_contract(
        probe, ContractInputs(upload_id="u1", dataset_version="v1", dataset_name="p", id_column="patient_id")
    )
    return any(c.count == "<11" for c in run_quality(probe.lazy(), contract).checks)


def _feasible_values(
    report: Any, truth: dict[str, int], lines: list[tuple[str, ...]], shown_ids: dict[str, str], rows: int
) -> dict[str, set[int]]:
    """Brute force: every integer assignment of the unknown cells that a reader of the report could
    not rule out. A reader knows the labelling rule, so if "<11" is ever used for 1..10 counts,
    another label must mean 11 or more; otherwise a suppressed count only means "at least 1".
    Zero is always shown as 0. Rest cells (never shown) are >= 0."""
    import itertools

    checks = _by_id(report)
    small_label = _scheme_has_small_label()

    def domain(cell: str, total: int) -> range:
        if cell in shown_ids:
            count = checks[shown_ids[cell]].count
            if not isinstance(count, str):
                return range(count, count + 1)
            if count == "<11":
                return range(1, _LIMIT)
            return range(_LIMIT if small_label else 1, total + 1)
        return range(0, total + 1)  # never-shown rest cell

    if report.row_count == "<11":
        row_options = range(1, _LIMIT)
    elif isinstance(report.row_count, str):  # hidden; its status is fixed, so it bounds nothing
        row_options = range(1, rows + _LIMIT + 4)
    else:
        row_options = range(rows, rows + 1)
    cells = sorted({c for line in lines for c in line})
    seen: dict[str, set[int]] = {c: set() for c in cells}
    for total in row_options:
        # Columns are independent given the total: enumerate each column (lines sharing cells) alone.
        groups: list[set[str]] = []
        for line in lines:
            merged = set(line)
            for g in [g for g in groups if g & merged]:
                merged |= g
                groups.remove(g)
            groups.append(merged)
        for group in groups:
            names = sorted(group)
            group_lines = [line for line in lines if set(line) <= group]
            fixed = {n: domain(n, total) for n in names}
            free = [n for n in names if len(fixed[n]) > 1]
            # One free cell of each line follows from its total; with one line, derive the last.
            derived = free.pop() if len(group_lines) == 1 and free else None
            for combo in itertools.product(*(fixed[n] for n in free)):
                value = {n: fixed[n][0] for n in names if n not in free and n != derived}
                value.update(zip(free, combo))
                if derived is not None:
                    value[derived] = total - sum(value[c] for c in group_lines[0] if c != derived)
                    if value[derived] not in fixed[derived]:
                        continue
                if all(sum(value[c] for c in line) == total for line in group_lines):
                    for n in names:
                        seen[n].add(value[n])
    return seen


def _assert_small_cells_robust(report: Any, truth: dict[str, int], lines, shown_ids, rows: int, label: str) -> None:
    feasible = _feasible_values(report, truth, lines, shown_ids, rows)
    for cell, value in truth.items():
        if 0 < value < _LIMIT:
            options = feasible[cell]
            assert value in options, (label, cell)
            if not isinstance(report.row_count, str):
                assert max(options) >= _LIMIT, (label, cell, value, sorted(options)[:12])
            else:  # tiny table, row count hidden: at minimum never a single value
                assert len(options) > 1, (label, cell, value, sorted(options))


_STATIN_LINES = [("m", "bad", "rest")]


@pytest.mark.parametrize(
    ("n_missing", "n_bad"),
    [(388, 11), (380, 15), (389, 0), (390, 0), (0, 10), (0, 11), (389, 10), (399, 0), (0, 399), (395, 4)],
    ids=[
        "m388_bad11_valid1",
        "m380_bad15_valid5",
        "m389_bad0_valid11",
        "m390_bad0_valid10",
        "m0_bad10",
        "m0_bad11",
        "m389_bad10_valid1",
        "m399_valid1",
        "bad399_valid1",
        "m395_bad4_valid1",
    ],
)
def test_runQuality_reviewerAdv3Cases_oneLabelAndNoSmallCountPinned(make_contract, n_missing, n_bad):
    # Arrange
    df = _column_cohort(n_missing, n_bad)
    n_valid = 400 - n_missing - n_bad

    # Act
    report = run_quality(df.lazy(), make_contract(df))

    # Assert: one label for every suppressed count, and nothing a reader can pin down.
    labels = {c.count for c in report.checks if isinstance(c.count, str)}
    assert labels <= {config.COUNT_HIDDEN}
    assert "<11" not in report.model_dump_json()
    for bad_id in ("statin_use_invalid", "bmi_range"):
        shown = {"m": "statin_use_missing" if bad_id == "statin_use_invalid" else "bmi_missing", "bad": bad_id}
        truth = {"m": n_missing, "bad": n_bad, "rest": n_valid}
        _assert_small_cells_robust(report, truth, _STATIN_LINES, shown, 400, bad_id)


def test_runQuality_idColumnTwelveRowsOneDistinct_distinctCountNotPinned(make_contract):
    # Arrange: reviewer's ID case, 388 missing IDs and 12 rows sharing one ID (one distinct patient).
    df = pl.DataFrame({"patient_id": [None] * 388 + ["S0400"] * 12, "x": list(range(400))})

    # Act
    report = run_quality(df.lazy(), make_contract(df))

    # Assert
    truth = {"m": 388, "bad": 11, "rest": 1}
    shown = {"m": "patient_id_not_null", "bad": "patient_id_no_duplicates"}
    _assert_small_cells_robust(report, truth, _STATIN_LINES, shown, 400, "id")


@pytest.mark.parametrize("rows", [5, 10, 11, 12, 13, 14])
def test_runQuality_tinyTables_everySmallCountRobust(make_contract, rows):
    # Arrange: tables at and just above the threshold, where the row total itself is small.
    df = _column_cohort(1, 1, rows=rows)

    # Act
    report = run_quality(df.lazy(), make_contract(df))

    # Assert
    truth = {"m": 1, "bad": 1, "rest": rows - 2}
    for bad_id, m_id in (("statin_use_invalid", "statin_use_missing"), ("bmi_range", "bmi_missing")):
        _assert_small_cells_robust(report, truth, _STATIN_LINES, {"m": m_id, "bad": bad_id}, rows, bad_id)


def test_runQuality_randomSmallTables_bruteForceNoSmallCountPinned(make_contract):
    # Arrange: seeded random cohorts; each column is a partition of the row total.
    import random

    rng = random.Random(20261103)
    for trial in range(120):
        rows = rng.randint(5, 24)
        ids = [f"P{rng.randint(0, rows)}" if rng.random() > 0.15 else None for _ in range(rows)]
        statin = [rng.choice([None, 1, 2, 7]) for _ in range(rows)]
        bmi = [rng.choice([None, float("nan"), 25.0, 500.0, 5.0]) for _ in range(rows)]
        df = pl.DataFrame({"patient_id": ids, STATIN: pl.Series(statin, dtype=pl.Int64), "bmi": bmi})
        present_ids = df["patient_id"].drop_nulls()
        truth = {
            "id_m": df["patient_id"].null_count(),
            "id_d": present_ids.len() - present_ids.n_unique(),
            "st_m": df[STATIN].null_count(),
            "st_i": df.filter(pl.col(STATIN).is_not_null() & ~pl.col(STATIN).is_in([1, 2])).height,
            "bmi_m": df.filter(pl.col("bmi").is_null() | pl.col("bmi").is_nan()).height,
            "bmi_o": df.filter(pl.col("bmi").is_not_nan() & ~pl.col("bmi").is_between(10, 90)).height,
        }
        lines = [("id_m", "id_d", "id_rest"), ("st_m", "st_i", "st_rest"), ("bmi_m", "bmi_o", "bmi_rest")]
        for m, bad, rest in lines:
            truth[rest] = rows - truth[m] - truth[bad]
        shown_ids = {
            "id_m": "patient_id_not_null",
            "id_d": "patient_id_no_duplicates",
            "st_m": "statin_use_missing",
            "st_i": "statin_use_invalid",
            "bmi_m": "bmi_missing",
            "bmi_o": "bmi_range",
        }

        # Act
        report = run_quality(df.lazy(), make_contract(df))

        # Assert
        checks = _by_id(report)
        assert all(checks[c].count == truth[k] or isinstance(checks[c].count, str) for k, c in shown_ids.items())
        assert all(checks[c].percent is None for c in shown_ids.values() if isinstance(checks[c].count, str))
        _assert_small_cells_robust(report, truth, lines, shown_ids, rows, f"trial {trial}")


def _rest_floor(kind: str, rows: int, cells: tuple[int, ...]) -> int:
    """What a reader knows about the never-shown rest: >= 0, and >= 1 distinct ID if any ID present."""
    return 1 if kind == "id" and rows - cells[0] > 0 else 0


def _full_report_feasible(
    contract: Any, checks: list[Any], raw: dict[str, int], report: Any, cells: tuple[str, ...], kind: str
) -> tuple[list[set[int]], set[int]]:
    """Values of ``cells`` (one column's counts) and of its rest that a reader cannot rule out.

    Every field of the report is a constraint: counts, percents, statuses, totals, messages and
    the summary (so also the UI's "N things to review"). Other columns stay at their true values,
    and so does the row total when it is shown (a hidden one is enumerated too), so the sets are a
    subset of what the reader allows: if a subset already holds a value of 11 or more, so does
    the full set.
    """
    import itertools

    from clinical_analytics.contracts.quality import report_from_counts

    true_rows = raw[ROWS_KEY]
    totals = range(1, true_rows + _LIMIT + 1) if isinstance(report.row_count, str) else [true_rows]
    key = report.model_dump_json(exclude={"run_ts"})
    seen: list[set[int]] = [set() for _ in cells]
    rests: set[int] = set()
    others = {k: v for k, v in raw.items() if k not in cells and k != ROWS_KEY}
    for rows in totals:
        if any(v > rows for v in others.values()):
            continue
        for values in itertools.product(range(rows + 1), repeat=len(cells)):
            rest = rows - sum(values)
            if rest < _rest_floor(kind, rows, values):
                continue
            candidate = {**raw, ROWS_KEY: rows, **dict(zip(cells, values))}
            built = report_from_counts(contract, checks, candidate, run_at=FIXED_TS)
            if built.model_dump_json(exclude={"run_ts"}) != key:
                continue
            for i, v in enumerate(values):
                seen[i].add(v)
            rests.add(rest)
    return seen, rests


def _assert_reader_cannot_pin(df: pl.DataFrame, contract: Any, columns: dict[str, tuple[tuple[str, ...], str]]) -> None:
    checks = compile_contract(contract, df.collect_schema())
    raw = measure(df.lazy(), checks)
    report = run_quality(df.lazy(), contract, run_at=FIXED_TS)
    rows = raw[ROWS_KEY]
    for label, (cells, kind) in columns.items():
        seen, rests = _full_report_feasible(contract, checks, raw, report, cells, kind)
        truths = [raw[c] for c in cells]
        for options, value, name in [*zip(seen, truths, cells), (rests, rows - sum(truths), f"{label} rest")]:
            assert value in options, (label, name)
            if 0 < value < _LIMIT:
                if not isinstance(report.row_count, str):
                    assert max(options) >= _LIMIT, (label, name, value, sorted(options))
                else:  # tiny table, row count hidden: at minimum never a single value
                    assert len(options) > 1, (label, name, value, sorted(options))


def test_runQuality_hiddenMissingWithStatus_twentyRows_countNotPinnedByStatus(make_contract):
    # Arrange: 1 of 20 missing; on 447bf51 "hidden" + "passed" (<= 5%) meant exactly 1.
    df = pl.DataFrame({"patient_id": [f"P{i}" for i in range(20)], "ldl": [None] + [1.0] * 19})
    contract = make_contract(df, created_at=FIXED_TS)

    # Act / Assert
    _assert_reader_cannot_pin(df, contract, {"ldl": (("ldl_missing",), "plain")})


def test_runQuality_hiddenCounts_statusIndependentOfHiddenValue(make_contract):
    # Arrange: two tables differing only in a hidden count (1 vs 2 missing of 20; 5% vs 10%).
    def report_for(n_missing: int) -> Any:
        df = pl.DataFrame(
            {"patient_id": [f"P{i}" for i in range(20)], "ldl": [None] * n_missing + [1.0] * (20 - n_missing)}
        )
        return run_quality(df.lazy(), make_contract(df), run_at=FIXED_TS)

    # Act
    one, two = report_for(1), report_for(2)

    # Assert: the whole report is the same, statuses and totals included.
    assert one.model_dump(exclude={"run_ts"}) == two.model_dump(exclude={"run_ts"})
    assert _by_id(one)["ldl_missing"].status == config.HIDDEN_STATUS


def test_runQuality_randomSmallTables_fullReportBruteForceNoSmallCountPinned(make_contract):
    # Arrange: seeded random cohorts small enough to enumerate every column assignment.
    import random

    rng = random.Random(4112026)
    for _trial in range(30):
        rows = rng.randint(5, 16)
        ids = [f"P{rng.randint(0, rows)}" if rng.random() > 0.15 else None for _ in range(rows)]
        statin = [rng.choice([None, 1, 2, 7]) for _ in range(rows)]
        bmi = [rng.choice([None, float("nan"), 25.0, 500.0, 5.0]) for _ in range(rows)]
        df = pl.DataFrame({"patient_id": ids, STATIN: pl.Series(statin, dtype=pl.Int64), "bmi": bmi})
        contract = make_contract(df, created_at=FIXED_TS)

        # Act / Assert
        _assert_reader_cannot_pin(
            df,
            contract,
            {
                "id": (("patient_id_not_null", "patient_id_no_duplicates"), "id"),
                "statin": (("statin_use_missing", "statin_use_invalid"), "plain"),
                "bmi": (("bmi_missing", "bmi_range"), "plain"),
            },
        )
