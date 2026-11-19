# SPDX-License-Identifier: MIT
"""Counts in data notes, and statistics that are shares, never give back 1-10 patients.

Two leaks an independent review reproduced on main with synthetic data:

1. "How many patients over 110?" answered "Total count: 20 patients. age has 15 values outside
   0–122 years": 5 patients aged 111-122. A note's count was checked against its own line only,
   never against the counts the answer shows.
2. "Average died by sex" (died coded 0/1) showed "F 0.06, n=100": 6 women who died, while the
   share form of the same question was withheld. A mean of a column that behaves like
   categories is a share, and is now withheld as one.
"""

from __future__ import annotations

from typing import Any

import polars as pl
import pytest

from clinical_analytics.analysis.compute import _value_min_cell
from clinical_analytics.analysis.grouping import NO_COUNT_TABLE
from clinical_analytics.contracts.quality import QualityCheckResult
from clinical_analytics.core.answer_checks import Grain, _report_note, missing_key_note
from clinical_analytics.core.query_plan import FilterSpec, QueryPlan
from clinical_analytics.core.result_presentation import (
    AMOUNT,
    SECONDARY_SUPPRESSED,
    _shown_counts,
    counted_note,
    present_analysis,
    text_note,
)
from tests.core.test_answer_data_checks import _ask, _service, _shown

IDS = [f"P{i:03d}" for i in range(200)]


def _count_answer(total: int, notes: list[Any]) -> dict[str, Any]:
    return {"type": "count", "total_count": total, "unit": "patients", "n_filtered_out": 0, "data_notes": notes}


def _grouped_mean(cells: list[Any], *, n: int = 100, n_rows: int | None = None) -> dict[str, Any]:
    """Mean of a 0/1 column in two groups of ``n``; group F has ``cells[0]`` as its fewest
    patients at one value, group M ``cells[1]``."""
    rows = n if n_rows is None else n_rows
    groups = [
        {"group": g, "value": 0.25, "n": n, "n_rows": rows, **({} if cell is None else {"table_min_cell": cell})}
        for g, cell in zip(("F", "M"), cells, strict=True)
    ]
    return {
        "type": "grouped_statistic",
        "statistic": "mean",
        "metric": "died",
        "group_col": "sex",
        "groups": groups,
        "n_used": 2 * n,
        "n_excluded": 2 * (rows - n),
        "n_filtered_out": 0,
    }


def _described(min_cell: Any, n: int = 200) -> dict[str, Any]:
    analysis = {
        "type": "descriptive",
        "focused_variable": "dose",
        "is_numeric": True,
        "mean": 0.42,
        "median": 0.5,
        "std": 0.3,
        "min": 0.0,
        "max": 1.0,
        "non_null_count": n,
        "null_count": 0,
        "n_filtered_out": 0,
    }
    if min_cell is not None:
        analysis["table_min_cell"] = min_cell
    return analysis


# --- 1. Note counts against the answer's shown counts ----------------------------------------


class TestNoteCountAgainstAnswer:
    @pytest.mark.parametrize(
        ("total", "count", "amount"),
        [
            pytest.param(20, 15, "some", id="review_repro_difference_5"),
            pytest.param(30, 20, "some", id="difference_10_withheld"),
            pytest.param(31, 20, "20", id="difference_11_shown"),
            pytest.param(50, 10, "some", id="count_10_withheld"),
            pytest.param(21, 11, "some", id="count_11_difference_10_withheld"),
            pytest.param(150, 11, "11", id="count_11_shown"),
            pytest.param(20, 20, "20", id="same_as_shown_count_adds_nothing"),
        ],
    )
    def test_presentAnalysis_noteCountNearShownCount_isSome(self, total: int, count: int, amount: str) -> None:
        # Arrange: the note's line is 200 values (rest = 200 - count)
        note = counted_note(f"age has {AMOUNT} values outside 0–122 years.", count, 200)

        # Act
        claim = present_analysis(_count_answer(total, [note]))["claim"]

        # Assert
        assert claim == f"Total count: {total} patients. age has {amount} values outside 0–122 years."

    @pytest.mark.parametrize(
        ("rest", "amount"),
        [pytest.param(10, "some", id="rest_10_withheld"), pytest.param(11, "189", id="rest_11_shown")],
    )
    def test_presentAnalysis_noteRestSmall_isSome(self, rest: int, amount: str) -> None:
        # Arrange: the answer shows 500, far from both the note's count and its rest
        note = counted_note(f"x is missing on {AMOUNT} rows.", 200 - rest, 200)

        # Act
        claim = present_analysis(_count_answer(500, [note]))["claim"]

        # Assert
        assert claim == f"Total count: 500 patients. x is missing on {amount} rows."

    def test_presentAnalysis_percentNoteNearShownCount_isSome(self) -> None:
        # Arrange: 15 of 200 missing is 7.5%, which with the row count is 15; the answer shows 20
        note = counted_note(f"Data check: age has {AMOUNT} of its values missing.", 15, 200, "percent")

        # Act
        near = present_analysis(_count_answer(20, [note]))["claim"]
        far = present_analysis(_count_answer(100, [note]))["claim"]

        # Assert
        assert near == "Total count: 20 patients. Data check: age has some of its values missing."
        assert far == "Total count: 100 patients. Data check: age has 7.5% of its values missing."

    def test_presentAnalysis_secondNoteNearFirstNotesCount_isSome(self) -> None:
        # Arrange: the first note writes 40; the second (45) would give back 5 beside it
        first = counted_note(f"a has {AMOUNT} values outside 0–9.", 40, 300)
        second = counted_note(f"b has {AMOUNT} values outside 0–9.", 45, 300)

        # Act
        claim = present_analysis(_count_answer(150, [first, second]))["claim"]

        # Assert
        assert claim == "Total count: 150 patients. a has 40 values outside 0–9. b has some values outside 0–9."

    def test_presentAnalysis_hiddenTotal_noteStillChecked(self) -> None:
        # Arrange: a total of 5 is shown as <11; the note's 15 is checked against 11 too
        note = counted_note(f"age has {AMOUNT} values outside 0–122 years.", 15, 200)

        # Act
        claim = present_analysis(_count_answer(5, [note]))["claim"]

        # Assert
        assert claim == "Total count: <11 patients. age has some values outside 0–122 years."

    @pytest.mark.parametrize(
        "note",
        [
            pytest.param(
                {"text": "x has {amount} rows.", "count": None, "total": 200, "form": "number"}, id="no_count"
            ),
            pytest.param({"text": "x has {amount} rows.", "count": "15", "total": 200, "form": "number"}, id="text"),
            pytest.param({"text": "x has 15 rows.", "count": 15, "total": 200, "form": "number"}, id="no_slot"),
            pytest.param({"text": "x {amount} {amount}", "count": 15, "total": 200, "form": "number"}, id="two"),
            pytest.param({"text": "x has {amount}.", "count": 15, "total": None, "form": "percent"}, id="pct"),
            pytest.param({"text": "x has {amount}.", "count": 15, "total": 200, "form": "words"}, id="form"),
            pytest.param(42, id="not_a_note"),
        ],
    )
    def test_presentAnalysis_malformedCountedNote_isDropped(self, note: Any) -> None:
        # Arrange
        analysis = _count_answer(500, [note])

        # Act
        claim = present_analysis(analysis)["claim"]

        # Assert
        assert claim == "Total count: 500 patients."

    @pytest.mark.parametrize(
        ("note", "kept"),
        [
            pytest.param("Data check: ldl: some values outside 0–400 mg/dL.", True, id="no_small_integer_kept"),
            pytest.param("Data check: x has 15 rows.", False, id="count_near_shown_20_dropped"),
            pytest.param("Data check: sex: some values are not in the codebook (1, 2).", False, id="small_dropped"),
            pytest.param("Data check: x has 60 rows.", True, id="far_integer_kept"),
            pytest.param("", False, id="empty_dropped"),
        ],
    )
    def test_presentAnalysis_plainNote_integersCheckedAsCounts(self, note: str, kept: bool) -> None:
        # Arrange: a plain note should carry no count; any integer in it is checked as one
        analysis = _count_answer(20, [note])

        # Act
        claim = present_analysis(analysis)["claim"]

        # Assert
        assert claim == ("Total count: 20 patients." + (f" {note}" if kept else ""))

    def test_presentAnalysis_implausibleDisclosureNearShownN_isSome(self) -> None:
        # Arrange: mean of 20 values (n=20) of which 15 are implausible: rest 5
        analysis = {
            "type": "descriptive",
            "focused_variable": "age",
            "is_numeric": True,
            "mean": 130.0,
            "median": 140.0,
            "std": 20.0,
            "min": 111.0,
            "max": 170.0,
            "non_null_count": 20,
            "null_count": 0,
            "n_filtered_out": 180,
            "table_min_cell": NO_COUNT_TABLE,
            "implausible": {"column": "age", "count": 15, "line_total": 20, "bounds": "0–122 years"},
        }

        # Act
        claim = present_analysis(analysis)["claim"]

        # Assert
        assert claim is not None
        assert claim.endswith("Includes some values of age outside 0–122 years (likely entry errors).")

    @pytest.mark.parametrize(
        ("text", "total", "form"),
        [
            pytest.param("no slot", 200, "number", id="no_placeholder"),
            pytest.param(f"{AMOUNT} and {AMOUNT}", 200, "number", id="two_placeholders"),
            pytest.param(f"{AMOUNT}", None, "percent", id="percent_without_total"),
            pytest.param(f"{AMOUNT}", 0, "percent", id="percent_of_zero"),
        ],
    )
    def test_countedNote_malformed_raises(self, text: str, total: int | None, form: Any) -> None:
        # Arrange / Act / Assert: a producer's mistake fails loudly, never ships a raw count
        with pytest.raises(ValueError):
            counted_note(text, 15, total, form)

    def test_shownCounts_shareBesideN_givesBothCountsBack(self) -> None:
        # Arrange: 6% of 100 is 6 with the value and 94 without; decimals and percents are not counts
        table = {"columns": ["proportion", "n"], "rows": [{"proportion": 0.06, "n": 100}]}

        # Act
        shown = _shown_counts("6.0% of patients had died=1 (n=100 patients); 95% CI 0.12 to 2020.", table)

        # Assert: the year is read as a count too (fail closed); 1 to 10 is never a shown count
        assert shown == {100, 94, 2020}

    @pytest.mark.parametrize(
        ("missing", "amount"),
        [pytest.param(15, "some", id="15_beside_20_shown"), pytest.param(40, "40", id="40_beside_20_shown")],
    )
    def test_missingKeyNote_againstShownTotal(self, missing: int, amount: str) -> None:
        # Arrange
        note = missing_key_note(Grain(key="patient_id", rows=300, patients=300 - missing, missing_keys=missing), False)

        # Act
        claim = present_analysis(_count_answer(20, [note]))["claim"]

        # Assert
        assert claim == f"Total count: 20 patients. Data check: patient_id is missing on {amount} rows."

    def test_ask_reviewRepro_patientsOver110_implausibleCountNotShown(self, make_semantic_layer: Any) -> None:
        # Arrange: 5 patients aged 115 and 15 aged over 122; 180 aged 40-89
        ages = [40 + i % 50 for i in range(200)]
        ages[:5] = [115] * 5
        ages[5:20] = [150 + i for i in range(15)]
        service = _service(make_semantic_layer, pl.DataFrame({"patient_id": IDS, "age": ages}))
        plan = QueryPlan(
            intent="COUNT",
            filters=[FilterSpec(column="age", operator=">", value=110)],
            scope="filtered",
            confidence=0.99,
        )
        held = _ask(service, plan)

        # Act
        result = _ask(service, plan, confirm_hash=held.plan_hash)

        # Assert: 20 is shown, so 15 would give back the 5 patients aged 111-122
        assert _shown(result) == (
            "ANSWER Total count: 20 patients. "
            "age has some values outside 0–122 years (likely entry errors), included as recorded."
        )

    def test_ask_implausibleCountFarFromAnswer_isShown(self, make_semantic_layer: Any) -> None:
        # Arrange: 15 implausible ages; the answer counts 200 patients
        ages = [40 + i % 50 for i in range(200)]
        ages[:15] = [150 + i for i in range(15)]
        service = _service(make_semantic_layer, pl.DataFrame({"patient_id": IDS, "age": ages}))
        plan = QueryPlan(
            intent="COUNT", filters=[FilterSpec(column="age", operator=">", value=0)], scope="filtered", confidence=0.99
        )
        held = _ask(service, plan)

        # Act
        result = _ask(service, plan, confirm_hash=held.plan_hash)

        # Assert
        assert _shown(result) == (
            "ANSWER Total count: 200 patients. "
            "age has 15 values outside 0–122 years (likely entry errors), included as recorded."
        )


# --- 2. A statistic of a column that behaves like categories is a share ------------------------


class TestShareLikeStatistics:
    @pytest.mark.parametrize(
        ("cells", "shown"),
        [
            pytest.param([6, 40], [False, True], id="review_repro_6_withheld"),
            pytest.param([10, 40], [False, True], id="cell_10_withheld"),
            pytest.param([11, 40], [True, True], id="cell_11_shown"),
            pytest.param([NO_COUNT_TABLE, NO_COUNT_TABLE], [True, True], id="measurement_shown"),
            pytest.param([None, 40], [False, True], id="not_recorded_withheld"),
            pytest.param(["6", 40], [False, True], id="not_a_count_withheld"),
        ],
    )
    def test_presentGroupedStatistic_shareLikeMean_followsShareRule(self, cells: list[Any], shown: list[bool]) -> None:
        # Arrange
        analysis = _grouped_mean(cells)

        # Act
        table = present_analysis(analysis)["table"]

        # Assert
        assert table is not None
        values = [row["mean died"] for row in table["rows"]]
        assert [value != SECONDARY_SUPPRESSED for value in values] == shown
        assert [row["n"] for row in table["rows"]] == [100, 100]

    def test_presentGroupedStatistic_reviewRepro_claimNamesNoShare(self) -> None:
        # Arrange
        analysis = _grouped_mean([6, 40])

        # Act
        claim = present_analysis(analysis)["claim"]

        # Assert
        assert claim is not None
        assert claim.startswith("Mean died by sex: F not shown, M 0.25 (n=200). Unadjusted.")
        assert "fewer than 11 patients have one of its values" in claim

    @pytest.mark.parametrize(
        ("cells", "female_missing", "n_shown"),
        [
            pytest.param([40, 40], 5, False, id="share_5_missing_in_group_hides_n"),
            pytest.param([40, 40], 11, True, id="share_11_missing_shown"),
            pytest.param([NO_COUNT_TABLE, NO_COUNT_TABLE], 5, True, id="measurement_unchanged"),
        ],
    )
    def test_presentGroupedStatistic_groupRowsWithoutValue_suppressedWithNForShares(
        self, cells: list[Any], female_missing: int, n_shown: bool
    ) -> None:
        # Arrange: F has `female_missing` rows without a value, M 20; a group's size (a COUNT by
        # group) minus its n gives them back
        analysis = _grouped_mean(cells)
        analysis["groups"][0]["n_rows"] = 100 + female_missing
        analysis["groups"][1]["n_rows"] = 120
        analysis["n_excluded"] = female_missing + 20

        # Act
        table = present_analysis(analysis)["table"]

        # Assert
        assert table is not None
        assert all((row["n"] != SECONDARY_SUPPRESSED) == n_shown for row in table["rows"])

    @pytest.mark.parametrize(
        ("min_cell", "shown"),
        [
            pytest.param(10, False, id="cell_10_withheld"),
            pytest.param(11, True, id="cell_11_shown"),
            pytest.param(None, False, id="not_recorded_withheld"),
            pytest.param(NO_COUNT_TABLE, True, id="measurement_shown"),
        ],
    )
    def test_presentDescriptive_shareLikeColumn_followsShareRule(self, min_cell: Any, shown: bool) -> None:
        # Arrange
        analysis = _described(min_cell)

        # Act
        presentation = present_analysis(analysis)

        # Assert
        assert presentation["table"] is not None
        row = presentation["table"]["rows"][0]
        stats = [row[k] for k in ("mean", "median", "std", "min", "max")]
        assert all((value != SECONDARY_SUPPRESSED) == shown for value in stats)
        assert row["n"] == 200
        if not shown:
            assert presentation["claim"] is not None
            assert presentation["claim"].startswith("Statistics for dose are not shown because fewer than 11")

    def test_presentDescriptive_shareLikeColumnWithNHidden_withheld(self) -> None:
        # Arrange: 5 missing hide n; the column behaves like categories, so its mean is a share
        analysis = {**_described(50, n=195), "null_count": 5}

        # Act
        presentation = present_analysis(analysis)

        # Assert
        assert presentation["claim"] == "Statistics for dose are not shown because n is hidden (n hidden)."

    @pytest.mark.parametrize(
        ("values", "expected"),
        [
            pytest.param([0.0] * 6 + [1.0] * 94, 6, id="two_values"),
            pytest.param([0.0] * 6 + [1.0] * 94 + [None, float("nan")], 6, id="nulls_and_nan_left_out"),
            pytest.param([float(i) for i in range(100)], NO_COUNT_TABLE, id="measurement"),
            pytest.param([None, None], NO_COUNT_TABLE, id="no_values"),
            pytest.param([0.0] * 11 + [0.5] * 11 + [1.0] * 11, 11, id="three_values_11_each"),
        ],
    )
    def test_valueMinCell_values_fewestPatientsAtOneValue(self, values: list[float | None], expected: Any) -> None:
        # Arrange
        series = pl.Series("dose", values, dtype=pl.Float64)

        # Act
        result = _value_min_cell(series)

        # Assert
        assert result == expected

    def test_valueMinCell_visitRows_countsPatients(self) -> None:
        # Arrange: 6 patients with value 1 over 12 visit rows; 50 patients with value 0
        series = pl.Series("died", [1.0] * 12 + [0.0] * 50)
        keys = pl.Series("pid", [f"D{i % 6}" for i in range(12)] + [f"A{i}" for i in range(50)])

        # Act
        result = _value_min_cell(series, keys)

        # Assert: 6 patients, not 12 rows
        assert result == 6

    @pytest.mark.parametrize(
        ("female_deaths", "female_shown"),
        [
            pytest.param(6, False, id="review_repro_6"),
            pytest.param(10, False, id="10"),
            pytest.param(11, True, id="11"),
        ],
    )
    def test_ask_reviewRepro_meanOfZeroOneBySex_followsShareRule(
        self, make_semantic_layer: Any, female_deaths: int, female_shown: bool
    ) -> None:
        # Arrange: 100 F (female_deaths died), 100 M (40 died), died coded 0/1
        died = [1] * female_deaths + [0] * (100 - female_deaths) + [1] * 40 + [0] * 60
        frame = pl.DataFrame({"patient_id": IDS, "sex": ["F"] * 100 + ["M"] * 100, "died": died})
        service = _service(make_semantic_layer, frame)
        plan = QueryPlan(intent="DESCRIBE", metric="died", group_by="sex", confidence=0.99)
        held = _ask(service, plan)

        # Act
        result = _ask(service, plan, confirm_hash=held.plan_hash)

        # Assert: n=100 stays shown; F's mean only when both its counts are 11 or more
        table = present_analysis(result.analysis)["table"]
        assert table["rows"] == [
            {"sex": "F", "mean died": female_deaths / 100 if female_shown else SECONDARY_SUPPRESSED, "n": 100},
            {"sex": "M", "mean died": 0.4, "n": 100},
        ]
        assert ("F not shown" in str(result.claim)) != female_shown

    def test_ask_meanOfZeroOneWithNulls_cellsCountPresentValuesOnly(self, make_semantic_layer: Any) -> None:
        # Arrange: F has 6 deaths, 94 survivors and 15 rows with no value (15 > 10, so n is shown)
        died: list[int | None] = [1] * 6 + [0] * 94 + [None] * 15 + [1] * 40 + [0] * 60
        sex = ["F"] * 115 + ["M"] * 100
        frame = pl.DataFrame({"patient_id": [f"P{i:03d}" for i in range(215)], "sex": sex, "died": died})
        service = _service(make_semantic_layer, frame)
        plan = QueryPlan(intent="DESCRIBE", metric="died", group_by="sex", confidence=0.99)
        held = _ask(service, plan)

        # Act
        result = _ask(service, plan, confirm_hash=held.plan_hash)

        # Assert
        assert [g.get("table_min_cell") for g in result.analysis["groups"]] == [6, 40]
        rows = present_analysis(result.analysis)["table"]["rows"]
        assert rows[0] == {"sex": "F", "mean died": SECONDARY_SUPPRESSED, "n": 100}

    @pytest.mark.parametrize(
        ("deaths", "shown"),
        [pytest.param(6, False, id="6_patients_over_12_visits"), pytest.param(11, True, id="11_patients")],
    )
    def test_ask_visitRows_meanOfZeroOne_countsPatients(
        self, make_semantic_layer: Any, deaths: int, shown: bool
    ) -> None:
        # Arrange: every F patient has two visit rows; deaths F patients died (2 x deaths rows)
        ids = [f"F{i:03d}" for i in range(60)] * 2 + [f"M{i:03d}" for i in range(60)] * 2
        sex = ["F"] * 120 + ["M"] * 120
        died = ([1] * deaths + [0] * (60 - deaths)) * 2 + ([1] * 30 + [0] * 30) * 2
        service = _service(make_semantic_layer, pl.DataFrame({"patient_id": ids, "sex": sex, "died": died}))
        plan = QueryPlan(intent="DESCRIBE", metric="died", group_by="sex", confidence=0.99)
        held = _ask(service, plan)

        # Act
        result = _ask(service, plan, confirm_hash=held.plan_hash)

        # Assert: 6 patients are 12 visit rows, and still withheld
        table = present_analysis(result.analysis)["table"]
        female = next(row for row in table["rows"] if row["sex"] == "F")
        assert (female["mean died"] != SECONDARY_SUPPRESSED) == shown
        assert result.analysis["groups"][0]["table_min_cell"] == deaths

    @pytest.mark.parametrize(
        ("rare", "shown"),
        [pytest.param(10, False, id="value_held_by_10"), pytest.param(11, True, id="value_held_by_11")],
    )
    def test_ask_describeFewValuedMeasure_followsShareRule(
        self, make_semantic_layer: Any, rare: int, shown: bool
    ) -> None:
        # Arrange: a dose of 0, 0.5, 1 or 2 mg (continuous to the classifier); 2 mg held by `rare`
        dose = [0.0] * 60 + [0.5] * 60 + [1.0] * 60 + [2.0] * rare
        frame = pl.DataFrame({"patient_id": [f"P{i:03d}" for i in range(len(dose))], "dose": dose})
        service = _service(make_semantic_layer, frame)
        plan = QueryPlan(intent="DESCRIBE", metric="dose", confidence=0.99)
        held = _ask(service, plan)

        # Act
        result = _ask(service, plan, confirm_hash=held.plan_hash if held.requires_confirmation else None)

        # Assert
        assert result.analysis["table_min_cell"] == rare
        row = present_analysis(result.analysis)["table"]["rows"][0]
        assert (row["mean"] != SECONDARY_SUPPRESSED) == shown


# --- Review round: holds, report notes, text notes, per-group shares, date groups --------------


class TestNoteSourcesAfterReview:
    def test_ask_heldImplausibleQuestion_holdNamesNoCount(self, make_semantic_layer: Any) -> None:
        # Arrange: the hold is followed by the answer once confirmed; its count beside that
        # answer's 20 would give back the 5 patients aged 111-122
        ages = [40 + i % 50 for i in range(200)]
        ages[:5] = [115] * 5
        ages[5:20] = [150 + i for i in range(15)]
        service = _service(make_semantic_layer, pl.DataFrame({"patient_id": IDS, "age": ages}))
        plan = QueryPlan(
            intent="COUNT",
            filters=[FilterSpec(column="age", operator=">", value=110)],
            scope="filtered",
            confidence=0.99,
        )

        # Act
        held = _ask(service, plan)

        # Assert
        assert held.requires_confirmation
        assert "age has some values outside 0–122 years" in str(held.failure_reason)
        assert "15" not in str(held.failure_reason)

    @pytest.mark.parametrize(
        ("check", "expected"),
        [
            pytest.param(
                {"metric": "rowCount", "column": None, "count": 320, "percent": None, "message": "320 rows"},
                "Data check: the number of rows does not meet the dataset's contract.",
                id="row_count_kept_without_number",
            ),
            pytest.param(
                {
                    "metric": "nullValues",
                    "column": "ldl",
                    "count": 40,
                    "percent": 10.0,
                    "message": "ldl: 10.0% missing",
                },
                "Data check: ldl: 10.0% missing.",
                id="percent_far_from_answer_shown",
            ),
            pytest.param(
                {
                    "metric": "range",
                    "column": "age",
                    "count": 15,
                    "percent": 3.8,
                    "message": "age: 15 values outside 0–122",
                },
                "Data check: age: some values outside 0–122.",
                id="count_near_answer_some",
            ),
            pytest.param(
                {
                    "metric": "invalidValues",
                    "column": "Visit 2 code",
                    "count": "hidden",
                    "percent": None,
                    "message": "Visit 2 code: some values are not in the codebook (1, 2)",
                },
                "Data check: Visit 2 code: some values are not in the codebook (1, 2).",
                id="hidden_count_text_note_kept_with_digits",
            ),
        ],
    )
    def test_reportNote_check_checkedAgainstAnswer(self, check: dict[str, Any], expected: str) -> None:
        # Arrange: a report of 400 rows; the answer shows 20
        result = QualityCheckResult(id="r1", severity="warning", status="warning", dimension=None, **check)

        # Act
        note = _report_note(result, 400)
        claim = present_analysis(_count_answer(20, [note]))["claim"]

        # Assert
        assert claim == f"Total count: 20 patients. {expected}"

    def test_reportNote_countNotFoundInMessage_dropped(self) -> None:
        # Arrange
        result = QualityCheckResult(
            id="r1", severity="warning", status="warning", metric="custom", column="x", count=40, message="x: odd"
        )

        # Act / Assert
        assert _report_note(result, 400) is None

    def test_textNote_withPlaceholder_raises(self) -> None:
        # Arrange / Act / Assert
        with pytest.raises(ValueError):
            text_note(f"x has {AMOUNT} rows")

    def test_shownCounts_nBesideExcluded_addsCohort(self) -> None:
        # Arrange / Act
        shown = _shown_counts("Mean ldl was 121.41 (n=206 patients; 13 excluded for missing data).", None)

        # Assert
        assert shown == {206, 13, 219}

    def test_presentAnalysis_implausibleWithoutLine_disclosedAsSome(self) -> None:
        # Arrange
        analysis = {
            **_described(NO_COUNT_TABLE),
            "implausible": {"column": "age", "count": 15, "bounds": "0–122 years"},
        }

        # Act
        claim = str(present_analysis(analysis)["claim"])

        # Assert
        assert claim.endswith("Includes some values of age outside 0–122 years (likely entry errors).")

    def test_ask_shareInOneGroupMeasureInAnother_shareWithheld(self, make_semantic_layer: Any) -> None:
        # Arrange: F holds only 0 and 1 (6 at 1); M spreads over 50 values, so pooled the metric
        # looks like a measurement
        score = [1.0] * 6 + [0.0] * 94 + [float(10 + i % 50) for i in range(100)]
        frame = pl.DataFrame({"patient_id": IDS, "sex": ["F"] * 100 + ["M"] * 100, "score": score})
        service = _service(make_semantic_layer, frame)
        plan = QueryPlan(intent="DESCRIBE", metric="score", group_by="sex", confidence=0.99)
        held = _ask(service, plan)

        # Act
        result = _ask(service, plan, confirm_hash=held.plan_hash if held.requires_confirmation else None)

        # Assert
        groups = {g["group"]: g.get("table_min_cell") for g in result.analysis["groups"]}
        assert groups == {"F": 6, "M": NO_COUNT_TABLE}
        rows = {row["sex"]: row for row in present_analysis(result.analysis)["table"]["rows"]}
        assert rows["F"]["mean score"] == SECONDARY_SUPPRESSED
        assert rows["M"]["mean score"] != SECONDARY_SUPPRESSED

    def test_ask_meanByDateColumn_largeCellsShown(self, make_semantic_layer: Any) -> None:
        # Arrange: two admission dates, each with 20 deaths and 80 survivors
        died = ([1] * 20 + [0] * 80) * 2
        dates = ["2020-01-01"] * 100 + ["2020-02-01"] * 100
        frame = pl.DataFrame({"patient_id": IDS, "admitted": dates, "died": died}).with_columns(
            pl.col("admitted").str.to_date()
        )
        service = _service(make_semantic_layer, frame)
        plan = QueryPlan(intent="DESCRIBE", metric="died", group_by="admitted", confidence=0.99)
        held = _ask(service, plan)

        # Act
        result = _ask(service, plan, confirm_hash=held.plan_hash if held.requires_confirmation else None)

        # Assert: each group's cells are found whatever type the date comes back as
        assert [g.get("table_min_cell") for g in result.analysis["groups"]] == [20, 20]
        values = [row["mean died"] for row in present_analysis(result.analysis)["table"]["rows"]]
        assert values == [0.2, 0.2]
