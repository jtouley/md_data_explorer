# SPDX-License-Identifier: MIT
"""Aggregate-data mode (ADR014): detecting estimate tables, totals, holds and estimate answers.

Synthetic tables only (``make_estimate_frame``); every number is made up.
"""

from __future__ import annotations

import ibis
import polars as pl
import pytest

from clinical_analytics.core import estimates
from clinical_analytics.core.query_plan import FilterSpec, QueryPlan
from clinical_analytics.core.result_presentation import SECONDARY_SUPPRESSED


def _detect(frame: pl.DataFrame) -> estimates.EstimateTable | None:
    return estimates.detect_estimate_table(ibis.memtable(frame), frame.height)


def _check(frame: pl.DataFrame, plan: QueryPlan, choices: dict | None = None) -> estimates.EstimateCheck:
    table = _detect(frame)
    assert table is not None
    view = ibis.memtable(frame)
    for spec in plan.filters:
        values = spec.value if isinstance(spec.value, list) else [spec.value]
        view = view.filter(view[spec.column].isin(values))
    return estimates.check_plan(plan, view, table, choices or {})


class TestDetection:
    def test_detect_ihmeShapedTable_keysBoundsAliasesAndTotals(self, make_estimate_frame):
        # Arrange
        frame = make_estimate_frame()

        # Act
        table = _detect(frame)

        # Assert
        assert table is not None
        assert [d.name for d in table.dimensions] == ["location_name", "cause_name", "sex", "year_id"]
        location = table.dimension("FIPS")
        assert location is not None and location.name == "location_name" and location.top == "Colorado"
        assert table.dimension("sex").top == "Both"  # type: ignore[union-attr]
        assert table.dimension("year_id").yearlike  # type: ignore[union-attr]
        assert [(v.name, v.lower, v.upper, v.kind) for v in table.values] == [("mx", "lower", "upper", "rate")]
        assert table.attributes == ("metric",)

    def test_detect_whoShapedSuffixBounds_attributeNotAKey(self):
        # Arrange: country x year, with _lo/_hi bounds and a region that depends on the country
        frame = pl.DataFrame(
            {
                "country": ["A", "A", "B", "B", "C", "C"],
                "iso3": ["AAA", "AAA", "BBB", "BBB", "CCC", "CCC"],
                "g_whoregion": ["AFR", "AFR", "AFR", "AFR", "EUR", "EUR"],
                "year": [2023, 2024, 2023, 2024, 2023, 2024],
                "e_inc_100k": [10.0, 11.0, 20.0, 21.0, 30.0, 31.0],
                "e_inc_100k_lo": [8.0, 9.0, 18.0, 19.0, 28.0, 29.0],
                "e_inc_100k_hi": [12.0, 13.0, 22.0, 23.0, 32.0, 33.0],
            }
        )

        # Act
        table = _detect(frame)

        # Assert
        assert table is not None
        assert [d.name for d in table.dimensions] == ["country", "year"]
        assert table.attributes == ("g_whoregion",)
        assert table.values[0].lower == "e_inc_100k_lo"

    @pytest.mark.parametrize(
        "frame",
        [
            pytest.param(
                pl.DataFrame({"patient_id": [f"P{i}" for i in range(30)], "heart_rate": [float(i) for i in range(30)]}),
                id="one_row_per_patient",
            ),
            pytest.param(
                pl.DataFrame({"sex": ["M", "F"] * 15, "ward": ["A", "B", "C"] * 10, "heart_rate": [70.0] * 30}),
                id="categories_do_not_key_rows",
            ),
            pytest.param(pl.DataFrame({"sex": ["M", "F"], "value": [1.0, 2.0]}), id="no_rate_bounds_or_total"),
            pytest.param(pl.DataFrame({"x": [1.0]}), id="one_row"),
        ],
    )
    def test_detect_patientOrPlainTables_notEstimates(self, frame):
        # Act
        table = _detect(frame)

        # Assert
        assert table is None


class TestParentLevels:
    @pytest.mark.parametrize(
        ("column", "levels", "expected_top"),
        [
            pytest.param("sex", ["Both", "Female", "Male"], "Both", id="both"),
            pytest.param("Sex", ["All Sexes", "Female", "Male"], "All Sexes", id="all_sexes"),
            pytest.param("sex", ["a", "f", "m"], "a", id="who_a"),
            pytest.param("State", ["Alabama", "United States"], "United States", id="national"),
            pytest.param("location", ["Colorado", "Baca County", "Otero County"], "Colorado", id="state_of_counties"),
            pytest.param("age_group", ["0-14", "15-24", "15plus", "0plus"], "0plus", id="widest_band"),
            pytest.param("cause", ["Heart disease", "Stroke"], None, id="no_total"),
            pytest.param("Date", ["2024-01", "2024-02"], None, id="dates_are_not_age_bands"),
            pytest.param("sex", [], None, id="empty"),
        ],
    )
    def test_parentLevels_namedBandsAndAreas_topFound(self, column, levels, expected_top):
        # Act
        _parents, top = estimates.parent_levels(levels, column)

        # Assert
        assert top == expected_top


class TestCheckPlan:
    def test_checkPlan_pooledMeanOverOpenDimensions_refusedNamingThem(self, make_estimate_frame):
        # Arrange: "Average mx by cause_name" leaves location, sex and year open
        plan = QueryPlan(intent="DESCRIBE", metric="mx", group_by="cause_name")

        # Act
        check = _check(make_estimate_frame(), plan)

        # Assert
        assert check.refusal is not None
        assert "location_name" in check.refusal and "Colorado is the total" in check.refusal
        assert "year_id" in check.refusal and "Name one value for each of year_id" in check.refusal

    def test_checkPlan_onlyTotalsOpen_holdOffersTotals(self, make_estimate_frame):
        # Arrange: sex is the only open dimension, and Both is its total
        plan = QueryPlan(
            intent="DESCRIBE",
            metric="mx",
            filters=[
                FilterSpec("location_name", "==", "Colorado"),
                FilterSpec("cause_name", "==", "Cardiovascular diseases"),
                FilterSpec("year_id", "==", 2014),
            ],
        )

        # Act
        check = _check(make_estimate_frame(), plan)

        # Assert
        assert check.refusal is None
        assert check.defaults == {"sex": "Both"}
        assert check.hold is not None and "Confirm to use sex = Both" in check.hold

    def test_checkPlan_savedChoice_appliedWithoutHold(self, make_estimate_frame):
        # Arrange
        plan = QueryPlan(
            intent="DESCRIBE",
            metric="mx",
            filters=[
                FilterSpec("location_name", "==", "Colorado"),
                FilterSpec("cause_name", "==", "Cardiovascular diseases"),
                FilterSpec("year_id", "==", 2014),
            ],
        )

        # Act
        check = _check(make_estimate_frame(), plan, {"sex": "Both"})

        # Assert
        assert check.hold is None and check.refusal is None
        assert check.chosen == {"sex": "Both"}

    def test_checkPlan_groupByDimensionWithTotal_totalLeftOut(self, make_estimate_frame):
        # Arrange
        plan = QueryPlan(
            intent="COMPARE_GROUPS",
            metric="mx",
            group_by="sex",
            filters=[
                FilterSpec("location_name", "==", "Colorado"),
                FilterSpec("cause_name", "==", "Cardiovascular diseases"),
                FilterSpec("year_id", "==", 2014),
            ],
        )

        # Act
        check = _check(make_estimate_frame(), plan)

        # Assert
        assert check.refusal is None and check.hold is None
        assert check.left_out == {"sex": ("Both",)}

    @pytest.mark.parametrize(
        ("plan", "expected"),
        [
            pytest.param(QueryPlan(intent="FIND_PREDICTORS", metric="mx"), "patient-level tests", id="model"),
            pytest.param(QueryPlan(intent="CORRELATIONS", metric="mx"), "patient-level tests", id="correlation"),
            pytest.param(
                QueryPlan(intent="DESCRIBE", proportion_column="sex", proportion_value="Male"),
                "share of rows",
                id="share",
            ),
            pytest.param(QueryPlan(intent="DESCRIBE", metric="year_id"), "not a value", id="metric_is_key"),
            pytest.param(QueryPlan(intent="DESCRIBE", metric="mx", group_by="metric"), "does not identify", id="attr"),
        ],
    )
    def test_checkPlan_patientShapedQuestions_refused(self, make_estimate_frame, plan, expected):
        # Act
        check = _check(make_estimate_frame(), plan)

        # Assert
        assert check.refusal is not None and expected in check.refusal


def _row(label: str, value: float | None, lower: float | None, upper: float | None) -> dict:
    return {"label": label, "key": (label,), "value": value, "lower": lower, "upper": upper}


class TestEstimateAnswer:
    @pytest.mark.parametrize(
        ("rows", "expected"),
        [
            pytest.param(
                [_row("Male", 238.2, 231.5, 244.6), _row("Female", 183.9, 177.3, 190.4)],
                "Their intervals don't overlap, so the estimates differ.",
                id="apart",
            ),
            pytest.param(
                [_row("Male", 200.0, 190.0, 210.0), _row("Female", 195.0, 185.0, 205.0)],
                "Their intervals overlap, so these estimates are not shown to differ.",
                id="overlap",
            ),
            pytest.param(
                [_row("Male", 200.0, 190.0, 210.0), _row("Female", 195.0, None, None)],
                "One of them has no interval",
                id="missing_bound",
            ),
        ],
    )
    def test_estimateAnswer_twoGroups_intervalsComparedNeverTested(self, rows, expected):
        # Arrange
        check = estimates.EstimateCheck(
            value=estimates.ValueColumn("mx", "lower", "upper", "rate"), group_dims=("sex",)
        )

        # Act
        answer = estimates.estimate_answer(rows, check, compare=True, lowest_first=False)

        # Assert
        assert expected in answer["claim"]
        assert "p=" not in answer["claim"] and "p<" not in answer["claim"]
        assert answer["table"]["columns"] == ["sex", "mx", "lower", "upper"]

    def test_estimateAnswer_rankingOverlapsNext_rankUncertain(self):
        # Arrange
        rows = [
            _row("A County", 201.1, 179.1, 224.2),
            _row("B County", 183.3, 160.0, 205.0),
            _row("C County", 90.0, 80.0, 100.0),
        ]
        check = estimates.EstimateCheck(
            value=estimates.ValueColumn("mx", "lower", "upper", "rate"),
            group_dims=("location_name",),
            left_out={"location_name": ("Colorado",)},
        )

        # Act
        answer = estimates.estimate_answer(rows, check, compare=False, lowest_first=False)

        # Assert
        assert "Highest: A County 201.10 (179.10 to 224.20)" in answer["claim"]
        assert "overlaps those of B County, so its rank among them is uncertain" in answer["claim"]
        assert "Colorado is the total of the other location_name values" in answer["claim"]

    @pytest.mark.parametrize(
        ("value", "hidden"),
        [
            pytest.param(0.0, False, id="zero"),
            pytest.param(1.0, True, id="one"),
            pytest.param(10.0, True, id="ten"),
            pytest.param(11.0, False, id="eleven"),
            pytest.param(None, False, id="null"),
        ],
    )
    def test_estimateAnswer_countColumnSmallValues_hidden(self, value, hidden):
        # Arrange: raw counts keep the small-cell rule on a declared dataset
        check = estimates.EstimateCheck(
            value=estimates.ValueColumn("Deaths", None, None, "count"), group_dims=("State",)
        )

        # Act
        answer = estimates.estimate_answer(
            [_row("Wyoming", value, None, None)], check, compare=False, lowest_first=False
        )

        # Assert
        shown = answer["table"]["rows"][0]["Deaths"]
        assert (shown == SECONDARY_SUPPRESSED) is hidden
        assert ("Counts from 1 to 10 are hidden." in answer["claim"]) is hidden


class TestGroundPlan:
    def test_groundPlan_yearsAndLevelsNamed_pinnedFromQuestion(self, make_estimate_frame):
        # Arrange
        table = _detect(make_estimate_frame())
        assert table is not None
        plan = QueryPlan(intent="DESCRIBE")

        # Act
        grounded = estimates.ground_plan(
            plan, table, "What was the cardiovascular disease mortality rate in Colorado in 2014 for both sexes?"
        )

        # Assert
        limits = {
            (f.column, f.operator, f.value if not isinstance(f.value, list) else tuple(f.value))
            for f in grounded.filters
        }
        assert limits == {
            ("cause_name", "==", "Cardiovascular diseases"),
            ("location_name", "==", "Colorado"),
            ("sex", "==", "Both"),
            ("year_id", "==", 2014),
        }
        assert grounded.metric == "mx"

    def test_groundPlan_countyRanking_groupsByLocationWithoutStateLimit(self, make_estimate_frame):
        # Arrange
        table = _detect(make_estimate_frame())
        assert table is not None
        plan = QueryPlan(intent="DESCRIBE", filters=[FilterSpec("location_name", "==", "Colorado")])

        # Act
        grounded = estimates.ground_plan(plan, table, "Which Colorado county had the highest mx in 2014?")

        # Assert
        assert grounded.group_by == "location_name"
        assert all(f.column != "location_name" for f in grounded.filters)

    def test_groundPlan_parseKeptAnotherLevel_questionLevelsWin(self, make_estimate_frame):
        # Arrange: the parse read "Place01" as Place02
        table = _detect(make_estimate_frame())
        assert table is not None
        plan = QueryPlan(intent="DESCRIBE", metric="mx", filters=[FilterSpec("location_name", "==", "Place02 County")])

        # Act
        grounded = estimates.ground_plan(plan, table, "mx in Place01 County and Colorado in 2014")

        # Assert
        kept = [f for f in grounded.filters if f.column == "location_name"]
        assert len(kept) == 1 and kept[0].operator == "IN"
        assert set(kept[0].value) == {"Place01 County", "Colorado"}  # type: ignore[arg-type]

    def test_groundPlan_countIntent_unchanged(self, make_estimate_frame):
        # Arrange
        table = _detect(make_estimate_frame())
        assert table is not None
        plan = QueryPlan(intent="COUNT", group_by="sex")

        # Act
        grounded = estimates.ground_plan(plan, table, "count by sex in 2014")

        # Assert
        assert grounded is plan


class TestNullKeysAndDuplicates:
    def test_checkPlan_nullKeyLevelBesideValue_dimensionStaysOpen(self):
        # Arrange: a key column with an empty level is two levels, not one (RESP-NET "Rate Type")
        frame = pl.DataFrame(
            {
                "State": ["A", "A", "B", "B"],
                "Rate Type": ["Observed", None, "Observed", None],
                "Rate": [1.0, 2.0, 3.0, 4.0],
            }
        )
        plan = QueryPlan(intent="DESCRIBE", metric="Rate", filters=[FilterSpec("State", "==", "A")])

        # Act
        check = _check(frame, plan)

        # Assert
        assert check.refusal is not None and "Rate Type" in check.refusal

    def test_estimateRows_twoRowsOneGroup_raises(self):
        # Arrange
        frame = pl.DataFrame({"sex": ["Male", "Male"], "mx": [1.0, 2.0]})
        table = estimates.EstimateTable(
            rows=2,
            dimensions=(estimates.Dimension("sex", 1),),
            values=(estimates.ValueColumn("mx", None, None, "rate"),),
        )
        check = estimates.EstimateCheck(value=table.values[0], group_dims=("sex",))

        # Act / Assert
        with pytest.raises(estimates.EstimateGrainError):
            estimates.estimate_rows(ibis.memtable(frame), check, table, lowest_first=False)


class TestIntervalConsistency:
    @pytest.mark.parametrize(
        ("lower", "value", "upper", "shown_bounds"),
        [
            pytest.param(1.0, 2.0, 3.0, (1.0, 3.0), id="inside"),
            pytest.param(2.0, 2.0, 2.0, (2.0, 2.0), id="boundary"),
            pytest.param(2.5, 2.0, 3.0, (None, None), id="lower_above_value"),
            pytest.param(1.0, 2.0, 1.5, (None, None), id="upper_below_value"),
            pytest.param(None, 2.0, 3.0, (None, None), id="one_bound_missing"),
        ],
    )
    def test_estimateRows_intervalMustContainValue_elseDropped(self, lower, value, upper, shown_bounds):
        # Arrange
        frame = pl.DataFrame(
            {"sex": ["Male"], "mx": [value], "lower": [lower], "upper": [upper]},
            schema={"sex": pl.String, "mx": pl.Float64, "lower": pl.Float64, "upper": pl.Float64},
        )
        table = estimates.EstimateTable(
            rows=1,
            dimensions=(estimates.Dimension("sex", 1),),
            values=(estimates.ValueColumn("mx", "lower", "upper", "rate"),),
        )
        check = estimates.EstimateCheck(value=table.values[0], group_dims=("sex",))

        # Act
        rows = estimates.estimate_rows(ibis.memtable(frame), check, table, lowest_first=False)
        answer = estimates.estimate_answer(rows, check, compare=False, lowest_first=False)

        # Assert
        assert (rows[0]["lower"], rows[0]["upper"]) == shown_bounds
        assert ("doesn't contain the estimate" in answer["claim"]) is (
            lower is not None and shown_bounds == (None, None)
        )
