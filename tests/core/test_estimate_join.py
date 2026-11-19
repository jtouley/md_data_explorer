# SPDX-License-Identifier: MIT
"""Questions across public aggregate tables (ADR014 §10): pinned, harmonized on state FIPS, year
and sex, never pooled, and held or refused whenever the join or the measure is ambiguous.

Synthetic tables only (``make_public_source``); every number is made up. Expected values come
from independent Polars reads of the same synthetic frames (oracles), never from the engine.
"""

from __future__ import annotations

import dataclasses

import polars as pl
import pytest

from clinical_analytics.contracts import config
from clinical_analytics.core import estimate_join as ej
from clinical_analytics.core.estimates import detect_frame
from clinical_analytics.core.result_presentation import SECONDARY_SUPPRESSED


def _gbd_value(source: ej.SourceInput, **where: object) -> dict[str, float]:
    """Oracle: the one GBD row matching ``where`` (fails when not exactly one)."""
    frame = source.frame.filter(pl.all_horizontal([pl.col(k) == v for k, v in where.items()]))
    assert frame.height == 1, where
    return frame.select("val", "lower", "upper").row(0, named=True)


class TestReadQuestion:
    @pytest.mark.parametrize(
        ("question", "analysis"),
        [
            ("How did the HCV death rate change from 2010 to 2023 in Michigan?", "change"),
            ("Where does Michigan rank for syphilis incidence in 2023?", "rank"),
            ("Where do GBD and AtlasPlus disagree most on syphilis in 2023?", "disagree"),
            ("Which states have the widest uncertainty intervals for HIV in 2023?", "precision"),
            ("GBD HCV deaths and WONDER deaths in Michigan, 2023", "table"),
        ],
    )
    def test_readQuestion_wording_analysis(self, question, analysis):
        # Act
        asked = ej.read_question(question)

        # Assert
        assert asked.analysis == analysis

    @pytest.mark.parametrize(
        ("question", "sexes", "by_sex"),
        [
            ("in Michigan women", ["Female"], False),
            ("in Michigan women compared with men", ["Female", "Male"], True),
            ("by sex", [], True),
            ("both sexes", ["Both"], False),
            ("in Michigan", [], False),
        ],
    )
    def test_readQuestion_sexWords_menNeverInsideWomen(self, question, sexes, by_sex):
        # Act
        asked = ej.read_question(question)

        # Assert
        assert asked.sexes == sexes
        assert asked.by_sex is by_sex

    def test_readQuestion_showUs_notTheNation(self):
        # Act
        asked = ej.read_question("Show us the HCV death rate in Michigan")

        # Assert
        assert asked.nation is False
        assert asked.focus == ["26"]

    def test_readQuestion_spanWithChange_endpointsOnly(self):
        # Act
        asked = ej.read_question("change from 2010 to 2023")

        # Assert
        assert asked.years == [2010, 2023]
        assert asked.trend is False


class TestRankRange:
    @pytest.mark.parametrize(
        ("value", "lower", "upper", "others", "expected"),
        [
            pytest.param(5.0, 4.0, 6.0, [(9.0, 8.0, 10.0), (1.0, 0.5, 1.5)], (2, 2, 2), id="apart"),
            pytest.param(5.0, 4.0, 6.0, [(6.5, 5.5, 7.5), (1.0, 0.5, 1.5)], (2, 1, 2), id="overlap_above"),
            pytest.param(5.0, 4.0, 6.0, [(4.5, 3.5, 5.5), (9.0, 8.0, 10.0)], (2, 2, 3), id="overlap_below"),
            pytest.param(5.0, None, None, [(4.5, 3.5, 5.5)], (1, 1, 2), id="no_own_bounds_point"),
            pytest.param(5.0, 4.0, 6.0, [], (1, 1, 1), id="alone"),
        ],
    )
    def test_rankRange_intervals_bestAndWorst(self, value, lower, upper, others, expected):
        # Act
        result = ej.rank_range(value, lower, upper, others, lowest_first=False)

        # Assert
        assert result == expected

    def test_rankRange_lowestFirst_mirrored(self):
        # Act
        result = ej.rank_range(5.0, 4.0, 6.0, [(1.0, 0.5, 1.5), (5.5, 4.5, 6.5)], lowest_first=True)

        # Assert
        assert result == (2, 2, 3)


@pytest.mark.slow
@pytest.mark.integration
class TestTraps:
    def test_answer_undeclaredTable_refusedNamingIt(self, make_public_source):
        # Arrange
        gbd = make_public_source("gbd", name="GBD", years=(2023,))
        wonder = make_public_source("wonder", name="WONDER", declared=False)

        # Act
        result = ej.answer([gbd, wonder], "GBD and WONDER HCV deaths in Michigan, 2023")

        # Assert
        assert result.status == "refused"
        assert result.message.startswith("WONDER wasn't marked")

    def test_answer_dailyStationTable_refusedAsNotEstimates(self, make_public_source):
        # Arrange: NOAA rainfall is one row per station-day, not a state-year estimate
        gbd = make_public_source("gbd", name="GBD", years=(2023,))
        noaa = make_public_source("noaa", name="NOAA")

        # Act
        result = ej.answer([gbd, noaa], "Does summer rainfall predict Legionella deaths in Michigan in 2023?")

        # Assert
        assert result.status == "refused"
        assert "NOAA doesn't look like a table of estimates" in result.message

    def test_answer_countyTable_refusedNotRolledUp(self, make_public_source):
        # Arrange
        gbd = make_public_source("gbd", name="GBD", years=(2023,))
        places = make_public_source("places", name="PLACES")

        # Act
        result = ej.answer([gbd, places], "GBD HCV death rate and PLACES smoking in Michigan, 2023")

        # Assert
        assert result.status == "refused"
        assert result.message.startswith("PLACES is by county or city")

    @pytest.mark.parametrize("question", ["Hepatitis C cases in Wayne County in 2024", "wayne county hep C 2023"])
    def test_answer_wayneOnOutWayneTable_refusedNamingDetroit(self, make_public_source, question):
        # Arrange
        local = make_public_source("mdhhs_local", name="MDHHS")

        # Act
        result = ej.answer([local], question)

        # Assert
        assert result.status == "refused"
        assert "Wayne County excluding Detroit" in result.message
        assert "It also has Detroit rows" in result.message

    def test_answer_gbdNamesBesideAtlasFips_joinedOnFipsNotGbdLocationId(self, make_public_source):
        # Arrange: AtlasPlus keys on FIPS text; GBD on names with its own location_id (not FIPS)
        gbd = make_public_source("gbd", name="GBD", causes=("Syphilis",), measures=("Incidence",), years=(2023,))
        atlas = make_public_source("atlasplus", name="AtlasPlus", years=(2023,))

        # Act
        result = ej.answer(
            [gbd, atlas],
            "GBD syphilis incidence rate and AtlasPlus rate in Michigan and Georgia, 2023, both sexes, all ages",
        )

        # Assert
        assert result.status == "answered", result.message
        rows = {r["state"]: r for r in result.table["rows"]}
        assert set(rows) == {"Michigan", "Georgia"}
        oracle = atlas.frame.filter(pl.col("FIPS") == "13").get_column("Rate per 100000").item()
        assert rows["Georgia"]["AtlasPlus: Rate per 100000"] == oracle
        expected = _gbd_value(gbd, location_name="Georgia", sex_name="Both", age_name="All ages", metric_name="Rate")
        assert rows["Georgia"]["GBD: val"] == expected["val"]

    def test_answer_wonderSuppressedAndSmallCounts_hiddenNeverZero(self, make_public_source):
        # Arrange
        wonder = make_public_source("wonder", name="WONDER", years=(2023,))

        # Act
        result = ej.answer([wonder], "WONDER HCV deaths by sex in the Great Lakes states, 2023")

        # Assert
        assert result.status == "answered", result.message
        shown = {(r["state"], r["sex"]): r["WONDER: Deaths"] for r in result.table["rows"]}
        source = wonder.frame.filter(pl.col("Gender").is_not_null())
        for row in source.iter_rows(named=True):
            key = (row["State"], row["Gender"])
            if key not in shown:
                continue
            if row["Deaths suppressed"] or (row["Deaths"] is not None and row["Deaths"] < config.MIN_CELL_SIZE):
                assert shown[key] == SECONDARY_SUPPRESSED
            else:
                assert shown[key] == row["Deaths"]
        assert any(v == SECONDARY_SUPPRESSED for v in shown.values())
        assert all(v != 0 for v in shown.values())

    def test_answer_gbdNumberMetric_smallModelledCountsHidden(self, make_public_source):
        # Arrange: GBD 'val' under metric = Number is a count whatever the column is called
        gbd = make_public_source("gbd", name="GBD", years=(2023,), seed=11)

        # Act
        result = ej.answer([gbd], "Number of HCV deaths in every state, 2023, both sexes, all ages, highest first")

        # Assert
        assert result.status == "answered", result.message
        oracle = gbd.frame.filter(
            (pl.col("metric_name") == "Number")
            & (pl.col("sex_name") == "Both")
            & (pl.col("age_name") == "All ages")
            & (pl.col("location_name") != "United States")
        )
        small = oracle.filter(pl.col("val") < config.MIN_CELL_SIZE).height
        ranked = {r["state"] for r in result.table["rows"]}
        assert small > 0
        assert len(ranked) == oracle.height - small
        assert any("1 to 10 are hidden" in n for n in result.notes)

    def test_answer_ageStandardizedVsCrude_refused(self, make_public_source):
        # Arrange
        gbd = make_public_source("gbd", name="GBD", causes=("Syphilis",), measures=("Incidence",), years=(2023,))
        atlas = make_public_source("atlasplus", name="AtlasPlus", years=(2023,))

        # Act
        result = ej.answer(
            [gbd, atlas],
            "Where do GBD and AtlasPlus syphilis rates disagree in 2023, both sexes, age-standardized?",
            {"AtlasPlus": {ej.STANDARDIZATION_CHOICE: "crude"}},
        )

        # Assert
        assert result.status == "refused"
        assert "age-standardized and AtlasPlus's is crude" in result.message

    def test_answer_twoAgeStandardizedSourcesDifferentStandards_refused(self, make_public_source):
        # Arrange
        gbd = make_public_source("gbd", name="GBD", years=(2023,))
        wonder = make_public_source("wonder", name="WONDER", years=(2023,))

        # Act
        result = ej.answer(
            [gbd, wonder], "Where do GBD and WONDER age-adjusted HCV death rates disagree in 2023 among women?"
        )

        # Assert
        assert result.status == "refused"
        assert "different standard populations" in result.message

    def test_answer_gbdAgeLeftOpen_refusedExplainingStandardization(self, make_public_source):
        # Arrange
        gbd = make_public_source("gbd", name="GBD", years=(2023,))

        # Act
        result = ej.answer([gbd], "HCV death rate in Michigan, 2023, both sexes")

        # Assert
        assert result.status == "refused"
        assert "age-standardized and all-ages (crude)" in result.message

    def test_answer_unknownStandardizationInComparison_heldThenAnswered(self, make_public_source):
        # Arrange
        gbd = make_public_source("gbd", name="GBD", causes=("Syphilis",), measures=("Incidence",), years=(2023,))
        atlas = make_public_source("atlasplus", name="AtlasPlus", years=(2023,))
        question = "Where do GBD syphilis incidence and AtlasPlus rates disagree in 2023, both sexes, all ages?"

        # Act
        held = ej.answer([gbd, atlas], question)
        option = held.options[0]
        answered = ej.answer([gbd, atlas], question, {option.dataset_id: {option.column: option.default}})

        # Assert
        assert held.status == "held"
        assert (option.dataset_id, option.column, option.default) == ("AtlasPlus", ej.STANDARDIZATION_CHOICE, "crude")
        assert answered.status == "answered", answered.message

    def test_answer_averageAcrossYears_neverPooled(self, make_public_source):
        # Arrange
        gbd = make_public_source("gbd", name="GBD", years=tuple(range(2010, 2024)))

        # Act
        result = ej.answer([gbd], "Average age-standardized HCV death rate in Michigan women, 2010 to 2023")

        # Assert
        assert result.status == "answered"
        assert result.message.startswith("No average or total across rows is computed")
        assert len(result.table["rows"]) == 14
        lowers = {r["GBD: val lower"] for r in result.table["rows"]}
        assert len(lowers) == 14  # every year keeps its own bound

    def test_answer_changeBetweenYears_noIntervalForTheChange(self, make_public_source):
        # Arrange
        gbd = make_public_source("gbd", name="GBD", years=tuple(range(2010, 2024)))

        # Act
        result = ej.answer(
            [gbd],
            "How did the age-standardized HCV death rate change from 2010 to 2023 in Michigan women compared with men?",
        )

        # Assert
        assert result.status == "answered", result.message
        rows = {r["group"]: r for r in result.table["rows"]}
        for sex in ("Female", "Male"):
            first = _gbd_value(
                gbd, location_name="Michigan", sex_name=sex, age_name="Age-standardized", metric_name="Rate", year=2010
            )
            last = _gbd_value(
                gbd, location_name="Michigan", sex_name=sex, age_name="Age-standardized", metric_name="Rate", year=2023
            )
            row = rows[f"Michigan · {sex}"]
            assert row["2010"] == first["val"]
            assert row["2023"] == last["val"]
            assert row["percent change"] == round((last["val"] - first["val"]) / first["val"] * 100, 1)
            assert "change interval" not in row
        assert config.ESTIMATE_JOIN_CHANGE_METHOD in result.notes

    def test_answer_sexSplitAgainstTableWithoutSex_refused(self, make_public_source):
        # Arrange
        gbd = make_public_source("gbd", name="GBD", causes=("Syphilis",), measures=("Incidence",), years=(2023,))
        atlas = make_public_source("atlasplus", name="AtlasPlus", years=(2023,))

        # Act
        result = ej.answer([gbd, atlas], "GBD syphilis incidence and AtlasPlus rate in Michigan women, 2023, all ages")

        # Assert
        assert result.status == "refused"
        assert "AtlasPlus has no breakdown by sex" in result.message

    def test_answer_trendBesideSingleYearTable_otherYearsLeftOutAndNamed(self, make_public_source):
        # Arrange: AIDSVu has one year here; only that year lines up
        gbd = make_public_source("gbd", name="GBD", causes=("HIV/AIDS",), years=(2021, 2022, 2023))
        aidsvu = make_public_source("aidsvu", name="AIDSVu", years=(2023,))

        # Act
        result = ej.answer(
            [gbd, aidsvu],
            "HIV death rate and AIDSVu PrEP rate in Michigan each year 2021 to 2023, both sexes, all ages",
        )

        # Assert
        assert result.status == "answered", result.message
        assert [r["year"] for r in result.table["rows"]] == [2023]
        assert "Joined rows: 2 only in GBD and 0 only in AIDSVu were left out." in result.notes

    def test_answer_trendBesideTableWithoutYear_refusedNotRepeated(self, make_public_source):
        # Arrange: a table with no year at all would repeat its one value beside every year
        gbd = make_public_source("gbd", name="GBD", causes=("HIV/AIDS",), years=(2021, 2022, 2023))
        aidsvu = make_public_source("aidsvu", name="AIDSVu", years=(2023,))
        frame = aidsvu.frame.drop("Year")
        no_year = dataclasses.replace(aidsvu, frame=frame, table=detect_frame(frame))

        # Act
        result = ej.answer(
            [gbd, no_year],
            "HIV death rate and AIDSVu PrEP rate in Michigan each year 2021 to 2023, both sexes, all ages",
        )

        # Assert
        assert result.status == "refused"
        assert "AIDSVu has no year" in result.message


@pytest.mark.slow
@pytest.mark.integration
class TestHoldsAndChoices:
    def test_answer_rankWithSexOpen_heldWithBothDefault(self, make_public_source):
        # Arrange
        gbd = make_public_source("gbd", name="GBD", years=(2023,))

        # Act
        result = ej.answer([gbd], "Where does Michigan rank for the age-standardized HCV death rate in 2023?")

        # Assert
        assert result.status == "held"
        assert [(o.column, o.default) for o in result.options] == [("sex_name", "Both")]

    def test_answer_seasonTable_heldThenAlignedToYear(self, make_public_source):
        # Arrange
        gbd = make_public_source(
            "gbd",
            name="GBD",
            causes=("Lower respiratory infections",),
            metrics=("Percent",),
            years=(2023,),
            rei="Smoking",
            ages=("All ages",),
        )
        flu = make_public_source("fluvaxview", name="FluVaxView")
        question = (
            "Pneumonia deaths attributable to smoking and flu vaccination coverage for "
            "adults 18 years and older in Michigan, 2023, both sexes"
        )

        # Act
        held = ej.answer([gbd, flu], question)
        chosen = {o.dataset_id: {o.column: o.default} for o in held.options}
        answered = ej.answer([gbd, flu], question, chosen)

        # Assert
        assert held.status == "held"
        assert chosen == {"FluVaxView": {"Season/Survey Year": "2022-23"}}
        assert answered.status == "answered", answered.message
        row = answered.table["rows"][0]
        oracle = flu.frame.filter(
            (pl.col("Geography") == "Michigan")
            & (pl.col("Season/Survey Year") == "2022-23")
            & (pl.col("Dimension") == "≥18 Years")
        )
        assert row["FluVaxView: Estimate (%)"] == oracle.get_column("Estimate (%)").item()
        assert row["FluVaxView: Estimate (%) lower"] == oracle.get_column("95% CI (%) lower").item()
        assert any("2022-23" in m and "year confirmed" in m for m in answered.method)

    def test_answer_seasonTableOverYears_refused(self, make_public_source):
        # Arrange
        gbd = make_public_source(
            "gbd",
            name="GBD",
            causes=("Lower respiratory infections",),
            metrics=("Percent",),
            years=(2022, 2023),
            rei="Smoking",
            ages=("All ages",),
        )
        flu = make_public_source("fluvaxview", name="FluVaxView")

        # Act
        result = ej.answer(
            [gbd, flu],
            "Smoking-attributable pneumonia deaths and flu vaccination for adults 18 "
            "years and older in Michigan each year 2022 to 2023, both sexes",
        )

        # Assert
        assert result.status == "refused"
        assert "flu season" in result.message

    def test_choiceProblem_unknownLevel_named(self, make_public_source):
        # Arrange
        gbd = make_public_source("gbd", name="GBD", years=(2023,))

        # Act
        problem = ej.choice_problem(gbd, {"sex_name": "Unknown"})
        fine = ej.choice_problem(gbd, {"sex_name": "Both", ej.STANDARDIZATION_CHOICE: "crude"})

        # Assert
        assert problem == "GBD: 'Unknown' is not a value of sex_name."
        assert fine is None


@pytest.mark.slow
@pytest.mark.integration
class TestAnalysesAgainstOracles:
    def test_answer_rankMichigan_pointRankAndRangeMatchOracle(self, make_public_source):
        # Arrange
        gbd = make_public_source("gbd", name="GBD", years=(2023,))

        # Act
        result = ej.answer(
            [gbd], "Where does Michigan rank for the age-standardized HCV death rate in 2023, both sexes?"
        )

        # Assert
        assert result.status == "answered", result.message
        states = gbd.frame.filter(
            (pl.col("sex_name") == "Both")
            & (pl.col("age_name") == "Age-standardized")
            & (pl.col("metric_name") == "Rate")
            & (pl.col("location_name") != "United States")
        )
        mi = states.filter(pl.col("location_name") == "Michigan").row(0, named=True)
        others = states.filter(pl.col("location_name") != "Michigan")
        point = 1 + others.filter(pl.col("val") > mi["val"]).height
        best = 1 + others.filter(pl.col("lower") > mi["upper"]).height
        worst = 1 + others.filter(pl.col("upper") >= mi["lower"]).height
        assert result.message.startswith(f"Michigan ranks {ej._ordinal(point)} of {states.height} states")
        if best != worst:
            assert f"from {ej._ordinal(best)} to {ej._ordinal(worst)}" in result.message

    def test_answer_modelVsSurveillance_verdictsMatchOracle(self, make_public_source):
        # Arrange
        gbd = make_public_source("gbd", name="GBD", causes=("Syphilis",), measures=("Incidence",), years=(2023,))
        atlas = make_public_source("atlasplus", name="AtlasPlus", years=(2023,))

        # Act
        result = ej.answer(
            [gbd, atlas],
            "Which states disagree most between GBD syphilis incidence and "
            "AtlasPlus rates in 2023, both sexes, all ages?",
            {"AtlasPlus": {ej.STANDARDIZATION_CHOICE: "crude"}},
        )

        # Assert
        assert result.status == "answered", result.message
        model = gbd.frame.filter(
            (pl.col("sex_name") == "Both") & (pl.col("age_name") == "All ages") & (pl.col("metric_name") == "Rate")
        ).select(pl.col("location_name").alias("Geography"), "val", "lower", "upper")
        joined = atlas.frame.filter(pl.col("Year") == 2023).join(model, on="Geography")
        above = joined.filter(pl.col("Rate per 100000") > pl.col("upper")).height
        below = joined.filter(pl.col("Rate per 100000") < pl.col("lower")).height
        assert result.message.startswith(
            f"Of {joined.height} rows, AtlasPlus's value is above GBD's interval in {above} and below it in {below}"
        )

    def test_answer_precisionAcrossCauses_widestFirstMatchesOracle(self, make_public_source):
        # Arrange
        causes = ("Syphilis", "Hepatitis C", "HIV/AIDS", "Tuberculosis")
        gbd = make_public_source("gbd", name="GBD", causes=causes, years=(2023,), sexes=("Both",))

        # Act
        result = ej.answer(
            [gbd],
            "Which states and causes have the widest uncertainty intervals for age-standardized death rates in 2023?",
        )

        # Assert
        assert result.status == "answered", result.message
        oracle = (
            gbd.frame.filter(
                (pl.col("age_name") == "Age-standardized")
                & (pl.col("metric_name") == "Rate")
                & (pl.col("location_name") != "United States")
            )
            .with_columns(((pl.col("upper") - pl.col("lower")) / pl.col("val")).round(3).alias("w"))
            .sort("w", descending=True)
        )
        assert len(result.table["rows"]) == oracle.height == 51 * len(causes)
        assert result.table["rows"][0]["relative width"] == oracle.get_column("w")[0]


@pytest.mark.slow
@pytest.mark.integration
class TestReviewRegressions:
    def test_answer_savedYearChoice_notReusedForAnotherYear(self, make_public_source):
        # Arrange: a pairing saved earlier (2022 shown beside 2023) must not pin a 2021 question
        gbd = make_public_source("gbd", name="GBD", causes=("HIV/AIDS",), years=(2021, 2022, 2023))
        aidsvu = make_public_source("aidsvu", name="AIDSVu", years=(2021, 2022), saved_choices={"Year": 2022})

        # Act
        result = ej.answer([gbd, aidsvu], "HIV death rate and AIDSVu PrEP rate in Michigan, 2021, both sexes, all ages")

        # Assert
        assert result.status == "answered", result.message
        mi = result.table["rows"][0]
        oracle = aidsvu.frame.filter((pl.col("State Abbreviation") == "MI") & (pl.col("Year") == 2021))
        assert mi["year"] == 2021
        assert mi["AIDSVu: PrEP Rate"] == oracle.get_column("PrEP Rate").item()

    def test_savableChoices_yearPairingDropped(self, make_public_source):
        # Arrange
        flu = make_public_source("fluvaxview", name="FluVaxView")

        # Act
        kept = ej.savable_choices(flu, {"Season/Survey Year": "2022-23", "Dimension": "≥18 Years"})

        # Assert
        assert kept == {"Dimension": "≥18 Years"}

    def test_answer_metricGroupedNumberRows_smallCountsHidden(self, make_public_source):
        # Arrange: metric left as a breakdown, so 'Number' rows sit beside 'Rate' rows
        gbd = make_public_source("gbd", name="GBD", years=(2023,), locations=("Michigan", "Ohio"), seed=2)

        # Act
        result = ej.answer([gbd], "GBD HCV deaths by metric in Michigan, 2023, both sexes, all ages")

        # Assert
        assert result.status == "answered", result.message
        number = gbd.frame.filter(
            (pl.col("location_name") == "Michigan")
            & (pl.col("sex_name") == "Both")
            & (pl.col("age_name") == "All ages")
            & (pl.col("metric_name") == "Number")
        ).row(0, named=True)
        shown = next(r for r in result.table["rows"] if r["metric_name"] == "Number")["GBD: val"]
        assert 0 < number["val"] < config.MIN_CELL_SIZE  # this seed's modelled count is small
        assert shown == SECONDARY_SUPPRESSED
