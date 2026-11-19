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

from clinical_analytics.contracts import config, geography
from clinical_analytics.contracts.geography import STATES
from clinical_analytics.core import estimate_join as ej
from clinical_analytics.core import estimates
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

    def test_answer_gbdNumberMetric_rankedAsPublished(self, make_public_source):
        # Arrange: GBD 'val' under metric = Number, including modelled counts of 1-10
        gbd = make_public_source("gbd", name="GBD", years=(2023,), seed=11)

        # Act
        result = ej.answer([gbd], "Number of HCV deaths in every state, 2023, both sexes, all ages, highest first")

        # Assert: every state ranked on its published value; nothing hidden by the app
        assert result.status == "answered", result.message
        oracle = gbd.frame.filter(
            (pl.col("metric_name") == "Number")
            & (pl.col("sex_name") == "Both")
            & (pl.col("age_name") == "All ages")
            & (pl.col("location_name") != "United States")
        )
        assert oracle.filter(pl.col("val") < config.MIN_CELL_SIZE).height > 0
        ranked = {r["state"]: r["val"] for r in result.table["rows"]}
        assert ranked == dict(oracle.select("location_name", "val").iter_rows())

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

    def test_answer_metricGroupedNumberRows_shownAsPublished(self, make_public_source):
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
        assert 0 < number["val"] < config.MIN_CELL_SIZE  # a modelled count of 1-10
        assert shown == number["val"]  # published, so shown as published


def _source(frame: pl.DataFrame, name: str) -> ej.SourceInput:
    """A declared join source from an in-memory synthetic frame (already typed)."""
    return ej.SourceInput(dataset_id=name, name=name, frame=frame, table=detect_frame(frame), declared=True)


class TestSecondReviewKinds:
    """Finding 1: a value is relaxed only when it positively reads as a rate or percent."""

    @pytest.mark.parametrize(
        ("column", "kind"),
        [
            ("Rate per 100000", "rate"),
            ("e_inc_100k", "rate"),
            ("Estimate (%)", "rate"),
            ("Crude Rate", "rate"),
            ("mx", "rate"),
            ("val", "count"),
            ("Hospitalizations", "count"),
            ("Estimated infections", "count"),
            ("Patients", "count"),
            ("Total", "count"),
            ("Admissions", "count"),
            ("Persons living with HIV", "count"),
            ("Incidence", "count"),
        ],
    )
    def test_valueKind_name_failsClosed(self, column, kind):
        # Act / Assert
        assert estimates.value_kind(column) == kind

    @pytest.mark.parametrize(
        ("level", "kind"),
        [("Rate", "rate"), ("Percent", "rate"), ("Crude Prevalence", "rate"), ("Number", "count"), ("Mean", "count")],
    )
    def test_metricLevelKind_level_failsClosed(self, level, kind):
        # Act / Assert
        assert estimates.metric_level_kind(level) == kind

    @pytest.mark.parametrize(
        "column",
        ["Hospitalizations", "Estimated infections", "Patients", "Total", "Admissions", "Persons living with HIV"],
    )
    def test_answer_publishedModelledCounts_shownAsPublished(self, column):
        # Arrange: modelled counts 2.5-9.5 in a declared public table (owner's decision, ADR014 §7)
        names = [s.name for s in STATES][:8]
        frame = pl.DataFrame(
            [
                {"State": s, "Year": y, column: 2.5 + i, f"{column} lower": 1.0 + i, f"{column} upper": 4.0 + i}
                for i, s in enumerate(names)
                for y in (2022, 2023)
            ]
        )

        # Act
        result = ej.answer([_source(frame, "T")], f"{column} by state in 2023")

        # Assert
        assert result.status == "answered", result.message
        shown = {r["state"]: r[f"T: {column}"] for r in result.table["rows"]}
        assert shown[names[0]] == 2.5
        assert shown[names[7]] == 9.5

    def test_answer_gbdOnlyNumberMetric_namedMetricReadShownAsPublished(self, make_public_table):
        # Arrange: metric_name is the constant "Number" beside metric_id = 1 (realistic GBD export)
        frame = make_public_table(
            "gbd", locations=("Michigan", "Ohio", "Indiana"), years=(2023,), metrics=("Number",), ages=("All ages",)
        ).with_columns(pl.col("val") / 100, pl.col("lower") / 100, pl.col("upper") / 100)
        source = _source(frame, "GBD")

        # Act
        result = ej.answer([source], "HCV deaths in Michigan, Ohio and Indiana, 2023, both sexes")

        # Assert: the named metric is read (never metric_id), and values are shown as published
        assert source.table is not None
        assert source.table.metric_levels == {"metric_name": {"Number": "count"}}
        assert result.status == "answered", result.message
        published = dict(frame.filter(pl.col("sex_name") == "Both").select("location_name", "val").iter_rows())
        assert {r["state"]: r["GBD: val"] for r in result.table["rows"]} == published


class TestSecondReviewPatientData:
    """Finding 4: a file with a column named as a patient ID is never an estimate table."""

    @pytest.mark.parametrize("id_column", ["mrn", "patient_id", "subject", "pid", "Patient ID", "MRN"])
    def test_detectFrame_visitFileWithPatientIdName_none(self, id_column):
        # Arrange
        frame = pl.DataFrame(
            {
                id_column: [f"MRN{p:04d}" for p in range(1, 41) for _ in range(3)],
                "visit_month": ["2024-01", "2024-02", "2024-03"] * 40,
                "site": ["Rural clinic" if p <= 4 else "Main hospital" for p in range(1, 41) for _ in range(3)],
                "hiv_viral_load_rate": [55.0 + (i % 50) for i in range(120)],
                "lower": [50.0] * 120,
                "upper": [120.0] * 120,
            }
        )

        # Act
        table = detect_frame(frame)

        # Assert
        assert table is None


class TestSecondReviewWrongAnswers:
    """Finding 5."""

    def test_answer_countryGeorgiaAmongCountries_leftOutNotRankedAsState(self):
        # Arrange: a GBD file with states and countries; the country Georgia is far higher
        names = [s.name for s in STATES if s.name != "Georgia"][:30] + ["Georgia", "Armenia", "Azerbaijan"]
        frame = pl.DataFrame(
            [
                {
                    "location_name": n,
                    "year": 2023,
                    "val": 500.0 if n == "Georgia" else 10.0 + i / 10,
                    "lower": 1.0,
                    "upper": 600.0,
                }  # fmt: skip
                for i, n in enumerate(names)
            ]
        ).with_columns(pl.col("lower").clip(upper_bound=pl.col("val")))

        # Act
        result = ej.answer([_source(frame, "G")], "Rank states by tuberculosis death rate in 2023")

        # Assert
        assert result.status == "answered", result.message
        assert "Georgia" not in {r["state"] for r in result.table["rows"]}
        assert any("Georgia was left out" in n for n in result.notes)

    @pytest.mark.parametrize(
        ("text", "groups"),
        [
            ("What is the rate in South Carolina?", []),
            ("rate in West Virginia", []),
            ("rate in the South", ["south"]),
            ("South Dakota and the Midwest", ["midwest"]),
        ],
    )
    def test_groupsNamed_wholeStateNamesFirst(self, text, groups):
        # Act / Assert
        assert geography.groups_named(text) == groups

    def test_detectFrame_wonderGenderAndCodeWithTotalRows_aliasesAndTotalLeftOut(self):
        # Arrange: WONDER total rows leave Gender and Gender Code empty
        rows = []
        for state in ("Michigan", "Ohio", "Iowa"):
            for sex, code, deaths in (("Female", "F", 60), ("Male", "M", 140), (None, None, 200)):
                rows.append({"State": state, "Year": 2015, "Gender": sex, "Gender Code": code, "Deaths": deaths})
        frame = pl.DataFrame(rows)

        # Act
        table = detect_frame(frame)

        # Assert
        assert table is not None
        gender = table.dimension("Gender")
        assert gender is not None and "Gender Code" in gender.aliases
        assert None in gender.parents

    def test_standardizationBySource_eachPhraseToItsTable(self, make_public_source):
        # Arrange
        gbd = make_public_source("gbd", name="GBD", years=(2023,))
        wonder = make_public_source("wonder", name="WONDER", years=(2023,))
        question = (
            "Does the WONDER crude death rate among women disagree with the GBD age-standardized estimate in 2023?"
        )

        # Act
        chosen = ej.standardization_by_source([gbd, wonder], question)
        result = ej.answer([gbd, wonder], question)

        # Assert
        assert chosen == {"WONDER": "crude", "GBD": "age-standardized"}
        assert result.status == "refused"
        assert "GBD's rate is age-standardized and WONDER's is crude" in result.message


def _visits(id_column: str, ids: list, value: str = "hiv_viral_load_rate", months: int = 3) -> pl.DataFrame:
    """Synthetic repeated visits per person, with a rate-named value (made-up numbers)."""
    return pl.DataFrame(
        {
            id_column: [i for i in ids for _ in range(months)],
            "visit_month": [f"2024-0{m}" for _ in ids for m in range(1, months + 1)],
            "site": ["Rural clinic" if n % 9 == 0 else "Main hospital" for n in range(len(ids)) for _ in range(months)],
            value: [55.0 + (n % 50) for n in range(len(ids) * months)],
            "lower": [50.0] * (len(ids) * months),
            "upper": [130.0] * (len(ids) * months),
        }
    )


class TestPatientFileGuard:
    """Item 3 (owner's decision): a patient-level file can never pass as public estimates.
    Structural, fail closed; the review found person, member_no, enc, Client, case_number and
    name accepted by a name list."""

    @pytest.mark.parametrize(
        "id_column",
        [
            "person",
            "member_no",
            "enc",
            "Client",
            "case_number",
            "name",
            "ClientID",
            "Encounter",
            "record_id",
            "first_name",
            "visit_no",
            "mrn",
            "PatientID",
            "subject",
        ],  # fmt: skip
    )
    def test_detectFrame_personOrEncounterColumn_neverEstimates(self, id_column):
        # Act
        table = detect_frame(_visits(id_column, [f"X{n:03d}" for n in range(40)]))

        # Assert
        assert table is None

    @pytest.mark.parametrize(
        ("label", "ids"),
        [
            pytest.param("code_like", [f"A{n:05d}" for n in range(15)], id="id_like_values"),
            pytest.param("numbers_as_text", [str(10000 + n) for n in range(15)], id="numeric_codes"),
            pytest.param("many_levels", [f"site-{chr(65 + n % 26)}{chr(65 + n // 26)}" for n in range(70)], id="many"),
            pytest.param(
                "repeated",
                [
                    f"{w} clinic"
                    for w in (
                        "north",
                        "south",
                        "east",
                        "west",
                        "river",
                        "hill",
                        "lake",
                        "park",
                        "bay",
                        "glen",
                        "oak",
                        "elm",
                        "ash",
                        "pine",
                        "fir",
                        "yew",
                        "bark",
                        "moss",
                        "fern",
                        "reed",
                        "sage",
                    )
                ],
                id="repeated_over_time",
            ),  # fmt: skip
        ],
    )
    def test_detectFrame_unrecognisedPeopleLikeKey_neverEstimates(self, label, ids):
        # Arrange: a key column named for nothing a public table is keyed by
        frame = _visits(label, ids)

        # Act
        table = detect_frame(frame)

        # Assert
        assert table is None

    def test_detectFrame_freeTextKey_neverEstimates(self):
        # Arrange: long free-text notes as a key (more than 20 levels)
        notes = [f"Patient reports intermittent symptoms since visit number {n}, follow up advised" for n in range(25)]

        # Act
        table = detect_frame(_visits("note", notes, months=2))

        # Assert
        assert table is None

    @pytest.mark.parametrize("disguise", ["location", "category", "indicator", "Year Code"])
    def test_detectFrame_idValuesUnderARecognisedName_neverEstimates(self, disguise):
        # Arrange: an MRN column renamed to a dimension a public table is keyed by
        frame = _visits(disguise, [f"MRN{n:04d}" for n in range(30)])

        # Act
        table = detect_frame(frame)

        # Assert
        assert table is None

    def test_detectFrame_categoriesOnlyWithConstantYear_neverEstimates(self):
        # Arrange: one patient per row by sex x age group x race, plus a one-value year column
        import itertools

        combos = list(itertools.product(("F", "M"), ("18-44", "45-64", "65+"), ("Black", "White", "Asian", "Other")))
        frame = pl.DataFrame(
            {
                "sex": [c[0] for c in combos],
                "age_group": [c[1] for c in combos],
                "race": [c[2] for c in combos],
                "year": [2023] * len(combos),
                "a1c_rate": [6.0 + n / 10 for n in range(len(combos))],
            }
        )

        # Act
        table = detect_frame(frame)

        # Assert
        assert table is None

    def test_detectFrame_weeklyTableWithNumericWeeks_stillEstimates(self):
        # Arrange: control, a public weekly table (weeks 1-52 are small numbers, not IDs)
        frame = pl.DataFrame(
            {
                "State": ["Michigan", "Ohio"] * 52,
                "week": [w for w in range(1, 53) for _ in range(2)],
                "rate": [1.0 + w / 10 for w in range(104)],
            }
        )

        # Act
        table = detect_frame(frame)

        # Assert
        assert table is not None

    def test_detectFrame_patientRowsShapedLikePublicTable_acceptedResidualRisk(self):
        # Arrange: no ID column; each row is one patient, unique by sex x age group x race x county,
        # with a rate. This is indistinguishable from a public county table by its structure: the
        # documented residual risk (ADR014 §7). The exact declaration value is the safeguard; an
        # undeclared upload of the same file never reaches the estimates path.
        import itertools

        combos = list(itertools.product(("F", "M"), ("18-44", "45-64", "65+"), ("Black", "White", "Asian"),
                                        ("Wayne County", "Kent County", "Oakland County")))[:40]  # fmt: skip
        frame = pl.DataFrame(
            {
                "sex": [c[0] for c in combos],
                "age_group": [c[1] for c in combos],
                "race": [c[2] for c in combos],
                "county": [c[3] for c in combos],
                "a1c_rate": [6.0 + n / 10 for n in range(len(combos))],
            }
        )

        # Act
        table = detect_frame(frame)

        # Assert
        assert table is not None

    @pytest.mark.parametrize(
        "shape",
        ["gbd", "wonder", "atlasplus", "aidsvu", "nndss", "fluvaxview", "places", "mdhhs_local"],
    )
    def test_detectFrame_publicShapes_stillEstimates(self, make_public_source, shape):
        # Act: control, every public table shape the fellow uploads is still accepted
        source = make_public_source(shape, name=shape)

        # Assert
        assert source.table is not None

    def test_answer_declaredBesidePatientFile_refused(self, make_public_source):
        # Arrange: item 4, a declared public table never sits beside a patient-level upload
        gbd = make_public_source("gbd", name="GBD", years=(2023,))
        patients = ej.SourceInput(
            "visits", "visits", _visits("mrn", [f"M{n}" for n in range(40)]), None, declared=False
        )

        # Act
        result = ej.answer([gbd, patients], "GBD HCV death rate and visits in Michigan, 2023")

        # Assert: refused before any level of the patient file is read or named
        assert result.status == "refused"
        assert result.message.startswith("visits wasn't marked")
        assert "M1" not in result.message


class TestVerificationReviewWrongAnswers:
    """Item 5 (N1-N5 and the "%" level)."""

    def test_valueKind_ageAdjustedDeathRate_isRate(self):
        # Act / Assert (N2)
        assert estimates.value_kind("Age-adjusted Death Rate") == "rate"

    def test_measureOf_metricIdBesideNamedMetric_namedMetricDecides(self, make_public_source):
        # Arrange (N1): metric_id = 3 beside metric_name = Rate (realistic GBD export)
        gbd = make_public_source("gbd", name="GBD", years=(2023,))
        value = gbd.table.value("val")

        # Act
        measure = ej.measure_of(gbd, value, {"metric_id": 3, "metric_name": "Rate", "age_name": "All ages"})

        # Assert
        assert measure.kind == "rate"
        assert measure.standardization == "crude"

    def test_measureOf_percentSignLevel_isPercent(self, make_public_source):
        # Arrange: a metric level written "%"
        gbd = make_public_source("gbd", name="GBD", years=(2023,))

        # Act
        measure = ej.measure_of(gbd, gbd.table.value("val"), {"metric_name": "%"})

        # Assert
        assert estimates.metric_level_kind("%") == "rate"
        assert measure.kind == "percent"

    def test_answer_georgiaBesideUsTerritories_staysTheState(self):
        # Arrange (N3): states plus PR, GU, VI, AS, MP: no foreign country, so Georgia is the state
        names = [s.name for s in STATES][:20] + ["Puerto Rico", "Guam", "U.S. Virgin Islands", "American Samoa",
                                                  "Northern Mariana Islands"]  # fmt: skip
        frame = pl.DataFrame([{"location_name": n, "year": 2023, "val": 10.0 + i, "lower": 9.0 + i, "upper": 11.0 + i}
                              for i, n in enumerate(names)])  # fmt: skip

        # Act
        result = ej.answer([_source(frame, "G")], "Rank states by the rate in 2023")

        # Assert
        assert result.status == "answered", result.message
        assert "Georgia" in {r["state"] for r in result.table["rows"]}
        assert not any("Georgia was left out" in n for n in result.notes)

    @pytest.mark.parametrize(
        ("question", "expected"),
        [
            ("Compare age-adjusted for WONDER, crude for GBD in 2023", {"WONDER": "age-standardized", "GBD": "crude"}),
            ("crude rates for GBD and age-adjusted rates for WONDER", {"GBD": "crude", "WONDER": "age-standardized"}),
            (
                "Does the WONDER crude death rate disagree with the GBD age-standardized estimate?",
                {"WONDER": "crude", "GBD": "age-standardized"},
            ),
        ],
    )
    def test_standardizationBySource_phraseBindsToItsTable(self, make_public_source, question, expected):
        # Arrange (N5)
        gbd = make_public_source("gbd", name="GBD", years=(2023,))
        wonder = make_public_source("wonder", name="WONDER", years=(2023,))

        # Act / Assert
        assert ej.standardization_by_source([gbd, wonder], question) == expected


def _monthly() -> ej.SourceInput:
    """A synthetic monthly table: ten states x twelve months of 2023 (made-up rates)."""
    rows = [
        {"State": s.name, "Month": f"2023-{m:02d}", "Rate per 100000": float(m * 10 + i)}
        for i, s in enumerate(STATES[:10])
        for m in range(1, 13)
    ]
    return _source(pl.DataFrame(rows), "monthly")


class TestAllowlistRoundJoins:
    """Round 4: months are not seasons, "respectively" binds in order, Georgia stays a state."""

    def test_roleOf_monthLevels_monthNotSeason(self):
        # Arrange
        table = _monthly().table
        assert table is not None

        # Act
        role = ej.role_of(table.dimension("Month"))

        # Assert
        assert role == "month"

    def test_answer_yearOnMonthlyTable_heldAllMonthsOrOneNeverDecember(self):
        # Act
        result = ej.answer([_monthly()], "Rate in Alaska in 2023")

        # Assert
        assert result.status == "held"
        [option] = result.options
        assert option.column == "Month"
        assert option.default == ej.ALL_MONTHS_CHOICE
        assert option.levels == (ej.ALL_MONTHS_CHOICE, *(f"2023-{m:02d}" for m in range(1, 13)))

    def test_answer_allMonthsConfirmed_eachMonthShown(self):
        # Arrange
        source = _monthly()
        oracle = source.frame.filter(pl.col("State") == "Alaska").sort("Month")

        # Act
        result = ej.answer([source], "Rate in Alaska in 2023", {"monthly": {"Month": ej.ALL_MONTHS_CHOICE}})

        # Assert
        assert result.status == "answered", result.message
        shown = [r["monthly: Rate per 100000"] for r in sorted(result.table["rows"], key=lambda r: r["month"])]
        assert shown == oracle.get_column("Rate per 100000").to_list()

    def test_answer_namedMonth_thatMonthOnly(self):
        # Arrange
        source = _monthly()
        oracle = source.frame.filter((pl.col("State") == "Alaska") & (pl.col("Month") == "2023-03"))

        # Act
        result = ej.answer([source], "Rate in Alaska in 2023-03")

        # Assert
        assert result.status == "answered", result.message
        assert [r["monthly: Rate per 100000"] for r in result.table["rows"]] == [
            oracle.get_column("Rate per 100000").item()
        ]

    @pytest.mark.parametrize(
        ("question", "expected"),
        [
            ("GBD and WONDER, age-standardized and crude respectively", {"GBD": "age-standardized", "WONDER": "crude"}),
            ("WONDER and GBD, crude and age-adjusted respectively", {"WONDER": "crude", "GBD": "age-standardized"}),
            ("GBD and WONDER, age-standardized respectively", {"GBD": None, "WONDER": None}),
            ("GBD and WONDER in 2019 and 2023, age-standardized and crude respectively", {"GBD": None, "WONDER": None}),
            ("GBD and WONDER for Ohio and Iowa, crude and age-adjusted respectively", {"GBD": None, "WONDER": None}),
            (
                "Age-adjusted for WONDER and crude for GBD, respectively",
                {"WONDER": "age-standardized", "GBD": "crude"},
            ),
        ],
    )
    def test_standardizationBySource_respectively_boundInOrder(self, question, expected):
        # Arrange
        sources = [
            ej.SourceInput(dataset_id=n, name=n, frame=pl.DataFrame(), table=None, declared=True)
            for n in ("WONDER", "GBD")
        ]

        # Act
        chosen = ej.standardization_by_source(sources, question)

        # Assert
        assert chosen == expected

    @pytest.mark.parametrize(
        "extra",
        [
            ["Puerto Rico", "Guam", "Virgin Islands, U.S."],
            ["Puerto Rico", "Virgin Islands of the U.S."],
            ["Northeast", "Midwest", "South", "West"],
            ["HHS Region 1", "HHS Region 2"],
            ["Unknown", "Other", "Total"],
        ],
    )
    def test_answer_usTableWithRegionsOrPlaceholders_georgiaKept(self, extra):
        # Arrange
        places = [s.name for s in STATES] + extra
        rows = [
            {"Indicator": "Primary and Secondary Syphilis", "Year": y, "Geography": p, "Rate per 100000": 10.0 + i / 10}
            for i, p in enumerate(places)
            for y in (2022, 2023)
        ]
        source = _source(pl.DataFrame(rows), "atlas")

        # Act
        result = ej.answer([source], "Rank states by syphilis rate in 2023", {"atlas": {}})
        if result.status == "held":
            result = ej.answer([source], "Rank states by syphilis rate in 2023", _held_defaults(result))

        # Assert
        assert result.status == "answered", result.message
        assert "Georgia" in {r["state"] for r in result.table["rows"]}

    def test_answer_tableListingCountries_georgiaLeftOut(self):
        # Arrange
        places = [s.name for s in STATES] + ["Armenia", "Azerbaijan"]
        rows = [{"Year": 2023, "Geography": p, "Rate per 100000": 10.0 + i / 10} for i, p in enumerate(places)]
        rows += [{"Year": 2022, "Geography": p, "Rate per 100000": 9.0 + i / 10} for i, p in enumerate(places)]
        source = _source(pl.DataFrame(rows), "mixed")

        # Act
        result = ej.answer([source], "Rank states by rate in 2023")

        # Assert
        assert result.status == "answered", result.message
        assert "Georgia" not in {r["state"] for r in result.table["rows"]}
        assert any("Georgia was left out" in n for n in result.notes)


def _held_defaults(result: ej.JoinAnswer) -> dict[str, dict[str, object]]:
    """The defaults a held answer offers, as the choices a reader would confirm."""
    chosen: dict[str, dict[str, object]] = {}
    for option in result.options:
        chosen.setdefault(option.dataset_id, {})[option.column] = option.default
    return chosen


def _annual(states: int = 10) -> ej.SourceInput:
    """A synthetic annual table with intervals: states x 2019-2023 (made-up rates)."""
    rows = [
        {"State": st.name, "Year": y, "Rate per 100000": float(10 + i), "lower": 9.0, "upper": 11.0}
        for i, st in enumerate(STATES[:states])
        for y in range(2019, 2024)
    ]
    return _source(pl.DataFrame(rows), "annual")


def _monthly_years() -> ej.SourceInput:
    rows = [
        {"State": st.name, "Month": f"{y}-{m:02d}", "Rate per 100000": float(m + i)}
        for i, st in enumerate(STATES[:10])
        for y in range(2019, 2024)
        for m in range(1, 13)
    ]
    return _source(pl.DataFrame(rows), "monthly")


class TestRoundFiveJoins:
    """Round 5 items 5-6: real intervals in the disagreement screen; months, seasons, Georgia."""

    def test_answer_disagreeWithOverlappingIntervals_noneDisagree(self):
        # Arrange: GBD 10 (9-11) against AtlasPlus 12 (8-16): the intervals overlap
        gbd = _source(
            pl.DataFrame(
                [
                    dict(measure_name="Incidence", location_name=st.name, sex_name="Both", age_name="All ages",
                         cause_name="Syphilis", metric_name="Rate", year=y, val=10.0, upper=11.0, lower=9.0)
                    for st in STATES[:30]
                    for y in (2022, 2023)
                ]
            ),
            "GBD",
        )  # fmt: skip
        atlas = _source(
            pl.DataFrame(
                [
                    {"Indicator": "Primary and Secondary Syphilis", "Year": y, "Geography": st.name, "Cases": 300,
                     "Rate per 100000": 12.0, "Lower 95% CI": 8.0, "Upper 95% CI": 16.0}
                    for st in STATES[:30]
                    for y in (2022, 2023)
                ]
            ),
            "ATLAS",
        )  # fmt: skip
        choices = {"ATLAS": {ej.VALUE_CHOICE: "Rate per 100000", ej.STANDARDIZATION_CHOICE: "crude"}}

        # Act
        result = ej.answer(
            [gbd, atlas], "Does the ATLAS rate disagree with GBD for syphilis in 2023 in all states?", choices
        )
        if result.status == "held":
            result = ej.answer(
                [gbd, atlas],
                "Does the ATLAS rate disagree with GBD for syphilis in 2023 in all states?",
                {**_held_defaults(result), **choices},
            )

        # Assert
        assert result.status == "answered", result.message
        rows = result.table["rows"]
        assert {r["reported vs interval"] for r in rows} == {"intervals overlap"}
        assert rows[0]["ATLAS lower"] == 8.0 and rows[0]["ATLAS upper"] == 16.0

    def test_answer_seasonTableWithOneYearColumn_alignedNoError(self):
        # Arrange: a season table that also carries a one-value Year column
        flu = _source(
            pl.DataFrame(
                [
                    {
                        "Geography": st.name,
                        "Season": se,
                        "Year": 2023,
                        "Estimate (%)": 50.0 + i,
                        "lower": 45.0,
                        "upper": 55.0,
                    }
                    for i, st in enumerate(STATES[:10])
                    for se in ("2022-23", "2023-24")
                ]
            ),
            "flu",
        )
        question = "Coverage and rate in Alaska in 2023"

        # Act
        held = ej.answer([flu, _annual()], question)
        answered = ej.answer([flu, _annual()], question, _held_defaults(held))

        # Assert
        assert held.status == "held"
        assert answered.status == "answered", answered.message

    def test_answer_byMonthOverYearRange_everyYearKept(self):
        # Act
        result = ej.answer([_monthly_years()], "Rate in Alaska by month from 2019 to 2023")

        # Assert
        assert result.status == "answered", result.message
        assert sorted({r["month"][:4] for r in result.table["rows"]}) == ["2019", "2020", "2021", "2022", "2023"]
        assert len(result.table["rows"]) == 60

    def test_answer_monthlyBesideAnnual_heldForOneMonthNeverAllMonths(self):
        # Arrange
        sources = [_monthly_years(), _annual()]
        question = "Rate in Alaska in 2023 in monthly and annual"

        # Act
        held = ej.answer(sources, question)
        chosen = ej.answer(sources, question, {"monthly": {"Month": "2023-03"}})

        # Assert
        assert held.status == "held"
        [option] = held.options
        assert ej.ALL_MONTHS_CHOICE not in option.levels
        assert option.levels == tuple(f"2023-{m:02d}" for m in range(1, 13))
        assert chosen.status == "answered", chosen.message
        oracle = sources[0].frame.filter((pl.col("State") == "Alaska") & (pl.col("Month") == "2023-03"))
        assert chosen.table["rows"][0]["monthly: Rate per 100000"] == oracle.get_column("Rate per 100000").item()

    @pytest.mark.parametrize("extra", ["Palau", "Marshall Islands", "Federated States of Micronesia"])
    def test_answer_freelyAssociatedState_georgiaKeptLevelNoted(self, extra):
        # Arrange
        places = [st.name for st in STATES] + [extra]
        source = _source(
            pl.DataFrame(
                [
                    {"Geography": p, "Year": y, "Rate per 100000": 1.0 + i}
                    for i, p in enumerate(places)
                    for y in (2022, 2023)
                ]
            ),
            "geo",
        )

        # Act
        result = ej.answer([source], "Rank states by rate in 2023")

        # Assert
        assert result.status == "answered", result.message
        assert "Georgia" in {r["state"] for r in result.table["rows"]}
        assert any(extra in n and "left out" in n for n in result.notes)


class TestRoundSixJoins:
    """Round 6 M1-M3: a named month against a year is said plainly; months never join as one year;
    a one-value year is confirmable; a table with no time is not annual; totals aren't territories."""

    def test_answer_namedMonthBesideAnnual_saidPlainly(self):
        # Arrange
        sources = [_monthly_years(), _annual()]

        # Act
        result = ej.answer(sources, "Rate in Alaska in 2023-03 in monthly and annual")

        # Assert
        assert result.status == "answered", result.message
        assert any("one month (2023-03)" in n and "whole year 2023" in n for n in result.notes)

    def test_answer_namedMonthDisagreeWithAnnual_refused(self):
        # Arrange
        sources = [_monthly_years(), _annual()]
        choices = {"monthly": {ej.STANDARDIZATION_CHOICE: "crude"}, "annual": {ej.STANDARDIZATION_CHOICE: "crude"}}

        # Act
        result = ej.answer(sources, "Does the monthly rate disagree with annual in Alaska in 2023-03?", choices)

        # Assert
        assert result.status == "refused"
        assert "single month" in result.message

    def test_answer_twoMonthlyTablesAtDifferentMonths_neverOneYear(self):
        # Arrange
        other = dataclasses.replace(_monthly_years(), dataset_id="monthly2", name="monthly2")
        sources = [_monthly_years(), other, _annual()]
        choices = {"monthly": {"Month": "2023-01"}, "monthly2": {"Month": "2023-03"}}

        # Act
        result = ej.answer(sources, "Rate in Alaska in 2023 in monthly, monthly2 and annual", choices)

        # Assert
        assert result.status == "refused"
        assert "different months" in result.message

    def test_answer_oneValueYearConfirmed_appliedNotRejected(self):
        # Arrange: a file whose only year is 2021, asked about 2022 beside an annual table
        one = _source(
            pl.DataFrame(
                [
                    {"State": st.name, "Year": 2021, "Rate per 100000": 7.0 + i, "Indicator": "Hepatitis C, chronic"}
                    for i, st in enumerate(STATES[:40])
                ]
            ),
            "ONE",
        )
        sources = [dataclasses.replace(_annual(), dataset_id="ANN", name="ANN"), one]
        question = "ANN and ONE rate in Alaska, 2022"

        # Act
        held = ej.answer(sources, question)
        chosen = _held_defaults(held)
        problems = [ej.choice_problem(one, chosen.get("ONE", {}))]
        answered = ej.answer(sources, question, chosen)

        # Assert
        assert held.status == "held"
        assert problems == [None]
        assert answered.status == "answered", answered.message

    def test_answer_monthlyBesideTableWithNoTime_notHeldAsAnnual(self):
        # Arrange: a table with no year at all is not an annual partner
        timeless = _source(
            pl.DataFrame([{"State": st.name, "Rate per 100000": 3.0 + i} for i, st in enumerate(STATES[:10])]),
            "timeless",
        )

        # Act
        result = ej.answer([_monthly_years(), timeless], "Rate in Alaska in 2023 in monthly and timeless")

        # Assert: refused for the missing month key, never held as if it were annual
        assert result.status == "refused"
        assert "no month" in result.message
        assert not result.options

    def test_harmonize_totalRow_notListedAsLeftOutPlace(self):
        # Arrange
        places = [st.name for st in STATES] + ["Total", "Guam"]
        source = _source(
            pl.DataFrame(
                [
                    {"Geography": p, "Year": y, "Rate per 100000": 1.0 + i}
                    for i, p in enumerate(places)
                    for y in (2022, 2023)
                ]
            ),
            "geo",
        )

        # Act
        result = ej.answer([source], "Rank states by rate in 2023")

        # Assert
        left_out = [n for n in result.notes if "left out (" in n]
        assert left_out and "Guam" in left_out[0] and "Total" not in left_out[0]


def _monthly_ci(name: str, factor: float = 1.0) -> ej.SourceInput:
    """Two years of monthly rates with consistent 95% CIs (made-up numbers)."""
    rows = [
        {
            "State": st.name,
            "Month": f"{y}-{m:02d}",
            "Rate per 100000": factor * (m + i + (y - 2022) * 10),
            "Lower 95% CI": factor * (m + i + (y - 2022) * 10 - 1),
            "Upper 95% CI": factor * (m + i + (y - 2022) * 10 + 1),
        }
        for i, st in enumerate(STATES[:10])
        for y in (2022, 2023)
        for m in range(1, 13)
    ]
    return _source(pl.DataFrame(rows), name)


class TestRoundSevenJoins:
    def test_answer_twoMonthlyTablesSameMonth_disagreeAnswered(self):
        # Arrange
        sources = [_monthly_ci("MA"), _monthly_ci("MB", factor=3.0)]
        choices = {
            "MA": {"Month": "2023-03", ej.STANDARDIZATION_CHOICE: "crude"},
            "MB": {"Month": "2023-03", ej.STANDARDIZATION_CHOICE: "crude"},
        }

        # Act
        result = ej.answer(sources, "Does the MA rate disagree with MB in all states in 2023?", choices)

        # Assert
        assert result.status == "answered", result.message
        assert not any("whole year" in n for n in result.notes)
        assert {r["reported vs interval"] for r in result.table["rows"]} == {"above"}

    def test_answer_twoMonthlyTablesSameMonth_changeComparesThatMonthEachYear(self):
        # Arrange
        sources = [_monthly_ci("MA"), _monthly_ci("MB", factor=3.0)]
        choices = {"MA": {"Month": "2023-03"}, "MB": {"Month": "2023-03"}}
        oracle = sources[0].frame.filter((pl.col("State") == "Alaska") & pl.col("Month").is_in(["2022-03", "2023-03"]))

        # Act
        result = ej.answer(sources, "How did the MA and MB rate change from 2022 to 2023 in Alaska?", choices)

        # Assert
        assert result.status == "answered", result.message
        assert not any("whole year" in n for n in result.notes)
        row = next(r for r in result.table["rows"] if r["table"] == "MA")
        assert [row["2022"], row["2023"]] == oracle.sort("Month").get_column("Rate per 100000").to_list()

    def test_answer_unknownAndOtherRows_namedWhenLeftOut(self):
        # Arrange
        places = [st.name for st in STATES] + ["Unknown", "Other", "Total"]
        source = _source(
            pl.DataFrame(
                [
                    {"Geography": p, "Year": y, "Rate per 100000": 1.0 + i}
                    for i, p in enumerate(places)
                    for y in (2022, 2023)
                ]
            ),
            "geo",
        )

        # Act
        result = ej.answer([source], "Rank states by rate in 2023")

        # Assert
        left_out = [n for n in result.notes if "left out (" in n]
        assert left_out and "Unknown" in left_out[0] and "Other" in left_out[0] and "Total" not in left_out[0]
