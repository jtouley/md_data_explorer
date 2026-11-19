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
        # A both-sexes count is also withheld when its Female or Male count is 1-10 (total minus the
        # part shown in another answer would give it back).
        parts = gbd.frame.filter(
            (pl.col("metric_name") == "Number") & (pl.col("age_name") == "All ages") & (pl.col("sex_name") != "Both")
        )
        small_part = set(parts.filter(pl.col("val") < config.MIN_CELL_SIZE).get_column("location_name").to_list())
        shown = oracle.filter((pl.col("val") >= config.MIN_CELL_SIZE) & ~pl.col("location_name").is_in(small_part))
        ranked = {r["state"] for r in result.table["rows"]}
        assert oracle.height - shown.height > 0
        assert ranked == set(shown.get_column("location_name").to_list())
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
    def test_answer_floatCountByAnyName_hiddenOnJoinPath(self, column):
        # Arrange: modelled counts 2.5-9.5 under a name that says neither rate nor count
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
        assert all(r[f"T: {column}"] == SECONDARY_SUPPRESSED for r in result.table["rows"])

    def test_answer_gbdOnlyNumberMetric_oneLevelMetricStillACount(self, make_public_table):
        # Arrange: metric_name is the constant "Number" (folded away as a one-value column)
        frame = make_public_table(
            "gbd", locations=("Michigan", "Ohio", "Indiana"), years=(2023,), metrics=("Number",), ages=("All ages",)
        ).with_columns(pl.col("val") / 100, pl.col("lower") / 100, pl.col("upper") / 100)
        source = _source(frame, "GBD")

        # Act
        result = ej.answer([source], "HCV deaths in Michigan, Ohio and Indiana, 2023, both sexes")

        # Assert
        assert source.table is not None
        assert source.table.metric_levels == {"metric_name": {"Number": "count"}}
        assert result.status == "answered", result.message
        assert all(r["GBD: val"] == SECONDARY_SUPPRESSED for r in result.table["rows"])


class TestSecondReviewWithheld:
    """Findings 2 and 3: nothing in a row with a small count, nor a total of it, is ever shown."""

    @staticmethod
    def _atlas_like() -> pl.DataFrame:
        cases = {"Michigan": 4, "Ohio": 7, "Wisconsin": 9}
        population = {"Michigan": 100_000, "Ohio": 350_000, "Wisconsin": 900_000}
        rows = []
        for s in [x.name for x in STATES][:20] + list(cases):
            c, p = cases.get(s, 150), population.get(s, 1_500_000)
            rows.append({"Geography": s, "Year": 2023, "Cases": c, "Population": p, "Rate per 100000": c / p * 1e5})
        return pl.DataFrame(rows).unique(subset=["Geography"], keep="first")

    @pytest.mark.parametrize("question", ["Syphilis rate per 100000 by state in 2023", "Population by state in 2023"])
    def test_answer_rateAndPopulationBesideSmallCount_withheld(self, question):
        # Arrange: rate x population would give the hidden count back
        frame = self._atlas_like()

        # Act
        result = ej.answer([_source(frame, "A")], question)

        # Assert
        assert result.status == "answered", result.message
        shown = {r["state"]: r for r in result.table["rows"]}
        for state in ("Michigan", "Ohio", "Wisconsin"):
            assert set(shown[state].values()) >= {SECONDARY_SUPPRESSED}
        assert config.ESTIMATE_WITHHELD_NOTE in result.notes

    def test_withheldMask_totalBesideSmallPart_totalWithheldEverywhere(self):
        # Arrange: Both = 46, Male = 41, Female = 5
        frame = pl.DataFrame(
            {
                "State": ["Alaska"] * 3 + ["Ohio"] * 3,
                "Sex": ["Both", "Female", "Male"] * 2,
                "Cases": [46, 5, 41, 80, 40, 40],
            }
        )
        table = detect_frame(frame)
        assert table is not None

        # Act
        mask = estimates.withheld_mask(frame, table)

        # Assert
        assert mask.to_list() == [True, True, False, False, False, False]

    def test_answer_totalAskedAlone_withheldWhenAPartIsSmall(self):
        # Arrange
        rows = []
        for i, s in enumerate([x.name for x in STATES][:20]):
            female = 5 if s == "Alaska" else 30 + i
            for sex, c in (("Both", 40 + i + female), ("Female", female), ("Male", 40 + i)):
                rows.append({"State": s, "Year": 2023, "Sex": sex, "Cases": c})
        source = _source(pl.DataFrame(rows), "S")

        # Act
        both = ej.answer([source], "How many cases in Alaska in 2023, both sexes?")
        men = ej.answer([source], "How many cases among men in Alaska in 2023?")

        # Assert: the man count is shown, the total never is (46 - 41 would give 5)
        assert both.table["rows"][0]["S: Cases"] == SECONDARY_SUPPRESSED
        assert men.table["rows"][0]["S: Cases"] == 41.0


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


class TestSecondReviewSharpEdges:
    def test_withheldMask_unnamedTotalBySum_withheldBesideSmallPart(self):
        # Arrange: "Statewide" is not a known total name, but it is the sum of the regions
        frame = pl.DataFrame(
            {
                "State": ["Alaska"] * 4 + ["Ohio"] * 4,
                "Region": ["Statewide", "North", "South", "West"] * 2,
                "Cases": [56, 5, 30, 21, 90, 30, 30, 30],
            }
        )
        table = detect_frame(frame)
        assert table is not None

        # Act
        mask = estimates.withheld_mask(frame, table)

        # Assert: Alaska's Statewide row (56) and its small region (5) are withheld; Ohio's are not
        assert mask.to_list() == [True, True, False, False, False, False, False, False]

    def test_isCountRow_countNamedColumnWithRateMetric_stillACount(self):
        # Arrange: a "Deaths" column on a row whose metric says Rate stays under the small-cell rule
        frame = pl.DataFrame({"State": ["Ohio", "Iowa"], "metric_name": ["Rate", "Rate"], "Deaths": [4.0, 50.0]})
        table = detect_frame(frame)
        assert table is not None
        deaths = table.value("Deaths")
        assert deaths is not None

        # Act
        is_count = estimates.is_count_row(deaths, table, {"metric_name": "Rate"})

        # Assert
        assert is_count is True
        assert estimates.withheld_mask(frame, table).to_list() == [True, False]
