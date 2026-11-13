"""Mixed units (DQ M3): two value clusters at a unit pair's factor, each plausible in its unit.

Synthetic values only: log-normal quantiles (tests.eval.dq_probes.lognormal_values), so every
distribution is deterministic and shaped like a lab result.
"""

from __future__ import annotations

import math

import numpy as np
import polars as pl
import pytest

from clinical_analytics.contracts import config
from clinical_analytics.contracts.builder import find_plausibility_range
from clinical_analytics.contracts.declarations import Declarations, OneUnitDeclaration
from clinical_analytics.contracts.unit_mix import detect_unit_mix, header_unit, unit_pair_for
from tests.eval.dq_probes import interleave, lognormal_values, mixed_glucose_frame, skewed_glucose_frame


def _mixed(variable: str, low_centre: float, low_n: int, high_n: int, log_sd: float, decimals: int = 2) -> pl.Series:
    """``low_n`` values in the pair's low unit and ``high_n`` of the same patients' kind in its
    high unit (centre converted with the pair's factor)."""
    pair = unit_pair_for(variable)
    assert pair is not None
    high_centre = low_centre * pair.scale + pair.offset
    values = interleave(
        lognormal_values(high_centre, log_sd, high_n, decimals), lognormal_values(low_centre, log_sd, low_n, decimals)
    )
    return pl.Series(variable, values)


class TestDetectUnitMix:
    @pytest.mark.parametrize(
        ("variable", "low_centre", "log_sd", "low_unit", "high_unit"),
        [
            pytest.param("glucose", 6.5, 0.25, "mmol/L", "mg/dL", id="glucose_mmol_mg"),
            pytest.param("cholesterol", 5.0, 0.2, "mmol/L", "mg/dL", id="cholesterol"),
            pytest.param("ldl", 3.0, 0.25, "mmol/L", "mg/dL", id="ldl"),
            pytest.param("hdl", 1.3, 0.25, "mmol/L", "mg/dL", id="hdl"),
            pytest.param("triglycerides", 1.7, 0.35, "mmol/L", "mg/dL", id="triglycerides"),
            pytest.param("creatinine", 1.0, 0.3, "mg/dL", "µmol/L", id="creatinine_mg_umol"),
            pytest.param("temperature", 37.0, 0.02, "°C", "°F", id="temperature_c_f"),
            pytest.param("weight", 75.0, 0.1, "kg", "lb", id="weight_kg_lb"),
            pytest.param("height", 67.0, 0.05, "in", "cm", id="height_in_cm"),
        ],
    )
    def test_detect_twoUnitsOfOnePair_foundWithBothUnits(
        self, variable: str, low_centre: float, log_sd: float, low_unit: str, high_unit: str
    ) -> None:
        # Arrange
        series = _mixed(variable, low_centre, low_n=60, high_n=140, log_sd=log_sd)

        # Act
        mix = detect_unit_mix(variable, series)

        # Assert
        assert mix is not None
        assert {mix.target, mix.other} == {low_unit, high_unit}
        assert (mix.low_count, mix.high_count) == (60, 140)
        assert mix.target == high_unit  # the larger cluster's unit

    @pytest.mark.parametrize(
        ("variable", "values"),
        [
            # Glucose of a diabetic clinic: normoglycaemic and hyperglycaemic groups, right-skewed.
            pytest.param("glucose", skewed_glucose_frame().get_column("glucose").to_list(), id="glucose_two_groups"),
            pytest.param("glucose", lognormal_values(130, 0.45, 2000, 0), id="glucose_wide_lognormal"),
            # Triglycerides are strongly right-skewed (fasting and post-prandial, chylomicronaemia).
            pytest.param("triglycerides", lognormal_values(150, 0.7, 500, 0), id="triglycerides_skewed"),
            pytest.param(
                "creatinine",
                interleave(lognormal_values(0.9, 0.2, 170), lognormal_values(4.0, 0.5, 30)),
                id="creatinine_ckd_tail",
            ),
            # Children beside adults, all in kg: 3.9x apart, not the 2.2x of kg/lb.
            pytest.param(
                "weight", interleave(lognormal_values(20, 0.35, 60), lognormal_values(78, 0.18, 140)), id="weight_kids"
            ),
            pytest.param("height", lognormal_values(120, 0.35, 300), id="height_children_wide"),
            pytest.param("temperature", lognormal_values(37.2, 0.03, 300), id="temperature_fever"),
        ],
    )
    def test_detect_singleUnitWideOrSkewed_notAMix(self, variable: str, values: list[float]) -> None:
        # Arrange
        series = pl.Series(variable, values)

        # Act
        mix = detect_unit_mix(variable, series)

        # Assert
        assert mix is None

    @pytest.mark.parametrize(
        "values",
        [
            pytest.param([], id="empty"),
            pytest.param([None] * 50, id="all_null"),
            pytest.param([6.0, 110.0], id="too_few_values"),
            pytest.param([100.0] * 200, id="one_value"),
            pytest.param([0.0] * 100 + [-1.0] * 100, id="no_positive_values"),
        ],
    )
    def test_detect_degenerateColumn_notAMix(self, values: list[float | None]) -> None:
        # Arrange
        series = pl.Series("glucose", values, dtype=pl.Float64)

        # Act / Assert
        assert detect_unit_mix("glucose", series) is None

    def test_detect_twoClustersAtWrongRatio_notAMix(self) -> None:
        # Arrange: clusters 4x apart (a post-load subgroup), not glucose's 18x
        series = pl.Series("glucose", interleave(lognormal_values(90, 0.1, 100), lognormal_values(360, 0.1, 100)))

        # Act / Assert
        assert detect_unit_mix("glucose", series) is None

    def test_detect_clusterImplausibleInItsUnit_notAMix(self) -> None:
        # Arrange: 18x apart, but the upper cluster (~9,000) is above any glucose in mg/dL
        series = pl.Series("glucose", interleave(lognormal_values(500, 0.1, 100), lognormal_values(9000, 0.1, 100)))

        # Act / Assert
        assert detect_unit_mix("glucose", series) is None

    @pytest.mark.parametrize(
        ("values", "low_count", "high_count"),
        [
            # On the branch head: "Mean glucose was 12.12" (true about 6.1 mmol/L), 0/200 detected.
            pytest.param(interleave(lognormal_values(6.0, 0.1, 38), [110.0, 145.0]), 38, 2, id="two_of_forty_mg_dl"),
            pytest.param([5.5, 100.0, 6.0, 110.0, 6.2], 3, 2, id="five_values"),
            pytest.param(lognormal_values(110, 0.2, 59) + [6.1], 1, 59, id="one_mmol_value_set_apart"),
        ],
    )
    def test_detect_smallMinorityAtTheFactor_foundBelowClusterMinimum(
        self, values: list[float], low_count: int, high_count: int
    ) -> None:
        # Arrange
        series = pl.Series("glucose", values)

        # Act
        mix = detect_unit_mix("glucose", series)

        # Assert
        assert mix is not None and mix.reason == "factor"
        assert (mix.low_count, mix.high_count) == (low_count, high_count)

    def test_detect_valuesImpossibleInMainUnit_foundWhateverTheFit(self) -> None:
        # Arrange: 60 and 62 are impossible as °C and plausible as °F, though not at 98.6
        series = pl.Series("temperature", [37.0 + (i % 7) / 10 for i in range(38)] + [60.0, 62.0])

        # Act
        mix = detect_unit_mix("temperature", series)

        # Assert
        assert mix is not None and mix.reason == "implausible"
        assert (mix.target, mix.other, mix.low_count, mix.high_count) == ("°C", "°F", 38, 2)
        assert mix.one_units == ()  # neither unit holds every value

    def test_detect_wideClusterTails_assignedByCutBetweenCentres(self) -> None:
        # Arrange: a narrow mmol/L cluster and a wide mg/dL one; the low tail (3.0) is far below
        # the narrow cluster, where only the wide curve has any density
        values = lognormal_values(6.5, 0.03, 100) + [3.0] + lognormal_values(110, 0.4, 100, 0)
        series = pl.Series("glucose", values)

        # Act
        mix = detect_unit_mix("glucose", series)

        # Assert: 3.0 is counted with the values below the cut, as mmol/L
        assert mix is not None
        assert (mix.low_count, mix.high_count) == (101, 100)

    def test_detect_childrenBesideAdultsInKg_heldWithBothOneUnitOptions(self) -> None:
        # Arrange: 28 kg beside 65 kg is 2.3x apart, inside the kg/lb window
        series = pl.Series("weight", interleave(lognormal_values(28, 0.15, 100), lognormal_values(65, 0.15, 100)))

        # Act
        mix = detect_unit_mix("weight", series)
        named = detect_unit_mix("weight_kg", series.alias("weight_kg"))

        # Assert: the doctor can say "all kg"; a header naming kg offers kg only
        assert mix is not None and set(mix.one_units) == {"kg", "lb"}
        assert named is not None and named.one_units == ("kg",) and named.target == "kg"

    def test_detect_thirtyKgBesideThirtyLb_caughtInEverySeededDraw(self) -> None:
        # Arrange: 30 weights near 75 kg and 30 near 165 lb (log-sd 0.18), 200 seeded draws. On
        # the branch head 27 of these 200 were missed (BIC gain under 10), and shown as answers.
        draws = []
        for seed in range(200):
            rng = np.random.default_rng(seed)
            kg = np.exp(rng.normal(math.log(75), 0.18, 30))
            lb = np.exp(rng.normal(math.log(165), 0.18, 30))
            draws.append(pl.Series("weight", np.round(np.concatenate([kg, lb]), 1)))

        # Act
        missed = [seed for seed, series in enumerate(draws) if detect_unit_mix("weight", series) is None]

        # Assert
        assert missed == []

    def test_detect_columnWithoutUnitPair_notChecked(self) -> None:
        # Arrange: crp has a plausibility range but no unit pair (mg/L only)
        series = pl.Series("crp", interleave(lognormal_values(5, 0.2, 100), lognormal_values(90, 0.2, 100)))

        # Act / Assert
        assert detect_unit_mix("crp", series) is None

    def test_detect_distinctValuesWithCounts_sameAsRows(self) -> None:
        # Arrange
        frame = mixed_glucose_frame()
        counted = frame.get_column("glucose").value_counts(name="n")

        # Act
        by_rows = detect_unit_mix("glucose", frame.get_column("glucose"))
        by_counts = detect_unit_mix("glucose", counted.get_column("glucose"), counted.get_column("n"))

        # Assert
        assert by_rows == by_counts
        assert by_rows is not None and by_rows.present == frame.height

    def test_detect_textNumbers_readAsNumbers(self) -> None:
        # Arrange
        values = [str(v) for v in mixed_glucose_frame().get_column("glucose").to_list()]

        # Act
        mix = detect_unit_mix("glucose", pl.Series("glucose", values))

        # Assert
        assert mix is not None and {mix.target, mix.other} == {"mg/dL", "mmol/L"}


class TestCutoffAndRule:
    def test_rule_separatedClusters_roundCutoffBetweenThem(self) -> None:
        # Arrange
        series = mixed_glucose_frame().get_column("glucose")

        # Act
        mix = detect_unit_mix("glucose", series)
        rule = mix.rule("v1") if mix else None

        # Assert
        assert rule is not None
        low_max = series.filter(series < 30).max()
        high_min = series.filter(series > 30).min()
        assert low_max < rule.cutoff < high_min
        assert rule.cutoff == 30.0  # a round number, never a patient's value
        assert (rule.side, rule.source_unit, rule.target_unit) == ("below", "mmol/L", "mg/dL")
        assert rule.scale == pytest.approx(18.016)
        assert rule.relation == "1 mmol/L = 18.016 mg/dL"
        assert rule.dataset_version == "v1"

    def test_rule_overlappingClusters_noRuleOffered(self) -> None:
        # Arrange: values in the valley between the clusters (24-30: 24 mmol/L or 24 mg/dL?)
        values = mixed_glucose_frame().get_column("glucose").to_list() + [24.0, 27.0, 30.0]

        # Act
        mix = detect_unit_mix("glucose", pl.Series("glucose", values))

        # Assert
        assert mix is not None
        assert mix.cutoff is None and mix.rule("v1") is None

    def test_rule_majorityInLowUnit_convertsHighValuesDown(self) -> None:
        # Arrange: 150 mmol/L, 50 mg/dL
        series = mixed_glucose_frame(mmol_rows=150, mg_rows=50).get_column("glucose")

        # Act
        mix = detect_unit_mix("glucose", series)
        rule = mix.rule(None) if mix else None

        # Assert
        assert rule is not None
        assert (rule.side, rule.source_unit, rule.target_unit) == ("at_or_above", "mg/dL", "mmol/L")
        assert rule.scale == pytest.approx(1 / 18.016)

    def test_rule_headerNamesUnit_convertsIntoHeaderUnit(self) -> None:
        # Arrange: most values are mg/dL, but the header says mmol/L
        series = mixed_glucose_frame().get_column("glucose").alias("Glucose (mmol/L)")

        # Act
        mix = detect_unit_mix("Glucose (mmol/L)", series)

        # Assert
        assert mix is not None and mix.target == "mmol/L"

    def test_rule_temperature_affineConversion(self) -> None:
        # Arrange
        series = _mixed("temperature", 37.0, low_n=150, high_n=50, log_sd=0.02, decimals=1)

        # Act
        mix = detect_unit_mix("temperature", series)
        rule = mix.rule(None) if mix else None

        # Assert
        assert rule is not None
        assert (rule.source_unit, rule.target_unit) == ("°F", "°C")
        assert 98.6 * rule.scale + rule.offset == pytest.approx(37.0)
        assert rule.relation == "°F = °C × 1.8 + 32"


class TestUnitPairs:
    @pytest.mark.parametrize(
        ("column", "variable"),
        [
            pytest.param("glucose", "glucose", id="bare"),
            pytest.param("Glucose (mg/dL)", "glucose", id="header_unit"),
            pytest.param("Serum creatinine", "creatinine", id="alias_with_words"),
            pytest.param("creatinine umol/L", "creatinine", id="creatinine_umol"),
            pytest.param("weight_kg", "weight", id="weight_kg"),
            pytest.param("Height (cm)", "height", id="height_cm"),
            pytest.param("Temp (°F)", "temperature", id="temp_degrees_f"),
            pytest.param("Temperature (°C)", "temperature", id="temperature_degrees_c"),
            pytest.param("temp_c", "temperature", id="temp_c"),
            pytest.param("temp_f", "temperature", id="temp_f"),
            pytest.param("Creatinine (µmol/L)", "creatinine", id="creatinine_micro_sign"),
            pytest.param("Creatinine (\u03bcmol/L)", "creatinine", id="creatinine_greek_mu"),
            pytest.param("creatinine_umol", "creatinine", id="creatinine_umol"),
            pytest.param("Glucose (mmol/L)", "glucose", id="glucose_mmol_l"),
            pytest.param("ht", None, id="ht_could_be_hypertension"),
            pytest.param("crp", None, id="no_pair"),
            pytest.param("age", None, id="not_unit_dependent"),
        ],
    )
    def test_unitPairFor_column_matchesConfiguredPair(self, column: str, variable: str | None) -> None:
        # Act
        pair = unit_pair_for(column)

        # Assert
        assert (pair.variable if pair else None) == variable

    @pytest.mark.parametrize(
        ("column", "unit"),
        [
            pytest.param("Glucose (mg/dL)", "mg/dL", id="mg_dl"),
            pytest.param("glucose_mmol_l", "mmol/L", id="mmol_l"),
            pytest.param("weight_lbs", "lb", id="lb"),
            pytest.param("Temp (°F)", "°F", id="degrees_f"),
            pytest.param("Temperature (°C)", "°C", id="degrees_c"),
            pytest.param("temp_c", "°C", id="temp_c"),
            pytest.param("temp_f", "°F", id="temp_f"),
            pytest.param("Creatinine (µmol/L)", "µmol/L", id="micro_mol"),
            pytest.param("creatinine_umol", "µmol/L", id="umol"),
            pytest.param("Creatinine (mg/dL)", "mg/dL", id="creatinine_mg_dl"),
            pytest.param("Glucose (mmol/L)", "mmol/L", id="glucose_mmol_l"),
            pytest.param("glucose", None, id="none"),
        ],
    )
    def test_headerUnit_unitSuffix_named(self, column: str, unit: str | None) -> None:
        # Arrange
        pair = unit_pair_for(column)
        assert pair is not None

        # Act / Assert
        assert header_unit(column, pair) == unit

    def test_unitPairs_everyFactorSourcedAndRangesConsistent(self) -> None:
        # Act / Assert
        for pair in config.UNIT_PAIRS:
            assert pair.source, pair.variable
            assert pair.low.maximum * pair.scale + pair.offset == pytest.approx(pair.high.maximum, rel=0.01)
            assert all(hint in config.UNIT_HINTS for hint in (*pair.low_hints, *pair.high_hints)) or pair.variable in {
                "temperature",
                "weight",
                "height",
            }

    def test_unitPairs_labFactorsMatchMolarMass(self) -> None:
        # Arrange: mg/dL per mmol/L is molar mass / 10 (Young 1987)
        molar_mass = {"glucose": 180.16, "cholesterol": 386.65, "ldl": 386.65, "hdl": 386.65}
        scales = {pair.variable: pair.scale for pair in config.UNIT_PAIRS}

        # Act / Assert
        for variable, mass in molar_mass.items():
            assert scales[variable] == pytest.approx(mass / 10, rel=1e-3)
        assert scales["creatinine"] == pytest.approx(10_000 / 113.12, rel=1e-3)
        assert scales["weight"] == pytest.approx(1 / 0.45359237)
        assert math.isclose(scales["height"], 2.54)


class TestHeaderUnits:
    @pytest.mark.parametrize(
        ("column", "variable", "low_centre", "log_sd"),
        [
            pytest.param("Temp (°F)", "temperature", 37.0, 0.02, id="temp_f"),
            pytest.param("temp_c", "temperature", 37.0, 0.02, id="temp_c"),
            pytest.param("Creatinine (µmol/L)", "creatinine", 1.0, 0.3, id="creatinine_umol"),
        ],
    )
    def test_detect_headerNamesOneUnit_mixStillFound(
        self, column: str, variable: str, low_centre: float, log_sd: float
    ) -> None:
        # Arrange: a header naming a unit is evidence, not proof; on the branch head these
        # headers switched the check off ("Mean Creatinine (µmol/L) was 49.37")
        series = _mixed(variable, low_centre, low_n=100, high_n=100, log_sd=log_sd).alias(column)

        # Act
        mix = detect_unit_mix(column, series)

        # Assert
        pair = unit_pair_for(column)
        assert mix is not None and pair is not None
        assert mix.target == header_unit(column, pair)

    def test_findPlausibilityRange_unitHeaders_rangeFound(self) -> None:
        # Act / Assert: the entry-error check reads these headers too
        assert {
            column: (found.variable if (found := find_plausibility_range(column)) else None)
            for column in ("Temp (°F)", "temp_c", "Creatinine (µmol/L)", "creatinine_umol")
        } == {
            "Temp (°F)": "temperature",
            "temp_c": "temperature",
            "Creatinine (µmol/L)": "creatinine",
            "creatinine_umol": "creatinine",
        }

    def test_unitHints_noHintThatNormalisedHeadersCannotMatch(self) -> None:
        # Act / Assert: headers are normalised to a-z and 0-9, so "µ" in a hint never matches
        hints = [*config.UNIT_HINTS, *(h for p in config.UNIT_PAIRS for h in (*p.low_hints, *p.high_hints))]
        assert all(hint.isascii() and hint.isalnum() for hint in hints)


class TestUnitDeclarations:
    def test_withUnit_secondRuleSameColumn_keptAfterTheFirst(self) -> None:
        # Arrange: the live layer applied both; a rebuilt one must apply both, in order
        mix = detect_unit_mix("glucose", mixed_glucose_frame().get_column("glucose"))
        assert mix is not None and (first := mix.rule("v1")) is not None
        second = first.model_copy(update={"cutoff": 3.0})

        # Act
        declarations = Declarations().with_unit(first).with_unit(second).with_unit(first)

        # Assert
        assert declarations.units_for("v1") == (first, second)

    def test_withOneUnit_sameColumnTwice_keepsTheLatest(self) -> None:
        # Act
        declarations = (
            Declarations()
            .with_one_unit(OneUnitDeclaration(column="weight", unit="lb", dataset_version="v1"))
            .with_one_unit(OneUnitDeclaration(column="weight", unit="kg", dataset_version="v1"))
        )

        # Assert
        assert declarations.one_unit_for("v1") == {"weight": "kg"}
        assert declarations.one_unit_for("v2") == {}

    def test_withUnit_sameColumnTwice_keepsOneRule(self) -> None:
        # Arrange
        mix = detect_unit_mix("glucose", mixed_glucose_frame().get_column("glucose"))
        assert mix is not None
        rule = mix.rule("v1")
        assert rule is not None

        # Act
        declarations = Declarations().with_unit(rule).with_unit(rule)

        # Assert
        assert declarations.units_for("v1") == (rule,)
        assert declarations.units_for("v2") == ()
