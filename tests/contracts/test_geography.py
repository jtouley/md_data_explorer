# SPDX-License-Identifier: MIT
"""State keys for joining public tables (ADR014 §10): names, postal codes and FIPS meet on one
two-digit FIPS text key; the nation, counties, Detroit and out-Wayne are never states."""

from __future__ import annotations

import pytest

from clinical_analytics.contracts import geography


class TestClassify:
    @pytest.mark.parametrize(
        ("level", "kind", "fips"),
        [
            pytest.param("Michigan", "state", "26", id="name"),
            pytest.param("  michigan ", "state", "26", id="name_case_space"),
            pytest.param("MI", "state", "26", id="usps"),
            pytest.param(26, "state", "26", id="fips_int"),
            pytest.param("26", "state", "26", id="fips_text"),
            pytest.param(6, "state", "06", id="fips_int_one_digit"),
            pytest.param("06", "state", "06", id="fips_leading_zero"),
            pytest.param("26000", "state", "26", id="state_as_place_code"),
            pytest.param(26.0, "state", "26", id="fips_float_whole"),
            pytest.param("Michigan, US", "state", "26", id="name_country_suffix"),
            pytest.param("District of Columbia", "state", "11", id="dc"),
            pytest.param("MICHIGAN", "state", "26", id="upper_case_nndss"),
            pytest.param("United States", "nation", None, id="nation"),
            pytest.param("US", "nation", None, id="nation_abbrev"),
            pytest.param("26163", "substate", None, id="county_fips"),
            pytest.param("Wayne County", "substate", None, id="county_name"),
            pytest.param("Detroit", "substate", None, id="detroit"),
            pytest.param("Out-Wayne", "substate", None, id="out_wayne"),
            pytest.param("Wayne County (excluding Detroit)", "substate", None, id="wayne_excluding"),
            pytest.param("Puerto Rico", "territory", None, id="territory"),
            pytest.param("GU", "territory", None, id="territory_usps"),
            pytest.param(99, "unknown", None, id="no_such_fips"),
            pytest.param(None, "unknown", None, id="null"),
            pytest.param("", "unknown", None, id="empty"),
            pytest.param(True, "unknown", None, id="bool"),
            pytest.param(26.5, "unknown", None, id="fractional"),
        ],
    )
    def test_classify_level_kindAndFips(self, level, kind, fips):
        # Act: a column named for a place (codes allowed)
        place = geography.classify(level, codes=True)

        # Assert
        assert place.kind == kind
        assert (place.state.fips if place.state else None) == fips

    @pytest.mark.parametrize("level", [26, "26", 6, "26000", "26163", 26.0])
    def test_classify_bareNumberOutsidePlaceColumn_neverAPlace(self, level):
        # Act: a patient number or a week is not a FIPS code
        place = geography.classify(level)

        # Assert
        assert place.kind == "unknown"

    @pytest.mark.parametrize(
        ("column", "codes"),
        [("FIPS", True), ("State Code", True), ("LocationID", True), ("patient_id", False), ("week", False)],
    )
    def test_isGeoCodeColumn_byName(self, column, codes):
        # Act / Assert
        assert geography.is_geo_code_column(column) is codes


class TestQuestionWords:
    def test_stateNamed_westVirginia_notAlsoVirginia(self):
        # Act
        named = geography.state_named("Compare West Virginia and Kentucky")

        # Assert
        assert [s.usps for s in named] == ["WV", "KY"]

    def test_stateNamed_bothVirginias_bothFound(self):
        # Act
        named = geography.state_named("Virginia versus West Virginia")

        # Assert
        assert {s.usps for s in named} == {"VA", "WV"}

    def test_groupsNamed_greatLakes_eightStates(self):
        # Act
        groups = geography.groups_named("Michigan and the other Great Lakes states")

        # Assert
        assert groups == ["great lakes"]
        assert len(geography.STATE_GROUPS["great lakes"]) == 8

    @pytest.mark.parametrize(
        ("level", "out_wayne", "detroit"),
        [
            ("Out-Wayne", True, False),
            ("Out Wayne", True, False),
            ("Wayne County excluding Detroit", True, False),
            ("Wayne County", False, False),
            ("Detroit", False, True),
            ("City of Detroit", False, True),
            ("Detroit Metro Area", False, False),
            (None, False, False),
        ],
    )
    def test_detroitAndOutWayne_recognised(self, level, out_wayne, detroit):
        # Act / Assert
        assert geography.is_out_wayne(level) is out_wayne
        assert geography.is_detroit(level) is detroit
