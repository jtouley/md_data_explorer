# SPDX-License-Identifier: MIT
"""The public-shape allowlist (ADR014 §7): the estimates path is open only when every key column
is a recognised public dimension by its values, and no other column looks like a person or
record. Fail closed. Synthetic frames only; every value is made up.

Regression tests for the round-4 probes (g1 guard, g2 grids, item 5 over-refusals, Georgia).
"""

from __future__ import annotations

import datetime as dt
import hashlib

import polars as pl
import pytest

from clinical_analytics.contracts import geography
from clinical_analytics.contracts.geography import STATES
from clinical_analytics.core import public_shape
from clinical_analytics.core.estimates import detect_frame

_STATE_NAMES = [s.name for s in STATES]
_MONTHS = ["2024-01", "2024-02", "2024-03"]


def _visits(idcols, n: int = 40, k: int = 3) -> pl.DataFrame:
    """Synthetic repeated visits per person: an ID-ish column, a month, sex and a lab value."""
    return pl.DataFrame(
        [
            {**idcols(p), "visit_month": _MONTHS[v], "sex": "Female" if p % 3 else "Male", "hba1c_pct": 5.0 + p % 6}
            for p in range(1, n + 1)
            for v in range(k)
        ]
    )


def _hash(p: int, n: int) -> str:
    return hashlib.sha1(str(p).encode()).hexdigest()[:n]  # noqa: S324 - synthetic IDs, not security


class TestClassifyValue:
    @pytest.mark.parametrize(
        ("value", "column", "expected"),
        [
            pytest.param("Alaska", "NAME", "place", id="census_NAME_state"),
            pytest.param("Region 4", "Region", "place", id="hhs_region"),
            pytest.param("HHS Region 10", "Region", "place", id="hhs_region_10"),
            pytest.param("Northeast", "Region", "place", id="census_region"),
            pytest.param("202401", "MMWR Week", "time", id="mmwr_week"),
            pytest.param("2023-01", "Month", "time", id="month"),
            pytest.param("2022-23", "Season", "time", id="flu_season"),
            pytest.param("18-34", "Age Group", "age", id="age_band"),
            pytest.param("Female", "Sex", "sex", id="sex"),
            pytest.param("Total", "Sex", "total", id="total"),
            pytest.param(None, "Sex", "placeholder", id="null_is_placeholder"),
            pytest.param("", "Gender", "placeholder", id="empty_is_placeholder"),
            pytest.param("MRN0001", "location", None, id="mrn_not_place"),
            pytest.param("1-1", "age_group", None, id="person_band_not_age"),
            pytest.param("Alabamaville", "State", None, id="prefix_never_matches"),
        ],
    )
    def test_classifyValue_wholeValueOnly_recognisedOrNone(self, value, column, expected):
        # Act
        kind = public_shape.classify_value(value, column)

        # Assert
        assert kind == expected


class TestKeyColumn:
    def test_keyColumn_icdCodesInCauseColumn_acceptedAsLabels(self):
        # Act
        verdict = public_shape.key_column("UCD - ICD-10 113 Cause List", ["I21", "I25", "C34"], {})

        # Assert
        assert verdict.ok

    @pytest.mark.parametrize(
        ("column", "values"),
        [
            pytest.param("location", [f"MRN{n:04d}" for n in range(10)], id="mrn"),
            pytest.param("group", [_hash(n, 6) for n in range(10)], id="hash"),
            pytest.param("cause", ["AB", "CD", "EF"], id="initials"),
            pytest.param("Indicator", ["Jane Smith", "John Brown"], id="person_names"),
        ],
    )
    def test_keyColumn_notPublicDimension_refused(self, column, values):
        # Act
        verdict = public_shape.key_column(column, values, {})

        # Assert
        assert not verdict.ok


class TestGuardRegressions:
    """Probe g1/g2: patient and visit files never pass, whatever their ID column is called."""

    @pytest.mark.parametrize(
        "idcols",
        [
            pytest.param(lambda p: {"person": f"P{p:04d}"}, id="person"),
            pytest.param(lambda p: {"member_no": f"M{p:05d}"}, id="member_no"),
            pytest.param(lambda p: {"enc": f"E{p:05d}"}, id="enc"),
            pytest.param(lambda p: {"Client": f"C{p:04d}"}, id="Client"),
            pytest.param(lambda p: {"location": f"MRN{p:04d}"}, id="location_mrn"),
            pytest.param(lambda p: {"Year Code": f"MRN{p:04d}"}, id="year_code_mrn"),
            pytest.param(lambda p: {"location": _hash(p, 6)}, id="location_hash6"),
            pytest.param(lambda p: {"location": _hash(p, 12)}, id="location_hash12"),
            pytest.param(lambda p: {"group": _hash(p, 6)}, id="group_hash6"),
            pytest.param(lambda p: {"cause": _hash(p, 6)}, id="cause_hash6"),
            pytest.param(lambda p: {"State Code": p}, id="state_code_ints"),
            pytest.param(lambda p: {"fips": f"{p:02d}"}, id="fips_text"),
            pytest.param(lambda p: {"Year Code": 1950 + p}, id="year_code_ints"),
            pytest.param(lambda p: {"cohort": 1950 + p}, id="cohort_years"),
            pytest.param(lambda p: {"site": "ABCD"[p % 4], "seq": p // 4 + 1}, id="site_seq"),
            pytest.param(lambda p: {"ward": "ABCD"[p % 4], "bed": p // 4 + 1}, id="ward_bed"),
            pytest.param(lambda p: {"age_group": f"{p}-{p}"}, id="age_group_person"),
            pytest.param(lambda p: {"age": p}, id="age_ints"),
            pytest.param(lambda p: {"week": p}, id="week_ints"),
        ],
    )
    def test_detectFrame_visitFileAnyIdColumn_neverEstimates(self, idcols):
        # Arrange
        frame = _visits(idcols)

        # Act
        table = detect_frame(frame)

        # Assert
        assert table is None

    @pytest.mark.parametrize(
        ("idcols", "n"),
        [
            pytest.param(lambda p: {"tag": f"T{p}"}, 9, id="tag_small_n"),
            pytest.param(lambda p: {"initials": "AB"[p % 2] + "CDEFGHJKLMNPQRS"[p - 1]}, 15, id="initials"),
        ],
    )
    def test_detectFrame_smallVisitFile_neverEstimates(self, idcols, n):
        # Act
        table = detect_frame(_visits(idcols, n=n))

        # Assert
        assert table is None

    @staticmethod
    def _grid() -> pl.DataFrame:
        """One patient per (age band, sex, site, month) cell: every key value is public-shaped
        except the site."""
        rows = []
        for b_index, band in enumerate(("18-34", "35-49", "50-64", "65+")):
            for sex in ("Female", "Male"):
                for s_index, site in enumerate(("North clinic", "South clinic", "East clinic")):
                    for m_index, month in enumerate(("2024-01", "2024-02", "2024-03", "2024-04")):
                        if (b_index + s_index + m_index) % 3:
                            rows.append(
                                {"age_group": band, "sex": sex, "site": site, "dx_month": month, "egfr": 60 + m_index}
                            )
        return pl.DataFrame(rows)

    def test_detectFrame_onePatientPerCellGrid_neverEstimates(self):
        # Act
        table = detect_frame(self._grid())

        # Assert: no place or time key with an interval, rate or total; "site" is not a label
        assert table is None

    @pytest.mark.parametrize("prefix", ["MRN", "Q"])
    def test_detectFrame_gridWithRecordIdAttribute_neverEstimates(self, prefix):
        # Arrange
        grid = self._grid()
        frame = grid.with_columns(pl.format(f"{prefix}{{}}", pl.int_range(1000, 1000 + grid.height)).alias("ref"))

        # Act
        table = detect_frame(frame)

        # Assert
        assert table is None

    def test_detectFrame_fewVisitsWithFakeTotalRow_neverEstimates(self):
        # Arrange
        rows = [{"ref": f"R{p}", "visit_month": m, "hba1c_pct": 5.0 + p} for p in range(1, 9) for m in _MONTHS]
        rows += [{"ref": "Total", "visit_month": m, "hba1c_pct": 7.5} for m in _MONTHS]

        # Act
        table = detect_frame(pl.DataFrame(rows))

        # Assert
        assert table is None


class TestOverRefusals:
    """Item 5: public tables the allowlist must still accept."""

    @pytest.mark.parametrize(
        ("frame", "keys"),
        [
            pytest.param(
                pl.DataFrame(
                    [
                        {"NAME": s, "state": f"{i + 1:02d}", "year": y, "B01001_001E": 1000.0 + i, "B01001_001M": 50.0}
                        for i, s in enumerate(_STATE_NAMES[:12])
                        for y in (2021, 2022)
                    ]
                ),
                ["NAME", "year"],
                id="census_NAME",
            ),
            pytest.param(
                pl.DataFrame(
                    [{"Region": f"Region {r}", "Season": "2023-24", "Rate per 100000": 1.0 + r} for r in range(1, 11)]
                ),
                ["Region"],
                id="hhs_region_one_row_each",
            ),
            pytest.param(
                pl.DataFrame(
                    [
                        {"State": s, "MMWR Week": 202400 + w, "Rate": 1.0 + w}
                        for s in _STATE_NAMES[:12]
                        for w in range(1, 9)
                    ]
                ),
                ["State", "MMWR Week"],
                id="mmwr_week",
            ),
            pytest.param(
                pl.DataFrame(
                    [
                        {"State": s, "Year": 2022, "ICD Code": c, "Crude Rate": 3.0, "Deaths": 200}
                        for s in _STATE_NAMES[:12]
                        for c in ("I21", "I25", "C34")
                    ]
                ),
                ["State", "ICD Code"],
                id="icd_cause_codes",
            ),
            pytest.param(
                pl.DataFrame(
                    [
                        {"State": s, "Year": y, "Patient Days": 12000 + i, "Rate per 1000 Patient Days": 2.5}
                        for i, s in enumerate(_STATE_NAMES[:12])
                        for y in (2022, 2023)
                    ]
                ),
                ["State", "Year"],
                id="patient_days_value",
            ),
            pytest.param(
                pl.DataFrame(
                    [
                        {"State": s, "Year": y, "Member Months": 12000 + i, "Visits per 1000 Member Months": 2.5}
                        for i, s in enumerate(_STATE_NAMES[:12])
                        for y in (2022, 2023)
                    ]
                ),
                ["State", "Year"],
                id="member_months_value",
            ),
            pytest.param(
                pl.DataFrame(
                    [
                        {"State": s, "Year": y, "Beneficiary Count": 12000 + i, "Percent with Diabetes": 20.5}
                        for i, s in enumerate(_STATE_NAMES[:12])
                        for y in (2022, 2023)
                    ]
                ),
                ["State", "Year"],
                id="beneficiary_count_value",
            ),
        ],
    )
    def test_detectFrame_publicShape_accepted(self, frame, keys):
        # Act
        table = detect_frame(frame)

        # Assert
        assert table is not None
        assert [d.name for d in table.dimensions] == keys


class TestGeorgia:
    """Item 4: regions, HHS regions, placeholders and Virgin Islands spellings are not countries."""

    @pytest.mark.parametrize(
        "level",
        [
            "Northeast",
            "Midwest",
            "South",
            "West",
            "HHS Region 1",
            "Region 10",
            "Unknown",
            "Other",
            "Total",
            "Virgin Islands, U.S.",
            "Virgin Islands of the U.S.",
            "U.S. Virgin Islands",
            "United States of America",
        ],
    )
    def test_classify_usPlaceOrPlaceholder_notCountry(self, level):
        # Act
        kind = geography.classify(level).kind

        # Assert
        assert kind != "country"

    @pytest.mark.parametrize("level", ["Armenia", "Azerbaijan", "France"])
    def test_classify_foreignCountry_country(self, level):
        # Act
        kind = geography.classify(level).kind

        # Assert
        assert kind == "country"


class TestResidualRisk:
    def test_detectFrame_stateNamesUsedAsPatientIds_acceptedResidualRisk(self):
        # Arrange: one "patient" per state name, three visit months: structurally a state x month
        # table of rates. Documented residual risk (ADR014 §7): the declaration is the safeguard.
        frame = _visits(lambda p: {"State": _STATE_NAMES[p]}).rename({"visit_month": "Month"})

        # Act
        table = detect_frame(frame)

        # Assert
        assert table is not None


class TestContactDetails:
    """Sharp-edges finding: an email or phone number passed as a label ("Indicator")."""

    @pytest.mark.parametrize("value", ["user0@example.org", "(313) 555-0100", "313-555-0100", "https://example.org/a"])
    def test_isLabel_contactDetail_notLabel(self, value):
        # Act
        label = public_shape.is_label(value, "Indicator")

        # Assert
        assert not label

    def test_detectFrame_emailsAsIndicator_neverEstimates(self):
        # Arrange
        frame = pl.DataFrame(
            [{"Indicator": f"user{p}@example.org", "State": s, "Rate": 1.0} for p in range(4) for s in _STATE_NAMES[:5]]
        )

        # Act
        table = detect_frame(frame)

        # Assert
        assert table is None


_LAST = ["LOPEZ", "SMITH", "NGUYEN", "PATEL", "KIM", "GARCIA", "JONES", "BROWN", "DAVIS", "MILLER"]
_FIRST = ["ANA", "JOHN", "MAI", "RAJ"]
_PEOPLE_AS_LABELS = [
    pytest.param("type", lambda p: f"Patient {p}", id="patient_n"),
    pytest.param("category", lambda p: f"Subject {p:02d}", id="subject_nn"),
    pytest.param("cause_name", lambda p: f"Resident {p}", id="resident_n"),
    pytest.param("indicator", lambda p: f"{_LAST[p % 10]}, {_FIRST[p // 10]}", id="last_first"),
    pytest.param("type", lambda p: f"{_FIRST[p // 10].lower()} {_LAST[p % 10].lower()}", id="lowercase_name"),
    pytest.param("type", lambda p: f"MRN.{p:04d}", id="mrn_dotted"),
    pytest.param("type", lambda p: f"Room A{p}", id="room"),
    pytest.param("measure", lambda p: f"Pаtient{p}", id="cyrillic_lookalike"),
    pytest.param("location", lambda p: "ＭＲＮ" + "".join(chr(0xFF10 + int(d)) for d in f"{p:04d}"), id="fullwidth"),
    pytest.param("cause", lambda p: f"I{20 + p}", id="icd_as_proxy"),
    # sharp-edges, round 5
    pytest.param("type", lambda p: f"{_LAST[p % 10].title()} {chr(65 + p)}", id="surname_initial"),
    pytest.param("type", lambda p: f"{chr(65 + p)}. {_LAST[p % 10].title()}", id="initial_surname"),
    pytest.param("type", lambda p: f"Pat\u200bient {p}", id="zero_width_split"),
    pytest.param("type", lambda p: f"Pat1ent {p}", id="digit_in_word"),
]


class TestRoundFiveGuard:
    """Round 5: a place key is required; labels never carry people; ID names refuse whatever the
    values; a blank is a placeholder, never a total."""

    @pytest.mark.parametrize(("column", "ident"), _PEOPLE_AS_LABELS)
    def test_detectFrame_peopleAsLabelsWithMonthOnly_neverEstimates(self, column, ident):
        # Arrange
        frame = pl.DataFrame(
            [{column: ident(p), "visit_month": m, "hba1c_pct": 5.0 + p % 6} for p in range(1, 31) for m in _MONTHS]
        )

        # Act
        table = detect_frame(frame)

        # Assert
        assert table is None

    @pytest.mark.parametrize(("column", "ident"), _PEOPLE_AS_LABELS)
    def test_detectFrame_peopleAsLabelsBesideStates_neverEstimates(self, column, ident):
        # Arrange: a place key does not let a person through as a label
        frame = pl.DataFrame(
            [
                {"State": s, column: ident(p), "Year": y, "Rate": 1.0, "lower": 0.5, "upper": 2.0}
                for p in range(1, 6)
                for s in ("Ohio", "Iowa")
                for y in (2022, 2023)
            ]
        )

        # Act
        table = detect_frame(frame)

        # Assert
        assert table is None

    @pytest.mark.parametrize(
        "labels",
        [
            pytest.param(["Syphilis", "Chlamydia", "COVID-19", "Influenza A(H1N1)"], id="disease_codes"),
            pytest.param(["Kidney disease", "Heart disease"], id="lowercase_labels"),
            pytest.param(["Vitamin D", "E. coli", "PM2.5", "Influenza A(H1N1)pdm09", "Hepatitis B"], id="named_codes"),
        ],
    )
    def test_detectFrame_publicLabelsBesideStates_accepted(self, labels):
        # Arrange
        frame = pl.DataFrame(
            [
                {"State": s, "cause_name": c, "Year": y, "Rate": 1.0, "lower": 0.5, "upper": 2.0}
                for c in labels
                for s in ("Ohio", "Iowa")
                for y in (2022, 2023)
            ]
        )

        # Act
        table = detect_frame(frame)

        # Assert
        assert table is not None

    def test_detectFrame_timeAndLabelsWithoutPlace_neverEstimates(self):
        # Arrange: a public-looking label x year table with no place at all
        frame = pl.DataFrame(
            [
                {"cause_name": c, "Year": y, "Rate": 1.0, "lower": 0.5, "upper": 2.0}
                for c in ("Syphilis", "Chlamydia")
                for y in (2021, 2022, 2023)
            ]
        )

        # Act
        table = detect_frame(frame)

        # Assert
        assert table is None

    def test_detectFrame_onePlaceForWholeFile_accepted(self):
        # Arrange: a state's own table: one state, so the place is a one-value column
        frame = pl.DataFrame(
            [
                {"State": "Michigan", "cause_name": c, "Year": y, "Rate": 1.0, "lower": 0.5, "upper": 2.0}
                for c in ("Syphilis", "Chlamydia")
                for y in (2021, 2022, 2023)
            ]
        )

        # Act
        table = detect_frame(frame)

        # Assert
        assert table is not None

    @pytest.mark.parametrize(
        ("id_column", "ident"),
        [
            pytest.param("subject_id", lambda p: str(2000 + p), id="subject_id_year_shaped"),
            pytest.param("patient_id", lambda p: str(202400 + p), id="patient_id_mmwr_shaped"),
            pytest.param("mrn", lambda p: _STATE_NAMES[p], id="mrn_state_names"),
        ],
    )
    def test_detectFrame_personIdNamedColumn_refusedWhateverValues(self, id_column, ident):
        # Arrange
        frame = pl.DataFrame(
            [
                {id_column: ident(p), "State": "Ohio", "event": e, "hba1c_pct": 6.0 + p / 10}
                for p in range(1, 31)
                for e in ("Screening", "Baseline", "Followup")
            ]
        )

        # Act
        table = detect_frame(frame)

        # Assert
        assert table is None

    def test_detectFrame_fullNameAttribute_refused(self):
        # Arrange: a public-shaped key, with a person's name beside each row
        frame = pl.DataFrame(
            [
                {"State": s, "Year": y, "full_name": f"{_LAST[i % 10]}, {_FIRST[i % 4]}", "Rate": 1.0}
                for i, s in enumerate(_STATE_NAMES[:20])
                for y in (2022, 2023)
            ]
        )

        # Act
        table = detect_frame(frame)

        # Assert
        assert table is None

    def test_detectFrame_demographicGridWithOneBlank_neverEstimates(self):
        # Arrange: one blank race cell must not read as a published total
        import itertools

        cells = list(
            itertools.product(["Female", "Male"], ["18-34", "35-49", "50-64", "65+"], ["White", "Black", "Asian"])
        )
        frame = pl.DataFrame(
            [
                {"sex": s, "age_group": a, "race": None if i == 0 else r, "a1c_pct": 6.0 + i / 10}
                for i, (s, a, r) in enumerate(cells)
            ]
        )

        # Act
        table = detect_frame(frame)

        # Assert
        assert table is None

    @pytest.mark.parametrize(
        ("period", "values"),
        [
            pytest.param("WEEK", range(1, 53), id="fluview_year_week"),
            pytest.param("Month", range(1, 13), id="year_month"),
        ],
    )
    def test_detectFrame_yearWithPeriodNumbers_accepted(self, period, values):
        # Arrange
        frame = pl.DataFrame(
            [
                {"REGION": s, "YEAR": y, period: w, "% WEIGHTED ILI": 1.0 + w / 10}
                for s in _STATE_NAMES[:5]
                for y in (2023, 2024)
                for w in values
            ]
        )

        # Act
        table = detect_frame(frame)

        # Assert
        assert table is not None
        assert {d.name for d in table.dimensions} == {"REGION", "YEAR", period}

    def test_detectFrame_yearBesideVisitMonth_neverEstimates(self):
        # Arrange
        frame = pl.DataFrame(
            [
                {"State": s, "Year": y, "visit_month": m, "Rate": 1.0}
                for s in _STATE_NAMES[:5]
                for y in (2022, 2023)
                for m in _MONTHS
            ]
        )

        # Act
        table = detect_frame(frame)

        # Assert
        assert table is None

    def test_detectFrame_nndssDivisionsAndUsResidents_accepted(self):
        # Arrange: NNDSS's abbreviated census divisions and its "US RESIDENTS" national row
        areas = ["US RESIDENTS", "New England", "Mid. Atlantic", "E.N. Central", "W.S. Central", "Pacific"]
        areas += [s.upper() for s in _STATE_NAMES[:20]]
        frame = pl.DataFrame(
            [
                {"Reporting Area": a, "MMWR Year": 2023, "MMWR Week": w, "Legionellosis, Cum 2023": 40 + i + w}
                for i, a in enumerate(areas)
                for w in (10, 11)
            ]
        )

        # Act
        table = detect_frame(frame)

        # Assert
        assert table is not None

    @pytest.mark.parametrize("level", ["Palau", "Marshall Islands", "Federated States of Micronesia"])
    def test_classify_freelyAssociatedState_territoryNotCountry(self, level):
        # Act
        kind = geography.classify(level).kind

        # Assert
        assert kind == "territory"


_TEN_STATES = ["Ohio", "Iowa", "Utah", "Maine", "Idaho", "Texas"]


def _state_years() -> pl.DataFrame:
    return pl.DataFrame(
        [
            {"State": s, "Year": y, "Rate per 100000": 1.0 + i}
            for i, s in enumerate(_TEN_STATES)
            for y in range(2016, 2024)
        ]
    )


class TestRoundSixGuard:
    """Round 6: the guard's job is a patient file declared public BY MISTAKE. Every column is
    checked whatever its type; names in any script; a person's own dates are never a time key."""

    @pytest.mark.parametrize(
        ("name", "extra", "accepted"),
        [
            pytest.param("ts", lambda n: pl.Series([dt.datetime(2024, 1, 1, h % 24, h // 24) for h in range(n)]),
                         False, id="datetime_varying"),
            pytest.param("dob", lambda n: pl.Series([dt.date(1950, 5, 1)] * n), False, id="date_named_dob"),
            pytest.param("visit_date", lambda n: pl.Series([dt.date(2024, 5, 1)] * n), False, id="date_named_visit"),
            pytest.param("codes", lambda n: pl.Series([[1, 2]] * n), False, id="list_fails_closed"),
            pytest.param("los", lambda n: pl.Series([dt.timedelta(days=3)] * n), False, id="duration_fails_closed"),
            pytest.param("Data As Of", lambda n: pl.Series([dt.date(2024, 5, 1)] * n), True, id="date_constant"),
            pytest.param("Rate suppressed", lambda n: pl.Series([False] * n), True, id="boolean_flag"),
        ],
    )  # fmt: skip
    def test_detectFrame_columnOfAnyType_checked(self, name, extra, accepted):
        # Arrange
        base = _state_years()
        frame = base.with_columns(extra(base.height).alias(name))

        # Act
        table = detect_frame(frame)

        # Assert
        assert (table is not None) is accepted

    def test_detectFrame_dateFixedByYearKey_accepted(self):
        # Arrange: a period-end date that the Year key fixes
        frame = _state_years().with_columns(
            (pl.col("Year").cast(pl.String) + "-12-31").str.to_date().alias("Period End")
        )

        # Act
        table = detect_frame(frame)

        # Assert
        assert table is not None

    @pytest.mark.parametrize(
        "names",
        [
            pytest.param(["José García", "Zoë Müller", "Ángel Ruiz", "Nguyễn Văn An"], id="accented"),
            pytest.param(
                ["Maria de la Cruz", "Dr. Ana Lopez", "D'Angelo Rossi", "Ana María Lopez"], id="particles_titles"
            ),
        ],
    )
    def test_detectFrame_unicodeNamesAsCategory_neverEstimates(self, names):
        # Arrange
        frame = pl.DataFrame(
            [{"State": "Texas", "Category": n, "Year": y, "a1c_percent": 6.5} for n in names for y in range(2016, 2024)]
        )

        # Act
        table = detect_frame(frame)

        # Assert
        assert table is None

    @pytest.mark.parametrize("column", ["Provider", "Attending Physician", "Nurse", "Doctor"])
    def test_detectFrame_clinicianColumnWithNames_refused(self, column):
        # Arrange
        base = _state_years()
        names = pl.Series([["José García", "Ana Lopez", "Zoë Müller"][i % 3] for i in range(base.height)])
        frame = base.with_columns(names.alias(column))

        # Act
        table = detect_frame(frame)

        # Assert
        assert table is None

    @pytest.mark.parametrize("value", ["Admission 004417", "12345-ABC", "Encounter 77812"])
    def test_labelShaped_wordWithLongNumberOrDigitsDashLetters_notLabel(self, value):
        # Act
        shaped = public_shape.label_shaped(value, "category")

        # Assert
        assert not shaped

    @pytest.mark.parametrize(
        "value",
        [
            "#Tuberculosis (A16-A19)",
            "Other diseases of respiratory system (J00-J06,J30-J39,J67,J70-J98)",
            "Shigella sonnei",
            "Escherichia coli O157:H7",
            "HHS Regions",
            "Census Regions",
            "Selected Hospitals",
            "Healthy People 2030",
        ],
    )
    def test_labelsOk_publishedVocabulary_accepted(self, value):
        # Act
        ok = public_shape.labels_ok([value], "cause")

        # Assert
        assert ok

    def test_detectFrame_birthYearByBirthMonthGrid_neverEstimates(self):
        # Arrange: counties x sex x birth_year x birth_month, one patient per cell
        frame = pl.DataFrame(
            [
                {"County": c, "Sex": s, "birth_year": by, "birth_month": bm, "bmi_rate": 20.0 + bm}
                for c in ("Alcona County, MI", "Alger County, MI")
                for s in ("Female", "Male")
                for by in (1950, 1951, 1952)
                for bm in range(1, 13)
            ]
        )

        # Act
        table = detect_frame(frame)

        # Assert
        assert table is None

    def test_detectFrame_stateYearVisitMonth_neverEstimates(self):
        # Arrange
        frame = pl.DataFrame(
            [
                {"State": s, "Year": 2024, "visit_month": m, "hba1c_pct": 6.0 + m / 10}
                for s in _TEN_STATES
                for m in range(1, 13)
            ]
        )

        # Act
        table = detect_frame(frame)

        # Assert
        assert table is None

    def test_detectFrame_onePlaceWithBlankCell_sameVerdictEveryRun(self):
        # Arrange: a state's own table, one State cell blank
        frame = pl.DataFrame(
            [
                {
                    "State": None if (c, y) == ("Syphilis", 2021) else "Michigan",
                    "cause_name": c,
                    "Year": y,
                    "Rate": 1.0,
                    "lower": 0.5,
                    "upper": 2.0,
                }
                for c in ("Syphilis", "Chlamydia")
                for y in (2021, 2022, 2023)
            ]
        )  # fmt: skip

        # Act
        verdicts = {detect_frame(frame) is not None for _ in range(8)}

        # Assert
        assert verdicts == {True}

    def test_detectFrame_placeInsideLabelKey_noPlaceRequirementMet(self):
        # Arrange: "Texas" among cause labels is not a place key
        frame = pl.DataFrame(
            [
                {"cause_name": c, "Year": y, "Rate": 1.0, "lower": 0.5, "upper": 2.0}
                for c in ("Syphilis", "Chlamydia", "Texas")
                for y in (2021, 2022, 2023)
            ]
        )

        # Act
        table = detect_frame(frame)

        # Assert
        assert table is None

    def test_detectFrame_fluviewRegionType_accepted(self):
        # Arrange: FluView ILINet with its REGION TYPE label
        frame = pl.DataFrame(
            [
                {
                    "REGION TYPE": "HHS Regions",
                    "REGION": f"Region {r}",
                    "YEAR": y,
                    "WEEK": w,
                    "% WEIGHTED ILI": 1.0 + w / 10,
                }
                for r in range(1, 11)
                for y in (2022, 2023)
                for w in range(1, 6)
            ]
        )

        # Act
        table = detect_frame(frame)

        # Assert
        assert table is not None
