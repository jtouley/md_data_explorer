"""covid_ms oracles: empty fields never become a group, undefined statistics never crash.

Synthetic CSVs only; the numbers below are made up.
"""

from __future__ import annotations

import math
from pathlib import Path

import polars as pl
import pytest
from polars.testing import assert_frame_equal


def _csv(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "synthetic.csv"
    path.write_text(text, encoding="utf-8")
    return path


def _answer(module, tmp_path: Path, oracle, text: str):
    return module.answers(_csv(tmp_path, text), {"q": oracle})["q"]


class TestScan:
    def test_scan_codedTextColumns_keptAsTextNotInferred(self, tmp_path, covid_ms_oracles):
        # Arrange
        csv = _csv(tmp_path, "age_in_cat,edss_in_cat2,sex\n1,0.0,female\n2,,male\n")
        expected = pl.DataFrame(
            {"age_in_cat": ["1", "2"], "edss_in_cat2": ["0.0", None], "sex": ["female", "male"]},
            schema={"age_in_cat": pl.String, "edss_in_cat2": pl.String, "sex": pl.String},
        )

        # Act
        frame = covid_ms_oracles.scan(csv).collect()

        # Assert
        assert_frame_equal(frame, expected)


class TestNullGroups:
    @pytest.mark.parametrize(
        ("builder", "expected_keys"),
        [
            pytest.param(lambda m: m._count_by("sex"), {"female", "male"}, id="count_by"),
            pytest.param(lambda m: m._rates("outcome", "sex"), {"female", "male"}, id="rates"),
            pytest.param(lambda m: m._stat_by("mean", "metric", "sex"), {"female", "male"}, id="mean_by"),
        ],
    )
    def test_groupedOracle_emptyGroupField_noNoneGroup(self, tmp_path, covid_ms_oracles, builder, expected_keys):
        # Arrange
        text = "sex,outcome,metric\nfemale,yes,1\nmale,no,2\n,yes,3\nfemale,,4\n"

        # Act
        answer = _answer(covid_ms_oracles, tmp_path, builder(covid_ms_oracles), text)

        # Assert
        assert set(answer.value) == expected_keys
        assert all(key[0] != "None" for key in answer.cell_patients)


class TestTwoByTwo:
    def test_chiSquare_nullRowsExcluded_matchesReferenceP(self, tmp_path, covid_ms_oracles):
        # Arrange: female 12 yes / 28 no, male 30 / 40, plus rows missing either field.
        # Reference p-values from scipy.stats.chi2_contingency (correction False / True).
        rows = ["female,yes"] * 12 + ["female,no"] * 28 + ["male,yes"] * 30 + ["male,no"] * 40
        rows += [",yes", "female,", ","]
        text = "sex,confirmed\n" + "\n".join(rows) + "\n"

        # Act
        answer = _answer(
            covid_ms_oracles, tmp_path, covid_ms_oracles._chi_square("sex", "confirmed", ("female", "male")), text
        )

        # Assert
        assert answer.undefined is None
        assert math.isclose(answer.value["pearson"], 0.18181652781452545, rel_tol=1e-9)
        assert math.isclose(answer.value["yates"], 0.2579723816158965, rel_tol=1e-9)
        assert sum(answer.cell_patients.values()) == 110

    def test_chiSquare_levelOutsideTable_oracleErrorNotCrash(self, tmp_path, covid_ms_oracles):
        # Arrange
        text = "sex,confirmed\nfemale,yes\nmale,no\nother,yes\n"

        # Act
        answer = _answer(
            covid_ms_oracles, tmp_path, covid_ms_oracles._chi_square("sex", "confirmed", ("female", "male")), text
        )

        # Assert
        assert answer.oracle_error is not None and "other" in answer.oracle_error

    @pytest.mark.parametrize(
        "rows",
        [
            pytest.param(["yes,yes", "yes,no", "no,no"], id="zero_cell"),
            pytest.param(["yes,no", "no,no"], id="zero_column"),
        ],
    )
    def test_oddsRatio_zeroCell_undefinedNotCrash(self, tmp_path, covid_ms_oracles, rows):
        # Arrange
        text = "exposed,outcome\n" + "\n".join(rows) + "\n"

        # Act
        answer = _answer(covid_ms_oracles, tmp_path, covid_ms_oracles._odds_ratio("exposed", "outcome"), text)

        # Assert
        assert answer.value is None
        assert answer.undefined is not None


class TestStatBy:
    def test_statBy_groupWithNoValues_valueNone(self, tmp_path, covid_ms_oracles):
        # Arrange: "NA" and "." are missing-value markers, not bad data.
        text = "sex,metric\nfemale,2010\nfemale,2012\nmale,NA\nmale,.\nmale,\n"

        # Act
        answer = _answer(covid_ms_oracles, tmp_path, covid_ms_oracles._stat_by("mean", "metric", "sex"), text)

        # Assert
        assert answer.oracle_error is None
        assert answer.value == {"female": 2011.0, "male": None}
        assert answer.cell_patients[("male",)] == 0

    def test_answers_nonNumericMetric_onlyThatQuestionErrors(self, tmp_path, covid_ms_oracles):
        # Arrange
        csv = _csv(tmp_path, "sex,metric\nfemale,2010\nmale,unknown year\n")
        oracles = {"mean": covid_ms_oracles._stat_by("mean", "metric", "sex"), "total": covid_ms_oracles._total}

        # Act
        answers = covid_ms_oracles.answers(csv, oracles)

        # Assert
        assert answers["mean"].oracle_error is not None and "1 value(s)" in answers["mean"].oracle_error
        assert answers["total"].value == 2
