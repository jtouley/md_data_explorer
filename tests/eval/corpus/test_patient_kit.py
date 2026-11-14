"""Patient-level oracle kit: patients are counted once, missing markers follow the publisher,
and every shipped patient-level question has an oracle. Synthetic tables only; numbers made up.
"""

from __future__ import annotations

import io
import math

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from clinical_analytics.eval.corpus import patient_kit as kit
from clinical_analytics.eval.corpus import run as run_module
from clinical_analytics.eval.corpus.fetch import load_corpus
from clinical_analytics.eval.corpus.score import OracleError

# Three patients over five rows: patient 1 has two infections, patient 3 none.
VISITS = "id,arm,age,status\n1,a,10,1\n1,a,10,1\n2,a,30,0\n2,a,30,1\n3,b,50,0\n"


def _frame(text: str) -> pl.LazyFrame:
    return pl.read_csv(io.StringIO(text), infer_schema=False).lazy()


def _answer(oracle: kit.Oracle, text: str):
    return kit.answers(_frame(text), {"q": oracle})["q"]


class TestGrain:
    @pytest.mark.parametrize(
        ("unit", "expected"),
        [
            pytest.param("patients", {"a": 2, "b": 1}, id="patients"),
            pytest.param("visits", {"a": 4, "b": 1}, id="rows"),
        ],
    )
    def test_countBy_unit_patientsDistinctVisitsRows(self, unit, expected):
        # Arrange
        oracle = kit.count_by("arm", unit=unit, patient="id")

        # Act
        answer = _answer(oracle, VISITS)

        # Assert
        assert answer.value == expected
        assert answer.cell_patients == {("a",): 2, ("b",): 1}

    def test_rates_withPatient_anyRowWithEventCountsPatientOnce(self):
        # Arrange
        oracle = kit.rates(pl.col("status") == "1", "arm", patient="id")

        # Act
        answer = _answer(oracle, VISITS)

        # Assert
        assert answer.value == {"a": 1.0, "b": 0.0}
        assert answer.cell_patients[("a", "events")] == 2

    def test_statBy_withPatient_repeatedAttributeCountsOnce(self):
        # Arrange: by rows the mean age is (10 + 10 + 10 + 40) / 4 = 17.5; by patient, 25.
        text = "id,arm,age\n1,a,10\n1,a,10\n1,a,10\n2,a,40\n"

        # Act
        answer = _answer(kit.stat_by("mean", "age", "arm", patient="id"), text)

        # Assert
        assert answer.value == {"a": 25.0}
        assert answer.cell_patients == {("a",): 2}

    def test_statBy_attributeVariesWithinPatient_oracleError(self):
        # Arrange
        text = "id,arm,age\n1,a,10\n1,a,11\n"

        # Act
        answer = _answer(kit.stat_by("mean", "age", "arm", patient="id"), text)

        # Assert
        assert answer.oracle_error is not None and "1 patient(s)" in answer.oracle_error


class TestMissingMarkers:
    @pytest.mark.parametrize("marker", ["", "NA", "nan", "?", "."])
    def test_countBy_missingMarkerGroup_leftOut(self, marker):
        # Arrange
        text = f"race,x\nCaucasian,1\n{marker},1\n"

        # Act
        answer = _answer(kit.count_by("race"), text)

        # Assert
        assert answer.value == {"Caucasian": 1}

    def test_countBy_noneText_isALevelNotMissing(self):
        # Arrange: Diabetes 130's A1Cresult "None" means "not measured".
        text = "A1Cresult,x\nNone,1\nNone,1\n>8,1\n"

        # Act
        answer = _answer(kit.count_by("A1Cresult"), text)

        # Assert
        assert answer.value == {"None": 2, ">8": 1}

    def test_statBy_nonNumericMetric_oracleError(self):
        # Arrange
        text = "sex,lab\nf,1.5\nf,high\nm,None\n"

        # Act
        answer = _answer(kit.stat_by("median", "lab", "sex"), text)

        # Assert
        assert answer.oracle_error is not None and "1 value(s)" in answer.oracle_error


class TestOddsRatio:
    def test_oddsRatio_twoByTwo_matchesWoolf(self):
        # Arrange: exposed 6 events / 4 non-events, unexposed 3 / 7.
        rows = ["1,1"] * 6 + ["1,0"] * 4 + ["0,1"] * 3 + ["0,0"] * 7
        text = "e,o\n" + "\n".join(rows) + "\n"
        oracle = kit.odds_ratio(pl.col("e") == "1", pl.col("o") == "1")

        # Act
        answer = _answer(oracle, text)

        # Assert
        assert math.isclose(answer.value["or"], 3.5)
        se = math.sqrt(1 / 6 + 1 / 4 + 1 / 3 + 1 / 7)
        assert math.isclose(answer.value["ci_low"], math.exp(math.log(3.5) - kit.Z_95 * se))

    def test_oddsRatio_zeroCell_undefined(self):
        # Arrange
        text = "e,o\n1,1\n1,0\n0,0\n"

        # Act
        answer = _answer(kit.odds_ratio(pl.col("e") == "1", pl.col("o") == "1"), text)

        # Assert
        assert answer.value is None and answer.undefined is not None


class TestReading:
    def test_asText_wholeFloatsLoseDecimalPoint(self):
        # Arrange
        frame = pl.LazyFrame({"code": [1.0, 2.0, None], "lab": [0.25, 3.0, 1.5], "n": [1, 2, 3]})
        expected = pl.DataFrame({"code": ["1", "2", None], "lab": ["0.25", "3", "1.5"], "n": ["1", "2", "3"]})

        # Act
        result = kit.as_text(frame).collect()

        # Assert
        assert_frame_equal(result, expected)

    def test_frameFromRows_sheetCells_textColumnsShortRowsPadded(self):
        # Arrange
        header = ["stage", "rx"]
        rows = [[3.0, "placebo"], [4.0, ""], [3.0]]
        expected = pl.DataFrame({"stage": ["3", "4", "3"], "rx": ["placebo", None, None]})

        # Act
        result = kit.frame_from_rows(header, rows).collect()

        # Assert
        assert_frame_equal(result, expected)

    @pytest.mark.parametrize("header", [pytest.param(["a", None], id="empty"), pytest.param(["a", "a"], id="repeated")])
    def test_frameFromRows_badHeader_oracleError(self, header):
        # Arrange / Act / Assert
        with pytest.raises(OracleError, match="header"):
            kit.frame_from_rows(header, [])


class TestShippedQuestionSets:
    @pytest.mark.parametrize("dataset", sorted(set(load_corpus()) - {"covid_ms"}))
    def test_questionSet_everyQuestionHasAnOracleAndNoOther(self, dataset):
        # Arrange
        questions = run_module.load_questions(dataset)

        # Act
        oracles = run_module.load_oracles(dataset).ORACLES

        # Assert
        assert [q["id"] for q in questions] == list(oracles)
        assert {q["dataset_id"] for q in questions} == {dataset}
        assert all(q["expect"] in ("answer", "held") for q in questions)
