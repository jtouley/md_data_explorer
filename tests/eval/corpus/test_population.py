"""Population (aggregate) datasets: the oracle kit never pools rows, the scorer grades pooled answers
WRONG, and every question file has an oracle for each question.

Synthetic tables only; the numbers below are made up.
"""

from __future__ import annotations

import importlib.util
import io
import zipfile
from typing import Any

import polars as pl
import pytest
import yaml

from clinical_analytics.eval.corpus import oracle_kit as kit
from clinical_analytics.eval.corpus.fetch import CorpusDataset, build_zip
from clinical_analytics.eval.corpus.run import QUESTIONS_DIR, QuestionResult, RunReport, load_questions, markdown
from clinical_analytics.eval.corpus.score import ApiOutcome, OracleAnswer, score

# Rows overlap on purpose: "Both" beside Female and Male, and "US" beside two states.
RATES = pl.DataFrame(
    {
        "place": ["US", "US", "US", "A", "A", "A", "B", "B", "B"],
        "sex": ["Both", "Female", "Male"] * 3,
        "rate": ["10.5", "9.0", "12.0", "20.0", "Not Available", "22.5", "5.0", "4.0", "6.0"],
        "low": ["10.0", "8.5", "11.0", "18.0", "", "20.0", "4.0", "3.0", "5.0"],
        "high": ["11.0", "9.5", "13.0", "22.0", "", "25.0", "6.0", "5.0", "7.0"],
    }
)


def _answer(oracle: kit.Oracle, frame: pl.DataFrame = RATES) -> OracleAnswer:
    return kit.answers({"t": frame.lazy()}, {"q": oracle})["q"]


def _table(column: str, rows: list[tuple[Any, ...]], value_columns: tuple[str, ...] = ("value",)) -> dict[str, Any]:
    columns = [column, *value_columns]
    return {"claim": "x", "table": {"columns": columns, "rows": [dict(zip(columns, row, strict=True)) for row in rows]}}


def _done(presentation: dict[str, Any]) -> ApiOutcome:
    return ApiOutcome("completed", presentation=presentation)


class TestOracleKit:
    def test_lookupBy_filterReachesGrain_oneValuePerGroup(self):
        # Act
        answer = _answer(kit.lookup_by("t", "sex", "rate", pl.col("place") == "US"))

        # Assert
        assert answer.oracle_error is None
        assert answer.value == {"Both": 10.5, "Female": 9.0, "Male": 12.0}

    def test_lookupBy_filterMissesDimension_oracleErrorNotPooled(self):
        # Act: every place for each sex is three rows per group.
        answer = _answer(kit.lookup_by("t", "sex", "rate", pl.lit(True)))

        # Assert
        assert answer.value is None
        assert "does not reach the table's grain" in str(answer.oracle_error)

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            pytest.param("1,234", 1234.0, id="thousands"),
            pytest.param(" 7.5 ", 7.5, id="spaces"),
            pytest.param("Not Available", None, id="not_available"),
            pytest.param("", None, id="empty"),
            pytest.param(None, None, id="null"),
            pytest.param("*", None, id="footnote_symbol"),
        ],
    )
    def test_number_publisherTokens_numberOrNull(self, text, expected):
        # Arrange
        frame = pl.DataFrame({"v": [text]}, schema={"v": pl.String})

        # Act
        result = frame.select(kit.number("v")).item()

        # Assert
        assert result == expected

    def test_lookupBy_unreadableValue_oracleError(self):
        # Arrange
        frame = pl.DataFrame({"g": ["a"], "v": ["12 (est.)"]})

        # Act
        answer = _answer(kit.lookup_by("t", "g", "v", pl.lit(True)), frame)

        # Assert
        assert "neither numbers nor missing-value markers" in str(answer.oracle_error)

    def test_lookupCiBy_topTwoSuppressedSkipped_rankedByEstimate(self):
        # Act
        answer = _answer(kit.lookup_ci_by("t", "place", ("rate", "low", "high"), pl.col("sex") == "Female", top=2))

        # Assert: A's female rate is suppressed, so the top two are US and B.
        assert answer.value == {
            "US": {"estimate": 9.0, "ci_low": 8.5, "ci_high": 9.5},
            "B": {"estimate": 4.0, "ci_low": 3.0, "ci_high": 5.0},
        }

    @pytest.mark.parametrize(
        ("kwargs", "error"),
        [
            pytest.param({"rows": 3}, "expected 3", id="rows"),
            pytest.param({"unique_by": "sex"}, "two rows for one sex", id="unique_by"),
        ],
    )
    def test_derivedBy_grainCheckFails_oracleError(self, kwargs, error):
        # Arrange: two Female rows for place A.
        frame = pl.concat([RATES, RATES.filter((pl.col("place") == "A") & (pl.col("sex") == "Female"))])

        # Act
        answer = _answer(kit.derived_by("t", "place", kit.number("rate").max(), pl.lit(True), **kwargs), frame)

        # Assert
        assert error in str(answer.oracle_error)

    def test_held_anyData_valueNoneWithReason(self):
        # Act
        answer = _answer(kit.held("t", "rows overlap"))

        # Assert
        assert answer.value is None
        assert answer.unit == "held: rows overlap"


VALUE_BY = {"id": "q", "kind": "value_by", "group_by": "place", "expect": "answer", "grain": "aggregate"}
TRUTH = OracleAnswer({"A": 22.5, "B": 6.0}, "rate")


class TestScorePopulation:
    @pytest.mark.parametrize(
        ("rows", "value_columns", "expected"),
        [
            pytest.param([("A", 22.5), ("B", 6.0)], ("mean",), "RIGHT", id="lookup_any_value_column"),
            pytest.param([("A", 22.5), ("B", 6.0)], ("sum Estimate",), "RIGHT", id="renamed_column"),
            pytest.param([("A", 64.5), ("B", 15.0)], ("sum",), "WRONG", id="pooled_both_with_sexes"),
            pytest.param([("A", 3), ("B", 3)], ("count",), "WRONG", id="row_counts"),
            pytest.param([("A", 22.5)], ("value",), "WRONG", id="row_missing"),
            pytest.param([("A", 22.5), ("B", 6.0), ("US", 12.0)], ("value",), "WRONG", id="extra_row"),
        ],
    )
    def test_score_valueBy_numbersShown_gradedAgainstRows(self, rows, value_columns, expected):
        # Act
        scored = score(VALUE_BY, TRUTH, _done(_table("place", rows, value_columns)))

        # Assert
        assert scored.grade == expected

    @pytest.mark.parametrize(
        ("rows", "expected"),
        [
            pytest.param([("A", 22.5), ("B", 6.0), ("C", 5.0)], "RIGHT", id="lower_rank_shown"),
            pytest.param([("A", 22.5), ("B", 6.0), ("C", 30.0)], "WRONG", id="higher_rank_left_out"),
        ],
    )
    def test_score_valueByTop_rowsBelowTopAllowed(self, rows, expected):
        # Arrange
        question = {**VALUE_BY, "top": 2}

        # Act
        scored = score(question, TRUTH, _done(_table("place", rows)))

        # Assert
        assert scored.grade == expected

    @pytest.mark.parametrize(
        ("claim", "unit", "expected"),
        [
            pytest.param("The share was 0.0523.", "proportion", "RIGHT", id="proportion"),
            pytest.param("The share was 5.23%.", "proportion", "RIGHT", id="percent"),
            pytest.param("The share was 5.23.", "rate", "WRONG", id="percent_not_allowed_for_rate"),
            pytest.param("Total count: 10,868 patients.", "proportion", "WRONG", id="row_count"),
            pytest.param("I cannot tell from this table.", "proportion", "HELD", id="no_number"),
        ],
    )
    def test_score_values_claimNumbers_matchedWithPercent(self, claim, unit, expected):
        # Arrange
        question = {"id": "q", "kind": "values", "expect": "answer", "grain": "aggregate"}
        truth = OracleAnswer({"share": 0.0523}, unit)

        # Act
        scored = score(question, truth, _done({"claim": claim, "table": None}))

        # Assert
        assert scored.grade == expected

    @pytest.mark.parametrize(
        ("row", "expected"),
        [
            pytest.param(("A", 22.5, 20.0, 25.0), "RIGHT", id="estimate_and_ci"),
            pytest.param(("A", 22.5, None, None), "WRONG", id="ci_missing"),
            pytest.param(("A", 22.5, 21.0, 24.0), "WRONG", id="ci_wrong"),
        ],
    )
    def test_score_ciBy_eachRowNeedsEstimateAndBounds(self, row, expected):
        # Arrange
        question = {**VALUE_BY, "kind": "ci_by"}
        truth = OracleAnswer({"A": {"estimate": 22.5, "ci_low": 20.0, "ci_high": 25.0}}, "rate")

        # Act
        scored = score(question, truth, _done(_table("place", [row], ("estimate", "low", "high"))))

        # Assert
        assert scored.grade == expected

    def test_score_aggregateGrainSmallCount_noLeak(self):
        # Arrange: three rows per place is a row count, not three patients.
        presentation = _table("place", [("A", 3), ("B", 3)], ("count",))

        # Act
        scored = score(VALUE_BY, TRUTH, _done(presentation))

        # Assert
        assert scored.leak is False

    @pytest.mark.parametrize(
        ("outcome", "expected"),
        [
            pytest.param(ApiOutcome("failed", failure_reason="rows overlap"), "RIGHT", id="refused"),
            pytest.param(_done({"claim": "Total count: 137700 patients.", "table": None}), "WRONG", id="pooled"),
        ],
    )
    def test_score_heldPooledQuestion_answeringIsWrong(self, outcome, expected):
        # Arrange
        question = {"id": "q", "kind": "values", "expect": "held", "grain": "aggregate"}

        # Act
        scored = score(question, OracleAnswer(None, "held: rows overlap"), outcome)

        # Assert
        assert scored.grade == expected


class TestExtractPackage:
    def test_buildZip_extractEntry_membersStrippedSortedDocsKept(self, tmp_path):
        # Arrange: a published archive with a folder prefix and a checksum file to leave out.
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr("demo-1.0/B.csv", "id\n1\n")
            archive.writestr("demo-1.0/A.csv", "id\n2\n")
            archive.writestr("demo-1.0/SHA256SUMS.txt", "x")
        files = tmp_path / "files"
        files.mkdir()
        (files / "demo.zip").write_bytes(buffer.getvalue())
        dataset = CorpusDataset.model_validate(
            {
                "name": "demo",
                "files": [
                    {"path": "demo.zip", "url": "https://example.org/demo.zip", "kind": "archive", "sha256": "0"}
                ],
                "upload_package": {
                    "zip": "demo_upload.zip",
                    "contents": [
                        {"from": "demo.zip", "make": "extract", "member": "demo-1.0/*.csv", "strip": "demo-1.0/"}
                    ],
                },
            }
        )

        # Act
        out = build_zip(dataset, files, tmp_path / "demo_upload.zip")

        # Assert
        with zipfile.ZipFile(out) as built:
            assert built.namelist() == ["A.csv", "B.csv"]
            assert built.read("B.csv") == b"id\n1\n"

    @pytest.mark.parametrize(
        "entry",
        [
            pytest.param({"from": "a.zip", "make": "extract"}, id="extract_without_member"),
            pytest.param({"from": "a.csv", "make": "copy"}, id="copy_without_name"),
            pytest.param({"name": "a", "from": "a.zip", "make": "extract", "member": "*"}, id="extract_with_name"),
        ],
    )
    def test_packageEntry_malformed_rejected(self, entry):
        # Arrange
        raw = {"name": "x", "files": [], "upload_package": {"zip": "x.zip", "contents": [entry]}}

        # Act / Assert
        with pytest.raises(ValueError, match="needs"):
            CorpusDataset.model_validate(raw)


def _question_files() -> list[str]:
    return sorted(path.stem for path in QUESTIONS_DIR.glob("*.yaml"))


def _oracle_ids(dataset: str) -> set[str]:
    path = QUESTIONS_DIR.parent / "oracles" / f"{dataset}.py"
    spec = importlib.util.spec_from_file_location(f"oracle_ids_{dataset}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return set(module.ORACLES)


class TestQuestionFiles:
    @pytest.mark.parametrize("dataset", _question_files())
    def test_questionFile_everyQuestion_hasOracleAndKnownKind(self, dataset):
        # Arrange
        questions = load_questions(dataset)

        # Act
        ids = [q["id"] for q in questions]

        # Assert
        assert len(ids) == len(set(ids))
        assert set(ids) == _oracle_ids(dataset)
        assert all(q.get("expect", "answer") in ("answer", "held") for q in questions)

    @pytest.mark.parametrize(
        "dataset",
        [d for d in _question_files() if yaml.safe_load((QUESTIONS_DIR / f"{d}.yaml").read_text()).get("grain")],
    )
    def test_populationQuestionFile_shape_heldOutThirdAndPoolingTrap(self, dataset):
        # Arrange
        questions = load_questions(dataset)

        # Act
        held_out = sum(bool(q.get("held_out")) for q in questions)
        families = {q["family"] for q in questions}

        # Assert
        assert 12 <= len(questions) <= 20
        assert 0.25 <= held_out / len(questions) <= 0.45
        assert "pooled_overlap" in families
        assert any(q["expect"] == "held" and q["family"] == "pooled_overlap" for q in questions)


class TestHeldOutReport:
    def test_markdown_heldOutQuestions_tallySeparately(self):
        # Arrange
        def result(qid: str, grade: str, held_out: bool) -> QuestionResult:
            return QuestionResult(
                qid, "f", "q", grade, False, "r", "completed", None, False, "not called", 1.0, False, held_out
            )

        report = RunReport("d", "none", "local", "abc", "2026-01-01T00:00:00+00:00", {})
        report.results = [result("a", "RIGHT", False), result("b", "WRONG", True), result("c", "HELD", True)]

        # Act
        text = markdown(report)

        # Assert
        assert report.counts(held_out=True) == {"RIGHT": 0, "WRONG": 1, "HELD": 1, "ERROR": 0, "LEAK": 0}
        assert "Held out: RIGHT 0 · WRONG 1 · HELD 1" in text
        assert "| b | held out | WRONG |" in text
