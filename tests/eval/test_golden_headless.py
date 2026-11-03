"""Generated golden questions on the synthetic cohort, answered through the headless service.

Safety property: every question is either answered with the ground-truth value or refused
(success False). A success with a different value is a silently wrong answer and fails the
suite. Ground truth is computed from tests/fixtures/synthetic/cohort.csv at test time.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from clinical_analytics.eval.catalog import DATASET_FILES, dataset_path, questions_for_dataset
from clinical_analytics.eval.headless import ask, classify, values_match

# Golden questions the system currently refuses. Pinned so a lost answer fails loudly;
# when a refusal starts being answered correctly, remove its id here.
GOLDEN_REFUSED_IDS: frozenset[str] = frozenset()
GOLDEN_MIN_ANSWERED_RATE = 0.95


class TestGoldenCatalog:
    def test_catalog_questions_queriesAndIdsUnique(self, project_root: Path) -> None:
        # Arrange
        questions = questions_for_dataset("cohort", project_root)

        # Act
        queries = [question["query"] for question in questions]
        ids = [question["id"] for question in questions]

        # Assert
        assert questions
        assert len(set(queries)) == len(queries)
        assert len(set(ids)) == len(ids)

    def test_catalog_groundTruth_isComputedForEveryQuestion(self, project_root: Path) -> None:
        # Arrange
        questions = questions_for_dataset("cohort", project_root)

        # Act
        missing = [question["id"] for question in questions if question["expected"] is None]

        # Assert
        assert missing == []

    def test_catalog_countQuestion_matchesSyntheticRowCount(self, project_root: Path, load_synthetic_dataset) -> None:
        # Arrange
        frame = load_synthetic_dataset("cohort")
        questions = questions_for_dataset("cohort", project_root)

        # Act
        count = next(question for question in questions if question["query"] == "how many patients?")

        # Assert
        assert count["expected"] == frame.height

    def test_catalog_datasets_pointAtCommittedSyntheticFiles(self, project_root: Path) -> None:
        # Arrange / Act
        paths = {dataset_id: dataset_path(dataset_id, project_root) for dataset_id in DATASET_FILES}

        # Assert
        for path in paths.values():
            assert path.parent == project_root / "tests" / "fixtures" / "synthetic"

    def test_catalog_unknownDataset_raisesKeyError(self, project_root: Path) -> None:
        # Arrange / Act / Assert
        with pytest.raises(KeyError, match="Unknown dataset"):
            dataset_path("mimic_patients", project_root)


class TestClassify:
    @pytest.mark.parametrize(
        ("success", "rows", "expected_outcome"),
        [
            pytest.param(True, 7, "answered", id="success_matching_value"),
            pytest.param(True, 8, "wrong", id="success_different_value"),
            pytest.param(False, 8, "refused", id="refused_even_with_value"),
            pytest.param(False, None, "refused", id="refused_no_payload"),
            pytest.param(True, None, "wrong", id="success_without_value"),
        ],
    )
    def test_classify_outcome_onlyMatchingSuccessIsAnswered(
        self, success: bool, rows: int | None, expected_outcome: str
    ) -> None:
        # Arrange
        question: dict[str, Any] = {"kind": "count"}
        payload = None if rows is None else {"result": rows}
        result = {"success": success, "payload": payload}

        # Act
        outcome, _actual = classify(question, 7, result)

        # Assert
        assert outcome == expected_outcome

    @pytest.mark.parametrize(
        ("expected", "actual", "match"),
        [
            pytest.param({"F": 1.0, "M": 2.0}, {"F": 1.0}, False, id="missing_group"),
            pytest.param({"F": 1.0}, {"F": 1.0, "M": 2.0}, False, id="extra_group"),
            pytest.param(1.0, 1.0 + 1e-3, False, id="outside_tolerance"),
            pytest.param(100.0, 100.0 + 1e-5, True, id="inside_tolerance"),
            pytest.param(None, 1.0, False, id="null_expected"),
            pytest.param(5, 5, True, id="exact_count"),
        ],
    )
    def test_valuesMatch_edgeCases_strictComparison(self, expected: Any, actual: Any, match: bool) -> None:
        # Arrange / Act
        result = values_match(expected, actual)

        # Assert
        assert result is match


class TestNoProductionHeadlessRoute:
    def test_apiRoutes_headlessAsk_isNotRegistered(self) -> None:
        # Arrange
        from clinical_analytics.api.main import app

        # Act
        paths = [getattr(route, "path", "") for route in app.routes]

        # Assert
        assert not any("headless" in path for path in paths)


@pytest.mark.slow
@pytest.mark.integration
def test_goldenSuite_everyQuestion_answeredCorrectlyOrRefused(project_root: Path) -> None:
    # Arrange
    questions = questions_for_dataset("cohort", project_root)

    # Act
    outcomes: dict[str, list[str]] = {"answered": [], "refused": [], "wrong": []}
    details: list[str] = []
    for question in questions:
        result = ask("cohort", question["query"], project_root)
        outcome, actual = classify(question, question["expected"], result)
        outcomes[outcome].append(question["id"])
        if outcome == "wrong":
            details.append(
                f"{question['id']} {question['query']!r} expected={question['expected']!r} actual={actual!r}"
            )

    # Assert
    assert details == [], "silently wrong answers"
    assert set(outcomes["refused"]) == GOLDEN_REFUSED_IDS
    assert len(outcomes["answered"]) / len(questions) >= GOLDEN_MIN_ANSWERED_RATE
