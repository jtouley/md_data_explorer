# SPDX-License-Identifier: MIT
"""Edges of the hold rule (``contracts.holds``), found by mutation testing: what a held answer
carries, which questions a patient-key question holds, and the sentence shown in its place.
Invented speech-therapy clinic columns only.
"""

from __future__ import annotations

import pytest

from clinical_analytics.contracts.holds import Hold, held_message, open_question_holds
from clinical_analytics.contracts.semantics import DatasetSemantics, Grain, OpenQuestion


def _semantics(*questions: OpenQuestion) -> DatasetSemantics:
    grain = Grain(kind="record", noun="session", rows=8, one_row_per_patient=False, source="default")
    return DatasetSemantics(rows=8, grain=grain, open_questions=questions)


KEY_QUESTION = OpenQuestion(
    id="patient_key:client_ref",
    kind="patient_key",
    question="Which column identifies the client?",
    column="client_ref",
    options=("client_ref", "family_ref"),
)


def test_openQuestionHolds_columnQuestion_holdCarriesItsColumnAndQuestion() -> None:
    # Arrange
    question = OpenQuestion(
        id="missing_code:fluency:99", kind="missing_code", question="Is 99 a missing fluency score?", column="fluency"
    )

    # Act
    held = open_question_holds(_semantics(question), columns={"fluency"}, counts_lines=False)

    # Assert
    assert held == [Hold("missing_code:fluency:99", "missing_code", "fluency", "Is 99 a missing fluency score?")]


@pytest.mark.parametrize(
    ("columns", "counts_lines", "held"),
    [
        pytest.param({"stammer_score"}, False, False, id="reads_another_column"),
        pytest.param(set(), False, False, id="reads_no_column"),
        pytest.param({"family_ref"}, False, True, id="reads_an_option"),
        pytest.param({"stammer_score"}, True, True, id="counts_lines"),
        pytest.param(None, False, True, id="reads_every_column"),
    ],
)
def test_openQuestionHolds_patientKeyQuestion_holdsOnlyWhatReadsItsColumns(
    columns: set[str] | None, counts_lines: bool, held: bool
) -> None:
    # Act
    found = open_question_holds(_semantics(KEY_QUESTION), columns=columns, counts_lines=counts_lines)

    # Assert
    assert bool(found) is held


def test_openQuestionHolds_kindWithNoRule_failsLoud() -> None:
    # Arrange: a kind outside every rule (bypassing the model's own check)
    question = OpenQuestion.model_construct(
        id="q", kind="voice_pitch", question="?", column="pitch", options=(), answered=False
    )

    # Act / Assert
    with pytest.raises(ValueError, match="No hold rule for question kind 'voice_pitch'"):
        open_question_holds(_semantics().model_copy(update={"open_questions": (question,)}), columns=None, counts_lines=False)


@pytest.mark.parametrize(
    ("questions", "expected"),
    [
        pytest.param(
            ("Is 99 missing?",),
            "This answer waits for you to answer 1 question about this file first: Is 99 missing?",
            id="one",
        ),
        pytest.param(
            ("Is 99 missing?", "Is each line one client?"),
            "This answer waits for you to answer 2 questions about this file first: "
            "Is 99 missing? Is each line one client?",
            id="two",
        ),
    ],
)
def test_heldMessage_questions_namedWithTheirCount(questions: tuple[str, ...], expected: str) -> None:
    # Arrange
    holds = [Hold(f"q{i}", "missing_code", "fluency", q) for i, q in enumerate(questions)]

    # Act / Assert
    assert held_message(holds) == expected
