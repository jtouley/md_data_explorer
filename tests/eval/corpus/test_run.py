"""Acceptance runner: one question's failure is graded ERROR and never stops the run.

A fake API on a fake clock stands in for the server; no process is started.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from clinical_analytics.eval.corpus.run import AcceptanceError, answer_patient_rows, ask_all
from clinical_analytics.eval.corpus.score import OracleAnswer

TIMEOUT_S = 10.0
TOTAL = OracleAnswer(1141, "patients", {("total",): 1141})


def _question(qid: str, query: str, **extra: Any) -> dict[str, Any]:
    return {"id": qid, "query": query, "kind": "count", "expect": "answer", **extra}


PARENT = _question("q-parent", "how many")
FOLLOW_UP = _question("q-child", "and the women", follow_up_of="q-parent")
NEXT = _question("q-next", "how many again")
TRUTH = {q["id"]: TOTAL for q in (PARENT, FOLLOW_UP, NEXT)}


class TestAskAll:
    @pytest.mark.parametrize(
        "failure",
        [
            pytest.param(AcceptanceError("POST /api/sessions -> 503: down"), id="http_error"),
            pytest.param({"detail": "unexpected envelope"}, id="no_session_id"),
        ],
    )
    def test_askAll_parentSessionFails_parentAndFollowUpErrorRunContinues(self, make_fake_api, failure):
        # Arrange
        api = make_fake_api(fail={"POST /api/sessions": failure})

        # Act
        results = ask_all(api, "ds", [PARENT, FOLLOW_UP, NEXT], TRUTH, "none", TIMEOUT_S)

        # Assert
        assert [r.grade for r in results] == ["ERROR", "ERROR", "RIGHT"]
        assert "q-parent has no session" in results[1].reason

    def test_askAll_queryResponseWithoutQueryId_gradedError(self, make_fake_api):
        # Arrange
        api = make_fake_api(fail={"POST /api/queries": {"detail": "schema changed"}})

        # Act
        results = ask_all(api, "ds", [PARENT, NEXT], TRUTH, "none", TIMEOUT_S)

        # Assert
        assert [r.grade for r in results] == ["ERROR", "ERROR"]
        assert "returned no query_id" in results[0].reason

    def test_askAll_timedOutQueryFinishesLater_itsLogStaysWithIt(self, make_fake_api):
        # Arrange: the first query takes 15s (timeout 10s) and its model tier falls back.
        api = make_fake_api(finish_at={"how many": 15.0}, falls_back={"how many"})

        # Act
        results = ask_all(api, "ds", [PARENT, NEXT], TRUTH, "llama3.2:3b", TIMEOUT_S)

        # Assert: the fallback is the timed-out question's, not the next one's.
        assert results[0].grade == "ERROR"
        assert results[0].llm.startswith("fell back")
        assert results[1].grade == "RIGHT"
        assert results[1].llm == "ran"

    def test_askAll_timedOutQueryStillRunning_nextQuestionWaitsGraceThenErrors(self, make_fake_api):
        # Arrange: the first query outlives both its timeout and the wait after it.
        api = make_fake_api(finish_at={"how many": 1000.0}, falls_back={"how many"})

        # Act
        results = ask_all(api, "ds", [PARENT, NEXT], TRUTH, "llama3.2:3b", TIMEOUT_S)

        # Assert
        assert [r.grade for r in results] == ["ERROR", "ERROR"]
        assert "still running" in results[0].reason
        assert "server still running timed-out query q1" in results[1].reason


class _AnswerApi:
    """Records the one-row-per-patient answer the runner posts and replies as the app does."""

    def __init__(self) -> None:
        self.sent: list[tuple[str, dict[str, Any]]] = []

    def post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        self.sent.append((path, payload))
        return {"dataset_id": "d1", "counts": "patients" if payload["one_row_per_patient"] else "rows"}


class TestAnswerPatientRows:
    @pytest.mark.parametrize(
        ("knows", "said", "answer"),
        [
            pytest.param(lambda path: True, "answered yes, so answers count patients", True, id="oracle_yes"),
            pytest.param(lambda path: False, "answered no, so answers count rows", False, id="oracle_no"),
        ],
    )
    def test_answerPatientRows_asked_oracleAnswersWithToken(self, tmp_path, knows, said, answer):
        # Arrange
        api, oracles = _AnswerApi(), SimpleNamespace(rows_are_patients=knows)
        uploaded = {"one_row_per_patient": {"status": "pending", "confirmation": "a" * 32}}

        # Act
        note = answer_patient_rows(api, "d1", uploaded, oracles, tmp_path / "t.csv")

        # Assert
        assert note == f"asked; {said}"
        assert api.sent == [
            ("/api/datasets/d1/one-row-per-patient", {"confirmation": "a" * 32, "one_row_per_patient": answer})
        ]

    @pytest.mark.parametrize(
        ("uploaded", "oracles", "said"),
        [
            pytest.param(
                {"one_row_per_patient": None}, SimpleNamespace(rows_are_patients=bool), "not asked", id="none"
            ),
            pytest.param({}, SimpleNamespace(rows_are_patients=bool), "not asked", id="older_app"),
            pytest.param(
                {"one_row_per_patient": {"confirmation": "a" * 32}},
                SimpleNamespace(),
                "asked; not answered (the oracle does not say), so answers count rows",
                id="oracle_silent",
            ),
        ],
    )
    def test_answerPatientRows_notAskedOrOracleSilent_nothingSent(self, tmp_path, uploaded, oracles, said):
        # Arrange
        api = _AnswerApi()

        # Act
        note = answer_patient_rows(api, "d1", uploaded, oracles, tmp_path / "t.csv")

        # Assert: never assumes yes
        assert note == said
        assert api.sent == []

    def test_answerPatientRows_appCountsOtherThing_acceptanceError(self, tmp_path):
        # Arrange
        api = _AnswerApi()
        api.post_json = lambda path, payload: {"counts": "rows"}  # type: ignore[method-assign]
        uploaded = {"one_row_per_patient": {"confirmation": "a" * 32}}

        # Act / Assert
        with pytest.raises(AcceptanceError, match="count 'rows'"):
            answer_patient_rows(api, "d1", uploaded, SimpleNamespace(rows_are_patients=lambda p: True), tmp_path)
