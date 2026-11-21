# SPDX-License-Identifier: MIT
"""The answer loop: the model writes the query, the app runs it, a failure goes back as feedback."""

from __future__ import annotations

from typing import Any

import pytest

from clinical_analytics.core.result_presentation import present_analysis
from clinical_analytics.model_query import llm
from clinical_analytics.model_query.engine import (
    MODEL_UNAVAILABLE_REASON,
    NO_QUERY_REASON,
    answer_question,
)
from clinical_analytics.model_query.plan import LISTING_REASON, unused_mentions
from clinical_analytics.sql import sandbox
from clinical_analytics.sql.sandbox import SandboxConnection
from tests.model_query import tables


def _plan(answer: str, **fields: Any) -> dict[str, Any]:
    base = {"thought": "t", "answer": answer, "where": None, "group_by": None}
    return base | {"numerator": None, "number": None, "outcome": None} | fields


class Recorded:
    """A stand-in model: gives the recorded replies in order and keeps the prompts it was sent."""

    def __init__(self, *replies: Any) -> None:
        self.replies = list(replies)
        self.prompts: list[tuple[str, str]] = []

    def __call__(self, system: str, user: str, schema: dict[str, Any]) -> Any:
        self.prompts.append((system, user))
        return self.replies.pop(0)


@pytest.fixture
def ask(make_query_table):
    """Factory: answer ``question`` on a synthetic table with the recorded ``model``."""

    def _ask(question: str, model: Recorded, table: str = "patients", **options: Any):
        frame = tables.patients() if table == "patients" else tables.visits()
        catalog, context = make_query_table(frame)
        return answer_question(
            question,
            context,
            model=model,
            open_sandbox=lambda: SandboxConnection(catalog),
            examples=options.pop("examples", []),
            **options,
        )

    return _ask


def test_answer_question_wordsThatMatchNoColumn_areAnsweredNotHeld(ask):
    # Arrange: "folks", "our", "registry", "please" name no column or value
    model = Recorded(_plan("count", where="admitted = 'yes'"))

    # Act
    answer = ask("How many folks in our registry ended up in hospital, please?", model)

    # Assert: answered on the first attempt, with the SQL and its plain reading
    assert (answer.status, answer.attempts) == ("answered", 1)
    assert answer.reading == "Counted patients where admitted = 'Yes'."
    assert answer.sql is not None and answer.sql.startswith("SELECT")
    assert present_analysis(answer.analysis)["claim"] == "Total count: 2 patients."
    assert answer.analysis is not None and answer.analysis["sql"] == answer.sql


def test_answer_question_planError_isFedBackAndTheRetryAnswers(ask):
    # Arrange
    model = Recorded(_plan("count", where="gender = 'male'"), _plan("count", where="sex = 'male'"))

    # Act
    answer = ask("How many men?", model)

    # Assert: the second prompt carries the failed reply and what was wrong with it
    assert (answer.status, answer.attempts) == ("answered", 2)
    retry_prompt = model.prompts[1][1]
    assert "There is no column named gender" in retry_prompt and '"where": "gender = \'male\'"' in retry_prompt
    assert answer.problems == (
        "There is no column named gender. The columns are: patient_id, sex, dx, admitted, confirmed, "
        "onset_year, comorbid, age_cat.",
    )


def test_answer_question_engineError_feedsTheEngineMessageBackButNeverShowsIt(ask):
    # Arrange: SQL that parses but fails in DuckDB (text compared with a list)
    bad = {"thought": "t", "answer": "sql", "sql": "SELECT COUNT(*) AS count FROM unified WHERE onset_year = [1]"}
    good = {"thought": "Count all.", "answer": "sql", "sql": "SELECT COUNT(*) AS count FROM unified"}
    model = Recorded(bad, good)

    # Act
    answer = ask("How many?", model, fmt="sql")

    # Assert: the model saw DuckDB's message; the recorded problem is the plain reason only
    assert (answer.status, answer.attempts) == ("answered", 2)
    assert "It could not be used:" in model.prompts[1][1]
    assert len(answer.problems) == 1 and answer.problems[0] in sandbox._COULD_NOT_RUN.values()
    assert present_analysis(answer.analysis)["claim"] == "Total count: 12 patients."


def test_answer_question_everyAttemptFails_isRefusedWithTheLastReason(ask):
    # Arrange
    model = Recorded(*[_plan("count", where="sex = 'unknown'")] * 3)

    # Act
    answer = ask("How many?", model, max_attempts=3)

    # Assert
    assert (answer.status, answer.attempts, answer.analysis) == ("refused", 3, None)
    assert answer.reason is not None and answer.reason.startswith(NO_QUERY_REASON)
    assert "Column sex has no value 'unknown'" in answer.reason


@pytest.mark.parametrize("payload", [None])
def test_answer_question_modelDoesNotAnswer_isUnavailable(ask, payload):
    # Act
    answer = ask("How many?", Recorded(payload))

    # Assert
    assert (answer.status, answer.reason) == ("unavailable", MODEL_UNAVAILABLE_REASON)


@pytest.mark.parametrize("payload", ["not json", ["a"], {"where": "sex = 'male'"}, {"answer": "poem"}])
def test_answer_question_replyNotUsable_isFedBack(ask, payload):
    # Arrange
    model = Recorded(payload, _plan("count"))

    # Act
    answer = ask("How many?", model)

    # Assert
    assert (answer.status, answer.attempts) == ("answered", 2)


@pytest.mark.parametrize(
    "question",
    [
        "List the patients who were admitted.",
        "Which patients were admitted more than once? I want their charts.",
        "Give me the record numbers of the admitted.",
    ],
)
def test_answer_question_asksForIndividualPatients_isRefusedWithoutTheModel(ask, question):
    # Arrange: the model is never called
    model = Recorded()

    # Act
    answer = ask(question, model)

    # Assert
    assert (answer.status, answer.reason, model.prompts) == ("refused", LISTING_REASON, [])


@pytest.mark.parametrize(
    "reply",
    [
        _plan("listing"),
        _plan("count", group_by="patient_id"),
        {"thought": "t", "answer": "sql", "sql": "SELECT patient_id, sex FROM unified"},
    ],
)
def test_answer_question_modelOrQueryListsPatients_isRefused(ask, reply):
    # Act
    answer = ask("Tell me about the admitted ones.", Recorded(reply), fmt="sql" if "sql" in reply else "plan")

    # Assert
    assert (answer.status, answer.reason) == ("refused", LISTING_REASON)


def test_answer_question_noRowsMatch_asksTheModelOnceThenReportsZero(ask):
    # Arrange: a condition no row meets, sent twice
    reply = _plan("count", where="dx = 'other' AND admitted = 'yes'")
    model = Recorded(reply, reply)

    # Act
    answer = ask("How many with other diagnoses were admitted?", model)

    # Assert: the model was asked to check its conditions; the same reply then stands as zero
    assert "No rows match the conditions" in model.prompts[1][1]
    assert present_analysis(answer.analysis)["claim"] == "Total count: 0 patients."


def test_answer_question_valueNamedButColumnUnused_hintsOnceAndTakesTheBetterReply(ask):
    # Arrange: the first reply drops "women"
    model = Recorded(_plan("count", where="confirmed = 'yes'"), _plan("count", where="confirmed = 'yes' AND sex = 'F'"))

    # Act
    answer = ask("How many women had a confirmed infection?", model)

    # Assert
    assert "The question mentions 'women' (column sex)" in model.prompts[1][1]
    assert answer.reading == "Counted patients where confirmed = 'Yes' AND sex = 'female'."
    assert present_analysis(answer.analysis)["claim"] == "Total count: 3 patients."


@pytest.mark.parametrize("second", [None, _plan("count", where="nope = 1"), _plan("count", where="confirmed = 'yes'")])
def test_answer_question_hintNotTaken_theFirstAnswerStands(ask, second):
    # Arrange: after the hint the model fails, errs, or repeats itself
    model = Recorded(_plan("count", where="confirmed = 'yes'"), second)

    # Act
    answer = ask("How many women had a confirmed infection?", model)

    # Assert: a hint never holds the question
    assert answer.status == "answered"
    assert present_analysis(answer.analysis)["claim"] == "Total count: 4 patients."


def test_unused_mentions_menOrWomenInTheQuestion_isFoundOnlyWhenSexIsUnused(make_query_table):
    # Arrange
    _catalog, context = make_query_table(tables.patients())
    question = "How many men with relapsing disease said yes?"

    # Act
    unused = unused_mentions(question, frozenset({"comorbid"}), context)
    none_unused = unused_mentions(question, frozenset({"sex"}), context)

    # Assert: only the sex words are read; "relapsing" and "yes" are left to the model
    assert unused == [("men", "sex")]
    assert none_unused == []


def test_answer_question_shareVariesWithinPatient_asksEverOrAlways(ask):
    # Act
    answer = ask(
        "What share of patients were screened?", Recorded(_plan("share", numerator="screened = 'yes'")), "visits"
    )

    # Assert: a real ambiguity, with concrete choices that each answer it when asked
    assert answer.status == "clarify"
    assert answer.reason is not None and "45 patients" in answer.reason
    assert [label.split(":")[0] for label, _question in answer.choices] == ["Ever", "Always"]
    assert answer.choices[0][1].startswith("What share of patients were screened? Count one when it was ever true")


# Hand count (tests/model_query/tables.py): 60 of 150 patients were screened at some visit, 15 at
# every visit (the one-visit patients screened at it); for 45 it differs between their visits.
@pytest.mark.parametrize(("choice", "rollup", "rate"), [(0, "any", 0.4), (1, "all", 0.1)])
def test_answer_question_choiceAskedAsQuestion_isAnsweredWithThatRollup(ask, choice, rollup, rate):
    # Arrange: the question a choice of the clarification asks
    reply = _plan("share", numerator="screened = 'yes'")
    asked = ask("What share of patients were screened?", Recorded(reply), "visits").choices[choice][1]

    # Act
    answer = ask(asked, Recorded(reply), "visits")

    # Assert
    assert answer.status == "answered"
    assert answer.analysis is not None and answer.analysis["rollup"] == rollup
    assert answer.analysis["proportion"] == pytest.approx(rate)


def test_answer_question_followUp_showsTheModelThePreviousTurn(ask):
    # Arrange
    previous = [("How many by diagnosis?", _plan("count", group_by="dx"))]
    model = Recorded(_plan("count", group_by="dx", where="sex = 'female'"))

    # Act
    answer = ask("Now only the women.", model, previous=previous)

    # Assert
    assert "How many by diagnosis?" in model.prompts[0][1]
    assert answer.reading == "Counted patients where sex = 'female', by dx."


def test_answer_question_noExamplesGiven_retrievesThemFromThePool(ask):
    # Arrange
    model = Recorded(_plan("count"))

    # Act
    ask("How many patients are there?", model, examples=None)

    # Assert
    assert "Examples on other tables" in model.prompts[0][1]


@pytest.mark.parametrize("allow_remote", [False, True])
def test_ollama_model_remoteUrl_isNeverCalled(monkeypatch, allow_remote):
    # Arrange: a configured address that is not this machine (even with the remote opt-in)
    monkeypatch.setattr(llm.nl_query_config, "OLLAMA_BASE_URL", "https://models.example.com")
    monkeypatch.setattr(llm.nl_query_config, "OLLAMA_ALLOW_REMOTE_HOST", allow_remote)

    # Act
    reply = llm.ollama_model("system", "user", {})

    # Assert: patient data stays on the machine
    assert reply is None
    assert llm.model_available() is False


def test_ollama_model_localServer_sendsSchemaAndDeterministicOptions(monkeypatch):
    # Arrange: the HTTP boundary of the app's Ollama client
    sent: dict[str, Any] = {}

    class Response:
        status_code = 200

        def json(self) -> dict[str, Any]:
            return {"response": '{"answer": "count"}'}

    class Session:
        """The proxy-free session OllamaClient uses (core.loopback)."""

        def post(self, url: str, json: dict[str, Any], timeout: float, allow_redirects: bool) -> Response:
            sent.update(url=url, payload=json, allow_redirects=allow_redirects)
            return Response()

        def get(self, *args: Any, **kwargs: Any) -> Response:
            return Response()

    monkeypatch.setattr(llm.nl_query_config, "OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    monkeypatch.setattr("clinical_analytics.core.llm_client.ollama_session", Session)
    schema = {"type": "object"}

    # Act
    reply = llm.ollama_model("system text", "user text", schema)

    # Assert
    assert reply == {"answer": "count"}
    assert (sent["url"], sent["allow_redirects"]) == ("http://127.0.0.1:11434/api/generate", False)
    assert sent["payload"]["format"] == schema
    assert sent["payload"]["system"] == "system text"
    assert sent["payload"]["options"]["temperature"] == 0
    assert sent["payload"]["options"]["num_ctx"] == llm.NUM_CTX
