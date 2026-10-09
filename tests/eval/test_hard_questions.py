"""Oracle-backed hard questions and clinician-shaped questions.

Expected values come from DuckDB SQL, not from the parser.
"""

from clinical_analytics.eval.catalog import DATASET_TABLES
from clinical_analytics.eval.hard_questions import all_hard_questions, oracle_value, questions_for_dataset
from clinical_analytics.eval.headless import actual_value, ask, values_match


def test_each_dataset_has_50_analytical_and_50_provider_questions() -> None:
    for dataset_id in DATASET_TABLES:
        questions = questions_for_dataset(dataset_id)
        families = {question["family"] for question in questions}
        assert families == {"analytical", "provider"}
        assert sum(question["family"] == "analytical" for question in questions) == 50
        assert sum(question["family"] == "provider" for question in questions) == 50
        assert len({question["query"] for question in questions}) == 100
        for question in questions:
            if question["family"] == "provider":
                assert question["citation"]


def test_oracle_sql_returns_a_value_for_every_hard_question() -> None:
    for question in all_hard_questions():
        assert oracle_value(question) is not None, question["id"]


def test_headless_matches_sql_oracle_for_hard_questions() -> None:
    failures: list[str] = []
    for question in all_hard_questions():
        expected = oracle_value(question)
        result = ask(question["dataset_id"], question["query"])
        actual = actual_value(
            question["kind"],
            question.get("stat"),
            question.get("group_by"),
            result.get("payload"),
        )
        if not result["success"] or not values_match(expected, actual):
            failures.append(f"{question['id']} expected={expected!r} actual={actual!r}")
    assert failures == []


def test_predictor_question_is_not_success() -> None:
    result = ask("gdsi", "what predicts recovery?")
    assert result["success"] is False


def test_unbound_death_word_is_not_success() -> None:
    """dod is the stored column. 'died' is not one of its values, so the count must not succeed."""
    result = ask("mimic_patients", "How many MIMIC patients died?")
    assert result["success"] is False


def test_query_service_unbound_death_is_an_error() -> None:
    """The service path must fail closed, not only the headless wrapper."""
    from clinical_analytics.eval.headless import service_for

    raw = service_for("mimic_patients").ask(
        "How many MIMIC patients died?",
        dataset_id="mimic_patients",
        dataset_version="golden",
    )
    assert any(item.get("severity") == "error" for item in raw.issues)


def test_grouped_count_stays_success() -> None:
    result = ask("gdsi", "how many patients by sex?")
    assert result["success"] is True
    assert result["group_by"] == "sex"


def test_non_numeric_correlation_is_not_success() -> None:
    result = ask("gdsi", "correlation between age and BMI")
    assert result["success"] is False
