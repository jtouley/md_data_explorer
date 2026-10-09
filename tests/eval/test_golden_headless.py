"""Golden questions answered through the headless service and API."""

from pathlib import Path

from fastapi.testclient import TestClient

from clinical_analytics.api.main import app
from clinical_analytics.eval.catalog import DATASET_TABLES, QUESTIONS_PER_DATASET, questions_for_dataset
from clinical_analytics.eval.headless import actual_value, ask, values_match

WORKSPACE = Path(__file__).resolve().parents[2]


class TestGoldenCatalog:
    def test_each_dataset_has_100_unique_questions(self) -> None:
        for dataset_id in DATASET_TABLES:
            questions = questions_for_dataset(dataset_id, WORKSPACE)
            queries = [question["query"] for question in questions]
            assert len(questions) == QUESTIONS_PER_DATASET
            assert len(set(queries)) == QUESTIONS_PER_DATASET


class TestHeadlessAsk:
    def test_count_patients_matches_row_count(self) -> None:
        questions = questions_for_dataset("mimic_patients", WORKSPACE)
        question = next(item for item in questions if item["query"] == "how many patients?")
        result = ask("mimic_patients", question["query"], WORKSPACE)
        actual = actual_value("count", None, None, result["payload"])
        assert result["success"]
        assert values_match(question["expected"], actual)

    def test_api_count_matches_cli_rows(self) -> None:
        client = TestClient(app)
        response = client.post(
            "/api/headless/ask",
            json={"dataset_id": "mimic_patients", "query": "how many patients?"},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["success"] is True
        assert body["intent"] == "COUNT"
        assert body["rows"]
