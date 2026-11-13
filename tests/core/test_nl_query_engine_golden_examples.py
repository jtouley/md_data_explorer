"""Tests for NL query engine golden examples feature.

Following AGENTS.md guidelines:
- AAA pattern (Arrange-Act-Assert)
- Descriptive test names: test_unit_scenario_expectedBehavior
- Test isolation (no shared mutable state)
"""

from pathlib import Path

import numpy as np
import pytest
import yaml


class TestNLQueryEngineGoldenExamples:
    """Test suite for golden examples embedding cache in NL query engine."""

    def test_load_golden_examples_config_loads_from_yaml(self, tmp_path):
        """Test that load_golden_examples_config loads questions from YAML file."""
        # Arrange: Create golden examples config
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        config_file = config_dir / "golden_examples.yaml"
        config_data = {
            "questions": [
                {
                    "id": "count_all",
                    "query": "how many patients?",
                    "expected_intent": "COUNT",
                    "expected_metric": None,
                    "expected_group_by": None,
                },
                {
                    "id": "describe_age",
                    "query": "what is the average age?",
                    "expected_intent": "DESCRIBE",
                    "expected_metric": "age",
                    "expected_group_by": None,
                },
            ]
        }
        config_file.write_text(yaml.dump(config_data))

        # Act: Load config
        from clinical_analytics.core.config_loader import load_golden_examples_config

        result = load_golden_examples_config(config_path=config_file)

        # Assert: Questions are loaded
        assert "questions" in result
        assert len(result["questions"]) == 2
        assert result["questions"][0]["id"] == "count_all"
        assert result["questions"][0]["query"] == "how many patients?"
        assert result["questions"][1]["id"] == "describe_age"

    def test_load_golden_examples_config_missing_file_uses_defaults(self):
        """Test that missing YAML file uses default golden questions location."""
        # Arrange: Non-existent config file
        config_file = Path("/nonexistent/config/golden_examples.yaml")

        # Act: Load config (should use defaults)
        from clinical_analytics.core.config_loader import load_golden_examples_config

        result = load_golden_examples_config(config_path=config_file)

        # Assert: Empty questions list (no default file)
        assert result == {"questions": []}

    def test_nl_query_engine_loads_golden_examples_at_init(self, make_semantic_layer):
        """Test that NL query engine loads golden examples at initialization."""
        # Arrange: Create semantic layer
        semantic = make_semantic_layer(
            dataset_name="test",
            data={"patient_id": [1, 2, 3], "age": [25, 35, 45]},
        )

        # Act: Initialize engine
        from clinical_analytics.core.nl_query_engine import NLQueryEngine

        engine = NLQueryEngine(semantic)

        # Assert: Golden embeddings are loaded (may be None if no config file)
        assert hasattr(engine, "_golden_embeddings")
        assert hasattr(engine, "_golden_examples")

    @pytest.mark.usefixtures("cached_sentence_transformer")  # needs the real embedding model
    def test_nl_query_engine_golden_embeddings_cached(self, make_semantic_layer, tmp_path):
        """Test that golden embeddings are cached in memory."""
        # Arrange: Create golden examples config
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        config_file = config_dir / "golden_examples.yaml"
        config_data = {
            "questions": [
                {
                    "id": "count_all",
                    "query": "how many patients?",
                    "expected_intent": "COUNT",
                },
            ]
        }
        config_file.write_text(yaml.dump(config_data))

        semantic = make_semantic_layer(
            dataset_name="test",
            data={"patient_id": [1, 2], "age": [25, 35]},
        )

        # Act: Initialize engine with config
        from clinical_analytics.core.nl_query_engine import NLQueryEngine

        engine = NLQueryEngine(semantic)
        # Clear cache and set new config path
        engine._golden_embeddings = None
        engine._golden_examples = None
        engine._golden_config_path = config_file
        engine._load_golden_examples()

        # Assert: Embeddings are cached
        assert engine._golden_embeddings is not None
        assert engine._golden_examples is not None
        assert len(engine._golden_examples) == 1

        # Act: Load again (should use cache)
        first_embeddings = engine._golden_embeddings
        engine._load_golden_examples()
        second_embeddings = engine._golden_embeddings

        # Assert: Same object returned (cached)
        assert first_embeddings is second_embeddings

    def test_nl_query_engine_semantic_match_uses_golden_examples(self, make_semantic_layer):
        """Test that semantic match uses golden examples for similarity."""
        # Arrange: Create semantic layer
        semantic = make_semantic_layer(
            dataset_name="test",
            data={"patient_id": [1, 2], "age": [25, 35]},
        )

        from clinical_analytics.core.nl_query_engine import NLQueryEngine

        engine = NLQueryEngine(semantic)

        # Act: Parse query similar to golden example
        # This should match the "count" pattern via golden examples
        intent = engine.parse_query("how many total patients?")

        # Assert: Intent type is COUNT (from golden example similarity)
        assert intent is not None
        assert intent.intent_type == "COUNT"
        assert intent.confidence >= 0.7

    @pytest.mark.usefixtures("cached_sentence_transformer")  # needs the real embedding model
    def test_semantic_match_golden_example_high_similarity_returns_intent(self, make_semantic_layer):
        """Test that queries with high similarity to golden examples use golden intent."""
        # Arrange: Create semantic layer
        semantic = make_semantic_layer(
            dataset_name="test",
            data={"patient_id": [1, 2, 3], "age": [25, 35, 45], "status": ["active", "inactive", "active"]},
        )

        from clinical_analytics.core.nl_query_engine import NLQueryEngine

        engine = NLQueryEngine(semantic)

        # Act: Parse query very similar to golden example "how many patients?"
        intent = engine._semantic_match("how many patients are there?")

        # Assert: Should match via golden example with high confidence
        assert intent is not None
        assert intent.intent_type == "COUNT"
        assert intent.confidence >= 0.8  # High confidence from golden match

    @pytest.mark.usefixtures("cached_sentence_transformer")  # needs the real embedding model
    def test_semantic_match_golden_example_with_variables(self, make_semantic_layer):
        """Test that golden examples with variables populate intent correctly."""
        # Arrange: Create semantic layer
        semantic = make_semantic_layer(
            dataset_name="test",
            data={"patient_id": [1, 2], "age": [25, 35], "gender": ["M", "F"]},
        )

        from clinical_analytics.core.nl_query_engine import NLQueryEngine

        engine = NLQueryEngine(semantic)

        # Act: Parse query similar to golden example "average age by gender"
        intent = engine._semantic_match("mean age by gender")

        # Assert: Should match DESCRIBE intent with variables
        assert intent is not None
        assert intent.intent_type == "DESCRIBE"
        # Variables should be extracted (either from golden or query)
        # Note: exact variable extraction depends on golden example metadata

    @pytest.mark.parametrize("noisy_duplicate", [0, 1], ids=["first_scores_lower", "second_scores_lower"])
    def test_semantic_match_duplicateGoldenQueryWithFloatNoise_sameExampleOnEveryMachine(
        self, make_semantic_layer, mock_encoder, noisy_duplicate
    ):
        """Two golden examples with the same query must not be chosen between by ~1e-7 score noise."""
        # Arrange: duplicates with different intents; one of them scores 1e-7 lower (CPU noise)
        from clinical_analytics.core.nl_query_engine import NLQueryEngine

        semantic = make_semantic_layer(dataset_name="tie_check", data={"patient_id": [1, 2], "age": [30, 40]})
        engine = NLQueryEngine(semantic)
        noise_angle = np.sqrt(2e-7)  # cos(angle) ~ 1 - 1e-7
        exact, noisy = [1.0, 0.0, 0.0], [np.cos(noise_angle), np.sin(noise_angle), 0.0]
        golden_embeddings = np.array([exact, exact, [0.0, 0.0, 1.0]], dtype=np.float64)
        golden_embeddings[noisy_duplicate] = noisy
        engine._golden_embeddings = golden_embeddings
        engine._golden_examples = [
            {"query": "same text", "expected_intent": "COUNT"},
            {"query": "same text", "expected_intent": "DESCRIBE"},
            {"query": "other text", "expected_intent": "CORRELATIONS"},
        ]
        mock_encoder.encode.side_effect = lambda texts, *args, **kwargs: np.array([exact], dtype=np.float64)
        engine.encoder = mock_encoder

        # Act
        intent = engine._semantic_match("same text")

        # Assert: the first duplicate wins whichever of the two the noise favours
        assert intent is not None
        assert intent.intent_type == "COUNT"
        assert intent.confidence == pytest.approx(1.0, abs=1e-6)


def _duplicate_queries(questions: list[dict]) -> dict[str, list[str]]:
    """Query text (case and surrounding/inner whitespace ignored) -> ids, for texts used more than once."""
    ids_by_text: dict[str, list[str]] = {}
    for question in questions:
        text = " ".join(str(question["query"]).lower().split())
        ids_by_text.setdefault(text, []).append(str(question.get("id")))
    return {text: ids for text, ids in ids_by_text.items() if len(ids) > 1}


@pytest.mark.parametrize(
    ("queries", "expected"),
    [
        ([], {}),
        (["how many patients?"], {}),
        (["how many patients?", "average age"], {}),
        (["remove the n/a", "remove the n/a"], {"remove the n/a": ["q0", "q1"]}),
        (["Remove  the N/A ", "remove the n/a", "average age"], {"remove the n/a": ["q0", "q1"]}),
    ],
    ids=["empty", "single", "distinct", "exact_duplicate", "case_and_whitespace_duplicate"],
)
def test_duplicate_queries_syntheticQuestions_reportsRepeatedTextOnly(queries, expected):
    # Arrange
    questions = [{"id": f"q{i}", "query": query} for i, query in enumerate(queries)]

    # Act
    duplicates = _duplicate_queries(questions)

    # Assert
    assert duplicates == expected


def test_shipped_golden_examples_queryText_isUnique():
    """One query text, one example: retrieval picks an example by its text alone.

    Two examples with the same text embed identically, so only the first can ever be retrieved;
    when they carry different intents the second is a contradiction that float noise could select.
    """
    # Arrange
    from clinical_analytics.core.config_loader import load_golden_examples_config

    questions = load_golden_examples_config()["questions"]

    # Act
    duplicates = _duplicate_queries(questions)

    # Assert
    assert questions, "the shipped golden examples file is empty or missing"
    assert duplicates == {}, f"duplicate query text in golden_examples.yaml (text -> ids): {duplicates}"


@pytest.mark.usefixtures("cached_sentence_transformer")  # needs the real embedding model
@pytest.mark.parametrize(
    ("history", "expected_intent"),
    [
        ([{"query": "count patients by statin", "intent": "COUNT", "group_by": "statin", "metric": None}], "COUNT"),
        (None, None),
        ([], None),
    ],
    ids=["after_a_count", "no_history", "empty_history"],
)
def test_semantic_match_removeTheNa_refinesPreviousCountOnlyWithContext(make_semantic_layer, history, expected_intent):
    """The one 'remove the n/a' example is the refinement of a count; without context it is not used."""
    # Arrange
    from clinical_analytics.core.nl_query_engine import NLQueryEngine

    semantic = make_semantic_layer(
        dataset_name="refinement_golden",
        data={"patient_id": [1, 2, 3, 4], "statin": [0, 1, 2, 1], "age": [45, 52, 38, 61]},
    )
    engine = NLQueryEngine(semantic)

    # Act
    intent = engine._semantic_match("remove the n/a", conversation_history=history)

    # Assert
    matched = None if intent is None or intent.confidence < 0.99 else intent.intent_type
    assert matched == expected_intent
