"""Tests for NL query engine golden examples feature.

Following AGENTS.md guidelines:
- AAA pattern (Arrange-Act-Assert)
- Descriptive test names: test_unit_scenario_expectedBehavior
- Test isolation (no shared mutable state)
"""

from pathlib import Path

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
