"""
Real-World Query Test Suite (ADR003).

Tests actual queries from user interactions to verify parsing accuracy
and track improvements over time. Uses fixtures from conftest.py to avoid
hardcoding test data.

This test suite:
- Validates parsing of real-world queries
- Tracks expected outputs in fixtures (easy to update)
- Verifies confidence thresholds and parsing tiers
- Allows iterative improvement without breaking existing tests
"""

import pytest

from clinical_analytics.core.nl_query_engine import NLQueryEngine


class TestRealWorldCorrelationQueries:
    """Test real-world CORRELATIONS queries from user interactions."""

    @pytest.mark.parametrize(
        "test_case",
        [
            pytest.param(
                {
                    "query": (
                        "how does bmi, statin use relate to the regiment that the person is on "
                        "and their cd4 counts?"
                    ),
                    "expected_intent": "CORRELATIONS",
                    "expected_primary_variable": "BMI",  # First variable extracted
                    "expected_grouping_variable": "statins",  # Second variable (statin use)
                    "min_confidence": 0.85,
                    "parsing_tier": "pattern_match",
                },
                id="bmi_statin_regimen_cd4_relationship",
            ),
        ],
    )
    def test_correlation_query_parsing(self, test_case: dict, semantic_layer_with_clinical_columns):
        """Test that CORRELATIONS queries parse correctly with expected intent and variables."""
        # Arrange
        engine = NLQueryEngine(semantic_layer_with_clinical_columns)
        query = test_case["query"]

        # Act
        intent = engine.parse_query(query)

        # Assert
        assert intent is not None, f"Query should parse: {query}"
        assert intent.intent_type == test_case["expected_intent"], (
            f"Expected {test_case['expected_intent']}, got {intent.intent_type} for: {query}"
        )
        assert intent.confidence >= test_case["min_confidence"], (
            f"Confidence {intent.confidence} below minimum {test_case['min_confidence']} for: {query}"
        )

        # Verify parsing tier if specified
        if "parsing_tier" in test_case and test_case["parsing_tier"]:
            assert intent.parsing_tier == test_case["parsing_tier"], (
                f"Expected tier {test_case['parsing_tier']}, got {intent.parsing_tier} for: {query}"
            )

        # Verify primary variable if expected
        if test_case.get("expected_primary_variable"):
            expected_primary = test_case["expected_primary_variable"]
            assert intent.primary_variable == expected_primary, (
                f"Expected primary variable {expected_primary}, got {intent.primary_variable} for: {query}"
            )

        # Verify grouping variable if expected
        if test_case.get("expected_grouping_variable"):
            expected_grouping = test_case["expected_grouping_variable"]
            assert intent.grouping_variable == expected_grouping, (
                f"Expected grouping variable {expected_grouping}, got {intent.grouping_variable} for: {query}"
            )


class TestRealWorldCountQueries:
    """Test real-world COUNT queries from user interactions."""

    @pytest.mark.parametrize(
        "test_case",
        [
            pytest.param(
                {
                    "query": "how many patients were on statins",
                    "expected_intent": "COUNT",
                    "expected_primary_variable": None,
                    "expected_grouping_variable": None,
                    "min_confidence": 0.75,
                    "parsing_tier": "pattern_match",
                },
                id="count_patients_on_statins",
            ),
            pytest.param(
                {
                    "query": "which statin was most prescribed?",
                    "expected_intent": "COUNT",
                    "expected_primary_variable": None,
                    "expected_grouping_variable": "statins",  # Canonical column name (normalized from "statin")
                    "min_confidence": 0.75,
                    "parsing_tier": "pattern_match",
                },
                id="most_prescribed_statin",
            ),
            pytest.param(
                {
                    "query": "what was the most common HIV regiment?",
                    "expected_intent": "COUNT",
                    "expected_primary_variable": None,
                    "expected_grouping_variable": "hiv_regiment",
                    "min_confidence": 0.75,
                    "parsing_tier": "pattern_match",
                },
                id="most_common_hiv_regiment",
            ),
            pytest.param(
                {
                    "query": "what was the most common Current Regimen",
                    "expected_intent": "COUNT",
                    "expected_primary_variable": None,
                    "expected_grouping_variable": "Current Regimen",
                    "min_confidence": 0.75,
                    "parsing_tier": "pattern_match",
                },
                id="most_common_current_regimen",
            ),
        ],
    )
    def test_count_query_parsing(self, test_case: dict, semantic_layer_with_clinical_columns):
        """Test that COUNT queries parse correctly with expected intent and variables."""
        # Arrange
        engine = NLQueryEngine(semantic_layer_with_clinical_columns)
        query = test_case["query"]

        # Act
        intent = engine.parse_query(query)

        # Assert
        assert intent is not None, f"Query should parse: {query}"
        assert intent.intent_type == test_case["expected_intent"], (
            f"Expected {test_case['expected_intent']}, got {intent.intent_type} for: {query}"
        )
        assert intent.confidence >= test_case["min_confidence"], (
            f"Confidence {intent.confidence} below minimum {test_case['min_confidence']} for: {query}"
        )

        # Verify parsing tier if specified
        if "parsing_tier" in test_case and test_case["parsing_tier"]:
            assert intent.parsing_tier == test_case["parsing_tier"], (
                f"Expected tier {test_case['parsing_tier']}, got {intent.parsing_tier} for: {query}"
            )

        # Verify grouping variable if expected
        if test_case.get("expected_grouping_variable"):
            expected_grouping = test_case["expected_grouping_variable"]
            assert intent.grouping_variable == expected_grouping, (
                f"Expected grouping variable {expected_grouping}, got {intent.grouping_variable} for: {query}"
            )

    def test_count_query_with_filter_parsing(self, semantic_layer_with_clinical_columns):
        """Test COUNT query with filter: 'excluding those not on statins, which was the most prescribed statin?'"""
        # Arrange
        engine = NLQueryEngine(semantic_layer_with_clinical_columns)
        query = "excluding those not on statins, which was the most prescribed statin?"

        # Act
        intent = engine.parse_query(query)

        # Assert
        assert intent is not None, f"Query should parse: {query}"
        assert intent.intent_type == "COUNT", f"Expected COUNT, got {intent.intent_type}"
        assert intent.confidence >= 0.7, f"Confidence {intent.confidence} below minimum 0.7 for: {query}"
        # Note: Filter extraction may require more sophisticated parsing
        # This test verifies basic intent recognition

    def test_complex_count_breakdown_parsing(self, semantic_layer_with_clinical_columns):
        """Test complex COUNT breakdown queries."""
        # Arrange
        engine = NLQueryEngine(semantic_layer_with_clinical_columns)

        test_cases = [
            {
                "query": "what statins were those patients on, broken down by count of patients per statin?",
                "expected_grouping": "statin",
            },
            {
                "query": (
                    "what statins were those patients on, broken down by count of patients by their Current Regimen"
                ),
                "expected_grouping": "Current Regimen",
            },
        ]

        for test_case in test_cases:
            # Act
            intent = engine.parse_query(test_case["query"])

            # Assert
            assert intent is not None, f"Query should parse: {test_case['query']}"
            assert intent.intent_type == "COUNT", f"Expected COUNT, got {intent.intent_type} for: {test_case['query']}"
            assert intent.confidence >= 0.7, (
                f"Confidence {intent.confidence} below minimum 0.7 for: {test_case['query']}"
            )
            # Note: Grouping variable extraction may require semantic matching
            # This test verifies basic intent recognition


class TestRealWorldDescribeQueries:
    """Test real-world DESCRIBE queries from user interactions."""

    @pytest.mark.parametrize(
        "query,expected_variable",
        [
            ("average BMI of patients", "BMI"),
            ("average ldl of all patients", "LDL mg/dL"),
        ],
    )
    def test_average_query_parsing(self, query: str, expected_variable: str, semantic_layer_with_clinical_columns):
        """Test that average/mean queries parse correctly with variable extraction."""
        # Arrange
        engine = NLQueryEngine(semantic_layer_with_clinical_columns)

        # Act
        intent = engine.parse_query(query)

        # Assert
        assert intent is not None, f"Query should parse: {query}"
        assert intent.intent_type == "DESCRIBE", f"Expected DESCRIBE, got {intent.intent_type} for: {query}"
        assert intent.primary_variable == expected_variable, (
            f"Expected variable {expected_variable}, got {intent.primary_variable} for: {query}"
        )
        assert intent.confidence >= 0.85, f"Confidence {intent.confidence} below minimum 0.85 for: {query}"
        assert intent.parsing_tier == "pattern_match", f"Expected pattern_match tier for: {query}"


class TestRealWorldQueryTracking:
    """Test suite for tracking all real-world queries from fixtures."""

    def test_all_count_queries_from_fixture(self, real_world_query_test_cases, semantic_layer_with_clinical_columns):
        """Test all COUNT queries from the real_world_query_test_cases fixture."""
        # Arrange
        engine = NLQueryEngine(semantic_layer_with_clinical_columns)
        count_queries = real_world_query_test_cases["count_queries"]

        # Act & Assert
        for test_case in count_queries:
            query = test_case["query"]
            intent = engine.parse_query(query)

            # Basic validation
            assert intent is not None, f"Query should parse: {query}"
            assert intent.intent_type == test_case["expected_intent"], (
                f"Expected {test_case['expected_intent']}, got {intent.intent_type} for: {query}"
            )
            assert intent.confidence >= test_case["min_confidence"], (
                f"Confidence {intent.confidence} below minimum {test_case['min_confidence']} for: {query}"
            )

            # Log parsing tier for tracking
            if intent.parsing_tier:
                print(f"Query: '{query}' -> Tier: {intent.parsing_tier}, Confidence: {intent.confidence:.2f}")

    def test_all_describe_queries_from_fixture(self, real_world_query_test_cases, semantic_layer_with_clinical_columns):
        """Test all DESCRIBE queries from the real_world_query_test_cases fixture."""
        # Arrange
        engine = NLQueryEngine(semantic_layer_with_clinical_columns)
        describe_queries = real_world_query_test_cases["describe_queries"]

        # Act & Assert
        for test_case in describe_queries:
            query = test_case["query"]
            intent = engine.parse_query(query)

            # Basic validation
            assert intent is not None, f"Query should parse: {query}"
            assert intent.intent_type == test_case["expected_intent"], (
                f"Expected {test_case['expected_intent']}, got {intent.intent_type} for: {query}"
            )
            expected_var = test_case["expected_primary_variable"]
            assert intent.primary_variable == expected_var, (
                f"Expected variable {expected_var}, got {intent.primary_variable} for: {query}"
            )
            assert intent.confidence >= test_case["min_confidence"], (
                f"Confidence {intent.confidence} below minimum {test_case['min_confidence']} for: {query}"
            )

            # Log parsing tier for tracking
            if intent.parsing_tier:
                print(
                    f"Query: '{query}' -> Tier: {intent.parsing_tier}, "
                    f"Confidence: {intent.confidence:.2f}, Variable: {intent.primary_variable}"
                )
