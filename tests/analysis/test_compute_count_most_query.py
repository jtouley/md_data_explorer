"""
Tests for 'most' query detection in compute_count_analysis.

Tests verify:
- 'most' queries are detected correctly
- Only top result is returned for 'most' queries
- Generic detection works for any query text
"""

import polars as pl
from clinical_analytics.analysis.compute import compute_count_analysis
from clinical_analytics.ui.components.question_engine import AnalysisContext, AnalysisIntent


class TestMostQueryDetection:
    """Test that 'most' queries are detected and handled correctly."""

    def test_most_query_returns_only_top_result(self):
        """Test that 'most' queries return only the top result."""
        # Arrange: DataFrame with grouping variable
        df = pl.DataFrame(
            {
                "treatment": [1, 1, 1, 2, 2, 3],  # Treatment 1 has 3, Treatment 2 has 2, Treatment 3 has 1
                "patient_id": [1, 2, 3, 4, 5, 6],
            }
        )

        context = AnalysisContext()
        context.inferred_intent = AnalysisIntent.COUNT
        context.grouping_variable = "treatment"
        context.query_text = "which treatment was most prescribed?"  # Contains "most"

        # Act
        result = compute_count_analysis(df, context)

        # Assert: Should return only top result
        assert result["type"] == "count"
        assert result["is_most_query"] is True
        assert len(result["group_counts"]) == 1, "Should return only top result for 'most' query"
        assert result["group_counts"][0]["treatment"] == 1, "Top result should be treatment 1 (count=3)"
        assert result["group_counts"][0]["count"] == 3

    def test_non_most_query_returns_all_results(self):
        """Test that non-'most' queries return all results."""
        # Arrange: Same DataFrame
        df = pl.DataFrame(
            {
                "treatment": [1, 1, 1, 2, 2, 3],
                "patient_id": [1, 2, 3, 4, 5, 6],
            }
        )

        context = AnalysisContext()
        context.inferred_intent = AnalysisIntent.COUNT
        context.grouping_variable = "treatment"
        context.query_text = "what treatments were prescribed?"  # No "most"

        # Act
        result = compute_count_analysis(df, context)

        # Assert: Should return all results
        assert result["type"] == "count"
        assert result.get("is_most_query", False) is False
        assert len(result["group_counts"]) == 3, "Should return all groups for non-'most' query"

    def test_most_detection_is_generic_works_for_any_query(self):
        """Test that 'most' detection is generic and works for any query text."""
        # Arrange
        df = pl.DataFrame(
            {
                "category": [1, 1, 2, 2, 2, 3],
                "patient_id": [1, 2, 3, 4, 5, 6],
            }
        )

        # Test various "most" query patterns
        most_queries = [
            "which category was most common?",
            "what category was most frequent?",
            "which category was most used?",
            "what was the most prescribed category?",
        ]

        for query_text in most_queries:
            context = AnalysisContext()
            context.inferred_intent = AnalysisIntent.COUNT
            context.grouping_variable = "category"
            context.query_text = query_text

            # Act
            result = compute_count_analysis(df, context)

            # Assert: Should detect as "most" query
            assert result["is_most_query"] is True, f"Should detect 'most' in: {query_text}"
            assert len(result["group_counts"]) == 1, f"Should return only top result for: {query_text}"

    def test_most_query_headline_shows_only_top_result(self):
        """Test that 'most' query headline shows only top result."""
        # Arrange
        df = pl.DataFrame(
            {
                "medication": [1, 1, 1, 2, 2],
                "patient_id": [1, 2, 3, 4, 5],
            }
        )

        context = AnalysisContext()
        context.inferred_intent = AnalysisIntent.COUNT
        context.grouping_variable = "medication"
        context.query_text = "which medication was most prescribed?"

        # Act
        result = compute_count_analysis(df, context)

        # Assert: Headline should show only top result
        assert result["is_most_query"] is True
        assert "1" in result["headline"] or "medication" in result["headline"].lower()
        assert "3" in result["headline"]  # Count of top result
