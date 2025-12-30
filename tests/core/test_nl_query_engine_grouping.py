"""
Tests for grouping variable extraction from compound queries.

Tests verify:
- "broken down by" patterns extract grouping variables correctly
- "per X" patterns extract grouping variables correctly
- "by X" patterns extract grouping variables correctly
- "which X" patterns still work (backward compatibility)
- Partial matching works when exact match fails
"""

from clinical_analytics.core.nl_query_engine import NLQueryEngine


class TestGroupingExtractionFromCompoundQueries:
    """Test grouping variable extraction from compound queries."""

    def test_extract_grouping_broken_down_by_per_pattern(self, mock_semantic_layer):
        """Test extraction of grouping from 'broken down by count of patients per X' pattern."""
        # Arrange: Create semantic layer with statin column
        mock = mock_semantic_layer(
            columns={
                "statin": "Statin Used:    0: n/a                       1: Atorvastatin  2: Rosuvastatin 3: Pravastatin   4: Pitavastatin  5: Simvastatin",
                "statin_used": "Statin Used:    0: n/a                       1: Atorvastatin  2: Rosuvastatin 3: Pravastatin   4: Pitavastatin  5: Simvastatin",
            }
        )
        engine = NLQueryEngine(mock)

        # Act: Extract grouping from query
        query = "what statins were those patients on, broken down by count of patients per statin?"
        grouping_var = engine._extract_grouping_from_compound_query(query)

        # Assert: Should extract "statin" or "statin_used"
        assert grouping_var is not None
        assert "statin" in grouping_var.lower()

    def test_extract_grouping_broken_down_by_simple_pattern(self, mock_semantic_layer):
        """Test extraction of grouping from 'broken down by X' pattern."""
        # Arrange
        mock = mock_semantic_layer(
            columns={
                "treatment": "Treatment Group",
                "treatment_group": "Treatment Group",
            }
        )
        engine = NLQueryEngine(mock)

        # Act
        query = "show results broken down by treatment"
        grouping_var = engine._extract_grouping_from_compound_query(query)

        # Assert
        assert grouping_var is not None
        assert "treatment" in grouping_var.lower()

    def test_extract_grouping_per_pattern(self, mock_semantic_layer):
        """Test extraction of grouping from 'per X' pattern at end of query."""
        # Arrange
        mock = mock_semantic_layer(
            columns={
                "statin": "Statin Used",
                "statin_used": "Statin Used",
            }
        )
        engine = NLQueryEngine(mock)

        # Act
        query = "count patients per statin"
        grouping_var = engine._extract_grouping_from_compound_query(query)

        # Assert
        assert grouping_var is not None
        assert "statin" in grouping_var.lower()

    def test_extract_grouping_by_pattern(self, mock_semantic_layer):
        """Test extraction of grouping from 'by X' pattern at end of query."""
        # Arrange
        mock = mock_semantic_layer(
            columns={
                "category": "Category",
                "category_type": "Category",
            }
        )
        engine = NLQueryEngine(mock)

        # Act
        query = "show distribution by category"
        grouping_var = engine._extract_grouping_from_compound_query(query)

        # Assert
        assert grouping_var is not None
        assert "category" in grouping_var.lower()

    def test_extract_grouping_which_pattern_still_works(self, mock_semantic_layer):
        """Test that 'which X' patterns still work (backward compatibility)."""
        # Arrange
        mock = mock_semantic_layer(
            columns={
                "statin": "Statin Used",
                "statin_used": "Statin Used",
            }
        )
        engine = NLQueryEngine(mock)

        # Act: Test with domain-specific term (not hardcoded)
        query = "which statin was most prescribed"
        grouping_var = engine._extract_grouping_from_compound_query(query)

        # Assert
        assert grouping_var is not None
        assert "statin" in grouping_var.lower()

    def test_extract_grouping_which_pattern_with_domain_specific_terms(self, mock_semantic_layer):
        """Test that 'which X was most Y' works with any domain-specific term (not hardcoded)."""
        # Arrange
        mock = mock_semantic_layer(
            columns={
                "treatment": "Treatment Type",
                "treatment_type": "Treatment Type",
            }
        )
        engine = NLQueryEngine(mock)

        # Act: Use domain-specific term "effective" (not in hardcoded list)
        query = "which treatment was most effective"
        grouping_var = engine._extract_grouping_from_compound_query(query)

        # Assert: Should still extract grouping variable (pattern is semantic, not keyword-based)
        assert grouping_var is not None
        assert "treatment" in grouping_var.lower()

    def test_extract_grouping_which_pattern_with_and(self, mock_semantic_layer):
        """Test 'and which X' pattern."""
        # Arrange
        mock = mock_semantic_layer(
            columns={
                "treatment": "Treatment Group",
                "treatment_group": "Treatment Group",
            }
        )
        engine = NLQueryEngine(mock)

        # Act
        query = "how many patients and which treatment was most common"
        grouping_var = engine._extract_grouping_from_compound_query(query)

        # Assert
        assert grouping_var is not None
        assert "treatment" in grouping_var.lower()

    def test_extract_grouping_returns_none_when_no_match(self, mock_semantic_layer):
        """Test that extraction returns None when no grouping pattern matches."""
        # Arrange
        mock = mock_semantic_layer(columns={})
        engine = NLQueryEngine(mock)

        # Act
        query = "what is the average age"
        grouping_var = engine._extract_grouping_from_compound_query(query)

        # Assert
        assert grouping_var is None

    def test_extract_grouping_prioritizes_broken_down_over_which(self, mock_semantic_layer):
        """Test that 'broken down by' patterns are checked before 'which' patterns."""
        # Arrange
        mock = mock_semantic_layer(
            columns={
                "statin": "Statin Used",
                "treatment": "Treatment Group",
            }
        )
        engine = NLQueryEngine(mock)

        # Act: Query has both "broken down by statin" and "which treatment"
        query = "show results broken down by statin and which treatment was used"
        grouping_var = engine._extract_grouping_from_compound_query(query)

        # Assert: Should extract "statin" (from "broken down by"), not "treatment" (from "which")
        assert grouping_var is not None
        assert "statin" in grouping_var.lower()
        assert "treatment" not in grouping_var.lower()

    def test_extract_grouping_uses_partial_match_when_fuzzy_fails(self, mock_semantic_layer):
        """Test that partial matching is used when fuzzy match confidence is too low."""
        # Arrange: Create semantic layer with column that has long name
        mock = mock_semantic_layer(
            columns={
                "statin_used_long_name": "Statin Used:    0: n/a                       1: Atorvastatin  2: Rosuvastatin 3: Pravastatin   4: Pitavastatin  5: Simvastatin",
            }
        )
        # Mock _fuzzy_match_variable to return low confidence
        engine = NLQueryEngine(mock)

        # Act: Use shortened term "statin" that should match via partial match
        query = "broken down by statin"
        grouping_var = engine._extract_grouping_from_compound_query(query)

        # Assert: Should still find it via partial matching
        assert grouping_var is not None
        assert "statin" in grouping_var.lower()
