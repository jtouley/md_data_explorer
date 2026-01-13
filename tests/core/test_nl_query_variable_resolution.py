"""Tests for variable name resolution in query plan creation.

This module tests that short variable aliases (e.g., "statin") are resolved
to their canonical column names before query execution.

Fixes: Column 'statin' not found error on semantic match tier.
"""

from unittest.mock import MagicMock

import pytest
from clinical_analytics.core.nl_query_engine import NLQueryEngine, QueryIntent
from clinical_analytics.core.query_plan import FilterSpec

# Actual column names from statin dataset
STATIN_COLUMN = (
    "Statin Used:    0: n/a                       1: Atorvastatin  "
    "2: Rosuvastatin 3: Pravastatin   4: Pitavastatin  5: Simvastatin"
)
LDL_COLUMN = "LDL mg/dL"


@pytest.fixture
def nl_engine_with_statin_column():
    """NLQueryEngine with semantic layer containing statin column aliases."""
    mock_semantic = MagicMock()
    mock_semantic.get_column_alias_index.return_value = {
        "statin": STATIN_COLUMN,
        "statin used": STATIN_COLUMN,
        "ldl": LDL_COLUMN,
    }
    mock_semantic.get_collision_suggestions.return_value = None
    mock_semantic._normalize_alias.side_effect = lambda x: x.lower().strip()

    return NLQueryEngine(mock_semantic)


class TestVariableResolution:
    """Test variable name resolution in query plan creation."""

    def test_intent_to_plan_resolves_short_alias_to_column_name(self, nl_engine_with_statin_column):
        """Short alias 'statin' should resolve to full column name."""
        # Arrange
        intent = QueryIntent(
            intent_type="COUNT",
            grouping_variable="statin",
            confidence=0.9,
        )

        # Act
        plan = nl_engine_with_statin_column._intent_to_plan(intent, dataset_version="test_v1")

        # Assert - should be resolved to canonical name
        assert plan.group_by == STATIN_COLUMN
        assert plan.group_by != "statin"

    def test_intent_to_plan_preserves_exact_column_name(self, nl_engine_with_statin_column):
        """Exact column name should pass through unchanged."""
        # Arrange
        intent = QueryIntent(
            intent_type="COUNT",
            grouping_variable=STATIN_COLUMN,
            confidence=0.9,
        )

        # Act
        plan = nl_engine_with_statin_column._intent_to_plan(intent, dataset_version="test_v1")

        # Assert
        assert plan.group_by == STATIN_COLUMN

    def test_intent_to_plan_resolves_metric_alias(self, nl_engine_with_statin_column):
        """Metric variable should also be resolved."""
        # Arrange
        intent = QueryIntent(
            intent_type="DESCRIBE",
            primary_variable="ldl",
            confidence=0.9,
        )

        # Act
        plan = nl_engine_with_statin_column._intent_to_plan(intent, dataset_version="test_v1")

        # Assert
        assert plan.metric == LDL_COLUMN

    def test_intent_to_plan_resolves_filter_column_names(self, nl_engine_with_statin_column):
        """Filter column names should be resolved."""
        # Arrange
        intent = QueryIntent(
            intent_type="COUNT",
            grouping_variable=STATIN_COLUMN,
            confidence=0.9,
            filters=[FilterSpec(column="statin", operator="!=", value=0)],
        )

        # Act
        plan = nl_engine_with_statin_column._intent_to_plan(intent, dataset_version="test_v1")

        # Assert
        assert plan.filters[0].column == STATIN_COLUMN

    def test_intent_to_plan_handles_none_variables(self, nl_engine_with_statin_column):
        """None variables should pass through as None."""
        # Arrange
        intent = QueryIntent(
            intent_type="COUNT",
            grouping_variable=None,
            primary_variable=None,
            confidence=0.9,
        )

        # Act
        plan = nl_engine_with_statin_column._intent_to_plan(intent, dataset_version="test_v1")

        # Assert
        assert plan.group_by is None
        assert plan.metric is None
