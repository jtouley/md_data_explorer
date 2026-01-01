"""
Tests for QueryPlan interpretation schema extension (ADR009 Phase 2).

Tests cover:
- QueryPlan schema with interpretation and confidence_explanation fields
- from_dict() handling of new fields with defaults
- Field preservation through serialization/deserialization
- Empty and None value handling
"""


from clinical_analytics.core.query_plan import QueryPlan


class TestQueryPlanInterpretationSchema:
    """Test QueryPlan schema extension with interpretation fields."""

    def test_queryplan_with_interpretation_creates_successfully(self):
        # Arrange & Act
        plan = QueryPlan(
            intent="DESCRIBE",
            metric="age",
            confidence=0.9,
            interpretation="This query asks for the average age of patients",
            confidence_explanation="High confidence because the query clearly mentions 'average age'",
        )

        # Assert
        assert plan.interpretation == "This query asks for the average age of patients"
        assert plan.confidence_explanation == "High confidence because the query clearly mentions 'average age'"

    def test_queryplan_without_interpretation_defaults_to_empty(self):
        # Arrange & Act
        plan = QueryPlan(
            intent="COUNT",
            entity_key="patient_id",
            confidence=0.8,
        )

        # Assert
        assert plan.interpretation == ""
        assert plan.confidence_explanation == ""

    def test_queryplan_from_dict_with_interpretation_preserves_fields(self):
        # Arrange
        data = {
            "intent": "COMPARE_GROUPS",
            "metric": "LDL",
            "group_by": "treatment",
            "confidence": 0.85,
            "interpretation": "Comparing LDL levels across treatment groups",
            "confidence_explanation": "Moderate confidence due to ambiguous treatment reference",
        }

        # Act
        plan = QueryPlan.from_dict(data)

        # Assert
        assert plan.interpretation == "Comparing LDL levels across treatment groups"
        assert plan.confidence_explanation == "Moderate confidence due to ambiguous treatment reference"

    def test_queryplan_from_dict_without_interpretation_uses_defaults(self):
        # Arrange
        data = {
            "intent": "DESCRIBE",
            "metric": "age",
            "confidence": 0.9,
            # No interpretation or confidence_explanation
        }

        # Act
        plan = QueryPlan.from_dict(data)

        # Assert
        assert plan.interpretation == ""
        assert plan.confidence_explanation == ""

    def test_queryplan_from_dict_with_empty_interpretation_preserves_empty(self):
        # Arrange
        data = {
            "intent": "DESCRIBE",
            "metric": "age",
            "confidence": 0.9,
            "interpretation": "",  # Explicitly empty
            "confidence_explanation": "",
        }

        # Act
        plan = QueryPlan.from_dict(data)

        # Assert
        assert plan.interpretation == ""
        assert plan.confidence_explanation == ""

    def test_queryplan_interpretation_accepts_long_text(self):
        # Arrange
        long_interpretation = (
            "This query is asking for a detailed analysis of patient demographics "
            "stratified by treatment group. The analysis will include statistical "
            "comparison of age, BMI, and comorbidity profiles between groups."
        )
        long_confidence = (
            "Confidence is moderate because the query mentions 'analysis' which could "
            "mean multiple different statistical approaches. The grouping variable is "
            "clear but the specific metrics to analyze are inferred."
        )

        # Act
        plan = QueryPlan(
            intent="COMPARE_GROUPS",
            confidence=0.7,
            interpretation=long_interpretation,
            confidence_explanation=long_confidence,
        )

        # Assert
        assert plan.interpretation == long_interpretation
        assert plan.confidence_explanation == long_confidence
        assert len(plan.interpretation) > 100
        assert len(plan.confidence_explanation) > 100

    def test_queryplan_with_interpretation_serializes_to_dict(self):
        # Arrange
        plan = QueryPlan(
            intent="DESCRIBE",
            metric="age",
            confidence=0.9,
            interpretation="Query explanation",
            confidence_explanation="Confidence reasoning",
        )

        # Act
        data = {
            "intent": plan.intent,
            "metric": plan.metric,
            "confidence": plan.confidence,
            "interpretation": plan.interpretation,
            "confidence_explanation": plan.confidence_explanation,
        }

        # Assert
        assert data["interpretation"] == "Query explanation"
        assert data["confidence_explanation"] == "Confidence reasoning"

    def test_queryplan_from_dict_roundtrip_preserves_interpretation(self):
        # Arrange
        original_data = {
            "intent": "FIND_PREDICTORS",
            "metric": "mortality",
            "confidence": 0.75,
            "interpretation": "Finding predictors of mortality",
            "confidence_explanation": "Good confidence in intent, moderate in metric",
        }

        # Act
        plan = QueryPlan.from_dict(original_data)
        roundtrip_data = {
            "intent": plan.intent,
            "metric": plan.metric,
            "confidence": plan.confidence,
            "interpretation": plan.interpretation,
            "confidence_explanation": plan.confidence_explanation,
        }

        # Assert
        assert roundtrip_data["interpretation"] == original_data["interpretation"]
        assert roundtrip_data["confidence_explanation"] == original_data["confidence_explanation"]

    def test_queryplan_with_both_follow_ups_and_interpretation(self):
        # Arrange & Act
        plan = QueryPlan(
            intent="DESCRIBE",
            metric="age",
            confidence=0.9,
            # Phase 1 fields
            follow_ups=["What predicts age?", "Compare by gender"],
            follow_up_explanation="Exploring age relationships",
            # Phase 2 fields
            interpretation="Analyzing patient age distribution",
            confidence_explanation="Clear query intent",
        )

        # Assert - Phase 1 fields preserved
        assert plan.follow_ups == ["What predicts age?", "Compare by gender"]
        assert plan.follow_up_explanation == "Exploring age relationships"
        # Assert - Phase 2 fields preserved
        assert plan.interpretation == "Analyzing patient age distribution"
        assert plan.confidence_explanation == "Clear query intent"
