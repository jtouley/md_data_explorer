"""
Tests for QueryPlan follow_ups schema extension (ADR009 Phase 1).

Tests cover:
- QueryPlan schema with follow_ups and follow_up_explanation fields
- from_dict() handling of new fields with defaults
- Field preservation through serialization/deserialization
- Empty and None value handling
"""

from clinical_analytics.core.query_plan import QueryPlan


class TestQueryPlanFollowUpsSchema:
    """Test QueryPlan schema extension with follow_ups fields."""

    def test_queryplan_with_follow_ups_creates_successfully(self):
        # Arrange & Act
        plan = QueryPlan(
            intent="DESCRIBE",
            metric="age",
            confidence=0.9,
            follow_ups=["What predicts mortality?", "Compare by treatment group"],
            follow_up_explanation="These questions explore the data further",
        )

        # Assert
        assert plan.follow_ups == ["What predicts mortality?", "Compare by treatment group"]
        assert plan.follow_up_explanation == "These questions explore the data further"

    def test_queryplan_without_follow_ups_defaults_to_empty(self):
        # Arrange & Act
        plan = QueryPlan(
            intent="COUNT",
            entity_key="patient_id",
            confidence=0.8,
        )

        # Assert
        assert plan.follow_ups == []
        assert plan.follow_up_explanation == ""

    def test_queryplan_from_dict_with_follow_ups_preserves_fields(self):
        # Arrange
        data = {
            "intent": "COMPARE_GROUPS",
            "metric": "LDL",
            "group_by": "treatment",
            "confidence": 0.85,
            "follow_ups": ["What is the effect size?", "Are there outliers?"],
            "follow_up_explanation": "Statistical and data quality questions",
        }

        # Act
        plan = QueryPlan.from_dict(data)

        # Assert
        assert plan.follow_ups == ["What is the effect size?", "Are there outliers?"]
        assert plan.follow_up_explanation == "Statistical and data quality questions"

    def test_queryplan_from_dict_without_follow_ups_uses_defaults(self):
        # Arrange
        data = {
            "intent": "DESCRIBE",
            "metric": "age",
            "confidence": 0.9,
            # No follow_ups or follow_up_explanation
        }

        # Act
        plan = QueryPlan.from_dict(data)

        # Assert
        assert plan.follow_ups == []
        assert plan.follow_up_explanation == ""

    def test_queryplan_from_dict_with_empty_follow_ups_preserves_empty_list(self):
        # Arrange
        data = {
            "intent": "DESCRIBE",
            "metric": "age",
            "confidence": 0.9,
            "follow_ups": [],  # Explicitly empty
            "follow_up_explanation": "",
        }

        # Act
        plan = QueryPlan.from_dict(data)

        # Assert
        assert plan.follow_ups == []
        assert plan.follow_up_explanation == ""

    def test_queryplan_follow_ups_accepts_single_question(self):
        # Arrange & Act
        plan = QueryPlan(
            intent="FIND_PREDICTORS",
            metric="mortality",
            confidence=0.75,
            follow_ups=["What is the most important predictor?"],
            follow_up_explanation="Asking about feature importance",
        )

        # Assert
        assert len(plan.follow_ups) == 1
        assert plan.follow_ups[0] == "What is the most important predictor?"

    def test_queryplan_follow_ups_accepts_multiple_questions(self):
        # Arrange & Act
        plan = QueryPlan(
            intent="DESCRIBE",
            metric="age",
            confidence=0.9,
            follow_ups=[
                "What is the age distribution by gender?",
                "Are there age outliers?",
                "What is the correlation with BMI?",
            ],
            follow_up_explanation="Exploring age relationships",
        )

        # Assert
        assert len(plan.follow_ups) == 3

    def test_queryplan_follow_up_explanation_accepts_long_text(self):
        # Arrange
        long_explanation = (
            "These follow-up questions help explore the data in multiple dimensions. "
            "The first question investigates the relationship between variables. "
            "The second question checks for data quality issues."
        )

        # Act
        plan = QueryPlan(
            intent="CORRELATIONS",
            confidence=0.8,
            follow_ups=["Question 1", "Question 2"],
            follow_up_explanation=long_explanation,
        )

        # Assert
        assert plan.follow_up_explanation == long_explanation
        assert len(plan.follow_up_explanation) > 100

    def test_queryplan_follow_ups_preserves_order(self):
        # Arrange
        questions_in_order = [
            "First question",
            "Second question",
            "Third question",
        ]

        # Act
        plan = QueryPlan(
            intent="DESCRIBE",
            metric="age",
            confidence=0.9,
            follow_ups=questions_in_order,
        )

        # Assert
        assert plan.follow_ups == questions_in_order
        assert plan.follow_ups[0] == "First question"
        assert plan.follow_ups[1] == "Second question"
        assert plan.follow_ups[2] == "Third question"

    def test_queryplan_with_follow_ups_serializes_to_dict(self):
        # Arrange
        plan = QueryPlan(
            intent="DESCRIBE",
            metric="age",
            confidence=0.9,
            follow_ups=["Question 1", "Question 2"],
            follow_up_explanation="Explanation text",
        )

        # Act
        data = {
            "intent": plan.intent,
            "metric": plan.metric,
            "confidence": plan.confidence,
            "follow_ups": plan.follow_ups,
            "follow_up_explanation": plan.follow_up_explanation,
        }

        # Assert
        assert data["follow_ups"] == ["Question 1", "Question 2"]
        assert data["follow_up_explanation"] == "Explanation text"

    def test_queryplan_from_dict_roundtrip_preserves_follow_ups(self):
        # Arrange
        original_data = {
            "intent": "COMPARE_GROUPS",
            "metric": "LDL",
            "group_by": "treatment",
            "confidence": 0.85,
            "follow_ups": ["Q1", "Q2", "Q3"],
            "follow_up_explanation": "Test explanation",
        }

        # Act
        plan = QueryPlan.from_dict(original_data)
        roundtrip_data = {
            "intent": plan.intent,
            "metric": plan.metric,
            "group_by": plan.group_by,
            "confidence": plan.confidence,
            "follow_ups": plan.follow_ups,
            "follow_up_explanation": plan.follow_up_explanation,
        }

        # Assert
        assert roundtrip_data["follow_ups"] == original_data["follow_ups"]
        assert roundtrip_data["follow_up_explanation"] == original_data["follow_up_explanation"]
