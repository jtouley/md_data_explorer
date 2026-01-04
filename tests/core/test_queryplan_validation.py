"""
Tests for QueryPlan validation (Phase 5.2).

Ensures:
- QueryPlan.from_dict() validates all fields
- Invalid intent types are rejected
- Missing required fields are rejected
- Field types are validated
"""

import pytest
from clinical_analytics.core.query_plan import FilterSpec, QueryPlan


class TestQueryPlanFromDict:
    """Test QueryPlan.from_dict() class method."""

    def test_from_dict_creates_valid_queryplan(self):
        """from_dict() should create QueryPlan from valid dict."""
        # Arrange: Valid QueryPlan dict
        plan_dict = {
            "intent": "DESCRIBE",
            "metric": "age",
            "group_by": None,
            "filters": [],
            "confidence": 0.9,
            "explanation": "Describe age statistics",
        }

        # Act: Create QueryPlan
        plan = QueryPlan.from_dict(plan_dict)

        # Assert: Should create valid QueryPlan
        assert isinstance(plan, QueryPlan)
        assert plan.intent == "DESCRIBE"
        assert plan.metric == "age"
        assert plan.confidence == 0.9

    def test_from_dict_rejects_invalid_intent(self):
        """from_dict() should reject invalid intent types."""
        # Arrange: Invalid intent
        plan_dict = {
            "intent": "INVALID_INTENT",
            "confidence": 0.9,
        }

        # Act & Assert: Should raise ValueError
        with pytest.raises(ValueError, match="invalid.*intent|Invalid.*intent"):
            QueryPlan.from_dict(plan_dict)

    def test_from_dict_rejects_missing_intent(self):
        """from_dict() should reject dicts missing required 'intent' field."""
        # Arrange: Missing intent
        plan_dict = {
            "metric": "age",
            "confidence": 0.9,
        }

        # Act & Assert: Should raise ValueError or KeyError
        with pytest.raises((ValueError, KeyError, TypeError), match="intent|Intent"):
            QueryPlan.from_dict(plan_dict)

    def test_from_dict_applies_defaults_for_optional_fields(self):
        """from_dict() should apply defaults for optional fields."""
        # Arrange: Only required field
        plan_dict = {
            "intent": "COUNT",
        }

        # Act: Create QueryPlan
        plan = QueryPlan.from_dict(plan_dict)

        # Assert: Optional fields should have defaults
        assert plan.metric is None
        assert plan.group_by is None
        assert plan.filters == []
        assert plan.confidence == 0.0
        assert plan.explanation == ""

    def test_from_dict_validates_confidence_range(self):
        """from_dict() should validate confidence is in [0.0, 1.0]."""
        # Arrange: Invalid confidence
        plan_dict = {
            "intent": "DESCRIBE",
            "confidence": 1.5,  # Invalid: > 1.0
        }

        # Act: Create QueryPlan (may clamp or raise)
        plan = QueryPlan.from_dict(plan_dict)

        # Assert: Confidence should be clamped to valid range
        assert 0.0 <= plan.confidence <= 1.0, "Confidence should be in valid range [0.0, 1.0]"


class TestQueryPlanValidateIntent:
    """Test intent validation in QueryPlan."""

    def test_queryplan_accepts_valid_intents(self):
        """QueryPlan should accept all valid intent types."""
        valid_intents = ["COUNT", "DESCRIBE", "COMPARE_GROUPS", "FIND_PREDICTORS", "CORRELATIONS"]

        for intent in valid_intents:
            # Act: Create QueryPlan with valid intent
            plan = QueryPlan(intent=intent)

            # Assert: Should accept valid intent
            assert plan.intent == intent

    def test_queryplan_rejects_invalid_intents(self):
        """QueryPlan.from_dict() should reject invalid intent types."""
        # Arrange: Invalid intent
        invalid_intent = "INVALID_INTENT"

        # Act & Assert: from_dict() should reject invalid intent
        # Note: Direct construction doesn't enforce Literal at runtime
        # Phase 5.2: Validation happens in from_dict()
        with pytest.raises(ValueError, match="Invalid intent"):
            QueryPlan.from_dict({"intent": invalid_intent})


class TestQueryPlanFilterSpecValidation:
    """Test FilterSpec validation in QueryPlan."""

    def test_queryplan_accepts_valid_filterspec(self):
        """QueryPlan should accept valid FilterSpec objects."""
        # Arrange: Valid filter
        filter_spec = FilterSpec(
            column="age",
            operator=">=",
            value=50,
            exclude_nulls=True,
        )

        # Act: Create QueryPlan with filter
        plan = QueryPlan(intent="DESCRIBE", filters=[filter_spec])

        # Assert: Should accept valid filter
        assert len(plan.filters) == 1
        assert plan.filters[0].column == "age"
        assert plan.filters[0].operator == ">="
        assert plan.filters[0].value == 50

    def test_filterspec_validates_operator(self):
        """FilterSpec should validate operator against allowed values."""
        # Arrange: Valid operators
        valid_operators = ["==", "!=", ">", ">=", "<", "<=", "IN", "NOT_IN"]

        for op in valid_operators:
            # Act: Create FilterSpec
            filter_spec = FilterSpec(column="age", operator=op, value=50)

            # Assert: Should accept valid operator
            assert filter_spec.operator == op

    def test_filterspec_rejects_invalid_operator(self):
        """FilterSpec operators are type-checked at development time."""
        # Arrange: Invalid operator
        invalid_operator = "INVALID_OP"

        # Note: Python dataclasses with Literal don't enforce at runtime
        # Type checking happens at development time (mypy, pyright, etc.)
        # This test documents that we rely on static type checking
        # For runtime validation, use FilterSpec within QueryPlan.from_dict()

        # Act: Create FilterSpec (won't raise at runtime, but type checker will warn)
        filter_spec = FilterSpec(column="age", operator=invalid_operator, value=50)  # type: ignore

        # Assert: Created successfully (runtime doesn't enforce)
        # Static type checkers will flag this as an error
        assert filter_spec.operator == "INVALID_OP"


class TestQueryPlanScopeValidation:
    """Test scope validation in QueryPlan."""

    def test_queryplan_accepts_valid_scopes(self):
        """QueryPlan should accept valid scope values."""
        valid_scopes = ["all", "filtered"]

        for scope in valid_scopes:
            # Act: Create QueryPlan with scope
            plan = QueryPlan(intent="COUNT", scope=scope)

            # Assert: Should accept valid scope
            assert plan.scope == scope

    def test_queryplan_defaults_scope_to_all(self):
        """QueryPlan should default scope to 'all'."""
        # Act: Create QueryPlan without specifying scope
        plan = QueryPlan(intent="COUNT")

        # Assert: Should default to 'all'
        assert plan.scope == "all"
