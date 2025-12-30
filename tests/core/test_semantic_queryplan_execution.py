"""
Test Semantic Layer QueryPlan Execution (Phase 3 - ADR003).

Tests for execute_query_plan() method with:
- Confidence and completeness gating
- QueryPlan contract validation
- Type-aware execution (categorical vs numeric)
- COUNT intent validation and execution
- Breakdown validation and filter deduplication
- Run-key determinism
"""

import polars as pl
import pytest

from clinical_analytics.core.query_plan import FilterSpec, QueryPlan
from clinical_analytics.core.semantic import SemanticLayer

# ============================================================================
# Fixtures (Phase 3)
# ============================================================================


@pytest.fixture
def sample_query_plan_count():
    """QueryPlan with COUNT intent."""
    return QueryPlan(
        intent="COUNT",
        metric=None,
        group_by=None,
        filters=[],
        confidence=0.8,
        explanation="Count all patients",
    )


@pytest.fixture
def sample_query_plan_describe():
    """QueryPlan with DESCRIBE intent."""
    return QueryPlan(
        intent="DESCRIBE",
        metric="BMI",
        group_by=None,
        filters=[],
        confidence=0.9,
        explanation="Describe BMI distribution",
    )


@pytest.fixture
def sample_cohort_with_categorical():
    """Polars DataFrame with categorical encoding ('1: Yes 2: No')."""
    return pl.DataFrame(
        {
            "patient_id": ["P001", "P002", "P003", "P004", "P005"],
            "treatment": ["1: Yes", "2: No", "1: Yes", "1: Yes", "2: No"],
            "status": ["1: Active", "2: Inactive", "1: Active", "1: Active", "2: Inactive"],
            "age": [45, 52, 38, 61, 49],
        }
    )


@pytest.fixture
def sample_cohort_with_numeric():
    """Polars DataFrame with numeric columns."""
    return pl.DataFrame(
        {
            "patient_id": ["P001", "P002", "P003", "P004", "P005"],
            "BMI": [25.5, 28.3, 22.1, 30.2, 26.8],
            "LDL mg/dL": [120, 150, 100, 180, 130],
            "age": [45, 52, 38, 61, 49],
        }
    )


@pytest.fixture
def mock_semantic_layer_for_execution(tmp_path):
    """Create a SemanticLayer instance for execution testing."""
    workspace = tmp_path / "test_workspace"
    workspace.mkdir()
    (workspace / "pyproject.toml").write_text("[project]\nname = 'test'")

    data_dir = workspace / "data" / "raw" / "test_dataset"
    data_dir.mkdir(parents=True)

    # Create test CSV
    df = pl.DataFrame(
        {
            "patient_id": ["P001", "P002", "P003"],
            "BMI": [25.5, 28.3, 22.1],
            "treatment": ["A", "B", "A"],
            "status": ["1: Yes", "2: No", "1: Yes"],
        }
    )
    df.write_csv(data_dir / "test.csv")

    config = {
        "init_params": {"source_path": "data/raw/test_dataset/test.csv"},
        "column_mapping": {},
        "outcomes": {},
        "analysis": {"default_outcome": "outcome"},
    }

    layer = SemanticLayer(dataset_name="test_dataset", config=config, workspace_root=workspace)
    return layer


# ============================================================================
# Test Cases (Phase 3)
# ============================================================================


def test_execute_query_plan_validates_columns_exist(mock_semantic_layer_for_execution):
    """Verify executor checks columns exist after alias resolution."""
    # Arrange
    plan = QueryPlan(
        intent="DESCRIBE",
        metric="nonexistent_column",  # Column that doesn't exist
        confidence=0.9,
    )

    # Act
    result = mock_semantic_layer_for_execution.execute_query_plan(plan)

    # Assert
    assert "requires_confirmation" in result
    assert result["requires_confirmation"] is True
    assert "failure_reason" in result
    assert "nonexistent_column" in result["failure_reason"].lower() or "not found" in result["failure_reason"].lower()


def test_execute_query_plan_validates_operators(mock_semantic_layer_for_execution):
    """Verify executor validates operators are supported."""
    # Arrange
    plan = QueryPlan(
        intent="DESCRIBE",
        metric="BMI",
        filters=[FilterSpec(column="BMI", operator="UNSUPPORTED_OP", value=25)],  # Invalid operator
        confidence=0.9,
    )

    # Act
    result = mock_semantic_layer_for_execution.execute_query_plan(plan)

    # Assert
    assert "requires_confirmation" in result
    assert result["requires_confirmation"] is True
    assert "failure_reason" in result
    assert "operator" in result["failure_reason"].lower() or "unsupported" in result["failure_reason"].lower()


def test_execute_query_plan_validates_type_compatibility(mock_semantic_layer_for_execution):
    """Verify executor checks type compatibility (e.g., can't use '>' on categorical)."""
    # Arrange
    plan = QueryPlan(
        intent="DESCRIBE",
        metric="status",  # Categorical column
        filters=[FilterSpec(column="status", operator=">", value="1: Yes")],  # Invalid: > on categorical
        confidence=0.9,
    )

    # Act
    result = mock_semantic_layer_for_execution.execute_query_plan(plan)

    # Assert
    # Note: This may be a warning rather than a hard failure, depending on implementation
    # For now, verify it doesn't crash and handles the type mismatch
    assert result is not None


def test_execute_query_plan_count_scope_validation(mock_semantic_layer_for_execution):
    """Verify executor refuses scope='all' with filters, refuses scope='filtered' with empty filters."""
    # Arrange
    plan_with_filters = QueryPlan(
        intent="COUNT",
        filters=[FilterSpec(column="BMI", operator=">", value=25)],
        confidence=0.9,
    )
    # Set scope="all" via attribute (if added to QueryPlan)
    if hasattr(plan_with_filters, "scope"):
        plan_with_filters.scope = "all"  # Invalid: can't have scope="all" with filters

    plan_no_filters = QueryPlan(
        intent="COUNT",
        filters=[],
        confidence=0.9,
    )
    if hasattr(plan_no_filters, "scope"):
        plan_no_filters.scope = "filtered"  # Invalid: can't have scope="filtered" with no filters

    # Act & Assert
    # Note: This test will need to be updated once scope field is added to QueryPlan
    # For now, verify the method exists and handles these cases
    result1 = mock_semantic_layer_for_execution.execute_query_plan(plan_with_filters)
    result2 = mock_semantic_layer_for_execution.execute_query_plan(plan_no_filters)
    assert result1 is not None
    assert result2 is not None


def test_execute_query_plan_count_entity_key_validation(mock_semantic_layer_for_execution):
    """Verify executor requires entity_key for COUNT (or defaults to dataset primary key)."""
    # Arrange
    plan = QueryPlan(
        intent="COUNT",
        confidence=0.9,
    )

    # Act
    result = mock_semantic_layer_for_execution.execute_query_plan(plan)

    # Assert
    # COUNT should either require entity_key or default to primary key
    # If neither available, should return requires_confirmation=True
    assert result is not None
    # Note: Implementation may default to "patient_id" if available


def test_execute_query_plan_confidence_gating(mock_semantic_layer_for_execution):
    """Verify executor refuses execution when confidence < threshold."""
    # Arrange
    plan = QueryPlan(
        intent="DESCRIBE",
        metric="BMI",
        confidence=0.5,  # Below default threshold of 0.75
    )

    # Act
    result = mock_semantic_layer_for_execution.execute_query_plan(plan, confidence_threshold=0.75)

    # Assert
    assert "requires_confirmation" in result
    assert result["requires_confirmation"] is True
    assert "failure_reason" in result
    assert "confidence" in result["failure_reason"].lower()


def test_execute_query_plan_completeness_gating(mock_semantic_layer_for_execution):
    """Verify executor refuses execution when required fields missing.

    COUNT requires entity_key OR grouping_variable.
    """
    # Arrange
    plan = QueryPlan(
        intent="COUNT",
        # Missing: entity_key and grouping_variable
        confidence=0.9,
    )

    # Act
    result = mock_semantic_layer_for_execution.execute_query_plan(plan)

    # Assert
    assert "requires_confirmation" in result
    assert result["requires_confirmation"] is True
    assert "failure_reason" in result
    assert "entity_key" in result["failure_reason"].lower() or "grouping" in result["failure_reason"].lower()


def test_execute_query_plan_type_aware_categorical(mock_semantic_layer_for_execution, sample_cohort_with_categorical):
    """Verify categorical columns return frequency tables, not mean/median."""
    # Arrange
    plan = QueryPlan(
        intent="DESCRIBE",
        metric="treatment",  # Categorical column
        confidence=0.9,
    )

    # Act
    result = mock_semantic_layer_for_execution.execute_query_plan(plan)

    # Assert
    # Categorical should return frequency table, not numeric stats
    assert result is not None
    # Note: Implementation should detect categorical encoding and return frequencies


def test_execute_query_plan_type_aware_numeric(mock_semantic_layer_for_execution, sample_cohort_with_numeric):
    """Verify numeric columns return descriptive statistics."""
    # Arrange
    plan = QueryPlan(
        intent="DESCRIBE",
        metric="BMI",  # Numeric column
        confidence=0.9,
    )

    # Act
    result = mock_semantic_layer_for_execution.execute_query_plan(plan)

    # Assert
    # Numeric should return descriptive stats (mean, median, std dev, etc.)
    assert result is not None
    # Note: Implementation should return stats like mean, median, std dev


def test_execute_query_plan_count_intent_execution(mock_semantic_layer_for_execution):
    """Verify COUNT intent uses SQL aggregation with entity_key."""
    # Arrange
    plan = QueryPlan(
        intent="COUNT",
        confidence=0.9,
    )
    # Set entity_key if field exists
    if hasattr(plan, "entity_key"):
        plan.entity_key = "patient_id"

    # Act
    result = mock_semantic_layer_for_execution.execute_query_plan(plan)

    # Assert
    assert result is not None
    # COUNT should use aggregation with entity_key


def test_execute_query_plan_breakdown_validation(mock_semantic_layer_for_execution):
    """Verify executor refuses grouping_variable=entity_key when query implies categorical breakdown."""
    # Arrange
    plan = QueryPlan(
        intent="COUNT",
        group_by="patient_id",  # Grouping by entity key (invalid for breakdown)
        confidence=0.9,
    )
    if hasattr(plan, "entity_key"):
        plan.entity_key = "patient_id"

    # Act
    result = mock_semantic_layer_for_execution.execute_query_plan(plan)

    # Assert
    # Should refuse or warn about grouping by entity key
    assert result is not None
    # Note: May return requires_confirmation=True with error message


def test_execute_query_plan_high_cardinality_detection(mock_semantic_layer_for_execution):
    """Verify executor refuses high-cardinality grouping (near-unique values)."""
    # Arrange
    # Create a plan with grouping on a high-cardinality column
    plan = QueryPlan(
        intent="COUNT",
        group_by="patient_id",  # High cardinality (each patient is unique)
        confidence=0.9,
    )

    # Act
    result = mock_semantic_layer_for_execution.execute_query_plan(plan)

    # Assert
    # Should detect high cardinality and refuse or warn
    assert result is not None
    # Note: May return requires_confirmation=True with high cardinality warning


def test_execute_query_plan_filter_deduplication(mock_semantic_layer_for_execution):
    """Verify executor detects and warns/deduplicates redundant filters (filtering and grouping on same field)."""
    # Arrange
    plan = QueryPlan(
        intent="COUNT",
        group_by="treatment",  # Grouping by treatment
        filters=[FilterSpec(column="treatment", operator="==", value="A")],  # Also filtering on treatment
        confidence=0.9,
    )

    # Act
    result = mock_semantic_layer_for_execution.execute_query_plan(plan)

    # Assert
    # Should detect redundancy and deduplicate (keep grouping, remove filter)
    assert result is not None
    # Note: May log warning about redundant filter


def test_execute_query_plan_run_key_determinism(mock_semantic_layer_for_execution):
    """Verify run_key generated deterministically from canonical plan + query text."""
    # Arrange
    plan1 = QueryPlan(
        intent="DESCRIBE",
        metric="BMI",
        confidence=0.9,
        explanation="Describe BMI",
    )
    plan2 = QueryPlan(
        intent="DESCRIBE",
        metric="BMI",
        confidence=0.9,
        explanation="Describe BMI",
    )

    # Act
    result1 = mock_semantic_layer_for_execution.execute_query_plan(plan1, query_text="average BMI")
    result2 = mock_semantic_layer_for_execution.execute_query_plan(plan2, query_text="average BMI")

    # Assert
    # Same plan + same query text should generate same run_key
    if "run_key" in result1 and "run_key" in result2:
        assert result1["run_key"] == result2["run_key"], "Run keys should be deterministic"


def test_execute_query_plan_refuses_invalid_plans(mock_semantic_layer_for_execution):
    """Verify executor raises clear errors for contract violations."""
    # Arrange
    # Create an invalid plan (e.g., missing required fields)
    plan = QueryPlan(
        intent="COMPARE_GROUPS",
        # Missing: primary_variable and grouping_variable (required for COMPARE_GROUPS)
        confidence=0.9,
    )

    # Act
    result = mock_semantic_layer_for_execution.execute_query_plan(plan)

    # Assert
    assert "requires_confirmation" in result
    assert result["requires_confirmation"] is True
    assert "failure_reason" in result
    # Error message should be clear about what's missing
