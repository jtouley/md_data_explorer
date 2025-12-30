"""
Test Trust UI (Phase 1 - ADR003).

Tests for trust verification expander, patient-level export, and cohort size calculation.
"""

import polars as pl
import pytest

from clinical_analytics.core.query_plan import FilterSpec, QueryPlan

# ============================================================================
# Fixtures (Phase 1)
# ============================================================================


@pytest.fixture
def sample_query_plan():
    """Factory for QueryPlan with various configurations."""

    def _make(
        intent: str = "DESCRIBE",
        metric: str | None = "age",
        group_by: str | None = None,
        filters: list[FilterSpec] | None = None,
        confidence: float = 0.9,
    ) -> QueryPlan:
        return QueryPlan(
            intent=intent,
            metric=metric,
            group_by=group_by,
            filters=filters or [],
            confidence=confidence,
            explanation="Test query plan",
        )

    return _make


@pytest.fixture
def sample_analysis_result():
    """Factory for analysis result dicts."""

    def _make(
        result_type: str = "descriptive",
        row_count: int = 100,
        non_null_count: int = 95,
        **kwargs,
    ) -> dict:
        base_result = {
            "type": result_type,
            "row_count": row_count,
            "non_null_count": non_null_count,
        }
        base_result.update(kwargs)
        return base_result

    return _make


@pytest.fixture
def mock_semantic_layer_with_aliases():
    """SemanticLayer with alias index populated."""
    import re
    from unittest.mock import MagicMock

    mock = MagicMock()

    # Match actual SemanticLayer normalization: lowercase, collapse whitespace, keep spaces
    def normalize(text: str) -> str:
        normalized = text.lower()
        normalized = re.sub(r"[^\w\s]", "", normalized)  # Remove punctuation
        normalized = re.sub(r"\s+", " ", normalized).strip()  # Collapse whitespace
        return normalized

    # Alias index maps normalized aliases to canonical names
    mock.get_column_alias_index.return_value = {
        "age": "age",
        "viral load": "viral_load",  # Normalized: "viral load" keeps space
        "cd4": "cd4_count",
    }
    mock._normalize_alias = normalize
    return mock


# ============================================================================
# Phase 1 Tests - Trust UI
# ============================================================================


def test_trust_ui_shows_query_plan_raw_fields(sample_query_plan):
    """Verify raw QueryPlan (intent, metric, filters) displayed."""
    from clinical_analytics.ui.components.trust_ui import TrustUI

    query_plan = sample_query_plan(
        intent="DESCRIBE",
        metric="age",
        filters=[FilterSpec(column="treatment", operator="==", value="A")],
    )

    # Trust UI should extract and display raw fields
    raw_fields = TrustUI._extract_raw_fields(query_plan)

    assert raw_fields["intent"] == "DESCRIBE"
    assert raw_fields["metric"] == "age"
    assert len(raw_fields["filters"]) == 1
    assert raw_fields["filters"][0]["column"] == "treatment"


def test_trust_ui_shows_alias_resolved_plan(sample_query_plan, mock_semantic_layer_with_aliases):
    """Verify canonical column names shown after alias resolution."""
    from clinical_analytics.ui.components.trust_ui import TrustUI

    query_plan = sample_query_plan(
        metric="viral load",  # This is an alias
        filters=[FilterSpec(column="cd4", operator=">", value=200)],
    )

    # Trust UI should resolve aliases to canonical names
    resolved_plan = TrustUI._resolve_aliases(query_plan, mock_semantic_layer_with_aliases)

    assert resolved_plan["metric"] == "viral_load"  # Canonical name
    assert resolved_plan["filters"][0]["column"] == "cd4_count"  # Canonical name


def test_trust_ui_shows_effective_execution(sample_query_plan, sample_cohort):
    """Verify effective execution display (dataset, entity_key, resolved columns, effective filters, cohort size)."""
    from clinical_analytics.ui.components.trust_ui import TrustUI

    query_plan = sample_query_plan(
        intent="COUNT",
        filters=[FilterSpec(column="treatment", operator="==", value="A")],
    )

    dataset_version = "test_dataset_v1"
    entity_key = "patient_id"

    # Trust UI should compute effective execution details
    effective_execution = TrustUI._compute_effective_execution(query_plan, sample_cohort, dataset_version, entity_key)

    assert effective_execution["dataset_version"] == "test_dataset_v1"
    assert effective_execution["entity_key"] == "patient_id"
    assert "effective_filters" in effective_execution
    assert "cohort_size" in effective_execution


def test_trust_ui_shows_run_key_and_audit_trail(sample_query_plan):
    """Verify run_key + query text displayed."""
    from clinical_analytics.ui.components.trust_ui import TrustUI

    query_plan = sample_query_plan()
    query_plan.run_key = "abc123xyz"

    query_text = "What is the average age?"

    # Trust UI should display run_key and audit trail
    audit_info = TrustUI._extract_audit_info(query_plan, query_text)

    assert audit_info["run_key"] == "abc123xyz"
    assert "query_text" in audit_info
    assert audit_info["query_text"] == "What is the average age?"


def test_trust_ui_patient_level_export_capped(sample_cohort, sample_query_plan):
    """Verify patient-level export limited to 100 rows by default."""
    from clinical_analytics.ui.components.trust_ui import TrustUI

    query_plan = sample_query_plan()

    # Large cohort (200 rows)
    large_cohort = pl.DataFrame(
        {
            "patient_id": [f"P{i:04d}" for i in range(200)],
            "age": [30 + i % 50 for i in range(200)],
            "treatment": ["A", "B"] * 100,
        }
    )

    # Export should be capped at 100 rows by default
    export_df = TrustUI._prepare_patient_export(large_cohort, query_plan, max_rows=100)

    assert export_df.height == 100


def test_trust_ui_patient_level_export_full_requires_confirmation(sample_cohort, sample_query_plan):
    """Verify full export requires explicit confirmation."""
    from clinical_analytics.ui.components.trust_ui import TrustUI

    query_plan = sample_query_plan()

    # Large cohort (200 rows)
    large_cohort = pl.DataFrame(
        {
            "patient_id": [f"P{i:04d}" for i in range(200)],
            "age": [30 + i % 50 for i in range(200)],
            "treatment": ["A", "B"] * 100,
        }
    )

    # Full export should require explicit max_rows=None
    export_df = TrustUI._prepare_patient_export(large_cohort, query_plan, max_rows=None)

    assert export_df.height == 200


def test_trust_ui_cohort_size_calculation(sample_cohort, sample_query_plan):
    """Verify `count_total` and `count_filtered` computed correctly for percentage reporting."""
    from clinical_analytics.ui.components.trust_ui import TrustUI

    query_plan = sample_query_plan(
        filters=[FilterSpec(column="treatment", operator="==", value="A")],
    )

    entity_key = "patient_id"

    # Cohort size calculation
    cohort_size = TrustUI._calculate_cohort_size(sample_cohort, query_plan.filters, entity_key)

    # Total: all 5 patients
    assert cohort_size["count_total"] == 5

    # Filtered: only patients with treatment="A" (3 patients: P1, P2, P5)
    assert cohort_size["count_filtered"] == 3

    # Percentage
    assert cohort_size["percentage"] == pytest.approx(60.0, rel=1e-2)


def test_trust_ui_tautology_detection():
    """Verify non-restrictive filters labeled or dropped from effective filters display."""
    from clinical_analytics.ui.components.trust_ui import TrustUI

    # Tautology: filtering by all valid values (no restriction)
    filters = [
        FilterSpec(column="treatment", operator="IN", value=["A", "B"]),  # All valid values
    ]

    cohort = pl.DataFrame(
        {
            "patient_id": [1, 2, 3, 4, 5],
            "treatment": ["A", "A", "B", "B", "A"],
        }
    )

    # Normalize filters (detect tautologies)
    effective_filters = TrustUI._normalize_effective_filters(filters, cohort)

    # Tautology should be labeled as "non-restrictive"
    assert effective_filters[0]["is_tautology"] is True
    assert "non-restrictive" in effective_filters[0].get("label", "").lower()


def test_trust_ui_integration_with_descriptive_analysis(ask_questions_page):
    """Verify trust UI appears in execute_analysis_with_idempotency()."""
    # NOTE: This test verifies integration points exist
    # Actual rendering is tested via Streamlit testing tools or manual inspection
    import inspect

    # Check that execute_analysis_with_idempotency has trust UI call
    source = inspect.getsource(ask_questions_page.execute_analysis_with_idempotency)
    assert "TrustUI" in source
    assert "render_verification" in source


def test_trust_ui_integration_with_comparison_analysis(ask_questions_page):
    """Verify trust UI appears in execute_analysis_with_idempotency()."""
    import inspect

    # Check that execute_analysis_with_idempotency has trust UI call
    source = inspect.getsource(ask_questions_page.execute_analysis_with_idempotency)
    assert "TrustUI" in source
    assert "render_verification" in source


def test_trust_ui_integration_with_count_analysis(ask_questions_page):
    """Verify trust UI appears in execute_analysis_with_idempotency()."""
    import inspect

    # Check that execute_analysis_with_idempotency has trust UI call
    source = inspect.getsource(ask_questions_page.execute_analysis_with_idempotency)
    assert "TrustUI" in source
    assert "render_verification" in source
