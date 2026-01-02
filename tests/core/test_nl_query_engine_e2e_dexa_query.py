"""DEPRECATED: DEXA-specific E2E tests.

This file has been replaced by tests/core/test_nl_query_engine_e2e_generic.py
which contains generic, extensible E2E tests that work with any dataset.

All tests in this file are disabled. They are kept for reference only.
"""

import pytest

# All tests disabled - replaced by generic tests in test_nl_query_engine_e2e_generic.py
pytestmark = pytest.mark.skip("Replaced by generic E2E tests in test_nl_query_engine_e2e_generic.py")


@pytest.fixture
def mock_dexa_semantic_layer():
    """DEPRECATED: Use mock_semantic_layer factory instead."""
    pytest.skip("Use mock_semantic_layer factory from conftest.py")


def test_e2e_dexa_query_which_regimen_lowest_viral_load(mock_dexa_semantic_layer):
    """DEPRECATED: Replaced by generic test_e2e_query_parses_with_grouping_and_metric."""
    pytest.skip("Replaced by generic E2E tests")


def test_e2e_dexa_query_to_analysis_context(mock_dexa_semantic_layer):
    """DEPRECATED: Replaced by generic test_e2e_query_converts_to_analysis_context."""
    pytest.skip("Replaced by generic E2E tests")


def test_e2e_dexa_query_logging_throughout_process(mock_dexa_semantic_layer):
    """DEPRECATED: Replaced by generic test_e2e_query_logging_throughout_process."""
    pytest.skip("Replaced by generic E2E tests")


def test_e2e_dexa_query_variable_matching_accuracy(mock_dexa_semantic_layer):
    """DEPRECATED: Replaced by generic tests - variable matching is tested generically."""
    pytest.skip("Replaced by generic E2E tests")


def test_e2e_dexa_query_execution_readiness(mock_dexa_semantic_layer):
    """DEPRECATED: Replaced by generic test_e2e_query_execution_readiness."""
    pytest.skip("Replaced by generic E2E tests")
