"""Tests for NL Query Engine configuration constants."""

import os
from unittest.mock import patch

from clinical_analytics.core.nl_query_config import (
    AUTO_EXECUTE_CONFIDENCE_THRESHOLD,
    CLARIFYING_QUESTIONS_THRESHOLD,
    FUZZY_MATCH_CUTOFF,
    SEMANTIC_SIMILARITY_THRESHOLD,
    TIER_1_PATTERN_MATCH_THRESHOLD,
    TIER_2_SEMANTIC_MATCH_THRESHOLD,
    TIER_TIMEOUT_SECONDS,
)


def test_config_constants_are_defined():
    """All configuration constants should be defined."""
    assert TIER_1_PATTERN_MATCH_THRESHOLD == 0.9
    assert TIER_2_SEMANTIC_MATCH_THRESHOLD == 0.75
    assert CLARIFYING_QUESTIONS_THRESHOLD == 0.5
    assert AUTO_EXECUTE_CONFIDENCE_THRESHOLD == TIER_2_SEMANTIC_MATCH_THRESHOLD
    assert TIER_TIMEOUT_SECONDS == 5.0
    assert SEMANTIC_SIMILARITY_THRESHOLD == 0.7
    assert FUZZY_MATCH_CUTOFF == 0.7


def test_auto_execute_threshold_matches_semantic_threshold():
    """AUTO_EXECUTE_CONFIDENCE_THRESHOLD should equal TIER_2_SEMANTIC_MATCH_THRESHOLD."""
    assert AUTO_EXECUTE_CONFIDENCE_THRESHOLD == TIER_2_SEMANTIC_MATCH_THRESHOLD


def test_feature_flags_default_to_true():
    """Feature flags should default to True when env var not set."""
    with patch.dict(os.environ, {}, clear=True):
        # Reload module to get fresh defaults
        import importlib

        import clinical_analytics.core.nl_query_config as config_module

        importlib.reload(config_module)
        assert config_module.ENABLE_CLARIFYING_QUESTIONS is True
        assert config_module.ENABLE_PROGRESSIVE_FEEDBACK is True


def test_feature_flags_respect_env_vars():
    """Feature flags should respect environment variables."""
    with patch.dict(os.environ, {"ENABLE_CLARIFYING_QUESTIONS": "false"}, clear=False):
        import importlib

        import clinical_analytics.core.nl_query_config as config_module

        importlib.reload(config_module)
        assert config_module.ENABLE_CLARIFYING_QUESTIONS is False

    with patch.dict(os.environ, {"ENABLE_PROGRESSIVE_FEEDBACK": "false"}, clear=False):
        import importlib

        import clinical_analytics.core.nl_query_config as config_module

        importlib.reload(config_module)
        assert config_module.ENABLE_PROGRESSIVE_FEEDBACK is False
