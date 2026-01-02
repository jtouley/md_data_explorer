"""Tests for prompt optimizer and self-learning system.

Following AGENTS.md guidelines:
- AAA pattern (Arrange-Act-Assert)
- Descriptive test names: test_unit_scenario_expectedBehavior
- No duplicate fixtures (use conftest.py)
- Test isolation (no shared mutable state)
"""

import json
from pathlib import Path

import pytest

from clinical_analytics.core.prompt_optimizer import (
    FailurePattern,
    LearningConfig,
    PromptOptimizer,
)


# Fixtures for test data
@pytest.fixture
def sample_learning_config():
    """Sample learning configuration for testing."""
    return LearningConfig(
        intent_keywords={
            "COUNT": ["how many", "count"],
            "DESCRIBE": ["average", "mean"],
            "FIND_PREDICTORS": ["predict", "predictor"],
        },
        refinement_phrases=["remove", "exclude", "only"],
        valid_intents=["COUNT", "DESCRIBE", "COMPARE_GROUPS", "FIND_PREDICTORS", "CORRELATIONS"],
        failure_patterns=[
            {
                "type": "invalid_intent",
                "detect": {"condition": "actual_intent not in valid_intents"},
                "priority": 1,
                "fix_template": "CRITICAL: Invalid intent '{invalid_intent}'. Use: {valid_intents}",
            },
            {
                "type": "refinement_ignored",
                "detect": {"condition": "has_conversation_history and actual_intent != expected_intent"},
                "priority": 2,
                "fix_template": "Refinement phrases: {refinement_phrases}",
            },
        ],
        prompt_template="Template: {dynamic_fixes}",
        logging_config={"enabled": True, "log_dir": None},  # Tests should override with tmp_path
    )


@pytest.fixture
def sample_test_results_with_invalid_intent():
    """Sample test results with invalid intent failures."""
    return [
        {
            "query": "remove the n/a",
            "expected_intent": "COUNT",
            "actual_intent": "REMOVE_NA",
            "passed": False,
        },
        {
            "query": "exclude unknowns",
            "expected_intent": "DESCRIBE",
            "actual_intent": "FILTER_OUT",
            "passed": False,
        },
        {
            "query": "count patients",
            "expected_intent": "COUNT",
            "actual_intent": "COUNT",
            "passed": True,
        },
    ]


@pytest.fixture
def sample_test_results_with_refinement_failures():
    """Sample test results with refinement failures."""
    return [
        {
            "query": "remove the n/a",
            "expected_intent": "COUNT",
            "actual_intent": "DESCRIBE",
            "conversation_history": [{"intent": "COUNT", "group_by": "statin"}],
            "passed": False,
        },
        {
            "query": "exclude missing",
            "expected_intent": "DESCRIBE",
            "actual_intent": "COUNT",
            "conversation_history": [{"intent": "DESCRIBE", "metric": "cholesterol"}],
            "passed": False,
        },
    ]


@pytest.fixture
def sample_test_results_all_passing():
    """Sample test results with all tests passing."""
    return [
        {
            "query": "count patients",
            "expected_intent": "COUNT",
            "actual_intent": "COUNT",
            "passed": True,
        },
        {
            "query": "average age",
            "expected_intent": "DESCRIBE",
            "actual_intent": "DESCRIBE",
            "passed": True,
        },
    ]


# Configuration loading tests
def test_learning_config_load_with_valid_config_returns_config(tmp_path):
    """Test that LearningConfig loads successfully from valid YAML file."""
    # Arrange: Create temporary config file
    config_file = tmp_path / "test_config.yaml"
    config_data = {
        "intent_keywords": {"COUNT": ["how many"]},
        "refinement_phrases": ["remove"],
        "valid_intents": ["COUNT", "DESCRIBE"],
        "failure_patterns": [],
        "prompt_template": "test template",
        "logging": {"enabled": True},
    }
    config_file.write_text(json.dumps(config_data))

    # Act: Load configuration (convert to YAML format first)
    import yaml

    config_file.write_text(yaml.dump(config_data))
    config = LearningConfig.load(config_file)

    # Assert: Configuration loaded correctly
    assert config.intent_keywords == {"COUNT": ["how many"]}
    assert config.refinement_phrases == ["remove"]
    assert config.valid_intents == ["COUNT", "DESCRIBE"]


def test_learning_config_load_with_missing_file_returns_empty_config():
    """Test that LearningConfig returns empty config when file not found."""
    # Arrange: Non-existent file path
    nonexistent_path = Path("/nonexistent/config.yaml")

    # Act: Load configuration
    config = LearningConfig.load(nonexistent_path)

    # Assert: Returns empty configuration
    assert config.intent_keywords == {}
    assert config.refinement_phrases == []
    assert config.valid_intents == []


# Pattern detection tests
def test_optimizer_analyze_failures_with_invalid_intent_detects_pattern(
    sample_learning_config, sample_test_results_with_invalid_intent
):
    """Test that optimizer detects invalid intent pattern from failures."""
    # Arrange: Create optimizer with config
    optimizer = PromptOptimizer(config=sample_learning_config)

    # Act: Analyze test results
    patterns = optimizer.analyze_failures(sample_test_results_with_invalid_intent)

    # Assert: Invalid intent pattern detected
    assert len(patterns) >= 1
    invalid_patterns = [p for p in patterns if p.pattern_type == "invalid_intent"]
    assert len(invalid_patterns) == 1
    assert invalid_patterns[0].count == 2  # Two invalid intents
    assert invalid_patterns[0].priority == 1  # Highest priority


def test_optimizer_analyze_failures_with_refinement_failures_detects_pattern(
    sample_learning_config, sample_test_results_with_refinement_failures
):
    """Test that optimizer detects refinement ignored pattern from failures."""
    # Arrange: Create optimizer with config
    optimizer = PromptOptimizer(config=sample_learning_config)

    # Act: Analyze test results
    patterns = optimizer.analyze_failures(sample_test_results_with_refinement_failures)

    # Assert: Refinement ignored pattern detected
    assert len(patterns) >= 1
    refinement_patterns = [p for p in patterns if p.pattern_type == "refinement_ignored"]
    assert len(refinement_patterns) == 1
    assert refinement_patterns[0].count == 2
    assert refinement_patterns[0].priority == 2


def test_optimizer_analyze_failures_with_all_passing_returns_empty_list(
    sample_learning_config, sample_test_results_all_passing
):
    """Test that optimizer returns no patterns when all tests pass."""
    # Arrange: Create optimizer with config
    optimizer = PromptOptimizer(config=sample_learning_config)

    # Act: Analyze test results
    patterns = optimizer.analyze_failures(sample_test_results_all_passing)

    # Assert: No patterns detected
    assert len(patterns) == 0


# Template generation tests
def test_optimizer_generate_fix_from_template_replaces_placeholders(sample_learning_config):
    """Test that fix generation replaces template placeholders correctly."""
    # Arrange: Create optimizer and template
    optimizer = PromptOptimizer(config=sample_learning_config)
    template = "Valid intents: {valid_intents}"
    failures = [{"actual_intent": "INVALID"}]

    # Act: Generate fix text
    fix_text = optimizer._generate_fix_from_template(template, failures, "invalid_intent")

    # Assert: Placeholders replaced
    assert "COUNT" in fix_text
    assert "DESCRIBE" in fix_text
    assert "{valid_intents}" not in fix_text


def test_optimizer_generate_prompt_additions_with_patterns_returns_text(
    sample_learning_config, sample_test_results_with_invalid_intent
):
    """Test that prompt additions are generated from failure patterns."""
    # Arrange: Create optimizer and analyze failures
    optimizer = PromptOptimizer(config=sample_learning_config)
    patterns = optimizer.analyze_failures(sample_test_results_with_invalid_intent)

    # Act: Generate prompt additions
    additions = optimizer.generate_improved_prompt_additions(patterns)

    # Assert: Additions contain fixes
    assert len(additions) > 0
    assert "CRITICAL" in additions or "Invalid intent" in additions


def test_optimizer_generate_prompt_additions_with_no_patterns_returns_empty(
    sample_learning_config,
):
    """Test that no prompt additions generated when no patterns detected."""
    # Arrange: Create optimizer with no patterns
    optimizer = PromptOptimizer(config=sample_learning_config)
    patterns = []

    # Act: Generate prompt additions
    additions = optimizer.generate_improved_prompt_additions(patterns)

    # Assert: Empty additions
    assert additions == ""


# Helper method tests
def test_optimizer_get_keyword_hints_with_matching_keyword_returns_intent(
    sample_learning_config,
):
    """Test that keyword hints return correct intent for matching query."""
    # Arrange: Create optimizer
    optimizer = PromptOptimizer(config=sample_learning_config)
    query = "how many patients are there"

    # Act: Get keyword hint
    hint = optimizer.get_keyword_hints_for_intent(query)

    # Assert: Returns COUNT intent
    assert hint == "COUNT"


def test_optimizer_get_keyword_hints_with_no_match_returns_none(sample_learning_config):
    """Test that keyword hints return None when no keyword matches."""
    # Arrange: Create optimizer
    optimizer = PromptOptimizer(config=sample_learning_config)
    query = "show me something"

    # Act: Get keyword hint
    hint = optimizer.get_keyword_hints_for_intent(query)

    # Assert: Returns None
    assert hint is None


def test_optimizer_is_refinement_query_with_refinement_phrase_returns_true(
    sample_learning_config,
):
    """Test that refinement detection returns True for refinement queries."""
    # Arrange: Create optimizer
    optimizer = PromptOptimizer(config=sample_learning_config)
    query = "remove the n/a values"

    # Act: Check if refinement
    is_refinement = optimizer.is_refinement_query(query)

    # Assert: Detected as refinement
    assert is_refinement is True


def test_optimizer_is_refinement_query_with_no_phrase_returns_false(
    sample_learning_config,
):
    """Test that refinement detection returns False for non-refinement queries."""
    # Arrange: Create optimizer
    optimizer = PromptOptimizer(config=sample_learning_config)
    query = "count all patients"

    # Act: Check if refinement
    is_refinement = optimizer.is_refinement_query(query)

    # Assert: Not detected as refinement
    assert is_refinement is False


# Logging tests
def test_optimizer_log_iteration_creates_log_file(tmp_path, sample_learning_config):
    """Test that iteration logging creates JSON log file."""
    # Arrange: Create optimizer with temp log dir
    log_dir = tmp_path / "logs"
    optimizer = PromptOptimizer(config=sample_learning_config, log_dir=log_dir)
    patterns = [
        FailurePattern(
            pattern_type="test_pattern",
            count=5,
            examples=[{"query": "test"}],
            suggested_fix="test fix",
            priority=1,
        )
    ]

    # Act: Log iteration
    optimizer.log_iteration(iteration=1, accuracy=0.75, patterns=patterns, prompt_additions="test additions")

    # Assert: Log file created
    log_file = log_dir / "iteration_01.json"
    assert log_file.exists()

    # Assert: Log file contains correct data
    log_data = json.loads(log_file.read_text())
    assert log_data["iteration"] == 1
    assert log_data["accuracy"] == 0.75
    assert len(log_data["patterns"]) == 1
    assert log_data["patterns"][0]["type"] == "test_pattern"


def test_optimizer_log_iteration_with_logging_disabled_creates_no_file(tmp_path, sample_learning_config):
    """Test that logging is skipped when disabled in config."""
    # Arrange: Create optimizer with logging disabled
    sample_learning_config.logging_config["enabled"] = False
    log_dir = tmp_path / "logs"
    optimizer = PromptOptimizer(config=sample_learning_config, log_dir=log_dir)

    # Act: Log iteration
    optimizer.log_iteration(iteration=1, accuracy=0.75, patterns=[], prompt_additions="")

    # Assert: No log file created
    assert not log_dir.exists() or len(list(log_dir.glob("*.json"))) == 0


# Condition evaluation tests
def test_optimizer_evaluate_condition_with_valid_condition_returns_true(
    sample_learning_config,
):
    """Test that condition evaluation returns True for matching conditions."""
    # Arrange: Create optimizer and failure record
    optimizer = PromptOptimizer(config=sample_learning_config)
    failure = {
        "actual_intent": "INVALID",
        "expected_intent": "COUNT",
        "conversation_history": None,
    }
    condition = "actual_intent not in valid_intents"

    # Act: Evaluate condition
    result = optimizer._evaluate_condition(failure, condition)

    # Assert: Condition evaluates to True
    assert result is True


def test_optimizer_evaluate_condition_with_invalid_condition_returns_false(
    sample_learning_config,
):
    """Test that condition evaluation returns False for non-matching conditions."""
    # Arrange: Create optimizer and failure record
    optimizer = PromptOptimizer(config=sample_learning_config)
    failure = {
        "actual_intent": "COUNT",
        "expected_intent": "COUNT",
        "conversation_history": None,
    }
    condition = "actual_intent not in valid_intents"

    # Act: Evaluate condition
    result = optimizer._evaluate_condition(failure, condition)

    # Assert: Condition evaluates to False
    assert result is False


# Integration test
def test_optimizer_full_workflow_analyzes_and_generates_fixes(
    tmp_path, sample_learning_config, sample_test_results_with_invalid_intent
):
    """Test full workflow: analyze failures, generate fixes, log iteration."""
    # Arrange: Create optimizer with temp log dir
    log_dir = tmp_path / "logs"
    optimizer = PromptOptimizer(config=sample_learning_config, log_dir=log_dir)

    # Act: Full workflow
    patterns = optimizer.analyze_failures(sample_test_results_with_invalid_intent)
    additions = optimizer.generate_improved_prompt_additions(patterns)
    optimizer.log_iteration(iteration=1, accuracy=0.67, patterns=patterns, prompt_additions=additions)

    # Assert: Patterns detected
    assert len(patterns) > 0

    # Assert: Additions generated
    assert len(additions) > 0

    # Assert: Log file created
    log_file = log_dir / "iteration_01.json"
    assert log_file.exists()
    log_data = json.loads(log_file.read_text())
    assert log_data["accuracy"] == 0.67


def test_optimizer_analyze_failures_with_none_intent_filters_none_values(sample_learning_config):
    """Test that optimizer filters None from invalid_intents set without crashing."""
    # Arrange: Create optimizer and test results with None actual_intent
    optimizer = PromptOptimizer(config=sample_learning_config)
    test_results = [
        {
            "query": "remove the n/a",
            "expected_intent": "COUNT",
            "actual_intent": None,  # None value that causes TypeError
            "passed": False,
        },
        {
            "query": "exclude unknowns",
            "expected_intent": "DESCRIBE",
            "actual_intent": None,  # Another None value
            "passed": False,
        },
        {
            "query": "filter out invalid",
            "expected_intent": "COUNT",
            "actual_intent": "FILTER_OUT",  # Valid invalid intent
            "passed": False,
        },
    ]

    # Act: Analyze failures (should not crash with TypeError)
    patterns = optimizer.analyze_failures(test_results)

    # Assert: Patterns detected without crash
    assert len(patterns) >= 1
    invalid_patterns = [p for p in patterns if p.pattern_type == "invalid_intent"]
    assert len(invalid_patterns) == 1

    # Assert: Generated fix only contains non-None invalid intents
    fix_text = invalid_patterns[0].suggested_fix
    assert "FILTER_OUT" in fix_text
    assert "None" not in fix_text  # None should be filtered out
