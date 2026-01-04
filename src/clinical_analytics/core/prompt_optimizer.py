"""Self-learning prompt optimization for NL query parsing.

This module analyzes golden question failures and automatically generates
improved prompts based on configurable pattern rules.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import structlog
import yaml

logger = structlog.get_logger(__name__)


@dataclass
class FailurePattern:
    """Pattern detected in test failures."""

    pattern_type: str
    count: int
    examples: list[dict[str, Any]]
    suggested_fix: str
    priority: int = 99


@dataclass
class LearningConfig:
    """Configuration for prompt learning."""

    intent_keywords: dict[str, list[str]] = field(default_factory=dict)
    refinement_phrases: list[str] = field(default_factory=list)
    valid_intents: list[str] = field(default_factory=list)
    failure_patterns: list[dict[str, Any]] = field(default_factory=list)
    prompt_template: str = ""
    logging_config: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, config_path: Path | None = None) -> "LearningConfig":
        """Load configuration from YAML file."""
        if config_path is None:
            # Get project root: prompt_optimizer.py → core/ → clinical_analytics/ → src/ → project_root
            current_file = Path(__file__)
            project_root = current_file.parent.parent.parent.parent
            config_path = project_root / "config" / "prompt_learning.yaml"

        if not config_path.exists():
            logger.warning("config_not_found", path=str(config_path))
            return cls()

        with open(config_path) as f:
            data = yaml.safe_load(f)

        return cls(
            intent_keywords=data.get("intent_keywords", {}),
            refinement_phrases=data.get("refinement_phrases", []),
            valid_intents=data.get("valid_intents", []),
            failure_patterns=data.get("failure_patterns", []),
            prompt_template=data.get("prompt_template", ""),
            logging_config=data.get("logging", {}),
        )


class PromptOptimizer:
    """Analyzes failures and suggests prompt improvements."""

    def __init__(
        self,
        config: LearningConfig | None = None,
        log_dir: Path | None = None,
    ):
        self.config = config or LearningConfig.load()
        self.log_dir = log_dir or Path(self.config.logging_config.get("log_dir", "/tmp/nl_query_learning"))
        self.log_dir.mkdir(exist_ok=True)
        self.iteration = 0

    def analyze_failures(self, test_results: list[dict[str, Any]]) -> list[FailurePattern]:
        """Analyze test results and identify failure patterns.

        Args:
            test_results: List of test results with expected/actual values

        Returns:
            List of detected failure patterns with suggested fixes
        """
        logger.info("analyzing_failures", total_tests=len(test_results))

        failures = [r for r in test_results if not r.get("passed", False)]
        if not failures:
            logger.info("no_failures_detected")
            return []

        patterns = []

        # Apply each configured failure pattern rule
        for rule in self.config.failure_patterns:
            detected = self._detect_pattern(failures, rule)
            if detected:
                patterns.append(detected)

        # Sort by priority
        patterns.sort(key=lambda p: p.priority)

        logger.info("failure_patterns_detected", pattern_count=len(patterns))
        return patterns

    def _detect_pattern(self, failures: list[dict[str, Any]], rule: dict[str, Any]) -> FailurePattern | None:
        """Detect a specific pattern based on rule configuration."""
        pattern_type = rule["type"]
        condition = rule["detect"]["condition"]
        priority = rule.get("priority", 99)

        # Filter failures matching the condition
        matching_failures = []
        for f in failures:
            if self._evaluate_condition(f, condition):
                matching_failures.append(f)

        if not matching_failures:
            return None

        # Generate examples
        examples = self._extract_examples(matching_failures, pattern_type)

        # Generate fix text from template
        fix_text = self._generate_fix_from_template(rule["fix_template"], matching_failures, pattern_type)

        return FailurePattern(
            pattern_type=pattern_type,
            count=len(matching_failures),
            examples=examples,
            suggested_fix=fix_text,
            priority=priority,
        )

    def _evaluate_condition(self, failure: dict[str, Any], condition: str) -> bool:
        """Evaluate a condition string against a failure record."""
        # Build context for eval
        ctx = {
            "actual_intent": failure.get("actual_intent"),
            "expected_intent": failure.get("expected_intent"),
            "actual_filters": failure.get("actual_filters"),
            "expected_filters": failure.get("expected_filters"),
            "actual_group_by": failure.get("actual_group_by"),
            "expected_group_by": failure.get("expected_group_by"),
            "has_conversation_history": bool(failure.get("conversation_history")),
            "valid_intents": self.config.valid_intents,
        }

        try:
            result = eval(condition, {"__builtins__": {}}, ctx)
            return bool(result)
        except Exception as e:
            logger.warning("condition_evaluation_failed", condition=condition, error=str(e))
            return False

    def _extract_examples(self, failures: list[dict[str, Any]], pattern_type: str) -> list[dict[str, Any]]:
        """Extract representative examples for a pattern."""
        examples = []
        for f in failures[:3]:  # Max 3 examples
            example = {"query": f.get("query", "")}

            if pattern_type == "invalid_intent":
                example["invalid_intent"] = f.get("actual_intent")
            elif pattern_type == "refinement_ignored":
                example["previous_intent"] = (
                    f.get("conversation_history", [{}])[-1].get("intent") if f.get("conversation_history") else None
                )
                example["expected_intent"] = f.get("expected_intent")
                example["actual_intent"] = f.get("actual_intent")
            elif pattern_type == "intent_mismatch":
                example["expected"] = f.get("expected_intent")
                example["actual"] = f.get("actual_intent")
            elif pattern_type == "missing_filters":
                example["expected_filters"] = f.get("expected_filters")

            examples.append(example)

        return examples

    def _generate_fix_from_template(
        self,
        template: str,
        failures: list[dict[str, Any]],
        pattern_type: str,
    ) -> str:
        """Generate fix text from template with dynamic values."""
        # Gather dynamic values
        replacements = {
            "valid_intents": ", ".join(self.config.valid_intents),
            "refinement_phrases": ", ".join(f'"{p}"' for p in self.config.refinement_phrases),
        }

        # Add pattern-specific replacements
        if pattern_type == "invalid_intent":
            invalid_intents = {str(f.get("actual_intent")) for f in failures if f.get("actual_intent") is not None}
            replacements["invalid_intent"] = ", ".join(invalid_intents)

        elif pattern_type == "intent_mismatch":
            # Generate keyword hints
            hints = []
            for intent, keywords in self.config.intent_keywords.items():
                hints.append(f"  - {', '.join(keywords)} → {intent}")
            replacements["keyword_hints"] = "\n".join(hints)

        # Replace placeholders in template
        fix_text = template
        for key, value in replacements.items():
            fix_text = fix_text.replace(f"{{{key}}}", str(value))

        return fix_text

    def generate_improved_prompt_additions(self, patterns: list[FailurePattern]) -> str:
        """Generate prompt additions based on failure patterns.

        Returns:
            Additional text to add to the system prompt
        """
        if not patterns:
            return ""

        additions = ["\n=== AUTO-GENERATED FIXES (Learning Iteration) ===\n"]

        for pattern in patterns:
            additions.append(pattern.suggested_fix)
            additions.append("")  # Blank line between patterns

        return "\n".join(additions)

    def log_iteration(
        self,
        iteration: int,
        accuracy: float,
        patterns: list[FailurePattern],
        prompt_additions: str,
    ) -> None:
        """Log iteration results for analysis."""
        if not self.config.logging_config.get("enabled", True):
            return

        log_file = self.log_dir / f"iteration_{iteration:02d}.json"

        data = {
            "iteration": iteration,
            "accuracy": accuracy,
            "patterns": [
                {
                    "type": p.pattern_type,
                    "count": p.count,
                    "priority": p.priority,
                    "fix": p.suggested_fix,
                    "examples": p.examples,
                }
                for p in patterns
            ],
            "prompt_additions": prompt_additions,
        }

        log_file.write_text(json.dumps(data, indent=2))
        logger.info(
            "iteration_logged",
            iteration=iteration,
            accuracy=accuracy,
            patterns=len(patterns),
            log_file=str(log_file),
        )

    def get_keyword_hints_for_intent(self, query: str) -> str | None:
        """Get intent hint based on keywords in query."""
        query_lower = query.lower()

        for intent, keywords in self.config.intent_keywords.items():
            for keyword in keywords:
                if keyword in query_lower:
                    return intent

        return None

    def is_refinement_query(self, query: str) -> bool:
        """Check if query appears to be a refinement."""
        query_lower = query.lower()
        return any(phrase in query_lower for phrase in self.config.refinement_phrases)
