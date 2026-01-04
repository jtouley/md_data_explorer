# NL Query Self-Improvement System

## Overview

The MD Data Explorer includes a **self-learning system** that automatically improves natural language query parsing by analyzing test failures and iteratively refining the LLM prompts.

## Architecture

### Components

1. **`PromptOptimizer`** (`src/clinical_analytics/core/prompt_optimizer.py`)
   - Analyzes test failures to detect patterns
   - Generates prompt improvements based on failure types
   - Logs iterations for analysis

2. **`LearningConfig`** (`src/clinical_analytics/core/prompt_learning_config.yaml`)
   - Configuration-driven pattern detection rules
   - Intent keyword mappings
   - Refinement phrase patterns
   - Fix templates for each pattern type

3. **`EvalHarness`** (`src/clinical_analytics/core/eval_harness.py`)
   - Runs golden questions regression tests
   - Provides structured evaluation results

4. **Self-Improvement Script** (`scripts/self_improve_nl_parsing.py`)
   - Orchestrates the improvement loop
   - Iterates until target accuracy reached

## Configuration

### Adding New Pattern Detection Rules

Edit `src/clinical_analytics/core/prompt_learning_config.yaml`:

```yaml
failure_patterns:
  - type: "custom_pattern"
    detect:
      condition: "actual_intent == 'WRONG' and expected_intent == 'RIGHT'"
    priority: 3
    fix_template: |
      When query contains X, use intent Y instead of Z.
```

### Adding Intent Keywords

```yaml
intent_keywords:
  FIND_PREDICTORS:
    - "predict"
    - "predictor"
    - "what causes"
    - "risk factors"
```

### Adding Refinement Phrases

```yaml
refinement_phrases:
  - "remove"
  - "exclude"
  - "without"
  - "get rid of"
```

## Usage

### Running Self-Improvement

```bash
# Run with default settings (95% target, 10 max iterations)
python scripts/self_improve_nl_parsing.py

# Custom target accuracy
python scripts/self_improve_nl_parsing.py --target-accuracy 0.98

# Custom max iterations
python scripts/self_improve_nl_parsing.py --max-iterations 20

# Custom log directory
python scripts/self_improve_nl_parsing.py --log-dir /tmp/my_logs
```

### Output

The script will:
1. Run golden questions evaluation
2. Analyze failures and detect patterns
3. Generate prompt improvements
4. Log iteration results to JSON files
5. Repeat until target accuracy reached or max iterations hit

**Example output:**

```
Iteration 1/10:
  Accuracy: 74.4% (target: 95.0%)
  Patterns detected: 3
    - invalid_intent: 2 failures
    - refinement_ignored: 6 failures
    - intent_mismatch: 4 failures

📝 Prompt improvements:
=== AUTO-GENERATED FIXES (Learning Iteration) ===

**VALIDATION RULE (CRITICAL):**
You MUST use EXACTLY one of these intents: COUNT, DESCRIBE, COMPARE_GROUPS, FIND_PREDICTORS, CORRELATIONS
ANY other intent (like 'REMOVE_NA', 'FILTER_OUT') is INVALID and will cause system failure.

**REFINEMENT DETECTION ALGORITHM:**
STEP 1: Check if query contains refinement phrase: "remove", "exclude", "only", ...
STEP 2: IF YES → This is a REFINEMENT
        - COPY previous_intent → current_intent (DO NOT CREATE NEW)
...
```

### Iteration Logs

Each iteration creates a JSON log file:

```
/tmp/nl_query_learning/
├── iteration_01.json
├── iteration_02.json
└── iteration_03.json
```

**Log format:**

```json
{
  "iteration": 1,
  "accuracy": 0.744,
  "patterns": [
    {
      "type": "invalid_intent",
      "count": 2,
      "priority": 1,
      "fix": "CRITICAL: Invalid intent...",
      "examples": [
        {"query": "remove the n/a", "invalid_intent": "REMOVE_NA"}
      ]
    }
  ],
  "prompt_additions": "=== AUTO-GENERATED FIXES ===\n..."
}
```

## Manual Integration

Currently, prompt improvements are **not automatically applied**. To integrate:

1. Run self-improvement script
2. Review generated prompt additions
3. Manually update `src/clinical_analytics/core/nl_query_engine.py`:
   - Add improvements to `_build_llm_prompt()` method
   - Update system prompt with new rules
4. Re-run golden questions test to verify improvement

### Example Integration

**Before:**

```python
def _build_llm_prompt(self, query: str, ...):
    system_prompt = """You are a medical data query parser.
    
    Return JSON matching the QueryPlan schema...
    """
```

**After:**

```python
def _build_llm_prompt(self, query: str, ...):
    system_prompt = """You are a medical data query parser.
    
    Return JSON matching the QueryPlan schema...
    
    === AUTO-GENERATED FIXES (Learning Iteration 1) ===
    
    **VALIDATION RULE (CRITICAL):**
    You MUST use EXACTLY one of these intents: COUNT, DESCRIBE, COMPARE_GROUPS...
    """
```

## Extending the System

### Adding New Failure Pattern Types

1. **Define pattern in config:**

```yaml
failure_patterns:
  - type: "missing_metric"
    detect:
      condition: "expected_metric and not actual_metric"
    priority: 4
    fix_template: |
      METRIC EXTRACTION:
      Look for the primary variable being analyzed.
      Examples:
      - "average age" → metric: "age"
      - "count patients" → metric: null (counting entities)
```

2. **Add test coverage:**

```python
# tests/core/test_prompt_optimizer.py
def test_optimizer_detects_missing_metric_pattern(sample_learning_config):
    failures = [
        {"expected_metric": "age", "actual_metric": None, "passed": False}
    ]
    optimizer = PromptOptimizer(config=sample_learning_config)
    patterns = optimizer.analyze_failures(failures)
    
    assert any(p.pattern_type == "missing_metric" for p in patterns)
```

### Customizing Fix Templates

Templates support placeholders that are automatically replaced:

- `{valid_intents}` → List of valid intent types
- `{refinement_phrases}` → List of refinement detection phrases
- `{invalid_intent}` → The invalid intent that was detected
- `{keyword_hints}` → Intent keyword mappings

## Best Practices

### 1. Start with Baseline

Run golden questions test to establish baseline accuracy:

```bash
make test  # Runs tests/eval/test_golden_questions.py
```

### 2. Iterative Refinement

Don't try to fix everything at once:
- Focus on highest priority patterns (priority 1, 2, 3)
- Apply fixes incrementally
- Re-test after each change

### 3. Update Golden Questions

When you fix a pattern, add new test cases to prevent regression:

```yaml
# tests/eval/golden_questions.yaml
golden_questions:
  - id: new_test_case
    query: "query that previously failed"
    expected_intent: COUNT
    notes: "Added after fixing pattern X in iteration Y"
```

### 4. Monitor Confidence Scores

Low confidence (<0.7) indicates:
- Query is ambiguous
- LLM needs more context
- Prompt may need clarification

### 5. Configuration Over Code

Prefer updating `prompt_learning_config.yaml` over hardcoding logic:
- Easier to maintain
- More transparent
- Faster iteration

## Troubleshooting

### Issue: Accuracy Not Improving

**Causes:**
- LLM model too small (3b vs 8b)
- Patterns not correctly detected
- Fix templates not specific enough

**Solutions:**
1. Check iteration logs for detected patterns
2. Verify condition evaluation in config
3. Make fix templates more explicit and shorter
4. Use larger LLM model (llama3.1:8b)

### Issue: Same Pattern Detected Every Iteration

**Cause:** Fix not actually applied to prompt

**Solution:** Manually integrate prompt_additions into nl_query_engine.py

### Issue: Tests Timeout

**Cause:** LLM requests timing out

**Solution:**
```yaml
# ollama_config.yaml
ollama:
  timeout_seconds: 60.0  # Increase timeout
```

## Metrics

Track these metrics across iterations:

- **Overall Accuracy**: % of all tests passing
- **Intent Accuracy**: % of intent classifications correct
- **Pattern Frequency**: How often each pattern appears
- **Confidence**: Average confidence across all parses

**Target thresholds:**
- Overall accuracy: ≥95%
- Intent accuracy: ≥90%
- Average confidence: ≥0.80

## Future Enhancements

Planned improvements:
1. **Automatic prompt integration** - Apply fixes without manual intervention
2. **A/B testing** - Compare prompt variants
3. **Few-shot example selection** - Dynamically select best examples
4. **Confidence calibration** - Tune confidence scoring
5. **Cross-dataset validation** - Test improvements across all datasets

## References

- **[ADR009](../implementation/ADR/ADR009.md)** - LLM-Enhanced UX architecture
- **[Prompt Optimizer Tests](../../tests/core/test_prompt_optimizer.py)** - Test coverage
- **[Golden Questions](../../tests/eval/golden_questions.yaml)** - Regression test suite
- **[Eval Harness](../eval_harness.py)** - Evaluation framework






