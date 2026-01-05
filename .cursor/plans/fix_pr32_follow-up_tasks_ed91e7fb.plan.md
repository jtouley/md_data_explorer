---
name: Fix PR32 Follow-up Tasks
overview: Fix test fixture signature mismatches and add accurate token counting with tiktoken for AutoContext
todos:
  - id: fixture-test-red
    content: Write failing test for n_patients parameter (Red phase)
    status: pending
  - id: fixture-implement-green
    content: Implement n_patients parameter in make_cohort_with_categorical (Green phase)
    status: pending
    dependencies:
      - fixture-test-red
  - id: fixture-tests-pass
    content: Fix failing API tests and verify all pass
    status: pending
    dependencies:
      - fixture-implement-green
  - id: fixture-commit
    content: Run make check and commit fixture changes with tests
    status: pending
    dependencies:
      - fixture-tests-pass
  - id: tiktoken-dependency
    content: Add tiktoken to pyproject.toml main dependencies
    status: pending
  - id: tiktoken-test-red
    content: Write failing test for accurate token counting (Red phase)
    status: pending
    dependencies:
      - tiktoken-dependency
  - id: tiktoken-implement-green
    content: Replace _estimate_tokens() with tiktoken implementation (Green phase)
    status: pending
    dependencies:
      - tiktoken-test-red
  - id: tiktoken-verify
    content: Run make test-core to verify no regressions
    status: pending
    dependencies:
      - tiktoken-implement-green
  - id: tiktoken-commit
    content: Run make check and commit token counting changes with tests
    status: pending
    dependencies:
      - tiktoken-verify
  - id: verify-all
    content: Run full test suite and verify all quality gates pass
    status: pending
    dependencies:
      - fixture-commit
      - tiktoken-commit
---

# Fix PR32 Follow-up Tasks

## Overview

Address two follow-up items from PR32 review:

1. Fix test fixture signature mismatches causing 4 API test failures in `test_semantic_layer_fastapi_compat.py`
2. Add tiktoken for accurate token counting in AutoContext (replacing character approximation)

## Task 1: Fix Test Fixture Signature Mismatches

### Current State

- `make_cohort_with_categorical` fixture accepts: `patient_ids`, `treatment`, `status`, `ages`
- Tests may be calling it with `n_patients` parameter (not currently supported)
- Some API tests are failing (4 failures in `test_semantic_layer_fastapi_compat.py`)

### Implementation Plan (TDD Workflow)

**Phase 1a: Write Failing Test (Red)**

1. Add test to `tests/conftest.py`:

```python
def test_make_cohort_with_categorical_n_patients_generates_defaults():
    """Test that n_patients parameter generates default patient IDs."""
    # Arrange
    n_patients = 5

    # Act
    cohort = make_cohort_with_categorical(n_patients=n_patients)

    # Assert
    assert cohort.height == 5
    assert cohort["patient_id"].to_list() == ["P001", "P002", "P003", "P004", "P005"]
```

2. Run test to verify failure:

```bash
make test-api PYTEST_ARGS="tests/conftest.py::test_make_cohort_with_categorical_n_patients -xvs"
```

3. Verify: Test fails with `TypeError: _make() got an unexpected keyword argument 'n_patients'`

**Phase 1b: Implement Fixture Enhancement (Green)**

1. Update `make_cohort_with_categorical` in `tests/conftest.py`:
   - Add `n_patients: int | None = None` parameter
   - Generate defaults: `patient_ids`, `treatment`, `status`, `ages`
   - Default generation logic:
     * `patient_ids`: `[f"P{i:03d}" for i in range(1, n_patients + 1)]`
     * `treatment`: `["control"] * n_patients`
     * `status`: `["active"] * n_patients`
     * `ages`: `[30 + i for i in range(n_patients)]` (ages 30-N)
   - Maintain backward compatibility with explicit parameters

2. Run test to verify pass:

```bash
make test-api PYTEST_ARGS="tests/conftest.py::test_make_cohort_with_categorical_n_patients -xvs"
```

3. Verify: Test passes (Green phase)

**Phase 1c: Fix Failing API Tests**

1. Run API tests to identify failures:

```bash
make test-api
```

2. Investigate 4 failures in `test_semantic_layer_fastapi_compat.py`:
   - Check test calls for incorrect fixture parameters
   - Update to use correct signature (either `patient_ids` or `n_patients`)

3. Run tests again:

```bash
make test-api
```

4. Verify: All API tests pass (0 failures)

**Phase 1d: Quality Gate**

```bash
make check  # Format, lint, type-check, test
```

**Phase 1e: Commit**

```bash
git commit -m "feat: Add n_patients parameter to make_cohort_with_categorical fixture

- Add n_patients parameter with default generation
- Generate patient_ids, treatment, status, ages from n_patients
- Maintain backward compatibility with explicit parameters
- Fix 4 failing API tests in test_semantic_layer_fastapi_compat.py
- Add test coverage for new parameter

All tests passing: X/Y
Following TDD: Red-Green-Refactor"
```

## Task 2: Add tiktoken for Accurate Token Counting

### Current State

- [`src/clinical_analytics/core/autocontext.py`](src/clinical_analytics/core/autocontext.py) uses character approximation: `len(text) // 4`
- Used in `_estimate_tokens()` function
- Called by `_enforce_token_budget()` to enforce 4000 token limit

### Implementation Plan (TDD Workflow)

**Phase 2a: Add tiktoken Dependency**

1. Add to `pyproject.toml` under `[project.dependencies]`:

```toml
"tiktoken>=0.5.0",
```

(Note: Main dependencies, not dev, because AutoContext is production code)

2. Install dependency:

```bash
uv sync
```

**Phase 2b: Write Failing Test (Red)**

1. Add test to `tests/core/test_autocontext.py`:

```python
def test_estimate_tokens_accuracy_with_tiktoken():
    """Test token estimation accuracy using tiktoken."""
    # Arrange
    # Known token counts from tiktoken cl100k_base encoding:
    # "Hello, world!" = 4 tokens
    # "The quick brown fox" = 4 tokens
    test_cases = [
        ("Hello, world!", 4),
        ("The quick brown fox", 4),
        ("A" * 100, 25),  # Approximate
    ]

    # Act & Assert
    for text, expected_tokens in test_cases:
        actual = _estimate_tokens(text)
        assert abs(actual - expected_tokens) <= 1, f"Token count for '{text}': expected ~{expected_tokens}, got {actual}"
```

2. Run test to verify failure:

```bash
make test-core PYTEST_ARGS="tests/core/test_autocontext.py::test_estimate_tokens -xvs"
```

3. Verify: Test fails (still using character approximation)

**Phase 2c: Implement tiktoken Token Counting (Green)**

1. Update `_estimate_tokens()` in `src/clinical_analytics/core/autocontext.py`:

```python
import tiktoken

def _estimate_tokens(text: str) -> int:
    """
    Estimate token count using tiktoken (accurate for OpenAI models).

    Uses cl100k_base encoding (standard for GPT-4).
    Falls back to character approximation if encoding fails.
    """
    try:
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    except Exception:
        # Fallback to approximation if encoding fails
        return len(text) // 4
```

2. Run test to verify pass:

```bash
make test-core PYTEST_ARGS="tests/core/test_autocontext.py::test_estimate_tokens -xvs"
```

3. Verify: Test passes (Green phase)

**Phase 2d: Run Full Core Tests**

```bash
make test-core
```

Verify: No regressions in AutoContext or other core functionality

**Phase 2e: Quality Gate**

```bash
make check  # Format, lint, type-check, test
```

**Phase 2f: Commit**

```bash
git commit -m "feat: Replace character approximation with tiktoken for accurate token counting

- Replace _estimate_tokens() with tiktoken cl100k_base encoding
- Maintain fallback to character approximation on error
- Add test coverage for token counting accuracy
- Verify token budget enforcement with accurate counts

All tests passing: X/Y
Following TDD: Red-Green-Refactor"
```

## Implementation Details

### Fixture Enhancement Pattern

```python
def _make(
    patient_ids: list[str] | None = None,
    treatment: list[str] | None = None,
    status: list[str] | None = None,
    ages: list[int] | None = None,
    n_patients: int | None = None,  # NEW parameter
) -> pl.DataFrame:
    # If n_patients provided, generate defaults
    if n_patients is not None:
        if patient_ids is None:
            patient_ids = [f"P{i:03d}" for i in range(1, n_patients + 1)]
        elif len(patient_ids) != n_patients:
            raise ValueError(f"patient_ids length ({len(patient_ids)}) != n_patients ({n_patients})")

        if treatment is None:
            treatment = ["control"] * n_patients
        if status is None:
            status = ["active"] * n_patients
        if ages is None:
            ages = [30 + i for i in range(n_patients)]

    # Rest of implementation...
```



### Token Counting Implementation

```python
import tiktoken

def _estimate_tokens(text: str) -> int:
    """
    Estimate token count using tiktoken (accurate for OpenAI models).

    Uses cl100k_base encoding (standard for GPT-4).
    Falls back to character approximation if encoding fails.
    """
    try:
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    except Exception:
        # Fallback to approximation if encoding fails
        return len(text) // 4
```



## Testing Strategy (TDD - Tests Written First)

### Fixture Tests (Written Before Implementation)

- Test fixture with `n_patients` parameter (Red phase → implement → Green phase)
- Test fixture with explicit `patient_ids` (backward compatibility)
- Test fixture with both parameters (should raise error if mismatch)
- Run full API test suite after implementation

### Token Counting Tests (Written Before Implementation)

- Test `_estimate_tokens()` with known text samples (Red phase → implement → Green phase)
- Verify accuracy vs character approximation
- Test error handling (encoding failures)
- Test token budget enforcement with accurate counts

**Note**: All tests follow TDD workflow: write failing test (Red) → implement → verify passing test (Green) → commit.

## Success Criteria

1. ✅ All API tests pass (currently 4 failures fixed)
2. ✅ Fixture supports `n_patients` parameter while maintaining backward compatibility
3. ✅ Token counting uses tiktoken for accuracy
4. ✅ Token budget enforcement works correctly with accurate counts
5. ✅ No regressions in existing tests
6. ✅ All quality gates pass (formatting, linting, type checking via `make check`)
7. ✅ TDD workflow followed: tests written before implementation for each phase
8. ✅ Each phase committed separately with implementation + tests together
9. ✅ Pre-commit hooks pass on all commits

## Files to Modify

- [`tests/conftest.py`](tests/conftest.py) - Enhance fixture signature
- [`tests/api/test_semantic_layer_fastapi_compat.py`](tests/api/test_semantic_layer_fastapi_compat.py) - Fix test failures
- [`src/clinical_analytics/core/autocontext.py`](src/clinical_analytics/core/autocontext.py) - Replace token estimation
- [`pyproject.toml`](pyproject.toml) - Add tiktoken dependency
- Test files for token counting (new or existing)

## Notes

- Fixture enhancement maintains backward compatibility with existing `patient_ids` parameter
- tiktoken is production-ready and widely used (OpenAI standard)
- Token counting accuracy improves AutoContext reliability for LLM context budgets
- Pre-commit hooks automatically enforce formatting, linting, and type checking on each commit
- TDD workflow (Red-Green-Refactor) ensures test coverage and prevents regressions
- Each phase has explicit commit point with implementation + tests together
- Use `make check` quality gate after each phase before committing
