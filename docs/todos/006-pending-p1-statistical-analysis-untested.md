---
status: pending
priority: p1
issue_id: "006"
tags: [code-review, testing, critical, research-integrity]
dependencies: []
estimated_effort: large
created_date: 2025-12-24
---

# Statistical Analysis Module Completely Untested

## Problem Statement

**Zero test coverage** for `src/clinical_analytics/analysis/stats.py` which performs logistic regression, survival analysis, and statistical computations on medical data. Untested statistical code on healthcare data poses **extreme research integrity risk** and could lead to incorrect clinical conclusions.

**Why it matters:**
- **Research Integrity:** Wrong statistical results = wrong clinical conclusions
- Publication retractions if errors discovered post-publication
- Patient safety if results inform clinical decisions
- Regulatory compliance (FDA, IRB requirements for validated methods)
- Scientific reproducibility requirements

**Impact:** Research conclusions may be invalid, publications at risk

## Findings

**Location:** `src/clinical_analytics/analysis/stats.py` (entire module)

**Current Test Coverage:**
```
stats.py: 0% coverage
- logistic_regression(): 0% tested
- survival_analysis(): 0% tested
- _prepare_data_for_regression(): 0% tested
- _create_feature_matrix(): 0% tested
```

**Critical Untested Code:**
```python
# Lines 45-98 - NO TESTS
def logistic_regression(
    df: pd.DataFrame,
    outcome_col: str,
    predictors: List[str],
    covariates: Optional[List[str]] = None
) -> Any:  # Returns statsmodels LogitResults
    """
    Perform logistic regression analysis.

    CRITICAL: Used for medical outcome predictions
    PROBLEM: No tests verify correctness
    """
    # Complex data transformations - UNTESTED
    # Feature matrix creation - UNTESTED
    # Model fitting - UNTESTED
    # Missing data handling - UNTESTED
    model = sm.Logit(y, X)
    result = model.fit()
    return result  # Returns model without validation
```

**Real-World Risk Examples:**

1. **Wrong missing data handling:**
   ```python
   # If this silently drops important cases...
   df_clean = df.dropna(subset=[outcome_col] + all_features)
   # Research conclusions may be biased
   ```

2. **Incorrect feature encoding:**
   ```python
   # If categorical variables encoded wrong...
   X = pd.get_dummies(df[all_features], drop_first=True)
   # Regression coefficients will be wrong
   ```

3. **Multicollinearity not checked:**
   ```python
   # If highly correlated predictors not detected...
   model = sm.Logit(y, X)
   # Results will be unstable and misleading
   ```

## Proposed Solutions

### Solution 1: Comprehensive Test Suite with Known Datasets (Recommended)
**Pros:**
- Validates against known statistical results
- Catches regressions in future changes
- Documents expected behavior
- Industry standard approach

**Cons:**
- Requires creating test datasets with known outcomes
- Time-intensive to get right

**Effort:** Large (12 hours)
**Risk:** Low

**Implementation:**
```python
# tests/analysis/test_stats.py
import pytest
import pandas as pd
import numpy as np
from clinical_analytics.analysis.stats import logistic_regression, survival_analysis

class TestLogisticRegression:
    """Test logistic regression implementation."""

    @pytest.fixture
    def simple_binary_data(self):
        """Create dataset with known logistic relationship."""
        np.random.seed(42)
        n = 1000

        # Create predictor with known effect size
        X = np.random.normal(0, 1, n)

        # True relationship: log_odds = -1 + 2*X
        # P(Y=1) = 1 / (1 + exp(-(-1 + 2*X)))
        log_odds = -1 + 2 * X
        prob = 1 / (1 + np.exp(-log_odds))
        y = np.random.binomial(1, prob)

        return pd.DataFrame({
            'outcome': y,
            'predictor': X,
            'age': np.random.normal(50, 10, n),  # Covariate
        })

    def test_logistic_regression_recovers_true_coefficients(self, simple_binary_data):
        """Test that logistic regression recovers known coefficients."""
        result = logistic_regression(
            df=simple_binary_data,
            outcome_col='outcome',
            predictors=['predictor'],
            covariates=['age']
        )

        # Check coefficient for predictor is close to true value (2.0)
        predictor_coef = result.params['predictor']
        assert 1.8 <= predictor_coef <= 2.2, \
            f"Expected coefficient ~2.0, got {predictor_coef}"

        # Check intercept is close to true value (-1.0)
        intercept = result.params['Intercept']
        assert -1.3 <= intercept <= -0.7, \
            f"Expected intercept ~-1.0, got {intercept}"

    def test_missing_data_handling(self, simple_binary_data):
        """Test that missing data is handled correctly."""
        # Add missing values
        df_with_missing = simple_binary_data.copy()
        df_with_missing.loc[0:10, 'predictor'] = np.nan

        result = logistic_regression(
            df=df_with_missing,
            outcome_col='outcome',
            predictors=['predictor']
        )

        # Should have dropped rows with missing values
        assert result.nobs == len(simple_binary_data) - 11

    def test_categorical_predictor_encoding(self):
        """Test that categorical variables are encoded correctly."""
        df = pd.DataFrame({
            'outcome': [0, 1, 1, 0, 1, 0] * 50,
            'treatment': ['A', 'B', 'C', 'A', 'B', 'C'] * 50
        })

        result = logistic_regression(
            df=df,
            outcome_col='outcome',
            predictors=['treatment']
        )

        # Should have k-1 dummy variables (2 for 3 categories)
        assert 'treatment_B' in result.params.index
        assert 'treatment_C' in result.params.index
        assert 'treatment_A' not in result.params.index  # Reference category

    def test_convergence_failure_handling(self):
        """Test that convergence failures are handled gracefully."""
        # Create data that causes convergence issues
        df = pd.DataFrame({
            'outcome': [0] * 50 + [1] * 50,  # Perfect separation
            'predictor': list(range(50)) + list(range(50, 100))
        })

        # Should raise informative error or return None
        with pytest.raises(ValueError, match="convergence") or \
             pytest.warns(UserWarning, match="convergence"):
            result = logistic_regression(
                df=df,
                outcome_col='outcome',
                predictors=['predictor']
            )

    def test_multicollinearity_detection(self):
        """Test that multicollinearity is detected and warned."""
        df = pd.DataFrame({
            'outcome': np.random.binomial(1, 0.5, 100),
            'x1': np.random.normal(0, 1, 100),
        })
        df['x2'] = df['x1'] + np.random.normal(0, 0.01, 100)  # Nearly identical

        # Should warn about multicollinearity
        with pytest.warns(UserWarning, match="multicollinearity|correlation"):
            result = logistic_regression(
                df=df,
                outcome_col='outcome',
                predictors=['x1', 'x2']
            )

class TestSurvivalAnalysis:
    """Test survival analysis implementation."""

    @pytest.fixture
    def simple_survival_data(self):
        """Create dataset with known survival relationship."""
        np.random.seed(42)
        n = 200

        # Create time-to-event data
        treatment = np.random.binomial(1, 0.5, n)

        # Exponential survival times (known hazard ratio = 0.5)
        baseline_hazard = 0.1
        hazard = baseline_hazard * np.exp(-0.693 * treatment)  # HR = 0.5
        times = np.random.exponential(1 / hazard)

        # Add censoring
        censoring_time = 10
        observed = times < censoring_time
        times = np.minimum(times, censoring_time)

        return pd.DataFrame({
            'time': times,
            'event': observed.astype(int),
            'treatment': treatment
        })

    def test_kaplan_meier_basic(self, simple_survival_data):
        """Test Kaplan-Meier estimator produces valid survival curve."""
        result = survival_analysis(
            df=simple_survival_data,
            time_col='time',
            event_col='event',
            method='kaplan_meier'
        )

        # Survival probability should start at 1.0
        assert result['survival_prob'].iloc[0] == 1.0

        # Survival probability should decrease monotonically
        assert (result['survival_prob'].diff().dropna() <= 0).all()

    def test_cox_regression_hazard_ratio(self, simple_survival_data):
        """Test Cox regression recovers known hazard ratio."""
        result = survival_analysis(
            df=simple_survival_data,
            time_col='time',
            event_col='event',
            covariates=['treatment'],
            method='cox'
        )

        # Hazard ratio for treatment should be ~0.5
        hr = np.exp(result.params['treatment'])
        assert 0.4 <= hr <= 0.6, f"Expected HR ~0.5, got {hr}"
```

### Solution 2: Property-Based Testing with Hypothesis
**Pros:**
- Tests wide range of inputs automatically
- Finds edge cases humans miss
- Less test code to maintain

**Cons:**
- Harder to debug failures
- May not catch domain-specific issues
- Requires learning Hypothesis framework

**Effort:** Large (10 hours)
**Risk:** Medium

### Solution 3: Integration Tests Only
**Pros:**
- Faster to write
- Tests real workflows
- Less maintenance

**Cons:**
- Doesn't isolate unit failures
- Slower test execution
- Harder to debug

**Effort:** Medium (6 hours)
**Risk:** High (inadequate coverage)

## Recommended Action

**Implement Solution 1** with:

1. **Test fixtures** with known statistical relationships
2. **Unit tests** for each statistical function
3. **Integration tests** for end-to-end workflows
4. **Regression tests** to prevent future breakage
5. **CI/CD integration** to run tests on every commit

**Test Categories:**
- Coefficient recovery tests (known relationships)
- Missing data handling tests
- Categorical variable encoding tests
- Convergence failure tests
- Multicollinearity detection tests
- Edge case tests (empty data, single row, etc.)
- Survival analysis tests (KM, Cox, log-rank)

## Technical Details

**New Test Files:**
```
tests/analysis/
├── __init__.py
├── test_stats.py           # Logistic regression tests
├── test_survival.py        # Survival analysis tests
├── test_profiling.py       # Data profiling tests
├── fixtures/
│   ├── __init__.py
│   ├── regression_data.py  # Test data generators
│   └── survival_data.py    # Survival test data
```

**CI/CD Integration:**
```yaml
# .github/workflows/tests.yml
- name: Run statistical tests
  run: |
    uv run pytest tests/analysis/ --cov=src/clinical_analytics/analysis --cov-fail-under=90
```

**Dependencies to Add:**
```toml
[project.optional-dependencies]
dev = [
    "pytest>=7.4.0",
    "pytest-cov>=4.1.0",
    "hypothesis>=6.0.0",  # Property-based testing
    "scipy>=1.11.0",      # Statistical validation
]
```

## Acceptance Criteria

- [ ] Test coverage for stats.py >= 90%
- [ ] All logistic regression functions have unit tests
- [ ] All survival analysis functions have unit tests
- [ ] Tests verify coefficient recovery on known data
- [ ] Tests verify missing data handling
- [ ] Tests verify categorical variable encoding
- [ ] Tests check for multicollinearity warnings
- [ ] Tests verify convergence failure handling
- [ ] Integration tests for end-to-end workflows
- [ ] CI/CD runs tests on every commit
- [ ] Tests fail if statistical assumptions violated
- [ ] Documentation updated with test data sources

## Work Log

### 2025-12-24
- **Action:** Testing review identified zero coverage for statistical analysis
- **Learning:** Medical research code requires rigorous testing for validity
- **Next:** Create test fixtures with known statistical relationships

## Resources

- **Statsmodels Documentation:** https://www.statsmodels.org/stable/index.html
- **Lifelines Documentation:** https://lifelines.readthedocs.io/
- **Testing Statistical Software:** Journal of Statistical Software best practices
- **Hypothesis Framework:** https://hypothesis.readthedocs.io/
- **FDA Statistical Software Validation:** https://www.fda.gov/regulatory-information/search-fda-guidance-documents/statistical-software-clarifying-acceptability
- **Related Finding:** Performance concerns in stats module (todo #012)
