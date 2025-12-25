# Statistical Tests

## Overview

The platform automatically selects appropriate statistical tests based on your question and variable types.

## Test Selection Logic

### Descriptive Statistics

**When Used**: "Descriptive statistics", "summary", "overview"

**Continuous Variables:**

- Mean ± SD
- Median (IQR)
- Min/Max
- Missing count

**Categorical Variables:**

- Frequency counts
- Percentages
- Missing count

### Group Comparisons

**Continuous Outcome:**

- **Two groups**: Independent t-test
- **>2 groups**: One-way ANOVA
- **Post-hoc**: Tukey HSD if ANOVA significant

**Binary Outcome:**

- **Two groups**: Chi-square test (or Fisher's exact if expected counts <5)
- **>2 groups**: Chi-square test

**Effect Sizes:**

- **t-test**: Cohen's d
- **Chi-square**: Odds ratio, relative risk

### Risk Factor Analysis

**When Used**: "What predicts...", "risk factors for..."

**Method**: Logistic regression (binary outcomes) or linear regression (continuous outcomes)

**Output:**

- Odds ratios (logistic) or coefficients (linear)
- 95% confidence intervals
- p-values for each predictor
- Model fit statistics (AUC, R²)

**Variable Selection:**

- All available predictors included by default
- Can specify subset in query

### Survival Analysis

**When Used**: "Survival", "time to event", "Kaplan-Meier"

**Methods:**

- **Univariate**: Kaplan-Meier curves + log-rank test
- **Multivariable**: Cox proportional hazards regression

**Output:**

- Survival curves by group
- Median survival times
- Hazard ratios with 95% CI
- Log-rank p-value

**Assumptions Checked:**

- Proportional hazards (Schoenfeld residuals)
- Warning displayed if violated

### Correlation Analysis

**When Used**: "Correlation", "relationship", "association"

**Methods:**

- **Both continuous**: Pearson correlation
- **Ordinal or non-normal**: Spearman correlation
- **Matrix**: Compute all pairwise correlations

**Output:**

- Correlation coefficient (r or ρ)
- p-value
- Scatter plot with regression line
- Confidence interval

## Assumptions and Checks

### Normality

**Tests**: Shapiro-Wilk (n<50) or Kolmogorov-Smirnov (n≥50)

**Fallback**: If violated, non-parametric alternatives used:

- Mann-Whitney U (instead of t-test)
- Kruskal-Wallis (instead of ANOVA)
- Spearman (instead of Pearson)

### Equal Variances

**Test**: Levene's test

**Fallback**: Welch's t-test if unequal variances

### Sample Size

**Minimum Requirements:**

- t-test: n≥20 per group
- Chi-square: Expected counts ≥5
- Regression: n≥10 per predictor
- Survival: ≥10 events per covariate

**Warnings**: Displayed if requirements not met

### Missing Data

**Handling:**

- **Complete case analysis**: Listwise deletion (default)
- **Warning**: If >10% missing
- **Future**: Multiple imputation for sensitivity analysis

## Interpreting Results

### P-values

- **p<0.001**: Strong evidence against null hypothesis
- **p<0.05**: Statistically significant (conventional threshold)
- **p>0.05**: No significant difference/association

**Important**: Statistical significance ≠ clinical significance. Always consider effect sizes.

### Effect Sizes

**Cohen's d (mean difference):**

- <0.2: Small
- 0.2-0.5: Medium
- >0.8: Large

**Odds Ratio:**

- OR=1: No association
- OR>1: Increased odds
- OR<1: Decreased odds

**Hazard Ratio:**

- HR=1: No difference in hazard
- HR>1: Increased hazard (worse survival)
- HR<1: Decreased hazard (better survival)

### Confidence Intervals

95% CI indicates the range within which the true effect likely lies.

- **CI excludes null value** (e.g., OR=1): Significant
- **CI includes null value**: Not significant

## Multiple Testing Correction

When performing multiple tests, the platform applies Bonferroni correction:

- Adjusted α = 0.05 / number of tests
- Prevents false positives from multiple comparisons
- Displayed in results if applicable

## Reporting Results

Results include all information needed for publication:

- Test statistic and degrees of freedom
- p-value (exact, not just <0.05)
- Effect size with 95% CI
- Sample sizes
- Test assumptions checked

**Example Output:**

```
Independent t-test: t(98) = 3.45, p = 0.001, Cohen's d = 0.69 (95% CI: 0.28-1.10)
Treatment group: 12.3 ± 2.1 (n=50)
Control group: 10.1 ± 2.3 (n=50)
```

## Next Steps

- Learn about [Question-Driven Analysis](question-driven-analysis.md)
- Understand [Result Interpretation](interpreting-results.md)
- Explore example analyses in [Quick Start](../getting-started/quick-start.md)
