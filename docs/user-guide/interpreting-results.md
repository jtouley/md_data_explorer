# Interpreting Results

## Plain Language Summaries

Every analysis includes a plain language summary that explains:

1. What test was performed
2. What the results mean
3. Whether findings are statistically significant
4. Clinical interpretation guidance

## Example Interpretations

### Group Comparison

**Statistical Output:**

```
Independent t-test: t(98) = 3.45, p = 0.001, Cohen's d = 0.69
Treatment A: 12.3 ± 2.1 (n=50)
Treatment B: 10.1 ± 2.3 (n=50)
```

**Plain Language:**

> Patients in Treatment A had significantly higher scores (mean 12.3) compared to Treatment B (mean 10.1). This difference is statistically significant (p = 0.001) with a moderate-to-large effect size (d = 0.69), suggesting a clinically meaningful difference of about 2.2 points between groups.

### Risk Factor Analysis

**Statistical Output:**

```
Logistic Regression Results:
Age: OR = 1.05 (95% CI: 1.02-1.08), p = 0.001
Treatment A vs B: OR = 0.42 (95% CI: 0.21-0.84), p = 0.014
Sex (Male vs Female): OR = 1.23 (95% CI: 0.67-2.26), p = 0.50
```

**Plain Language:**

> Age is a significant predictor of mortality. Each additional year of age increases the odds of mortality by 5% (OR=1.05, p=0.001). Treatment A reduces mortality risk by 58% compared to Treatment B (OR=0.42, p=0.014). Sex was not significantly associated with mortality (p=0.50).

### Survival Analysis

**Statistical Output:**

```
Kaplan-Meier Analysis:
Log-rank test: χ²(1) = 8.45, p = 0.004
Median survival:
  Treatment A: 385 days (95% CI: 320-450)
  Treatment B: 280 days (95% CI: 210-350)
```

**Plain Language:**

> Treatment A demonstrated significantly better survival than Treatment B (log-rank p = 0.004). Median survival was 385 days for Treatment A versus 280 days for Treatment B, representing a clinically important difference of 105 days.

## Visualizations

### What to Look For

**Box Plots (Group Comparisons):**

- Overlapping boxes suggest small difference
- Non-overlapping notches suggest significant difference
- Look for outliers (dots beyond whiskers)

**Survival Curves:**

- Curves that diverge suggest differential survival
- Wider confidence bands indicate more uncertainty
- Look for curve crossings (may violate proportional hazards)

**Forest Plots (Risk Factors):**

- Points to the right of 1.0 (OR/HR) indicate increased risk
- Confidence intervals crossing 1.0 are not significant
- Longer bars indicate more uncertainty

**Scatter Plots (Correlations):**

- Tight clustering around line indicates strong correlation
- Spread indicates weak correlation
- Look for outliers or non-linear patterns

## Statistical vs. Clinical Significance

### Statistical Significance

- **Definition**: Unlikely to have occurred by chance (p<0.05)
- **Depends on**: Sample size, variability

### Clinical Significance

- **Definition**: Large enough difference to matter in practice
- **Depends on**: Effect size, clinical context

**Example:**

```
Age difference: 0.5 years, p = 0.001 (n=10,000)
```

- **Statistically significant**: Yes (p<0.05)
- **Clinically significant**: Probably not (0.5 years is trivial)

### Guidelines

**Small Sample (n<50):**

- Large effects needed for significance
- Lack of significance doesn't prove no effect
- Consider confidence intervals

**Large Sample (n>500):**

- Even tiny effects become significant
- Focus on effect sizes, not just p-values
- Consider clinical meaningfulness

## Common Pitfalls

### Multiple Testing

When performing many tests:

- Some will be significant by chance (5% false positive rate)
- Look for consistent patterns across related tests
- Apply corrections (Bonferroni) when appropriate

### Confounding

An apparent association may be due to a third variable:

- Example: Coffee → Cancer could be confounded by smoking
- Use multivariable models to adjust for confounders
- Be cautious with observational data

### Causation vs. Association

**Association**: Two variables are related

**Causation**: One variable causes changes in another

**Requirements for causation:**

1. Association (statistical)
2. Temporal precedence (cause before effect)
3. No plausible alternative explanation
4. Dose-response relationship
5. Biological plausibility

Most platform analyses show **association only**. Randomized controlled trials are needed to establish causation.

### Overfitting

In risk factor analysis with many predictors:

- Model may fit noise, not true patterns
- Look for cross-validation or split-sample validation
- Be skeptical of perfect predictions (AUC=1.0)

## Reporting in Publications

Include the following:

1. **Sample size**: Total n and n per group
2. **Descriptive statistics**: Mean±SD or median (IQR)
3. **Test used**: e.g., "Independent t-test" or "Chi-square test"
4. **Test statistic**: e.g., t(98) = 3.45
5. **P-value**: Exact value, not just <0.05
6. **Effect size**: Cohen's d, OR, HR with 95% CI
7. **Assumptions**: Note if violated and how handled

**Example:**

> We compared mortality between treatment groups using a chi-square test. Mortality was significantly lower in Treatment A (12%, 6/50) versus Treatment B (32%, 16/50), χ²(1) = 5.81, p = 0.016, OR = 0.29 (95% CI: 0.10-0.82), representing a 71% reduction in mortality risk.

## When to Seek Expert Help

Consult a statistician if:

- Sample size is very small (n<30)
- Assumptions are severely violated
- Missing data >20%
- Complex interactions suspected
- Time-varying covariates in survival models
- Preparing for publication in high-impact journal

## Next Steps

- Review [Statistical Tests](statistical-tests.md) to understand methods
- Practice with example datasets in [Quick Start](../getting-started/quick-start.md)
- Explore [Question-Driven Analysis](question-driven-analysis.md) for advanced queries
