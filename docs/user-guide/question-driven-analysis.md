# Question-Driven Analysis

## Overview

Instead of navigating through menus and selecting analysis types, simply ask your question in plain English. The platform uses natural language understanding to:

1. Determine what type of analysis you need
2. Extract relevant variables from your question
3. Run the appropriate statistical test
4. Present results with interpretation

## How It Works

### Three-Tier Query Understanding

**Tier 1: Pattern Matching**

Fast regex-based matching for common query patterns:

- "compare X by Y" → Group comparison
- "what predicts X" → Risk factor analysis
- "survival" → Kaplan-Meier analysis
- "correlation" → Correlation analysis

**Tier 2: Semantic Embeddings**

Uses sentence-transformers to match your question with similar queries:

- Handles variations in phrasing
- Works with synonyms and related terms
- Confidence threshold: 75%

**Tier 3: LLM Fallback** (Optional)

For complex or ambiguous queries:

- Uses local Ollama LLM (privacy-preserving, no data leaves your machine)
- Structured prompts with semantic layer context (RAG)
- Requests structured JSON output
- Automatically installed and configured on first run
- Only used when Tier 1 and Tier 2 fail
- Confidence threshold: 0.5 for parsing, 0.75 for auto-execution

### Variable Matching

The platform fuzzy-matches variable names in your question to actual column names:

- "age" → `age_years`, `patient_age`
- "died" → `mortality`, `death`, `deceased`
- "treatment" → `treatment_arm`, `tx_group`

## Example Queries

### Descriptive Statistics

```
"Descriptive statistics"
"Show me summary statistics"
"Overview of the data"
```

**What it does:** Generates summary statistics for all variables (mean, median, SD for continuous; counts for categorical).

### Group Comparisons

```
"Compare survival by treatment arm"
"Is mortality different between groups?"
"Compare age across treatment arms"
```

**What it does:**

- Continuous outcomes: t-test or ANOVA
- Binary outcomes: Chi-square test
- Displays effect sizes and confidence intervals

### Risk Prediction

```
"What predicts mortality?"
"Risk factors for readmission"
"Predictors of ICU admission"
```

**What it does:** Logistic regression with all available predictors, showing odds ratios and p-values.

### Survival Analysis

```
"Survival analysis"
"Kaplan-Meier curves"
"Time to event by treatment"
```

**What it does:** Kaplan-Meier curves with log-rank test, Cox proportional hazards if covariates are specified.

### Correlations

```
"Correlation between age and outcome"
"Show relationships between variables"
"Association between X and Y"
"How does BMI relate to CD4 counts and statin use?"
```

**What it does:** 
- Automatically filters to numeric variables only (text/categorical variables are excluded)
- Computes correlation matrix for all numeric variables mentioned
- Highlights strong correlations (|r| ≥ 0.5) and moderate correlations (0.3 ≤ |r| < 0.5)
- Shows key findings in plain language
- Displays correlation matrix visualization (for small variable sets)

## Confidence Scores

The platform shows how confident it is in understanding your question:

- **>75%**: High confidence (green checkmark)
- **50-75%**: Medium confidence (yellow warning)
- **<50%**: Low confidence (asks clarifying questions)

You can always verify the interpreted intent before running the analysis.

## Clarifying Questions

If your query is ambiguous, the platform asks follow-up questions:

```
You: "Compare outcomes"

Platform: "Which outcome would you like to compare?"
  • mortality
  • readmission
  • icu_admission

Platform: "Compare by which variable?"
  • treatment_arm
  • age_group
  • sex
```

This ensures you get the right analysis.

## Advanced Features

### Filters

Add conditions to your query:

```
"Compare survival by treatment for patients over 65"
"What predicts mortality in ICU patients only"
```

### Multiple Variables

Specify multiple predictors or outcomes:

```
"Compare mortality and readmission by treatment"
"Risk factors: include age, sex, and comorbidities"
```

### Time Constraints

For survival analysis:

```
"30-day mortality"
"Survival at 1 year"
"Time to readmission within 90 days"
```

## Tips for Better Results

1. **Be Specific**: "Compare mortality by treatment" is better than "Compare groups"
2. **Use Variable Names**: Use terms from your dataset when possible
3. **Start Simple**: Begin with basic queries, then add complexity
4. **Check Intent**: Review the interpreted analysis type before running
5. **Iterate**: Refine your question based on results

## Your Data is Saved

**Don't worry about losing your work:** Your uploaded data is automatically saved on your computer. You can:

- Close the browser and come back later - your data is still there
- Refresh the page - your analysis continues where you left off
- Work on the same dataset across multiple sessions
- Your query history is preserved (coming soon)

You only need to upload your data once.

## Fallback to Structured Questions

If the NL query engine can't understand your question (confidence <50%), you can use structured questions:

1. Select analysis type from dropdown
2. Choose variables explicitly
3. Set filters manually

This ensures you can always complete your analysis, even for complex or unusual requests.

## Next Steps

- Learn about [Statistical Tests](statistical-tests.md)
- Understand [Result Interpretation](interpreting-results.md)
- Explore the [NL Query Engine Architecture](../architecture/nl-query-engine.md)
