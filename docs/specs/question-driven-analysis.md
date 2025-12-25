# Question-Driven Analysis Specification

## Overview

Transform the platform from menu-driven to question-driven analysis where users ask questions in natural language instead of selecting analysis types from dropdowns.

## Problem Statement

Current workflow requires:

1. User selects "Compare Groups" from menu
2. User selects outcome variable
3. User selects grouping variable
4. User clicks "Run Analysis"

**Pain Points:**

- Requires knowledge of statistical test names
- Multiple clicks to specify intent
- Not intuitive for clinicians

## Proposed Solution

Users type free-form questions:

```
"Compare survival by treatment arm"
→ Automatically infers: group comparison, survival outcome, treatment grouping
```

## Requirements

### Functional Requirements

**FR1: Natural Language Query Parsing**

- Accept free-form text input
- Parse into structured QueryIntent
- Extract variables mentioned in query
- Confidence score > 75% for automatic execution

**FR2: Intent Classification**

Support these analysis types:

- DESCRIBE: Descriptive statistics
- COMPARE_GROUPS: Group comparisons
- FIND_PREDICTORS: Risk factor analysis
- SURVIVAL: Survival analysis
- CORRELATIONS: Correlation analysis

**FR3: Variable Extraction**

- Fuzzy match query terms to column names
- Handle synonyms (age → age_years, died → mortality)
- Extract outcome, grouping, predictor variables

**FR4: Confidence Scoring**

- >75%: Auto-execute
- 50-75%: Show interpretation, ask confirmation
- <50%: Ask clarifying questions

**FR5: Fallback to Structured Input**

If NL parsing fails, allow:

- Selection of analysis type from dropdown
- Explicit variable selection
- Manual filter specification

### Non-Functional Requirements

**NFR1: Performance**

- Query parsing: <500ms
- Total time to results: <5s

**NFR2: Privacy**

- Tier 1 & 2: Fully local (no API calls)
- Tier 3 (LLM fallback): Only anonymized metadata sent

**NFR3: Accuracy**

- 85%+ correct intent classification on test set
- 90%+ correct variable extraction

**NFR4: Usability**

- Users can complete analysis without training
- Clear explanation of interpreted intent
- Easy correction of misinterpretations

## Architecture

### Three-Tier Parsing

**Tier 1: Pattern Matching (Regex)**

- Fast (<1ms)
- High precision (>95%)
- Limited coverage (~40% of queries)

**Tier 2: Semantic Embeddings**

- Medium speed (~50ms)
- Good precision (~85%)
- Wide coverage (~80% of queries)

**Tier 3: LLM Fallback**

- Slower (~2-5s)
- Variable precision (~70%)
- Full coverage (100%)

### Components

1. **NLQueryEngine**: Main parsing class
2. **QueryIntent**: Structured intent data class
3. **TemplateLibrary**: Pre-computed query templates
4. **VariableMatcher**: Fuzzy variable name matching

## User Interface

### Query Input

```
┌────────────────────────────────────────────┐
│ 💬 Ask your question                       │
│                                             │
│ ┌─────────────────────────────────────────┐│
│ │ e.g., compare survival by treatment    ││
│ └─────────────────────────────────────────┘│
│                                             │
│ Examples:                                   │
│ • Compare survival by treatment arm         │
│ • What predicts mortality?                  │
│ • Show correlation between age and outcome  │
└────────────────────────────────────────────┘
```

### Intent Confirmation

```
┌────────────────────────────────────────────┐
│ ✅ I understand! (Confidence: 92%)         │
│                                             │
│ 🔍 How I interpreted your question:        │
│                                             │
│ Analysis Type: Compare Groups               │
│ Primary Variable: survival_days             │
│ Grouping Variable: treatment_arm            │
│                                             │
│ [✓ Looks good] [✗ Let me clarify]         │
└────────────────────────────────────────────┘
```

## Example Queries

### Valid Queries

- "Compare survival by treatment arm"
- "What predicts mortality?"
- "Show me correlation between age and outcome"
- "Descriptive statistics"
- "Survival analysis stratified by treatment"
- "Risk factors for readmission"

### Ambiguous Queries (Require Clarification)

- "Compare outcomes" → Which outcome? By which variable?
- "Show differences" → Between what groups?
- "Predictors" → Predictors of what outcome?

## Testing

### Test Cases

**TC1: Basic Group Comparison**

```
Input: "compare mortality by treatment"
Expected: QueryIntent(
    intent_type="COMPARE_GROUPS",
    primary_variable="mortality",
    grouping_variable="treatment_arm",
    confidence > 0.9
)
```

**TC2: Risk Factor Analysis**

```
Input: "what predicts mortality"
Expected: QueryIntent(
    intent_type="FIND_PREDICTORS",
    primary_variable="mortality",
    predictor_variables=[all available],
    confidence > 0.9
)
```

**TC3: Synonym Handling**

```
Input: "compare death by tx"
Expected: QueryIntent(
    intent_type="COMPARE_GROUPS",
    primary_variable="mortality",  # matched "death"
    grouping_variable="treatment_arm",  # matched "tx"
    confidence > 0.8
)
```

### Acceptance Criteria

- [ ] 85%+ accuracy on 50-query test set
- [ ] <500ms parsing latency (95th percentile)
- [ ] Fallback to structured input works
- [ ] User testing shows improved UX vs menu-driven

## Implementation Plan

1. **Phase 1: Core NL Engine** (3 days)
   - NLQueryEngine class
   - Tier 1 pattern matching
   - Tier 2 semantic embeddings

2. **Phase 2: UI Integration** (2 days)
   - Free-form text input component
   - Intent confirmation UI
   - Clarification question flow

3. **Phase 3: Testing** (2 days)
   - Unit tests for parsing
   - Integration tests end-to-end
   - User acceptance testing

## Future Enhancements

- **Multi-step queries**: "Compare survival by treatment, then stratify by age"
- **Conditional filters**: "Show mortality for ICU patients only"
- **Time constraints**: "30-day mortality" → extract timeframe
- **Learning from corrections**: Update weights based on user feedback

## References

- [NL Query Best Practices](../research/NL_QUERY_BEST_PRACTICES.md)
- [Architecture Overview](../architecture/overview.md)
- [NL Query Engine](../architecture/nl-query-engine.md)
