# Clinical Analytics Platform

**Spec-driven clinical analytics with question-driven semantic analysis**

## Overview

The Clinical Analytics Platform enables clinicians and researchers to analyze clinical datasets using natural language queries. Just upload your data and ask questions - no manual configuration required.

### Key Features

- **Question-Driven Analysis**: Ask questions in plain English instead of clicking through menus
- **Automatic Schema Inference**: Upload any CSV/Excel file - we detect patient IDs, outcomes, and time variables automatically
- **Semantic Layer**: Config-driven analysis that understands your data structure
- **Multi-Table Support**: Handle complex datasets like MIMIC-IV with automatic relationship detection
- **Statistical Analysis**: Descriptive statistics, group comparisons, survival analysis, risk prediction, and correlations

## Quick Start

```bash
# Install
pip install clinical-analytics

# Run
clinical-analytics serve

# Upload your data and start asking questions!
```

### Example Queries

- "Compare survival by treatment arm"
- "What predicts mortality?"
- "Show me correlation between age and outcome"
- "Descriptive statistics for all patients"

## Architecture

The platform combines:

1. **Natural Language Query Engine**: Three-tier parsing (pattern matching → semantic embeddings → LLM fallback)
2. **Semantic Layer**: Ibis-based SQL generation with outcome definitions
3. **Dataset Registry**: Automatic schema inference and multi-table handling
4. **Statistical Engine**: Comprehensive clinical analytics suite

## Documentation Structure

- **[Getting Started](getting-started/installation.md)**: Installation and initial setup
- **[User Guide](user-guide/question-driven-analysis.md)**: How to use the platform
- **[Architecture](architecture/overview.md)**: System design and components
- **[API Reference](api-reference/core.md)**: Developer documentation
- **[Development](development/contributing.md)**: Contributing guidelines

## Vision

Transform clinical data analysis from manual configuration and menu navigation to intelligent, question-driven insights. The platform should:

1. **Understand Intent**: Parse natural language questions to determine analysis type
2. **Infer Structure**: Automatically detect dataset schema without manual mapping
3. **Execute Analysis**: Generate appropriate statistical tests and visualizations
4. **Explain Results**: Provide plain language interpretation of findings

## Status

Currently in active development. Phase 0 (data upload and basic analysis) is complete. Phase 1 (question-driven NL queries) is in progress.

See [Implementation Plan](implementation/IMPLEMENTATION_PLAN.md) for roadmap details.
