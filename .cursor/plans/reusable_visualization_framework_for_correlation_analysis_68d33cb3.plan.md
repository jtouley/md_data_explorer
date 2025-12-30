---
name: Reusable Visualization Framework for Correlation Analysis
overview: Create a flexible, reusable visualization utility framework that provides general-purpose charting functions (heatmaps, scatter plots, etc.) that any analysis type can use, rather than hardcoding specific visualizations per analysis type.
todos:
  - id: create_viz_utils
    content: Create src/clinical_analytics/ui/components/visualizations.py with reusable visualization utilities (create_correlation_heatmap, create_scatter_plot)
    status: pending
  - id: add_plotly_dependency
    content: Add plotly>=5.0.0 to pyproject.toml dependencies (optional or required based on default backend choice)
    status: pending
  - id: update_correlation_render
    content: Update render_relationship_analysis() to use visualization utilities instead of inline seaborn code
    status: pending
    dependencies:
      - create_viz_utils
  - id: add_scatter_plots
    content: Add scatter plot visualizations for strong correlations in render_relationship_analysis()
    status: pending
    dependencies:
      - create_viz_utils
      - update_correlation_render
  - id: test_visualizations
    content: Test visualization utilities with various inputs and ensure deterministic output
    status: pending
    dependencies:
      - create_viz_utils
      - update_correlation_render
---

#Reusable Visualization Framework for Correlation Analysis

## Overview

Create a flexible, reusable visualization utility framework that provides general-purpose charting functions. These utilities can be used by any analysis type (correlations, comparisons, predictors, etc.) without hardcoding specific visualizations per analysis.

## Current State

- Correlation analysis has basic matplotlib/seaborn heatmap in expander
- Other analyses (survival, predictors) have basic matplotlib plots
- No reusable visualization utilities
- Visualizations are hardcoded in each render function

## Architecture

### New Module: `src/clinical_analytics/ui/components/visualizations.py`

Create a new module with reusable visualization functions that:

- Work with serializable result dictionaries (maintains deterministic framework)
- Support both matplotlib/seaborn (static) and plotly (interactive) backends
- Are configurable but not opinionated about when/how to use them
- Can be imported and used by any render function

### Visualization Utilities to Create

1. **`create_correlation_heatmap()`** - Enhanced heatmap with better styling

- Input: correlation matrix (DataFrame or dict), optional p-values
- Output: matplotlib Figure or plotly Figure
- Configurable: colormap, annotations, significance markers, size

2. **`create_scatter_plot()`** - Scatter plot with optional regression line

- Input: x/y data (arrays or column names + DataFrame)
- Output: matplotlib Figure or plotly Figure
- Configurable: regression line, confidence intervals, labels

3. **`create_distribution_plot()`** - Distribution visualization

- Input: data array or column name + DataFrame
- Output: matplotlib Figure or plotly Figure
- Configurable: histogram, KDE, box plot

4. **`create_network_graph()`** - Network visualization for correlation clusters (optional)

- Input: correlation data, threshold
- Output: matplotlib Figure or plotly Figure

### Key Design Principles

1. **Flexible Input**: Functions accept either:

- Direct data (arrays, DataFrames) when available
- Column names + DataFrame when data needs to be fetched
- Result dict keys when working with serialized results

2. **Backend Agnostic**: Functions can return either matplotlib or plotly figures

- Parameter: `backend="matplotlib"` or `backend="plotly"`
- Default: matplotlib for compatibility

3. **No Hardcoded Logic**: Utilities don't make decisions about:

- When to show visualizations
- Which visualizations to show
- How many visualizations to show
- These decisions stay in render functions

4. **Deterministic**: All functions work with serializable data structures

- No direct access to session state
- Can work with result dictionaries
- Same inputs = same outputs

## Implementation Plan

### Phase 1: Create Visualization Utilities Module

**File**: `src/clinical_analytics/ui/components/visualizations.py`Create reusable functions:

- `create_correlation_heatmap()` - Enhanced heatmap
- `create_scatter_plot()` - Scatter with regression
- Helper functions for styling, formatting

**Dependencies**: Add `plotly>=5.0.0` to `pyproject.toml` (optional, for interactive charts)

### Phase 2: Update Correlation Render Function

**File**: `src/clinical_analytics/ui/pages/3_💬_Ask_Questions.py`Update `render_relationship_analysis()` to:

- Use `create_correlation_heatmap()` instead of inline seaborn code
- Optionally use `create_scatter_plot()` for strong correlations
- Keep all decision logic (when to show, what to show) in render function

### Phase 3: Make Utilities Available to Other Analyses

Other render functions can now import and use:

- `create_scatter_plot()` for predictor analysis
- `create_distribution_plot()` for descriptive analysis
- `create_correlation_heatmap()` for any correlation-like data

## Files to Modify

1. **New**: `src/clinical_analytics/ui/components/visualizations.py`

- Reusable visualization utilities

2. **Modify**: `src/clinical_analytics/ui/pages/3_💬_Ask_Questions.py`

- Update `render_relationship_analysis()` to use utilities
- Import visualization utilities

3. **Modify**: `pyproject.toml`

- Add `plotly>=5.0.0` as optional dependency (or make it required if we want interactive by default)

## Example Usage Pattern

```python
from clinical_analytics.ui.components.visualizations import (
    create_correlation_heatmap,
    create_scatter_plot,
)

def render_relationship_analysis(result: dict) -> None:
    # ... existing text output ...
    
    # Decision: Show heatmap if <= 6 variables
    if len(variables) <= 6 and corr_data:
        corr_matrix = _build_matrix_from_corr_data(corr_data, variables)
        fig = create_correlation_heatmap(
            corr_matrix,
            backend="matplotlib",  # or "plotly" for interactive
            annot=True,
            cmap="coolwarm",
        )
        st.pyplot(fig)  # or st.plotly_chart(fig)
    
    # Decision: Show scatter plots for top 3 strong correlations
    if strong_correlations:
        for corr in strong_correlations[:3]:
            # Need to fetch data - render function handles this
            fig = create_scatter_plot(
                x_data=cohort[corr["var1"]],
                y_data=cohort[corr["var2"]],
                x_label=corr["var1"],
                y_label=corr["var2"],
                show_regression=True,
            )
            st.pyplot(fig)
```



## Key Benefits

1. **Flexible**: Any analysis can use any visualization utility
2. **DRY**: No code duplication across render functions
3. **Maintainable**: Visualization improvements benefit all analyses
4. **Deterministic**: Works with serializable result dicts
5. **Extensible**: Easy to add new visualization types

## Testing Considerations

- Test utilities with various input formats