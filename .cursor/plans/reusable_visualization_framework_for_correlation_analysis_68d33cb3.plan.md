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
    content: Write comprehensive test suite in tests/ui/test_visualizations.py following TDD workflow (write failing tests first, implement, verify pass)
    status: pending
    dependencies:
      - create_viz_utils
  - id: quality_gates_phase1
    content: Run quality gates for Phase 1 (make format && make lint-fix && make type-check && make test-ui && make check)
    status: pending
    dependencies:
      - create_viz_utils
      - test_visualizations
  - id: quality_gates_phase2
    content: Run quality gates for Phase 2 (make format && make lint-fix && make type-check && make test-ui && make check)
    status: pending
    dependencies:
      - update_correlation_render
      - add_scatter_plots
  - id: extend_chart_spec
    content: Extend ChartSpec to support heatmap and scatter types for correlations
    status: pending
    dependencies:
      - create_viz_utils
  - id: llm_visualization_decision
    content: Implement LLM-driven visualization recommendation function based on query, conversation, and result characteristics
    status: pending
    dependencies:
      - extend_chart_spec
  - id: user_viz_request_extraction
    content: Extract user-requested visualization types from query text (e.g., "show me a scatter plot")
    status: pending
    dependencies:
      - extend_chart_spec
  - id: dynamic_render_selection
    content: Update render_relationship_analysis() to use ChartSpec for dynamic visualization selection
    status: pending
    dependencies:
      - llm_visualization_decision
      - user_viz_request_extraction
      - update_correlation_render
  - id: quality_gates_phase3
    content: Run quality gates for Phase 3 (make format && make lint-fix && make type-check && make test-core && make check)
    status: pending
    dependencies:
      - extend_chart_spec
  - id: quality_gates_phase4
    content: Run quality gates for Phase 4 (make format && make lint-fix && make type-check && make test-core && make check)
    status: pending
    dependencies:
      - llm_visualization_decision
  - id: quality_gates_phase5
    content: Run quality gates for Phase 5 (make format && make lint-fix && make type-check && make test-core && make check)
    status: pending
    dependencies:
      - user_viz_request_extraction
  - id: quality_gates_phase6
    content: Run quality gates for Phase 6 (make format && make lint-fix && make type-check && make test-ui && make check)
    status: pending
    dependencies:
      - dynamic_render_selection
---

#Reusable Visualization Framework for Correlation Analysis

## Overview

Create a flexible, reusable visualization utility framework that provides general-purpose charting functions. These utilities can be used by any analysis type (correlations, comparisons, predictors, etc.) without hardcoding specific visualizations per analysis.

**LLM Integration**: The framework integrates with the LLM query engine to dynamically select visualizations based on:
- User's question and intent
- Conversation history and context
- Result data characteristics
- User-requested visualization types in conversation

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

- **Input**: `corr_matrix: pl.DataFrame` (Polars DataFrame with variable names as index/columns, correlation values as cells) OR `corr_data: list[dict[str, Any]]` with structure `[{"var1": str, "var2": str, "correlation": float}]`, optional `p_values: pl.DataFrame | list[dict[str, Any]] | None`
- **Output**: `matplotlib.figure.Figure | plotly.graph_objects.Figure` (type depends on backend parameter)
- **Type Signature**:
  ```python
  def create_correlation_heatmap(
      corr_matrix: pl.DataFrame | list[dict[str, Any]],
      backend: Literal["matplotlib", "plotly"] = "matplotlib",
      annot: bool = True,
      cmap: str = "coolwarm",
      p_values: pl.DataFrame | list[dict[str, Any]] | None = None,
      figsize: tuple[float, float] = (8, 6),
  ) -> matplotlib.figure.Figure | plotly.graph_objects.Figure:
  ```

- **Configurable**: colormap, annotations, significance markers, size
- **Polars-First**: Accepts `pl.DataFrame`, converts to pandas only at visualization boundary (with `# PANDAS EXCEPTION` comment)
- **Error Handling**: Validates input shape/types, raises `ValueError` for invalid data. Checks plotly availability when `backend="plotly"`, falls back to matplotlib if not installed.

2. **`create_scatter_plot()`** - Scatter plot with optional regression line

- **Input**: `x_data: pl.Series | np.ndarray | list[float]`, `y_data: pl.Series | np.ndarray | list[float]`, `x_label: str`, `y_label: str`
- **Output**: `matplotlib.figure.Figure | plotly.graph_objects.Figure`
- **Type Signature**:
  ```python
  def create_scatter_plot(
      x_data: pl.Series | np.ndarray | list[float],
      y_data: pl.Series | np.ndarray | list[float],
      x_label: str,
      y_label: str,
      backend: Literal["matplotlib", "plotly"] = "matplotlib",
      show_regression: bool = False,
      show_confidence: bool = False,
      figsize: tuple[float, float] = (8, 6),
  ) -> matplotlib.figure.Figure | plotly.graph_objects.Figure:
  ```

- **Configurable**: regression line, confidence intervals, labels
- **Data Fetching Strategy**: Render function fetches cohort data and passes arrays to visualization utility (Option A from review)

3. **`create_distribution_plot()`** - Distribution visualization

- **Input**: `data: pl.Series | np.ndarray | list[float]`, `label: str`
- **Output**: `matplotlib.figure.Figure | plotly.graph_objects.Figure`
- **Type Signature**:
  ```python
  def create_distribution_plot(
      data: pl.Series | np.ndarray | list[float],
      label: str,
      backend: Literal["matplotlib", "plotly"] = "matplotlib",
      plot_type: Literal["histogram", "kde", "box"] = "histogram",
      figsize: tuple[float, float] = (8, 6),
  ) -> matplotlib.figure.Figure | plotly.graph_objects.Figure:
  ```

- **Configurable**: histogram, KDE, box plot

4. **`create_network_graph()`** - Network visualization for correlation clusters (optional, future enhancement)

- **Input**: `corr_data: list[dict[str, Any]]`, `threshold: float`
- **Output**: `matplotlib.figure.Figure | plotly.graph_objects.Figure`

### Key Design Principles

1. **Polars-First**: Functions accept `pl.DataFrame` and `pl.Series` (Polars-first principle)

- Convert to pandas only at visualization boundary when required by matplotlib/seaborn
- Use `# PANDAS EXCEPTION: Required for matplotlib/seaborn visualization` comment
- Accept numpy arrays or lists for flexibility, but prefer Polars types

2. **Flexible Input**: Functions accept either:

- Direct Polars data (`pl.DataFrame`, `pl.Series`) when available
- Numpy arrays or lists when data is pre-extracted
- Correlation data as list of dicts: `[{"var1": str, "var2": str, "correlation": float}]`

3. **Backend Agnostic**: Functions can return either matplotlib or plotly figures

- Parameter: `backend: Literal["matplotlib", "plotly"] = "matplotlib"`
- Default: matplotlib for compatibility
- Check plotly availability when `backend="plotly"`, fallback to matplotlib with warning if not installed

4. **No Hardcoded Logic**: Utilities don't make decisions about:

- When to show visualizations
- Which visualizations to show
- How many visualizations to show
- These decisions stay in render functions

5. **Deterministic**: All functions work with serializable data structures

- No direct access to session state
- Can work with result dictionaries
- Same inputs = same outputs

6. **Error Handling**: Robust validation and clear error messages

- Validate input data shape/types, raise `ValueError` with clear message
- Handle edge cases: empty data, single variable, all NaN correlations
- Check plotly availability, provide fallback or clear error

## Implementation Plan

### Phase 1: Create Visualization Utilities Module (TDD)

**Test-First Workflow**:
1. Write failing tests for `create_correlation_heatmap()` and `create_scatter_plot()` in `tests/ui/test_visualizations.py`
2. Run `make test-ui PYTEST_ARGS="tests/ui/test_visualizations.py -xvs"` to verify tests fail (Red phase)
3. Implement functions with type hints
4. Run `make test-ui PYTEST_ARGS="tests/ui/test_visualizations.py -xvs"` to verify tests pass (Green phase)
5. Run quality gates: `make format && make lint-fix && make type-check && make test-ui`
6. Run `make check` before commit

**File**: `src/clinical_analytics/ui/components/visualizations.py`

**Functions to Create**:
- `create_correlation_heatmap()` - Enhanced heatmap with type hints (see Architecture section)
- `create_scatter_plot()` - Scatter with regression (see Architecture section)
- `_build_matrix_from_corr_data(corr_data: list[dict[str, Any]], variables: list[str]) -> pl.DataFrame` - Helper to convert correlation list to matrix
- `_format_correlation_label(value: float) -> str` - Format correlation value for display (e.g., "0.75" → "0.75**")
- `_get_significance_marker(p_value: float) -> str` - Get significance marker (e.g., "< 0.001" → "***")
- `_check_plotly_available() -> bool` - Check if plotly is installed

**Dependencies**: Add `plotly>=5.0.0` to `pyproject.toml` as optional dependency (in `[project.optional-dependencies]`)

**Success Criteria**:
- ✅ All visualization functions have explicit type hints
- ✅ Functions accept `pl.DataFrame` (Polars-first)
- ✅ Convert to pandas only at visualization boundary with `# PANDAS EXCEPTION` comment
- ✅ All tests pass (unit tests for each function)
- ✅ Quality gates pass (`make check`)
- ✅ No linting errors
- ✅ Type checking passes

### Phase 2: Update Correlation Render Function (TDD)

**Test-First Workflow**:
1. Write tests for updated `render_relationship_analysis()` in `tests/ui/test_ask_questions.py` (or appropriate test file)
2. Run `make test-ui PYTEST_ARGS="tests/ui/test_ask_questions.py::test_render_relationship_analysis -xvs"` to verify tests fail (Red phase)
3. Refactor `render_relationship_analysis()` to use visualization utilities
4. Run tests to verify pass (Green phase)
5. Run quality gates: `make format && make lint-fix && make type-check && make test-ui`
6. Run `make check` before commit

**File**: `src/clinical_analytics/ui/pages/03_💬_Ask_Questions.py`

**Changes**:
- Import visualization utilities: `from clinical_analytics.ui.components.visualizations import create_correlation_heatmap, create_scatter_plot, _build_matrix_from_corr_data`
- Replace inline seaborn heatmap code (lines 896-908) with `create_correlation_heatmap()` call
- Add scatter plots for strong correlations (top 3) using `create_scatter_plot()`
- Remove seaborn import (`import seaborn as sns`) after migration
- Keep all decision logic (when to show, what to show) in render function
- Fetch cohort data in render function and pass arrays to visualization utilities

**Success Criteria**:
- ✅ `render_relationship_analysis()` uses visualization utilities
- ✅ All existing tests pass (no regressions)
- ✅ No seaborn imports remain in file
- ✅ Quality gates pass (`make check`)
- ✅ Visualization output matches previous behavior (visual regression test if possible)

### Phase 3: LLM-Driven ChartSpec Generation for Correlations (TDD)

**Test-First Workflow**:
1. Write failing tests for `generate_chart_spec()` with CORRELATIONS intent in `tests/core/test_chart_spec.py`
2. Run `make test-core PYTEST_ARGS="tests/core/test_chart_spec.py -xvs"` to verify tests fail (Red phase)
3. Extend `ChartSpec` to support correlation visualizations
4. Update `generate_chart_spec()` to handle CORRELATIONS intent
5. Run tests to verify pass (Green phase)
6. Run quality gates: `make format && make lint-fix && make type-check && make test-core`
7. Run `make check` before commit

**File**: `src/clinical_analytics/core/query_plan.py`

**Changes**:
- Extend `ChartSpec.type` to include `"heatmap"` and `"scatter"` for correlations:
  ```python
  type: Literal["bar", "line", "hist", "heatmap", "scatter"]
  ```
- Update `generate_chart_spec()` to handle CORRELATIONS intent:
  ```python
  # CORRELATIONS -> heatmap for matrix, scatter for strong correlations
  if plan.intent == "CORRELATIONS":
      return ChartSpec(
          type="heatmap",
          x=None,
          y=None,
          group_by=None,
          title="Correlation Matrix",
      )
  ```

**Success Criteria**:
- ✅ `ChartSpec` supports `"heatmap"` and `"scatter"` types
- ✅ `generate_chart_spec()` returns ChartSpec for CORRELATIONS intent
- ✅ All existing tests pass (no regressions)
- ✅ Quality gates pass (`make check`)

### Phase 4: LLM Visualization Decision Framework (TDD)

**Test-First Workflow**:
1. Write failing tests for `generate_visualization_recommendation()` in `tests/core/test_visualization_decision.py`
2. Run `make test-core PYTEST_ARGS="tests/core/test_visualization_decision.py -xvs"` to verify tests fail (Red phase)
3. Add `VISUALIZATION_RECOMMENDATION` to `LLMFeature` enum in `src/clinical_analytics/core/llm_feature.py`
4. Add timeout constant to `src/clinical_analytics/core/nl_query_config.py`: `LLM_TIMEOUT_VISUALIZATION_RECOMMENDATION_S = 15.0`
5. Implement LLM-driven visualization recommendation function
6. Run tests to verify pass (Green phase)
7. Run quality gates: `make format && make lint-fix && make type-check && make test-core`
8. Run `make check` before commit

**New File**: `src/clinical_analytics/core/visualization_decision.py`

**Functions to Create**:
- `generate_visualization_recommendation(
    query_text: str,
    query_plan: QueryPlan,
    result: dict[str, Any],
    conversation_history: list[dict] | None = None,
    user_requested_type: str | None = None,
) -> ChartSpec | None` - LLM-driven visualization recommendation
- `_build_visualization_prompt(
    query_text: str,
    query_plan: QueryPlan,
    result_summary: dict[str, Any],
    conversation_history: list[dict] | None,
    user_requested_type: str | None,
) -> tuple[str, str]` - Build LLM prompt for visualization decision
- `_sanitize_result_for_visualization_prompt(result: dict[str, Any]) -> dict[str, Any]` - Sanitize result data for LLM prompt (remove large data, keep summary stats)

**LLMFeature Enum Update**:
- Add to `src/clinical_analytics/core/llm_feature.py`:
  ```python
  VISUALIZATION_RECOMMENDATION = "visualization_recommendation"
  ```

**Timeout Configuration**:
- Add to `src/clinical_analytics/core/nl_query_config.py`:
  ```python
  LLM_TIMEOUT_VISUALIZATION_RECOMMENDATION_S: float = float(os.getenv("LLM_TIMEOUT_VISUALIZATION_RECOMMENDATION_S", "15.0"))
  ```

**Result Sanitization**:
- Follow pattern from `result_interpretation.py`
- Include: `n_observations`, `variables` (list of names), `strong_correlations` (count only), `moderate_correlations` (count only)
- Exclude: Full correlation matrices, raw data, PII, large arrays

**LLM Prompt Strategy**:
```python
system_prompt = """You are a data visualization expert. Recommend the best visualization type based on:
1. User's question and intent
2. Conversation history (what they've been exploring)
3. Result data characteristics (number of variables, correlation strength, etc.)
4. User-requested visualization type (if specified)

Available visualization types:
- heatmap: For correlation matrices (best for 3-20 variables)
- scatter: For exploring relationships between 2 variables (best for strong correlations)
- bar: For comparing groups or categories
- hist: For distributions
- line: For trends over time

Decision Guidelines:
- If user explicitly requests a type, recommend that type
- For correlations with <=6 variables: heatmap is usually best
- For correlations with strong relationships (|r| > 0.7): scatter plots are useful
- For many variables (>20): consider limiting to top correlations only
- Return null if no visualization would add value

Return JSON with:
{
  "visualization_type": "heatmap" | "scatter" | "bar" | "hist" | "line" | null,
  "reasoning": "Brief explanation of why this visualization is recommended",
  "config": {
    "show_regression": true/false (for scatter),
    "annot": true/false (for heatmap),
    "max_variables": number (for heatmap, if limiting display),
    "max_scatters": number (for scatter, if limiting display)
  }
}
"""

user_prompt = f"""Query: {query_text}
Intent: {query_plan.intent}
Result Summary: {result_summary}
Conversation History: {conversation_history}
User Requested Type: {user_requested_type}

Recommend visualization type and configuration."""
```

**Decision Framework Logic**:
1. **User Request Priority**: If user explicitly requests visualization type (e.g., "show me a scatter plot"), honor it
2. **Query Intent**: Use QueryPlan intent to suggest appropriate visualization
3. **Result Characteristics**: Analyze result data (number of variables, correlation strength, etc.)
4. **Conversation Context**: Consider previous visualizations and exploration patterns
5. **Fallback**: Default to deterministic `generate_chart_spec()` if LLM unavailable

**Error Handling**:
- Use `call_llm()` with `feature=LLMFeature.VISUALIZATION_RECOMMENDATION`
- Timeout: `LLM_TIMEOUT_VISUALIZATION_RECOMMENDATION_S = 15.0`
- If LLM fails (unavailable, timeout, JSON parse failure): Fall back to deterministic `generate_chart_spec(plan)`
- Never return `None` - always provide a ChartSpec (either LLM-generated or deterministic)

**Integration Point in `execute_query_plan()`**:
- Location: After `generate_chart_spec()` (after line 1419 in `src/clinical_analytics/core/semantic.py`)
- Integration code:
  ```python
  # After line 1419: chart_spec = {...} if chart_spec_obj else None

  # Phase 4: LLM enhancement (if available and enabled)
  if chart_spec and ENABLE_LLM_VISUALIZATION_RECOMMENDATION:
      # Sanitize result for LLM prompt
      sanitized_result = _sanitize_result_for_visualization_prompt(result_dict)

      # Generate LLM recommendation
      llm_spec = generate_visualization_recommendation(
          query_text=query_text,
          query_plan=plan,
          result=sanitized_result,
          conversation_history=conversation_history,
          user_requested_type=plan.requested_visualization,
      )

      # Override deterministic spec if LLM recommendation exists
      if llm_spec:
          chart_spec = {
              "type": llm_spec.type,
              "x": llm_spec.x,
              "y": llm_spec.y,
              "group_by": llm_spec.group_by,
              "title": llm_spec.title,
              "config": llm_spec.config if hasattr(llm_spec, "config") else {},
          }
  ```
- Merge Strategy: LLM recommendation overrides deterministic spec if both exist
- Config Flag: Add `ENABLE_LLM_VISUALIZATION_RECOMMENDATION` to `nl_query_config.py` (default: `True`)

**Success Criteria**:
- ✅ `VISUALIZATION_RECOMMENDATION` added to `LLMFeature` enum
- ✅ Timeout constant added to `nl_query_config.py`
- ✅ LLM generates visualization recommendations based on context
- ✅ Result sanitization function implemented (removes large data, keeps summary stats)
- ✅ User-requested visualization types are honored
- ✅ Falls back gracefully to deterministic `generate_chart_spec()` if LLM unavailable
- ✅ Error handling covers unavailable, timeout, and JSON parse failure scenarios
- ✅ Integration point in `execute_query_plan()` specified and implemented
- ✅ All tests pass (including LLM mocking and fallback scenarios)
- ✅ Quality gates pass (`make check`)

### Phase 5: User-Requested Visualization Types in Conversation (TDD)

**Test-First Workflow**:
1. Write failing tests for visualization type extraction from queries in `tests/core/test_visualization_decision.py`
2. Run `make test-core PYTEST_ARGS="tests/core/test_visualization_decision.py::test_extract_user_visualization_request -xvs"` to verify tests fail (Red phase)
3. Add `requested_visualization` field to `QueryPlan` dataclass
4. Implement extraction of user-requested visualization types from query text
5. Run tests to verify pass (Green phase)
6. Run quality gates: `make format && make lint-fix && make type-check && make test-core`
7. Run `make check` before commit

**File**: `src/clinical_analytics/core/visualization_decision.py`

**Functions to Create**:
- `extract_user_visualization_request(query_text: str) -> str | None` - Extract visualization type from query text
- Pattern matching for phrases like:
  - "show me a [heatmap/scatter plot/bar chart]"
  - "visualize as [heatmap/scatter/bar]"
  - "I want to see a [heatmap/scatter plot]"
  - "display as [heatmap/scatter]"
  - "as a [heatmap/scatter]"
  - "in a [heatmap/scatter plot]"

**QueryPlan Field Addition**:
- Add to `src/clinical_analytics/core/query_plan.py`:
  ```python
  requested_visualization: str | None = None  # User-requested visualization type (e.g., "scatter", "heatmap")
  ```
- Valid values: `"heatmap" | "scatter" | "bar" | "hist" | "line" | None`
- Update `QueryPlan.from_dict()` to handle this field
- Add validation in `QueryPlan.__post_init__()` to ensure valid values
- Document field in QueryPlan docstring

**Integration Point in `parse_query()`**:
- Location: Early in `NLQueryEngine.parse_query()`, after query normalization, before intent parsing
- Integration code:
  ```python
  # In NLQueryEngine.parse_query(), after query normalization
  normalized_query = normalize_query(query)

  # Phase 5: Extract user-requested visualization type
  requested_viz = extract_user_visualization_request(normalized_query)

  # Continue with existing parsing logic...
  query_intent = self._parse_with_tiers(normalized_query, ...)

  # Store in QueryIntent (add field if needed) or propagate directly to QueryPlan
  ```
- Propagation: Store in `QueryIntent` first (add `requested_visualization: str | None` field), then propagate to `QueryPlan` during `_intent_to_plan()` conversion
- Passed to `generate_visualization_recommendation()` as `user_requested_type` parameter

**Success Criteria**:
- ✅ `requested_visualization` field added to `QueryPlan` dataclass
- ✅ Field validation implemented (valid values only)
- ✅ `QueryPlan.from_dict()` handles new field
- ✅ User-requested visualization types extracted from queries
- ✅ Integration point in `parse_query()` specified and implemented
- ✅ Stored in QueryPlan for downstream use
- ✅ All tests pass (including various phrasings)
- ✅ Quality gates pass (`make check`)

### Phase 6: Dynamic Visualization Selection in Render Functions (TDD)

**Test-First Workflow**:
1. Write failing tests for `render_relationship_analysis()` using ChartSpec in `tests/ui/test_ask_questions.py`
2. Run `make test-ui PYTEST_ARGS="tests/ui/test_ask_questions.py::test_render_relationship_analysis_with_chart_spec -xvs"` to verify tests fail (Red phase)
3. Update `render_relationship_analysis()` to use ChartSpec for dynamic visualization selection
4. Run tests to verify pass (Green phase)
5. Run quality gates: `make format && make lint-fix && make type-check && make test-ui`
6. Run `make check` before commit

**File**: `src/clinical_analytics/ui/pages/03_💬_Ask_Questions.py`

**Changes**:
- Update `render_relationship_analysis()` to read `result.get("chart_spec")`
- Use ChartSpec to determine which visualizations to show:
  ```python
  chart_spec = result.get("chart_spec")
  if chart_spec and chart_spec.get("type") == "heatmap":
      # Show heatmap using create_correlation_heatmap()
  elif chart_spec and chart_spec.get("type") == "scatter":
      # Show scatter plots using create_scatter_plot()
  else:
      # Fallback to default behavior (existing logic)
  ```
- **Multiple Visualization Types**: ChartSpec supports single type only (not list). If LLM recommends multiple types, choose one based on priority:
  1. User-requested type (if specified in QueryPlan)
  2. LLM primary recommendation (first type in LLM response)
  3. Deterministic `generate_chart_spec()` output
- Honor user-requested visualization types from QueryPlan
- Handle ChartSpec config: `chart_spec.get("config", {})` for visualization parameters

**Success Criteria**:
- ✅ Render functions use ChartSpec for visualization selection
- ✅ LLM recommendations are honored
- ✅ User-requested types are respected
- ✅ Multiple visualization type handling implemented (single type priority logic)
- ✅ ChartSpec config parameters are used (show_regression, annot, max_scatters, etc.)
- ✅ Fallback to default behavior if ChartSpec missing
- ✅ All existing tests pass (no regressions)
- ✅ Quality gates pass (`make check`)

## Files to Modify

1. **New**: `src/clinical_analytics/ui/components/visualizations.py`

- Reusable visualization utilities

2. **Modify**: `src/clinical_analytics/ui/pages/3_💬_Ask_Questions.py`

- Update `render_relationship_analysis()` to use utilities
- Import visualization utilities

3. **Modify**: `pyproject.toml`

- Add `plotly>=5.0.0` to `[project.optional-dependencies]` (optional, for interactive charts)
- Default backend is matplotlib for compatibility
- Plotly is checked at runtime, falls back to matplotlib if not installed

4. **New**: `tests/ui/test_visualizations.py`

- Comprehensive test suite for visualization utilities
- Unit tests (fast) and integration tests (slow)
- Follows AAA pattern with descriptive names

5. **Modify**: `src/clinical_analytics/core/query_plan.py`

- Extend `ChartSpec.type` to include `"heatmap"` and `"scatter"`
- Update `generate_chart_spec()` to handle CORRELATIONS intent
- Add `requested_visualization: str | None` field to `QueryPlan` dataclass
- Update `QueryPlan.from_dict()` to handle new field
- Add validation for `requested_visualization` field

6. **Modify**: `src/clinical_analytics/core/llm_feature.py`

- Add `VISUALIZATION_RECOMMENDATION = "visualization_recommendation"` to `LLMFeature` enum

7. **Modify**: `src/clinical_analytics/core/nl_query_config.py`

- Add `LLM_TIMEOUT_VISUALIZATION_RECOMMENDATION_S = 15.0` timeout constant
- Add `ENABLE_LLM_VISUALIZATION_RECOMMENDATION` config flag (default: `True`)

8. **Modify**: `src/clinical_analytics/core/semantic.py`

- Integrate `generate_visualization_recommendation()` call in `execute_query_plan()` (after line 1419)
- Add result sanitization before LLM call
- Implement LLM recommendation override logic

9. **Modify**: `src/clinical_analytics/core/nl_query_engine.py`

- Integrate `extract_user_visualization_request()` call in `parse_query()` (early, after normalization)
- Propagate `requested_visualization` from QueryIntent to QueryPlan

10. **New**: `src/clinical_analytics/core/visualization_decision.py`

- LLM-driven visualization recommendation function
- User visualization request extraction
- Result sanitization for LLM prompts
- Visualization decision framework

11. **New**: `tests/core/test_visualization_decision.py`

- Test suite for LLM visualization decision framework
- Tests for user request extraction (various phrasings)
- Tests for LLM recommendation generation
- Tests for LLM fallback scenarios (unavailable, timeout, JSON parse failure)
- LLM mocking examples

## Example Usage Pattern

### Basic Usage (Phase 2)

```python
from clinical_analytics.ui.components.visualizations import (
    create_correlation_heatmap,
    create_scatter_plot,
    _build_matrix_from_corr_data,
)
import polars as pl
import streamlit as st

def render_relationship_analysis(result: dict) -> None:
    # ... existing text output ...

    # Decision: Show heatmap if <= 6 variables
    if len(variables) <= 6 and corr_data:
        # Build correlation matrix from correlation data (list of dicts)
        corr_matrix = _build_matrix_from_corr_data(corr_data, variables)

        # Create heatmap (accepts pl.DataFrame, converts to pandas internally)
        fig = create_correlation_heatmap(
            corr_matrix,  # pl.DataFrame
            backend="matplotlib",  # or "plotly" for interactive
            annot=True,
            cmap="coolwarm",
        )
        st.pyplot(fig)  # or st.plotly_chart(fig) if backend="plotly"

    # Decision: Show scatter plots for top 3 strong correlations
    if strong_correlations:
        # Fetch cohort data (render function handles data fetching)
        cohort = get_cohort_data()  # Returns pl.DataFrame

        for corr in strong_correlations[:3]:
            # Extract data arrays from Polars DataFrame
            x_data = cohort.select(pl.col(corr["var1"])).to_series().to_numpy()
            y_data = cohort.select(pl.col(corr["var2"])).to_series().to_numpy()

            # Create scatter plot
            fig = create_scatter_plot(
                x_data=x_data,  # numpy array
                y_data=y_data,  # numpy array
                x_label=corr["var1"],
                y_label=corr["var2"],
                show_regression=True,
                backend="matplotlib",
            )
            st.pyplot(fig)
```

### LLM-Driven Usage (Phase 6)

```python
from clinical_analytics.ui.components.visualizations import (
    create_correlation_heatmap,
    create_scatter_plot,
    _build_matrix_from_corr_data,
)
from clinical_analytics.core.visualization_decision import generate_visualization_recommendation
import polars as pl
import streamlit as st

def render_relationship_analysis(
    result: dict,
    query_plan: QueryPlan | None = None,
    query_text: str | None = None,
    conversation_history: list[dict] | None = None,
) -> None:
    # ... existing text output ...

    # Phase 6: Use ChartSpec from result (LLM-generated or deterministic)
    chart_spec = result.get("chart_spec")

    if chart_spec and chart_spec.get("type") == "heatmap":
        # LLM recommended heatmap or user requested it
        if corr_data:
            corr_matrix = _build_matrix_from_corr_data(corr_data, variables)
            fig = create_correlation_heatmap(
                corr_matrix,
                backend=chart_spec.get("backend", "matplotlib"),
                annot=chart_spec.get("config", {}).get("annot", True),
                cmap="coolwarm",
            )
            st.pyplot(fig) if chart_spec.get("backend") != "plotly" else st.plotly_chart(fig)

    elif chart_spec and chart_spec.get("type") == "scatter":
        # LLM recommended scatter plots or user requested them
        if strong_correlations:
            cohort = get_cohort_data()
            max_scatters = chart_spec.get("config", {}).get("max_scatters", 3)
            for corr in strong_correlations[:max_scatters]:
                x_data = cohort.select(pl.col(corr["var1"])).to_series().to_numpy()
                y_data = cohort.select(pl.col(corr["var2"])).to_series().to_numpy()
                fig = create_scatter_plot(
                    x_data=x_data,
                    y_data=y_data,
                    x_label=corr["var1"],
                    y_label=corr["var2"],
                    show_regression=chart_spec.get("config", {}).get("show_regression", True),
                    backend=chart_spec.get("backend", "matplotlib"),
                )
                st.pyplot(fig) if chart_spec.get("backend") != "plotly" else st.plotly_chart(fig)

    else:
        # Fallback to default behavior (Phase 2 logic)
        # ... existing default visualization logic ...
```

### User-Requested Visualization Example

**User Query**: "Show me correlations between age and blood pressure as a scatter plot"

**Flow**:
1. `extract_user_visualization_request()` extracts `"scatter"` from query
2. Stored in `QueryPlan.requested_visualization = "scatter"`
3. `generate_visualization_recommendation()` honors user request
4. `ChartSpec.type = "scatter"` in result
5. `render_relationship_analysis()` shows scatter plot instead of heatmap

## Key Benefits

1. **Flexible**: Any analysis can use any visualization utility
2. **DRY**: No code duplication across render functions
3. **Maintainable**: Visualization improvements benefit all analyses
4. **Deterministic**: Works with serializable result dicts
5. **Extensible**: Easy to add new visualization types
6. **LLM-Driven**: Visualizations adapt to user questions and conversation context
7. **User-Controlled**: Users can request specific visualization types in conversation
8. **Context-Aware**: Visualization recommendations consider conversation history and result characteristics

## Testing Considerations

### Test File Location

**File**: `tests/ui/test_visualizations.py`

### Test Structure (AAA Pattern)

All tests follow Arrange-Act-Assert pattern with descriptive names: `test_unit_scenario_expectedBehavior`

### Test Coverage

#### Unit Tests (Fast)

1. **`test_create_correlation_heatmap_with_polars_dataframe_returns_figure()`**
   - Arrange: Create `pl.DataFrame` correlation matrix
   - Act: Call `create_correlation_heatmap()`
   - Assert: Returns matplotlib Figure, has correct shape

2. **`test_create_correlation_heatmap_with_dict_list_returns_figure()`**
   - Arrange: Create correlation data as list of dicts
   - Act: Call `create_correlation_heatmap()` (should convert internally)
   - Assert: Returns matplotlib Figure

3. **`test_create_correlation_heatmap_plotly_backend_returns_plotly_figure()`**
   - Arrange: Create correlation matrix, set `backend="plotly"`
   - Act: Call `create_correlation_heatmap()`
   - Assert: Returns plotly Figure (or falls back to matplotlib if plotly not installed)

4. **`test_create_correlation_heatmap_empty_data_raises_valueerror()`**
   - Arrange: Empty correlation matrix
   - Act: Call `create_correlation_heatmap()`
   - Assert: Raises `ValueError` with clear message

5. **`test_create_scatter_plot_with_arrays_returns_figure()`**
   - Arrange: Create x/y numpy arrays
   - Act: Call `create_scatter_plot()`
   - Assert: Returns matplotlib Figure

6. **`test_create_scatter_plot_with_regression_includes_line()`**
   - Arrange: Create x/y data, set `show_regression=True`
   - Act: Call `create_scatter_plot()`
   - Assert: Figure contains regression line

7. **`test_build_matrix_from_corr_data_creates_symmetric_matrix()`**
   - Arrange: Correlation data as list of dicts
   - Act: Call `_build_matrix_from_corr_data()`
   - Assert: Returns symmetric `pl.DataFrame` with correct values

#### Integration Tests (Marked Slow)

8. **`test_visualizations_with_real_correlation_data()`** (`@pytest.mark.slow`, `@pytest.mark.integration`)
   - Arrange: Use real correlation data from `compute_relationship_analysis()`
   - Act: Create visualizations
   - Assert: Visualizations render without errors

9. **`test_render_relationship_analysis_uses_visualization_utilities()`** (`@pytest.mark.slow`, `@pytest.mark.integration`)
   - Arrange: Mock result dict with correlation data
   - Act: Call `render_relationship_analysis()`
   - Assert: Visualization utilities are called, no seaborn imports

#### LLM Integration Tests (Phase 4/5)

10. **`test_generate_visualization_recommendation_with_llm_success_returns_chartspec()`** (`@pytest.mark.slow`, `@pytest.mark.integration`)
    - Arrange: Mock `call_llm()` to return valid ChartSpec recommendation
    - Act: Call `generate_visualization_recommendation()`
    - Assert: Returns ChartSpec with correct type

11. **`test_generate_visualization_recommendation_llm_unavailable_falls_back_to_deterministic()`** (`@pytest.mark.slow`, `@pytest.mark.integration`)
    - Arrange: Mock `call_llm()` to return error (unavailable)
    - Act: Call `generate_visualization_recommendation()`
    - Assert: Falls back to `generate_chart_spec()` output

12. **`test_generate_visualization_recommendation_llm_timeout_falls_back_to_deterministic()`** (`@pytest.mark.slow`, `@pytest.mark.integration`)
    - Arrange: Mock `call_llm()` to return timeout
    - Act: Call `generate_visualization_recommendation()`
    - Assert: Falls back to `generate_chart_spec()` output

13. **`test_generate_visualization_recommendation_llm_json_parse_failure_falls_back()`** (`@pytest.mark.slow`, `@pytest.mark.integration`)
    - Arrange: Mock `call_llm()` to return invalid JSON
    - Act: Call `generate_visualization_recommendation()`
    - Assert: Falls back to `generate_chart_spec()` output

14. **`test_extract_user_visualization_request_various_phrasings()`**
    - Arrange: Various query phrasings ("show me a scatter plot", "visualize as heatmap", etc.)
    - Act: Call `extract_user_visualization_request()`
    - Assert: Correct visualization type extracted

15. **`test_extract_user_visualization_request_no_request_returns_none()`**
    - Arrange: Query without visualization request
    - Act: Call `extract_user_visualization_request()`
    - Assert: Returns None

16. **`test_queryplan_requested_visualization_field_validation()`**
    - Arrange: QueryPlan with invalid `requested_visualization` value
    - Act: Create QueryPlan
    - Assert: Raises ValueError or uses default None

#### Deterministic Tests

17. **`test_correlation_heatmap_deterministic_output()`**
    - Arrange: Fixed correlation matrix
    - Act: Create heatmap twice
    - Assert: Figures are identical (compare key properties or hash)

### LLM Mocking Strategy

**For Phase 4/5 Tests**:
```python
from unittest.mock import patch, MagicMock
from clinical_analytics.core.llm_feature import LLMFeature, LLMCallResult

@patch("clinical_analytics.core.visualization_decision.call_llm")
def test_generate_visualization_recommendation_success(mock_call_llm):
    # Arrange: Mock successful LLM call
    mock_call_llm.return_value = LLMCallResult(
        raw_text='{"visualization_type": "scatter", "reasoning": "Strong correlation", "config": {}}',
        payload={"visualization_type": "scatter", "reasoning": "Strong correlation", "config": {}},
        latency_ms=500.0,
        timed_out=False,
        error=None,
    )

    # Act
    result = generate_visualization_recommendation(...)

    # Assert
    assert result is not None
    assert result.type == "scatter"
    mock_call_llm.assert_called_once_with(
        feature=LLMFeature.VISUALIZATION_RECOMMENDATION,
        system=...,
        user=...,
        timeout_s=15.0,
    )
```

**For Fallback Tests**:
```python
@patch("clinical_analytics.core.visualization_decision.call_llm")
@patch("clinical_analytics.core.visualization_decision.generate_chart_spec")
def test_llm_unavailable_falls_back(mock_generate_chart_spec, mock_call_llm):
    # Arrange: LLM unavailable
    mock_call_llm.return_value = LLMCallResult(
        raw_text=None,
        payload=None,
        latency_ms=10.0,
        timed_out=False,
        error="ollama_unavailable",
    )
    mock_generate_chart_spec.return_value = ChartSpec(type="heatmap", ...)

    # Act
    result = generate_visualization_recommendation(...)

    # Assert: Falls back to deterministic
    assert result is not None
    assert result.type == "heatmap"
    mock_generate_chart_spec.assert_called_once()
```

### Test Execution

**During Development**:
```bash
# Run single test file
make test-ui PYTEST_ARGS="tests/ui/test_visualizations.py -xvs"

# Run specific test
make test-ui PYTEST_ARGS="tests/ui/test_visualizations.py::test_create_correlation_heatmap_with_polars_dataframe_returns_figure -xvs"

# Run fast tests only (skip slow)
make test-fast
```

**Before Commit**:
```bash
# Full quality gate
make check
```

### Test Fixtures

Use shared fixtures from `tests/conftest.py`:
- `sample_cohort` - Sample Polars DataFrame for testing
- `sample_correlation_data` - Sample correlation data structure
- `mock_session_state` - Mock Streamlit session state (if needed)

### Test Markers

- `@pytest.mark.slow` - Tests that load data or take >1s
- `@pytest.mark.integration` - Tests that require real data/connections

**Always use both** for data-loading tests: `@pytest.mark.slow` + `@pytest.mark.integration`

## Plan Update Summary

**Date**: 2025-01-27
**Updated Based On**: Plan review feedback

### Changes Made

1. **Added TDD Workflow**: Each phase now specifies test-first development (write failing test → implement → verify pass)

2. **Specified Polars-First Compliance**:
   - Functions accept `pl.DataFrame` and `pl.Series`
   - Convert to pandas only at visualization boundary with `# PANDAS EXCEPTION` comment
   - Updated example code to use Polars syntax

3. **Added Explicit Type Hints**: All functions now have complete type signatures with return types

4. **Added Quality Gates**: Each phase specifies mandatory quality checks (`make format`, `make lint-fix`, `make type-check`, `make test-ui`, `make check`)

5. **Fixed Phase 3**: Removed non-actionable phase, noted that utilities are already available via import

6. **Specified Input/Output Contracts**:
   - Documented exact structure for correlation data (list of dicts)
   - Added helper function: `_build_matrix_from_corr_data()`
   - Documented data fetching strategy (render function fetches, passes arrays)

7. **Added Error Handling**:
   - Plotly availability check with fallback
   - Input validation with clear error messages
   - Edge case handling (empty data, single variable, all NaN)

8. **Added Success Criteria**: Each phase now has explicit success criteria

9. **Expanded Testing Section**:
   - Comprehensive test examples (unit, integration, deterministic)
   - Test file location and structure
   - Test execution commands
   - Test markers and fixtures

10. **Updated Example Code**: Uses Polars syntax (`cohort.select(pl.col(...)).to_series().to_numpy()`)

### LLM Integration Summary (Added 2025-01-27)

**New Phases Added**:
- **Phase 3**: Extend ChartSpec to support correlation visualizations (heatmap, scatter)
- **Phase 4**: LLM-driven visualization decision framework based on query, conversation, and result characteristics
- **Phase 5**: Extract user-requested visualization types from conversation (e.g., "show me a scatter plot")
- **Phase 6**: Dynamic visualization selection in render functions using ChartSpec

**LLM Integration Points**:
1. **Query Parsing**: Extract visualization requests from user queries
2. **Result Processing**: Generate visualization recommendations after result computation
3. **Context Awareness**: Consider conversation history and previous visualizations
4. **User Control**: Honor explicit visualization type requests

**Decision Framework**:
- Priority 1: User-requested visualization type (explicit request)
- Priority 2: LLM recommendation (based on query, context, result characteristics)
- Priority 3: Deterministic fallback (default rules if LLM unavailable)

**Architecture Flow**:
```
User Query → NLQueryEngine.parse_query()
  → Extract visualization request → QueryPlan.requested_visualization
  → Execute Query → Result computed
  → generate_visualization_recommendation() (LLM)
  → ChartSpec generated → result["chart_spec"]
  → render_relationship_analysis() reads ChartSpec
  → Calls visualization utilities dynamically
```

### Plan Update Summary (Review-Based Changes - 2025-01-27)

**Updated Based On**: Plan review feedback addressing blocking issues

**Changes Made**:

1. **Added LLMFeature Enum Value** (Phase 4):
   - Specify adding `VISUALIZATION_RECOMMENDATION = "visualization_recommendation"` to `LLMFeature` enum
   - Add timeout constant: `LLM_TIMEOUT_VISUALIZATION_RECOMMENDATION_S = 15.0` to `nl_query_config.py`
   - Add config flag: `ENABLE_LLM_VISUALIZATION_RECOMMENDATION` (default: `True`)

2. **Added QueryPlan Field Specification** (Phase 5):
   - Specify adding `requested_visualization: str | None = None` to `QueryPlan` dataclass
   - Add validation for valid values: `"heatmap" | "scatter" | "bar" | "hist" | "line" | None`
   - Update `QueryPlan.from_dict()` to handle new field
   - Add validation in `__post_init__()`

3. **Specified Exact Integration Point** (Phase 4):
   - Exact location: After line 1419 in `semantic.py` (after `generate_chart_spec()`)
   - Added code example showing integration with result sanitization
   - Specified merge strategy: LLM recommendation overrides deterministic spec

4. **Added LLM Error Handling** (Phase 4):
   - Timeout: `LLM_TIMEOUT_VISUALIZATION_RECOMMENDATION_S = 15.0`
   - Fallback: Use deterministic `generate_chart_spec()` if LLM fails (unavailable, timeout, JSON parse failure)
   - Never return `None` - always provide ChartSpec

5. **Added Result Sanitization** (Phase 4):
   - Function: `_sanitize_result_for_visualization_prompt()`
   - Follow pattern from `result_interpretation.py`
   - Include: `n_observations`, `variables` (names only), correlation counts
   - Exclude: Full correlation matrices, raw data, PII

6. **Specified Phase 5 Integration Point** (Phase 5):
   - Exact location: Early in `parse_query()`, after query normalization, before intent parsing
   - Added code example showing integration
   - Specified propagation: QueryIntent → QueryPlan during `_intent_to_plan()` conversion

7. **Specified Multiple Visualization Types Handling** (Phase 6):
   - ChartSpec supports single type only (not list)
   - Priority: User request > LLM recommendation > Deterministic
   - Documented decision

8. **Added LLM Mocking Examples** (Testing):
   - Added 7 new test cases for LLM integration (success, unavailable, timeout, JSON parse failure, extraction)
   - Added LLM mocking strategy with code examples
   - Added fallback test scenarios

9. **Updated Files to Modify List**:
   - Added `llm_feature.py` (LLMFeature enum)
   - Added `nl_query_config.py` (timeout and config flag)
   - Added `semantic.py` (integration point)
   - Added `nl_query_engine.py` (Phase 5 integration)

### Ready for Execution

The plan is now **READY TO EXECUTE** (all blocking issues from review addressed). All required specifications are in place for spec-driven execution, including:
- Complete LLM integration with error handling
- Exact integration points with code examples
- Result sanitization for LLM prompts
- QueryPlan field specifications
- Comprehensive test requirements with LLM mocking
