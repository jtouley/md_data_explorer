---
name: Performance Tracking and Reporting System
overview: Implement comprehensive test performance tracking system with automated monitoring, regression detection, reporting tools, and documentation updates
todos:
  - id: "1"
    content: Create performance tracking module structure (__init__.py, plugin.py, storage.py, regression.py, reporter.py)
    status: completed
  - id: "2"
    content: Implement storage.py - JSON read/write utilities for performance data and baseline
    status: completed
    dependencies:
      - "1"
  - id: "3"
    content: Implement plugin.py - Pytest hooks (pytest_runtest_setup/teardown) to track test durations
    status: completed
    dependencies:
      - "2"
  - id: "4"
    content: Register performance plugin in tests/conftest.py using pytest_configure hook
    status: completed
    dependencies:
      - "3"
  - id: "5"
    content: Implement reporter.py - Generate markdown reports from performance data
    status: completed
    dependencies:
      - "2"
  - id: "6"
    content: Implement regression.py - Compare current performance against baseline with thresholds
    status: completed
    dependencies:
      - "2"
  - id: "7"
    content: Create scripts/generate_performance_report.py - CLI tool for report generation
    status: completed
    dependencies:
      - "5"
      - "6"
  - id: "7.5"
    content: Add integration test - End-to-end workflow (track → report → baseline → regression)
    status: pending
    dependencies:
      - "7"
  - id: "8"
    content: Create tests/test_performance_regression.py - Regression test suite
    status: completed
    dependencies:
      - "6"
      - "7.5"
  - id: "9"
    content: Add performance configuration to pyproject.toml (thresholds, file paths)
    status: completed
  - id: "10"
    content: Add Makefile commands (test-performance, performance-report, performance-baseline, etc.)
    status: completed
    dependencies:
      - "7"
  - id: "11"
    content: Update tests/PERFORMANCE.md with automated tracking, regression, and reporting sections
    status: pending
    dependencies:
      - "5"
      - "6"
      - "7"
  - id: "12"
    content: Add .performance_data.json and .performance_data_worker_*.json to .gitignore
    status: pending
  - id: "13"
    content: Write tests for performance system (test_plugin.py, test_storage.py, test_regression.py, test_reporter.py)
    status: pending
    dependencies:
      - "3"
      - "5"
      - "6"
  - id: "14"
    content: Run full test suite and verify performance tracking works end-to-end
    status: pending
    dependencies:
      - "4"
      - "8"
      - "13"
---

# Perfo

rmance Tracking and Reporting System

## Overview

Implement a comprehensive test performance tracking system that automates performance monitoring, detects regressions, generates reports, and maintains accurate documentation.

## Architecture

```javascript
tests/
├── performance/
│   ├── __init__.py
│   ├── plugin.py              # Pytest plugin for tracking durations
│   ├── storage.py              # JSON storage for performance data
│   ├── regression.py           # Regression test utilities
│   └── reporter.py             # Report generation
├── test_performance_regression.py  # Regression tests
└── conftest.py                 # Add performance tracking hooks

scripts/
└── generate_performance_report.py  # CLI tool for reports

tests/
└── PERFORMANCE.md              # Updated documentation
```



## Implementation Components

### 1. Automated Performance Tracking

**File**: `tests/performance/plugin.py`Pytest plugin that tracks test durations. **Tracking is opt-in via `--track-performance` flag** to avoid overhead on normal test runs.**Core Functionality**:

- Hooks into `pytest_runtest_setup` and `pytest_runtest_teardown` to track durations
- Stores results in JSON format: `tests/.performance_data.json`
- Tracks: test name, duration, markers (slow/integration), module, timestamp, status
- **Excludes `tests/performance/test_*.py` from tracking** to avoid recursive tracking

**pytest-xdist Parallel Execution Strategy**:With pytest-xdist, each worker runs in a separate process with no shared memory. The plugin uses **per-worker file writing with aggregation**:

1. **During test execution**: Each worker writes to separate file: `.performance_data_worker_{worker_id}.json`
2. **After all workers complete**: `pytest_sessionfinish` hook (runs only on master) aggregates all worker files:

- Reads all `.performance_data_worker_*.json` files
- Merges test results (combines all tests from all workers)
- Calculates summary statistics (total tests, slow tests, average duration)
- Writes final `.performance_data.json` file
- Cleans up worker files

**Implementation Pattern**:

```python
def pytest_runtest_setup(item):
    """Track test start time."""
    if not _is_tracking_enabled():
        return
    if _should_exclude_test(item):
        return
    item._performance_start = time.perf_counter()

def pytest_runtest_teardown(item, nextitem):
    """Track test end time and write to worker file."""
    if not _is_tracking_enabled():
        return
    if _should_exclude_test(item):
        return
    duration = time.perf_counter() - item._performance_start
    _write_to_worker_file(item, duration)

def pytest_sessionfinish(session, exitstatus):
    """Aggregate worker files into final performance data."""
    if not _is_tracking_enabled():
        return
    if _is_worker_process():
        return  # Only master aggregates
    _aggregate_worker_files()
    _write_final_performance_data()
```

**Key Features**:

- Session-scoped tracking (one file per test run)
- Aggregates results across parallel workers safely
- Filters by markers (slow/integration)
- Tracks both individual test and suite-level metrics
- Opt-in via `--track-performance` flag

**Integration**:

- Register plugin in `tests/conftest.py` using `pytest_configure` hook (conditional on `--track-performance` flag)
- Store data in `tests/.performance_data.json` (gitignored)
- Worker files (`.performance_data_worker_*.json`) are also gitignored

### 2. Performance Regression Tests

**File**: `tests/test_performance_regression.py`Test suite that validates performance hasn't regressed beyond acceptable thresholds.**Core Functionality**:

- Loads baseline performance data from `tests/.performance_baseline.json`
- Compares current run (from `.performance_data.json`) against baseline
- Fails if performance degrades beyond thresholds:
- Individual tests: >20% slower than baseline (configurable)
- Suite-level: >15% slower than baseline (configurable)
- Provides clear error messages with before/after comparisons showing:
- Test name
- Baseline duration vs. current duration
- Percentage increase
- Threshold exceeded

**Baseline Management Workflow**:

1. **Initial Baseline Creation**:

- Run tests with tracking: `make test-performance`
- Create baseline from current run: `make performance-baseline`
- This creates `tests/.performance_baseline.json` from `.performance_data.json`
- Baseline is stored in git (tracked)

2. **Updating Baseline**:

- After performance improvements or test changes
- Run: `make performance-baseline` (overwrites existing baseline)
- Commit updated baseline to git

3. **Error Handling**:

- If baseline file doesn't exist: Regression tests skip with clear message: "Baseline not found. Run 'make performance-baseline' to create initial baseline."
- If performance data file doesn't exist: Regression tests skip with message: "No performance data found. Run tests with '--track-performance' flag first."

**Baseline Schema** (see Implementation Details section for full schema)**Thresholds**:

- Configurable via `pyproject.toml` `[tool.performance]` section
- Default: 20% for individual tests, 15% for suite-level

**Test Structure**:

```python
@pytest.mark.slow
@pytest.mark.integration
class TestPerformanceRegression:
    def test_individual_test_performance(self):
        """Verify individual tests haven't regressed."""
        
    def test_suite_performance(self):
        """Verify test suite performance."""
        
    def test_slow_test_count(self):
        """Verify slow test count hasn't increased unexpectedly."""
```



### 3. Performance Reporting Tool

**File**: `scripts/generate_performance_report.py`CLI tool that:

- Parses `.performance_data.json` files
- Generates markdown reports
- Updates `tests/PERFORMANCE.md` automatically
- Provides command-line options:
- `--update-docs` - Update PERFORMANCE.md
- `--compare-baseline` - Compare against baseline
- `--format json|markdown` - Output format
- `--threshold SECONDS` - Filter slow tests

**Report Sections**:

- Summary statistics (total tests, slow tests, average duration)
- Slowest tests (>30 seconds)
- Performance trends (if historical data available)
- Regression warnings
- Recommendations

**Makefile Integration**:

- `make performance-report` - Generate report
- `make performance-update-docs` - Update PERFORMANCE.md

### 4. Documentation Updates

**File**: `tests/PERFORMANCE.md`Updates:

- Add "Automated Performance Tracking" section
- Document baseline update process
- Add regression test documentation
- Include CLI tool usage examples
- Add performance data file locations
- Document threshold configuration

**Structure Improvements**:

- Add table of contents
- Organize by component (tracking, regression, reporting)
- Add troubleshooting section
- Include examples for common workflows

## Implementation Details

### Performance Data Schema

```json
{
  "run_id": "2025-01-15T10:30:00",
  "tests": [
    {
      "nodeid": "tests/core/test_mapper.py::TestColumnMapper::test_mapper_initialization_with_config",
      "duration": 30.5,
      "markers": ["slow", "integration"],
      "module": "core",
      "status": "passed"
    }
  ],
  "summary": {
    "total_tests": 150,
    "slow_tests": 12,
    "total_duration": 245.3,
    "average_duration": 1.64
  }
}
```



### Baseline Schema

```json
{
  "baseline_date": "2025-01-15",
  "tests": {
    "tests/core/test_mapper.py::TestColumnMapper::test_mapper_initialization_with_config": {
      "duration": 30.0,
      "threshold": 36.0
    }
  },
  "suite_metrics": {
    "core": {"total_duration": 180.0, "threshold": 207.0}
  }
}
```



### Configuration

Add to `pyproject.toml`:

```toml
[tool.performance]
# Regression thresholds (percentage increase)
individual_test_threshold = 20.0
suite_threshold = 15.0
slow_test_threshold_seconds = 30.0
baseline_file = "tests/.performance_baseline.json"
data_file = "tests/.performance_data.json"
```



## Makefile Commands

Add to `Makefile` (aligned with project conventions):

- `make test-performance` - Run tests with performance tracking enabled (`--track-performance` flag)
- `make performance-report` - Generate performance report from latest run
- `make performance-update-docs` - Update PERFORMANCE.md with current benchmarks
- `make performance-baseline` - Create or update performance baseline from latest run
- `make performance-regression` - Run regression tests only (requires baseline and performance data)

**Implementation**:

```makefile
test-performance: ensure-venv ## Run tests with performance tracking
	@echo "$(GREEN)Running tests with performance tracking...$(NC)"
	$(PYTEST) $(TEST_DIR) -v --track-performance

performance-report: ## Generate performance report
	@echo "$(GREEN)Generating performance report...$(NC)"
	$(PYTHON_RUN) scripts/generate_performance_report.py --format markdown

performance-update-docs: ## Update PERFORMANCE.md with current benchmarks
	@echo "$(GREEN)Updating PERFORMANCE.md...$(NC)"
	$(PYTHON_RUN) scripts/generate_performance_report.py --update-docs

performance-baseline: ## Create or update performance baseline
	@echo "$(GREEN)Creating performance baseline...$(NC)"
	@if [ ! -f tests/.performance_data.json ]; then \
		echo "$(RED)Error: No performance data found. Run 'make test-performance' first.$(NC)"; \
		exit 1; \
	fi
	$(PYTHON_RUN) scripts/generate_performance_report.py --create-baseline

performance-regression: ensure-venv ## Run performance regression tests
	@echo "$(GREEN)Running performance regression tests...$(NC)"
	$(PYTEST) tests/test_performance_regression.py -v
```



## Testing Strategy

**Test the Performance System**:

- Unit tests for storage/parsing logic
- Integration tests for plugin hooks
- Regression tests for regression detection
- CLI tests for report generation

**Test Files**:

- `tests/performance/test_plugin.py`
- `tests/performance/test_storage.py`
- `tests/performance/test_regression.py`
- `tests/performance/test_reporter.py`

## File Changes

### New Files

1. `tests/performance/__init__.py`
2. `tests/performance/plugin.py` - Pytest plugin
3. `tests/performance/storage.py` - JSON storage utilities
4. `tests/performance/regression.py` - Regression detection
5. `tests/performance/reporter.py` - Report generation
6. `tests/test_performance_regression.py` - Regression test suite
7. `scripts/generate_performance_report.py` - CLI tool
8. `tests/performance/test_plugin.py` - Plugin tests
9. `tests/performance/test_storage.py` - Storage tests
10. `tests/performance/test_regression.py` - Regression tests
11. `tests/performance/test_reporter.py` - Reporter tests

### Modified Files

1. `tests/conftest.py` - Register performance plugin
2. `tests/PERFORMANCE.md` - Update with new sections
3. `pyproject.toml` - Add performance configuration
4. `Makefile` - Add performance commands
5. `.gitignore` - Add `.performance_data.json`

## TDD Workflow (MANDATORY)

> **Reference**: [.cursor/rules/104-plan-execution-hygiene.mdc](.cursor/rules/104-plan-execution-hygiene.mdc), [.cursor/rules/101-testing-hygiene.mdc](.cursor/rules/101-testing-hygiene.mdc)**Every component MUST follow this workflow**:

1. **Write Failing Test First (Red Phase)**

- Use AAA pattern (Arrange-Act-Assert)
- Test naming: `test_unit_scenario_expectedBehavior`
- Use shared fixtures from `conftest.py` (check first before creating new fixtures)
- Run test immediately: `make test-fast` or module-specific command
- Verify test fails for the RIGHT reason (not setup error)

2. **Implement Minimum Code (Green Phase)**

- Write only code needed to pass the test
- Keep it simple

3. **Run Test Again (Green Phase)**

- Same command as step 1
- Verify test passes

4. **Fix Quality Issues (Refactor Phase)**

- Run: `make format`
- Run: `make lint-fix`
- Fix any remaining issues manually
- Run: `make type-check` (if applicable)

5. **Run Module Test Suite**

- Use module-specific command: `make test-core`, `make test-analysis`, etc.
- Or: `make test-fast` for quick feedback
- Verify no regressions

6. **Commit with Tests**

- Include both implementation AND tests in same commit
- Commit message format: `feat: [description] - Add comprehensive test suite (X tests passing)`

**Quality Gates Per Phase**:

- `make format` / `make format-check`
- `make lint-fix` / `make lint`
- `make type-check` (for Python code)
- Module-specific test run (e.g., `make test-fast`)

**Makefile Command Usage Per Phase**:

- Todos #2-6: Use `make test-fast` for quick feedback during development
- Todo #13: Use `make test-fast` with filter for performance system tests
- Todo #14: Use `make test` for full suite verification

## Implementation Order

1. **Storage Layer** (`storage.py`) - JSON read/write utilities

- **TDD**: Write `test_storage.py` first with failing tests
- **Success Criteria**: Can read/write JSON, handles missing files gracefully, validates schema

2. **Plugin** (`plugin.py`) - Pytest hooks for tracking

- **TDD**: Write `test_plugin.py` first with failing tests
- **Success Criteria**: Tracks durations accurately, handles parallel execution, doesn't break existing tests, excludes performance system tests

3. **Reporter** (`reporter.py`) - Report generation

- **TDD**: Write `test_reporter.py` first with failing tests
- **Success Criteria**: Generates accurate reports, handles edge cases (no data, empty baseline)

4. **Regression Detection** (`regression.py`) - Comparison logic

- **TDD**: Write `test_regression.py` first with failing tests
- **Success Criteria**: Detects regressions correctly, provides clear error messages with before/after comparisons

5. **CLI Tool** (`generate_performance_report.py`) - Command-line interface

- **TDD**: Write CLI tests first
- **Success Criteria**: All command-line options work, handles errors gracefully

6. **Integration Test** - End-to-end workflow verification

- **New Phase**: Add integration test between CLI tool and regression tests
- **Success Criteria**: Full workflow (track → report → baseline → regression) works end-to-end

7. **Regression Tests** (`test_performance_regression.py`) - Test suite

- **TDD**: Write regression tests following AAA pattern
- **Success Criteria**: Tests fail when performance degrades, skip gracefully when baseline missing

8. **Documentation** (`PERFORMANCE.md`) - Update docs

- **Success Criteria**: All new sections documented, examples included, troubleshooting section added

9. **Tests for Performance System** - Test the tracking system itself

- **TDD**: Already written in steps 1-4 above
- **Success Criteria**: 80%+ test coverage for performance system

10. **Final Integration** - Wire everything together

    - **Success Criteria**: All components work together, full test suite passes

## Success Criteria

### Overall Success Criteria

- [ ] Performance data automatically collected on every test run **when `--track-performance` flag is used**
- [ ] Regression tests fail when performance degrades >20% **with clear error message showing before/after comparison**
- [ ] CLI tool generates accurate reports **with all specified sections populated**
- [ ] PERFORMANCE.md automatically updated with current benchmarks **via `make performance-update-docs`**
- [ ] All components have comprehensive test coverage **≥80% for performance system**

### Per-Component Success Criteria

**Storage (`storage.py`)**:

- [ ] Can read/write JSON files correctly
- [ ] Handles missing files gracefully (returns empty structure, logs warning)
- [ ] Validates schema on load
- [ ] Handles corrupted JSON files (logs error, returns empty structure)

**Plugin (`plugin.py`)**:

- [ ] Tracks durations accurately (within 1ms precision)
- [ ] Handles parallel execution (pytest-xdist) correctly
- [ ] Aggregates worker files properly
- [ ] Doesn't break existing tests (all tests pass with plugin enabled)
- [ ] Excludes `tests/performance/test_*.py` from tracking
- [ ] Only tracks when `--track-performance` flag is set

**Reporter (`reporter.py`)**:

- [ ] Generates accurate markdown reports
- [ ] Handles edge cases (no data, empty baseline)
- [ ] All specified sections populated correctly
- [ ] Updates PERFORMANCE.md correctly when `--update-docs` flag used

**Regression Detection (`regression.py`)**:

- [ ] Detects regressions correctly (>threshold)
- [ ] Provides clear error messages with before/after comparisons
- [ ] Handles missing baseline gracefully (skips with clear message)
- [ ] Calculates percentage increases correctly

**CLI Tool (`generate_performance_report.py`)**:

- [ ] All command-line options work correctly
- [ ] Handles errors gracefully (missing files, invalid arguments)
- [ ] Generates reports in specified format (json/markdown)
- [ ] Creates baseline correctly

**Regression Tests (`test_performance_regression.py`)**:

- [ ] Tests fail when performance degrades beyond threshold
- [ ] Tests skip gracefully when baseline missing
- [ ] Error messages are clear and actionable