---
name: Comprehensive Test Performance System
overview: Implement comprehensive test performance system that both tracks performance AND optimizes slow tests. Includes automated monitoring, regression detection, reporting tools, LLM mocking, test data caching, selective dataset loading, fixture scope optimization, and documentation updates.

**Note**: All test commands run in parallel by default (`-n auto`) for maximum speed. Integration tests run serially to avoid conflicts with external services.
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
  - id: "15"
    content: Phase 2.1: Add mock_llm_calls fixture to conftest.py for unit tests
    status: completed
    dependencies:
      - "14"
  - id: "16"
    content: Phase 2.1: Refactor test_nl_query_refinement.py to use mocked LLM (create integration test file for real LLM)
    status: completed
    dependencies:
      - "15"
  - id: "17"
    content: Phase 2.1: Refactor test_nl_query_engine_filter_extraction.py to use mocked LLM
    status: completed
    dependencies:
      - "15"
  - id: "18"
    content: Phase 2.1: Verify LLM unit tests run <1s each (30-50x speedup)
    status: completed
    dependencies:
      - "16"
      - "17"
  - id: "19"
    content: Phase 2.2: Create tests/fixtures/cache.py with content-based hashing for test data caching
    status: pending
    dependencies:
      - "18"
  - id: "20"
    content: Phase 2.2: Implement DataFrame caching (parquet files in tests/.test_cache/)
    status: pending
    dependencies:
      - "19"
  - id: "21"
    content: Phase 2.2: Implement Excel file caching for expensive fixtures
    status: pending
    dependencies:
      - "19"
  - id: "22"
    content: Phase 2.2: Update fixtures to use cached data when available
    status: pending
    dependencies:
      - "20"
      - "21"
  - id: "23"
    content: Phase 2.2: Measure caching impact (target: 50-80% reduction in data loading time)
    status: pending
    dependencies:
      - "22"
  - id: "24"
    content: Phase 2.3: Enhance discovered_datasets fixture to support lazy loading
    status: pending
    dependencies:
      - "23"
  - id: "25"
    content: Phase 2.3: Add get_dataset_by_name() helper for selective dataset loading
    status: pending
    dependencies:
      - "24"
  - id: "26"
    content: Phase 2.3: Update tests to request specific datasets instead of loading all
    status: pending
    dependencies:
      - "25"
  - id: "27"
    content: Phase 2.3: Measure selective loading impact (target: 30-50% reduction in setup time)
    status: pending
    dependencies:
      - "26"
  - id: "28"
    content: Phase 2.4: Profile fixtures to identify expensive ones using performance tracking data
    status: pending
    dependencies:
      - "14"
  - id: "29"
    content: Phase 2.4: Convert expensive immutable fixtures to module/session scope
    status: pending
    dependencies:
      - "28"
  - id: "30"
    content: Phase 2.4: Verify test isolation maintained after scope changes
    status: pending
    dependencies:
      - "29"
  - id: "31"
    content: Phase 2.4: Measure scope optimization impact (target: 10-30% overall speedup)
    status: pending
    dependencies:
      - "30"
  - id: "32"
    content: Phase 2.5: Audit tests for parallel-safety (file I/O, shared state)
    status: pending
    dependencies:
      - "31"
  - id: "33"
    content: Phase 2.5: Mark non-parallel-safe tests with @pytest.mark.serial
    status: pending
    dependencies:
      - "32"
  - id: "34"
    content: Phase 2.5: Optimize Makefile parallel execution commands
    status: pending
    dependencies:
      - "33"
  - id: "35"
    content: Phase 3: Create scripts/categorize_slow_tests.py using performance tracking data
    status: pending
    dependencies:
      - "14"
  - id: "36"
    content: Phase 3: Auto-categorize tests >30s with @pytest.mark.slow
    status: pending
    dependencies:
      - "35"
  - id: "37"
    content: Phase 3: Generate report of uncategorized slow tests
    status: pending
    dependencies:
      - "36"
  - id: "38"
    content: Final: Measure total performance improvement and update PERFORMANCE.md with results
    status: pending
    dependencies:
      - "18"
      - "23"
      - "27"
      - "31"
      - "34"
      - "37"
---

# Comprehensive Test Performance System

## Overview

Implement a comprehensive test performance system that **both tracks performance AND optimizes slow tests**. This plan includes:

1. **Phase 1: Performance Tracking & Monitoring** - Automated tracking, regression detection, reporting
2. **Phase 2: Actual Performance Optimizations** - Mock LLM calls, test data caching, selective loading, fixture optimization
3. **Phase 3: Automated Categorization** - Use tracking data to improve test-fast effectiveness

**Key Goal**: Not just track slow tests, but actually **fix them** to achieve measurable speedups (30-50x for LLM tests, 50-80% for data loading, 10-30% overall).

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



## Implementation Phases

### Phase 1: Performance Tracking & Monitoring

#### 1. Automated Performance Tracking

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

#### 2. Performance Regression Tests

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



#### 3. Performance Reporting Tool

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

#### 4. Documentation Updates

### Phase 2: Actual Performance Optimizations (CRITICAL - FIXES SLOW TESTS)

**Goal**: Actually optimize slow tests, not just track them. Achieve measurable speedups through mocking, caching, and scope optimization.

#### 2.1: Mock LLM Calls in Unit Tests (HIGHEST IMPACT - DO FIRST)

**Problem**: Tests in `test_nl_query_refinement.py` and `test_nl_query_engine_filter_extraction.py` make real LLM calls (10-30 seconds each), causing:
- Slow test execution (10-30s per test)
- Flaky tests (depends on Ollama availability)
- Expensive CI/CD runs

**Solution**: Mock LLM calls in unit tests, keep real calls only for integration tests.

**Implementation**:

1. **Add `mock_llm_calls` fixture to `conftest.py`** (explicit fixture, NOT autouse):
```python
@pytest.fixture
def mock_llm_calls():
    """
    Explicit fixture to mock LLM calls for unit tests.
    
    Usage: Add 'mock_llm_calls' to test function parameters.
    Integration tests should NOT use this fixture to get real LLM calls.
    """
    from unittest.mock import patch
    from clinical_analytics.core.llm_feature import LLMCallResult, LLMFeature
    
    with patch("clinical_analytics.core.llm_feature.call_llm") as mock_call_llm:
        # Mock responses for all LLMFeature types
        def _mock_call_llm(feature, system, user, timeout_s, model=None):
            if feature == LLMFeature.PARSE:
                return LLMCallResult(
                    raw_text='{"intent": "DESCRIBE", "confidence": 0.8}',
                    payload={"intent": "DESCRIBE", "confidence": 0.8},
                    latency_ms=10.0,
                    timed_out=False,
                    error=None,
                )
            elif feature == LLMFeature.FILTER_EXTRACTION:
                return LLMCallResult(
                    raw_text='{"filters": []}',
                    payload={"filters": []},
                    latency_ms=10.0,
                    timed_out=False,
                    error=None,
                )
            elif feature == LLMFeature.FOLLOWUPS:
                return LLMCallResult(
                    raw_text='{"follow_ups": []}',
                    payload={"follow_ups": []},
                    latency_ms=10.0,
                    timed_out=False,
                    error=None,
                )
            elif feature == LLMFeature.RESULT_INTERPRETATION:
                return LLMCallResult(
                    raw_text='{"interpretation": "Test interpretation"}',
                    payload={"interpretation": "Test interpretation"},
                    latency_ms=10.0,
                    timed_out=False,
                    error=None,
                )
            elif feature == LLMFeature.ERROR_TRANSLATION:
                return LLMCallResult(
                    raw_text='{"translation": "Test error translation"}',
                    payload={"translation": "Test error translation"},
                    latency_ms=10.0,
                    timed_out=False,
                    error=None,
                )
            # Default fallback
            return LLMCallResult(
                raw_text='{}',
                payload={},
                latency_ms=10.0,
                timed_out=False,
                error=None,
            )
        
        mock_call_llm.side_effect = _mock_call_llm
        yield mock_call_llm
```

**Alternative approach using pytest_configure hook** (if explicit fixture injection is too verbose):
```python
def pytest_configure(config):
    """Configure LLM mocking based on markers."""
    # Register custom marker
    config.addinivalue_line("markers", "real_llm: Use real LLM calls (for integration tests)")

@pytest.fixture(autouse=True)
def mock_llm_calls_unless_marked(request):
    """
    Automatically mock LLM calls UNLESS test is marked with @pytest.mark.real_llm.
    
    Integration tests must explicitly mark with @pytest.mark.real_llm to get real LLM.
    """
    if request.node.get_closest_marker("real_llm"):
        # Integration test - use real LLM
        yield
        return
    
    # Unit test - use mocked LLM
    # ... (same mock implementation as above)
    yield
```

**Decision**: Use explicit fixture approach (first option) for clarity and explicit opt-in. Integration tests simply don't use the fixture.

2. **Refactor unit tests to use mocks**:
   - `tests/core/test_nl_query_refinement.py` - Remove real LLM calls, use mocked responses
   - `tests/core/test_nl_query_engine_filter_extraction.py` - Remove real LLM calls, use mocked responses

3. **Create integration test files**:
   - `tests/core/test_nl_query_refinement_integration.py` - Real LLM tests (marked `@pytest.mark.slow` and `@pytest.mark.integration`)
   - `tests/core/test_nl_query_engine_filter_extraction_integration.py` - Real LLM tests

**Integration Test Requirements**:
- **Coverage threshold**: 80% of real LLM scenarios must be covered
- **Required scenarios**:
  - Happy path: Successful query parsing with various intents
  - Error cases: LLM timeout, JSON parse failures, Ollama unavailable
  - Edge cases: Ambiguous queries, complex filters, refinement queries
- **How to run**: `make test-integration-llm` (new Makefile command, filters `@pytest.mark.integration` + `@pytest.mark.real_llm`)
- **Test structure**: Separate integration test files (`*_integration.py`) marked with both `@pytest.mark.slow` and `@pytest.mark.integration`

**Success Criteria**:
- [ ] Unit tests run <1s each (30-50x speedup from 10-30s)
- [ ] Integration tests still verify real LLM behavior (80% scenario coverage)
- [ ] All existing tests pass
- [ ] No flaky tests from Ollama availability
- [ ] Mock responses match real LLM response patterns (verified in integration tests)

**Expected Impact**: 10-30s → <1s per test (30-50x speedup for LLM tests)

#### 2.2: Test Data Caching (HIGH IMPACT)

**Problem**: Tests regenerate the same data files repeatedly:
- Excel files recreated for each test
- Large CSV files regenerated
- Polars DataFrames recreated from scratch

**Solution**: Cache test data with content-based hashing.

**Implementation**:

1. **Create `tests/fixtures/cache.py`**:
```python
import hashlib
import json
from pathlib import Path
import polars as pl

CACHE_DIR = Path("tests/.test_cache")

def get_cache_key(data: dict | pl.DataFrame, config: dict | None = None) -> str:
    """Generate content-based hash for caching."""
    if isinstance(data, pl.DataFrame):
        # Hash DataFrame schema and sample rows
        key_data = {
            "schema": dict(data.schema),
            "shape": data.shape,
            "sample": data.head(10).to_dict(as_series=False),
        }
    else:
        key_data = data
    
    if config:
        key_data["config"] = config
    
    key_str = json.dumps(key_data, sort_keys=True)
    return hashlib.sha256(key_str.encode()).hexdigest()

def cache_dataframe(df: pl.DataFrame, cache_key: str) -> Path:
    """Cache DataFrame as parquet file."""
    cache_file = CACHE_DIR / f"{cache_key}.parquet"
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(cache_file)
    return cache_file

def load_cached_dataframe(cache_key: str) -> pl.DataFrame | None:
    """Load cached DataFrame if exists."""
    cache_file = CACHE_DIR / f"{cache_key}.parquet"
    if cache_file.exists():
        return pl.read_parquet(cache_file)
    return None
```

2. **Update expensive fixtures to use caching**:
   - `synthetic_dexa_excel_file` - Cache Excel file generation
   - `synthetic_statin_excel_file` - Cache Excel file generation
   - `synthetic_complex_excel_file` - Cache Excel file generation
   - Large DataFrame fixtures - Cache as parquet

3. **Cache invalidation strategy**:
   
   **Invalidation triggers**:
   - **Content hash mismatch**: If hash of input data/config differs from cached hash, regenerate
   - **File modification time**: If source files (fixture code, test data) modified after cache creation, invalidate
   - **Git commit hash**: Store git commit hash with cache, invalidate if commit changes (optional, for CI/CD)
   - **Manual flag**: `tests/.test_cache/.invalidate` file presence triggers full cache clear
   - **Cache version**: Increment cache version in code to force invalidation of all caches
   
   **Corruption recovery**:
   - Check cache file integrity on load (try/catch parquet read)
   - If corrupted: Delete corrupted file, log warning, regenerate
   - Add `cache_valid` flag to cache metadata (JSON file alongside parquet)
   
   **Cache size management**:
   - Maximum cache size: 500MB (configurable)
   - LRU eviction: Remove least recently used cache entries when limit reached
   - Cache cleanup: `make test-cache-clear` removes all cached files
   - Cache stats: Track cache hit/miss rates for monitoring
   
   **Implementation**:
   ```python
   def get_cache_key(data: dict | pl.DataFrame, config: dict | None = None) -> str:
       """Generate content-based hash with version."""
       CACHE_VERSION = "v1"  # Increment to invalidate all caches
       # ... hash logic ...
       return f"{CACHE_VERSION}_{hashlib.sha256(...).hexdigest()}"
   
   def load_cached_dataframe(cache_key: str) -> pl.DataFrame | None:
       """Load with corruption recovery."""
       cache_file = CACHE_DIR / f"{cache_key}.parquet"
       metadata_file = CACHE_DIR / f"{cache_key}.meta.json"
       
       if not cache_file.exists():
           return None
       
       # Check manual invalidation flag
       if (CACHE_DIR / ".invalidate").exists():
           return None
       
       # Check metadata for validity
       try:
           if metadata_file.exists():
               metadata = json.loads(metadata_file.read_text())
               if not metadata.get("cache_valid", True):
                   return None
       except Exception:
           pass
       
       # Try to load with corruption recovery
       try:
           return pl.read_parquet(cache_file)
       except Exception as e:
           logger.warning(f"Cache corruption detected, regenerating: {e}")
           cache_file.unlink(missing_ok=True)
           metadata_file.unlink(missing_ok=True)
           return None
   ```
   
   **Makefile command**:
   ```makefile
   test-cache-clear: ## Clear test data cache
   	@echo "$(GREEN)Clearing test cache...$(NC)"
   	rm -rf tests/.test_cache
   	@echo "$(GREEN)Cache cleared$(NC)"
   ```

**Success Criteria**:
- [ ] Cached data loads 50-80% faster than regeneration
- [ ] Cache invalidation works correctly
- [ ] No test failures from stale cache
- [ ] Cache directory added to `.gitignore`

**Expected Impact**: 50-80% reduction in data loading time

#### 2.3: Selective Dataset Loading (MEDIUM IMPACT)

**Problem**: Tests load all datasets even when only one is needed:
- `discovered_datasets` fixture loads all dataset configs
- Tests that only need one dataset pay cost of loading all

**Solution**: Add new fixture for lazy loading while maintaining backward compatibility with existing `discovered_datasets` fixture.

**Migration Strategy** (CRITICAL - prevents breaking existing tests):

1. **Audit existing usage** (BEFORE making changes):
   ```bash
   # Find all usages of discovered_datasets fixture
   grep -r "discovered_datasets" tests/ --include="*.py" | wc -l
   # Expected: ~20+ files
   ```
   
2. **Add NEW fixture (backward compatible)**:
   ```python
   # Keep existing discovered_datasets fixture unchanged (returns dict)
   @pytest.fixture(scope="session")
   def discovered_datasets():
       """Session-scoped dataset discovery (BACKWARD COMPATIBLE - returns dict)."""
       # ... existing implementation unchanged ...
       return {
           "available": available,
           "configs": configs,
           "all_datasets": all_datasets,
       }
   
   # Add NEW fixture for lazy loading
   @pytest.fixture(scope="session")
   def dataset_registry():
       """Session-scoped dataset registry for lazy loading."""
       DatasetRegistry.reset()
       DatasetRegistry.discover_datasets()
       DatasetRegistry.load_config()
       return DatasetRegistry
   
   @pytest.fixture
   def get_dataset_by_name(dataset_registry):
       """Helper to load specific dataset by name (lazy loading)."""
       def _get(name: str):
           dataset = dataset_registry.get_dataset(name)
           if dataset and not dataset.validate():
               pytest.skip(f"{name} data not available")
           return dataset
       return _get
   ```

3. **Migration approach** (incremental, non-breaking):
   - **Option A (Recommended)**: New tests use `get_dataset_by_name`, old tests continue using `discovered_datasets`
   - **Option B**: Migrate tests incrementally (update 5-10 tests at a time, verify passing)
   - **Option C**: Create migration script to update all tests automatically, then verify all pass

4. **Validation after migration**:
   - All existing tests pass (no regressions)
   - New lazy loading works correctly
   - Performance improvement measured (30-50% reduction in setup time)

**Implementation**:

1. **Add new fixtures** (without changing existing `discovered_datasets`):
   - Add `dataset_registry` fixture (returns DatasetRegistry)
   - Add `get_dataset_by_name` helper fixture

2. **Update tests incrementally**:
   - Start with new tests: Use `get_dataset_by_name` for selective loading
   - Migrate existing tests gradually: Update 5-10 tests, verify passing, commit
   - Final state: All tests use lazy loading, `discovered_datasets` can be deprecated (future)

3. **Verify backward compatibility**:
   - All tests using `discovered_datasets` continue to work
   - No test failures introduced
   - Performance improvement verified

**Success Criteria**:
- [ ] Tests load only needed datasets
- [ ] Setup time reduced by 30-50%
- [ ] All existing tests pass

**Expected Impact**: 30-50% reduction in test setup time

#### 2.4: Fixture Scope Optimization (MEDIUM IMPACT)

**Problem**: Expensive fixtures recreated for every test:
- Workspace setup repeated
- Large DataFrames regenerated
- Excel files recreated

**Solution**: Convert immutable, expensive fixtures to module/session scope.

**Profiling Methodology**:

1. **Extract metrics from performance tracking data**:
   - Run `make test-performance` to generate `.performance_data.json`
   - Extract fixture creation time (if fixture-level tracking available)
   - Extract test duration (includes fixture setup + test execution)
   - Count fixture reuse (how many tests use same fixture)
   
2. **Identify expensive fixtures**:
   - **Threshold criteria**: Fixture is "expensive" if:
     - Creation time > 1 second (if measurable)
     - OR test duration > 5 seconds AND fixture used by >5 tests
     - OR fixture creates large files (>10MB) or performs I/O
   - **Tooling**: 
     - Use `pytest-profiling` plugin: `pytest --profile -o profile.txt`
     - OR analyze `.performance_data.json` to find slow tests, then identify fixtures used
     - OR add fixture-level tracking to performance plugin (future enhancement)
   
3. **Profiling process**:
   ```bash
   # Step 1: Run tests with profiling
   make test-performance
   
   # Step 2: Analyze performance data
   python scripts/generate_performance_report.py --analyze-fixtures
   
   # Step 3: Identify candidates
   # Look for:
   # - Tests with duration >5s that share common fixtures
   # - Fixtures used by >5 tests with average test duration >2s
   # - Fixtures that create files or perform I/O
   ```

**Implementation**:

1. **Profile fixtures using performance tracking data**:
   - Use `make test-performance` to identify expensive fixtures
   - Analyze `.performance_data.json` for slow tests
   - Identify common fixtures across slow tests
   - Apply threshold criteria (>1s creation OR >5 tests with >2s duration)

2. **Convert to module/session scope**:
   - Identify immutable fixtures (don't change between tests)
   - Convert to `@pytest.fixture(scope="module")` or `scope="session"`
   - Use `tmp_path_factory` for module-scoped temp files

3. **Verify test isolation**:
   - Run tests multiple times
   - Run in random order
   - Verify no shared state issues

**Success Criteria**:
- [ ] Expensive fixtures converted to appropriate scope
- [ ] Test isolation maintained
- [ ] Overall test speedup of 10-30%
- [ ] No flaky tests introduced

**Expected Impact**: 10-30% overall test speedup

#### 2.5: Parallel Execution Safety Audit and Fix (CRITICAL - BLOCKING)

**Problem**: Parallel execution is currently default, but safety issues identified:
- Hardcoded filesystem paths (`/tmp/nl_query_learning/`, `/tmp/test_logs`) can collide in parallel runs
- No `@pytest.mark.serial` markers for unsafe tests
- No parallel-safety verification (flake rate, random order testing)
- Risk of nondeterministic failures in parallel execution

**Solution**: Switch to serial-by-default, fix issues, verify safety, then re-enable parallel.

**Implementation**:

**Phase 1: Fix Hardcoded Paths** (IMMEDIATE):
- Replace `/tmp/nl_query_learning/` with `tmp_path` fixture in `test_nl_query_engine_self_improvement.py`
- Replace `/tmp/test_logs` with `tmp_path` fixture in `test_prompt_optimizer.py`
- Audit all tests for hardcoded paths (grep for `/tmp/`, `/var/`, absolute paths)

**Phase 2: Add Serial Markers**:
- Identify tests that can't run in parallel (shared DB, ports, file locks)
- Mark with `@pytest.mark.serial`
- Update Makefile to exclude serial tests from parallel runs

**Phase 3: Verify Parallel-Safety**:
- Run with random order: `pytest --random-order`
- Run multiple times: `for i in {1..10}; do pytest -n auto; done`
- Monitor flake rate (target: <1% over 100+ runs)
- Check for shared state issues

**Phase 4: Re-enable Parallel as Default** (AFTER VERIFICATION):
- Only after: zero hardcoded paths, serial markers in place, flake rate <1%
- Switch Makefile back to parallel-by-default
- Document parallel-safety guarantees

**Current Status**: 
- ⚠️ **BLOCKING**: Parallel is default but unsafe
- **Action**: Switch to serial-by-default immediately, then fix issues

**Success Criteria**:
- [ ] All hardcoded paths replaced with `tmp_path` fixtures
- [ ] Serial markers on all unsafe tests
- [ ] Flake rate <1% over 100+ parallel runs
- [ ] Random order testing passes consistently
- [ ] Parallel re-enabled as default (after verification)

**Expected Impact**: Safe parallel execution with 2-4x speedup (after fixes)

### Phase 3: Automated Test Categorization (LOW EFFORT, HIGH VALUE)

**Goal**: Use performance tracking data to automatically categorize slow tests and improve `test-fast` effectiveness.

**Implementation**:

1. **Create `scripts/categorize_slow_tests.py`**:
   - Read `.performance_data.json`
   - Identify tests >30 seconds without `@pytest.mark.slow`
   - Generate report of uncategorized slow tests
   - Optionally auto-add `@pytest.mark.slow` (with confirmation)

2. **Integrate with performance report**:
   - Add section to `make performance-report` output
   - Show uncategorized slow tests
   - Recommend adding `@pytest.mark.slow`

**Success Criteria**:
- [ ] Identifies uncategorized slow tests
- [ ] Generates actionable reports
- [ ] Improves `test-fast` effectiveness

**Expected Impact**: Better test categorization, more effective `test-fast` runs

### Phase 1 Completion Gate (REQUIRED BEFORE PHASE 2)

**Validation Checklist** (must complete before starting Phase 2):
- [ ] All Phase 1 todos marked complete (1-14)
- [ ] Performance tracking verified working: `make test-performance` generates `.performance_data.json`
- [ ] Baseline created: `make performance-baseline` creates `.performance_baseline.json`
- [ ] Baseline committed to git
- [ ] Regression tests passing: `make performance-regression` passes
- [ ] Performance report generated: `make performance-report` works
- [ ] Documentation updated: `make performance-update-docs` updates PERFORMANCE.md

**Gate Command**:
```bash
# Verify Phase 1 complete
make test-performance
make performance-baseline
make performance-regression  # Should pass
make performance-report
git status  # Verify baseline committed
```

**Only proceed to Phase 2 after all Phase 1 validation passes.**

### Rollback Strategy

**If optimizations break tests or cause regressions:**

1. **Disable optimizations**:
   - **LLM Mocking**: Remove `mock_llm_calls` fixture from test parameters (revert to real LLM)
   - **Caching**: Set environment variable `DISABLE_TEST_CACHE=1` (add to cache.py)
   - **Lazy Loading**: Revert to using `discovered_datasets` fixture (keep old behavior)
   - **Scope Changes**: Revert fixture scope changes (git revert specific commits)

2. **Clear cache if causing issues**:
   ```bash
   make test-cache-clear
   ```

3. **Revert changes**:
   - **Git revert**: `git revert <commit-hash>` for specific optimization commits
   - **Manual undo**: Restore original fixture implementations from git history
   - **Feature flags**: Add environment variables to disable optimizations without code changes

4. **Validation before committing optimizations**:
   - Run full test suite: `make test` (all tests must pass)
   - Verify performance improvement: Compare before/after metrics
   - Check for flaky tests: Run tests multiple times, verify stability
   - Verify test isolation: Run tests in random order, verify no shared state

5. **Rollback validation**:
   - After rollback, verify all tests pass
   - Verify performance returns to baseline
   - Document rollback reason in commit message

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

test-cache-clear: ## Clear test data cache
	@echo "$(GREEN)Clearing test cache...$(NC)"
	rm -rf tests/.test_cache
	@echo "$(GREEN)Cache cleared$(NC)"

test-integration-llm: ensure-venv ## Run LLM integration tests (real LLM calls)
	@echo "$(GREEN)Running LLM integration tests...$(NC)"
	$(PYTEST) $(TEST_DIR) -v -m "integration and real_llm"
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

1. `tests/conftest.py` - Register performance plugin, add mock_llm_calls fixture, enhance discovered_datasets
2. `tests/PERFORMANCE.md` - Update with new sections and optimization results
3. `pyproject.toml` - Add performance configuration
4. `Makefile` - Add performance commands
5. `.gitignore` - Add `.performance_data.json`, `.performance_data_worker_*.json`, `tests/.test_cache/`

### New Files (Phase 2 Optimizations)

1. `tests/fixtures/cache.py` - Test data caching utilities
2. `tests/core/test_nl_query_refinement_integration.py` - Real LLM integration tests
3. `tests/core/test_nl_query_engine_filter_extraction_integration.py` - Real LLM integration tests
4. `scripts/categorize_slow_tests.py` - Automated test categorization

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

### Phase 1: Performance Tracking (Complete tracking system first)

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

### Phase 2: Performance Optimizations (After tracking is working)

**CRITICAL**: Phase 2 optimizations should be implemented **after** Phase 1 tracking is complete, so we can measure the impact of each optimization.

**Phase 1 Completion Gate**: Verify all Phase 1 validation checklist items pass before starting Phase 2 (see "Phase 1 Completion Gate" section above).

**Order of Implementation** (by impact):

1. **2.1: Mock LLM Calls** (HIGHEST IMPACT - DO FIRST)
   - **TDD**: Write test for mock_llm_calls fixture first
   - **Success Criteria**: Unit tests <1s each, integration tests still work
   - **Expected**: 30-50x speedup for LLM tests

2. **2.2: Test Data Caching** (HIGH IMPACT)
   - **TDD**: Write tests for cache utilities first
   - **Success Criteria**: 50-80% faster data loading
   - **Expected**: Significant reduction in fixture creation time

3. **2.3: Selective Dataset Loading** (MEDIUM IMPACT)
   - **TDD**: Write tests for lazy loading helpers first
   - **Success Criteria**: 30-50% reduction in setup time
   - **Expected**: Faster test initialization

4. **2.4: Fixture Scope Optimization** (MEDIUM IMPACT)
   - **TDD**: Profile first, then optimize
   - **Success Criteria**: 10-30% overall speedup
   - **Expected**: Reduced fixture recreation overhead

5. **2.5: Parallel Execution Safety Fix** (CRITICAL - BLOCKING)
   - **TDD**: Write tests for parallel-safety detection
   - **Phase 1**: Fix hardcoded paths (IMMEDIATE)
   - **Phase 2**: Add serial markers
   - **Phase 3**: Verify parallel-safety
   - **Phase 4**: Re-enable parallel as default (after verification)
   - **Success Criteria**: Zero hardcoded paths, flake rate <1%, safe parallel execution
   - **Expected**: Safe 2-4x speedup after fixes

### Phase 3: Automated Categorization (After optimizations)

1. **Categorization Script**
   - **TDD**: Write tests for categorization logic first
   - **Success Criteria**: Identifies uncategorized slow tests
   - **Expected**: Better test-fast effectiveness

## Success Criteria

### Phase 1: Performance Tracking Success Criteria

- [ ] Performance data automatically collected on every test run **when `--track-performance` flag is used**
- [ ] Regression tests fail when performance degrades >20% **with clear error message showing before/after comparison**
- [ ] CLI tool generates accurate reports **with all specified sections populated**
- [ ] PERFORMANCE.md automatically updated with current benchmarks **via `make performance-update-docs`**
- [ ] All components have comprehensive test coverage **≥80% for performance system**

### Phase 2: Performance Optimization Success Criteria

**2.1: LLM Mocking**:
- [ ] Unit tests run <1s each (30-50x speedup from 10-30s)
- [ ] Integration tests still verify real LLM behavior
- [ ] No flaky tests from Ollama availability

**2.2: Test Data Caching**:
- [ ] Cached data loads 50-80% faster than regeneration
- [ ] Cache invalidation works correctly
- [ ] No test failures from stale cache

**2.3: Selective Dataset Loading**:
- [ ] Tests load only needed datasets
- [ ] Setup time reduced by 30-50%

**2.4: Fixture Scope Optimization**:
- [ ] Expensive fixtures converted to appropriate scope
- [ ] Overall test speedup of 10-30%
- [ ] Test isolation maintained

**2.5: Parallel Execution**:
- [ ] Maintain 2-4x parallel speedup
- [ ] Serial tests properly excluded

### Phase 3: Automated Categorization Success Criteria

- [ ] Identifies uncategorized slow tests automatically
- [ ] Generates actionable reports
- [ ] Improves `test-fast` effectiveness

### Overall Success Criteria

- [ ] **Measurable performance improvements**: 30-50x for LLM tests, 50-80% for data loading, 10-30% overall
- [ ] **Performance tracking works end-to-end**: Track → Report → Baseline → Regression
- [ ] **All optimizations verified**: Before/after metrics documented in PERFORMANCE.md
- [ ] **No regressions**: All existing tests pass after optimizations

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