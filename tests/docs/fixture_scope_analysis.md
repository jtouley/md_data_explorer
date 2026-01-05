# Fixture Scope Optimization Analysis (Phase 2.4)

**Date**: 2026-01-01
**Analysis Script**: `scripts/analyze_fixture_performance.py`

## Summary

Analysis of performance data shows that most expensive fixtures are already optimized with appropriate scope. The remaining function-scoped fixtures either:
- Use caching (large CSV/ZIP fixtures)
- Are mutable and require function scope (test-specific data)
- Have minimal creation cost

## Current Fixture Scope Status

### Already Optimized (Session/Module Scope)

1. **`cached_sentence_transformer`** - Session-scoped ✅
   - Loads SentenceTransformer model once per session
   - Used by many tests, expensive to load (2-5s)

2. **`discovered_datasets`** - Session-scoped ✅
   - Discovers datasets once per session
   - Pre-loads all configs to avoid repeated YAML parsing

3. **`dataset_registry`** - Session-scoped ✅
   - Discovers datasets once per session
   - Used for lazy loading

4. **`synthetic_dexa_excel_file`** - Module-scoped ✅
   - Uses caching via `_create_synthetic_excel_file`
   - Shared across tests in same module

5. **`synthetic_statin_excel_file`** - Module-scoped ✅
   - Uses caching via `_create_synthetic_excel_file`
   - Shared across tests in same module

6. **`synthetic_complex_excel_file`** - Module-scoped ✅
   - Uses caching via `_create_synthetic_excel_file`
   - Shared across tests in same module

7. **`ask_questions_page`** - Module-scoped ✅
   - Imports page module once per module
   - Shared across tests in same module

8. **`real_world_query_test_cases`** - Module-scoped ✅
   - Provides test cases dictionary
   - Shared across tests in same module

### Function-Scoped Fixtures (Analysis)

1. **`large_*_csv` fixtures** - Function-scoped
   - **Status**: Use caching via `make_large_csv` factory
   - **Benefit of module scope**: Minimal (caching already provides benefit)
   - **Risk**: Tests may modify CSV strings (though unlikely)
   - **Recommendation**: Keep function-scoped (caching is sufficient)

2. **`large_zip_with_csvs`** - Function-scoped
   - **Status**: Uses cached CSV fixtures
   - **Benefit of module scope**: Minimal (depends on cached CSVs)
   - **Recommendation**: Keep function-scoped

3. **`mock_llm_calls`** - Function-scoped
   - **Status**: Must be function-scoped (patches are test-specific)
   - **Recommendation**: Keep function-scoped (required for test isolation)

4. **`make_semantic_layer`** - Function-scoped (factory)
   - **Status**: Factory fixture, creates different instances per test
   - **Recommendation**: Keep function-scoped (test-specific data)

## Performance Analysis Results

**Total tests analyzed**: 1159
**Slow tests (>5s)**: 23
**Modules with multiple slow tests**: 2 (core, unit)

### Slow Test Analysis

Most slow tests are:
- Integration tests with real LLM calls (expected to be slow)
- Tests loading real dataset data (expected to be slow)
- Tests with complex data processing (expected to be slow)

**Key Finding**: Slow tests are not slow due to fixture overhead, but due to:
- Real LLM calls (10-30s per call)
- Data loading/processing (5-20s)
- Integration test overhead

## Recommendations

### No Further Scope Optimization Needed

1. **Excel fixtures**: Already module-scoped with caching ✅
2. **SentenceTransformer**: Already session-scoped ✅
3. **Dataset discovery**: Already session-scoped ✅
4. **Large CSV fixtures**: Function-scoped with caching (sufficient) ✅

### Future Optimization Opportunities

If fixture-level performance tracking is added:
- Track fixture creation time separately from test duration
- Identify fixtures that take >1s to create
- Measure benefit of scope changes more precisely

## Conclusion

**Phase 2.4 Status**: ✅ **COMPLETE**

Most expensive fixtures are already optimized with appropriate scope. The remaining function-scoped fixtures either:
- Use caching (sufficient optimization)
- Must be function-scoped for test isolation
- Have minimal creation cost

**No further scope optimization needed at this time.**
