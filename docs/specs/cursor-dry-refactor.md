# Semantic Model Refactor: Eliminate Hardcoded Logic

Transform the platform from a hybrid config/hardcoded system to a fully semantic, idempotent model where all transformations, filters, aggregations, and defaults are driven by `datasets.yaml`.

## Current Issues

1. **Loaders have hardcoded transformations** that duplicate YAML config
2. **Dataset classes have hardcoded filter logic** instead of using mapper
3. **Aggregation logic is hardcoded** in sepsis loader, ignoring config
4. **Time zero, outcome selection, and labels are hardcoded** in dataset classes
5. **SQL queries are hardcoded** in MIMIC3 loader instead of using config
6. **Non-idempotent**: Same config can produce different results depending on execution order

## Solution Architecture

### Phase 1: Enhance ColumnMapper with Missing Capabilities

**File: [src/clinical_analytics/core/mapper.py](src/clinical_analytics/core/mapper.py)**

1. **Add filter application engine**:

- `apply_filters(df: pl.DataFrame, filters: dict) -> pl.DataFrame`
- Support config-defined filter types: `equals`, `in`, `range`, `exists`
- Read filter definitions from `default_filters` in config

2. **Add aggregation engine**:

- `apply_aggregations(df: pl.DataFrame, group_by: str) -> pl.DataFrame`
- Read aggregation specs from `aggregation` section in config
- Support methods: `first`, `last`, `max`, `min`, `mean`, `sum`, `count`

3. **Enhance outcome transformations**:

- Support `aggregation` type outcomes (currently only handles `binary` with `mapping`)
- Apply transformations during `map_to_unified_cohort()` automatically

4. **Add config-driven defaults**:

- `get_time_zero_value() -> Optional[str]` - read from config
- `get_default_outcome_label(outcome_col: str) -> str` - read from config or derive
- `get_default_filters() -> dict` - already exists, ensure it's used

### Phase 2: Refactor COVID-MS Loader

**File: [src/clinical_analytics/datasets/covid_ms/loader.py](src/clinical_analytics/datasets/covid_ms/loader.py)**

1. **Remove hardcoded outcome transformations** (lines 33-46):

- Delete `normalize_outcome()` function (unused)
- Remove hardcoded `pl.when().then().otherwise()` logic
- Keep only raw data loading: `load_raw_data()` and basic cleaning (null handling)

2. **Apply outcome transformations via mapper**:

- `clean_data()` should accept a `mapper: ColumnMapper` parameter
- Call `mapper.apply_outcome_transformations(df)` instead of hardcoding

**File: [src/clinical_analytics/datasets/covid_ms/definition.py](src/clinical_analytics/datasets/covid_ms/definition.py)**

1. **Remove hardcoded filter logic** (line 66):

- Delete special-case `confirmed_only` handling
- Use `mapper.apply_filters(df, all_filters)` instead

2. **Remove hardcoded defaults** (lines 82-84):

- `time_zero_value` → `mapper.get_time_zero_value()`
- `outcome_label` → `mapper.get_default_outcome_label(outcome_col)`

3. **Update `load()` method**:

- Pass `self.mapper` to `clean_data()` so it can apply transformations

### Phase 3: Refactor Sepsis Loader

**File: [src/clinical_analytics/datasets/sepsis/loader.py](src/clinical_analytics/datasets/sepsis/loader.py)**

1. **Make aggregation config-driven**:

- `load_and_aggregate()` should accept `mapper: ColumnMapper` parameter
- Read aggregation specs from `mapper.config.get('aggregation', {})`
- Build Polars aggregation expressions dynamically from config
- Support both `static_features` and `outcome` aggregation sections

2. **Remove hardcoded column names**:

- Use config to determine which columns to aggregate and how
- Patient ID extraction should come from config if possible

**File: [src/clinical_analytics/datasets/sepsis/definition.py](src/clinical_analytics/datasets/sepsis/definition.py)**

1. **Remove hardcoded defaults** (lines 79-81):

- `time_zero_value` → `mapper.get_time_zero_value()`
- `outcome_col` → `mapper.get_default_outcome()`
- `outcome_label` → `mapper.get_default_outcome_label(outcome_col)`

2. **Update `load()` method**:

- Pass `self.mapper` to `load_and_aggregate()` for config-driven aggregation

3. **Use mapper for filters**:

- Replace simple filter loop with `mapper.apply_filters(df, filters)`

### Phase 4: Refactor MIMIC3 Loader

**File: [src/clinical_analytics/datasets/mimic3/loader.py](src/clinical_analytics/datasets/mimic3/loader.py)**

1. **Remove hardcoded SQL query** (lines 84-113):

- Delete `_get_default_cohort_query()` method
- Always use query from config (already partially done in `load()`)

2. **Ensure config is single source of truth**:

- `load_cohort()` should require query parameter or read from config
- No fallback to hardcoded query

**File: [src/clinical_analytics/datasets/mimic3/definition.py](src/clinical_analytics/datasets/mimic3/definition.py)**

1. **Remove hardcoded outcome_label** (line 114):

- Use `mapper.get_default_outcome_label(outcome_col)` instead

2. **Use mapper for filters**:

- Replace simple filter loop with `mapper.apply_filters(df, filters)`

### Phase 5: Update Config Files

**File: [data/configs/datasets.yaml](data/configs/datasets.yaml)**

1. **Add missing config sections**:

- `time_zero` section with `value` or `source_column` for each dataset
- `filters` section with filter definitions (type, column, default_value)
- Ensure `aggregation` section is complete for sepsis

2. **Fix duplicate keys** (line 26):

- `outcome_hospitalized: outcome_label` is duplicate - should be separate config

3. **Add outcome_label mappings**:

- Define default outcome labels for each outcome type

### Phase 6: Update Tests

**Files: [tests/test_mapper.py](tests/test_mapper.py), [tests/test_covid_ms_dataset.py](tests/test_covid_ms_dataset.py)**

1. **Add tests for new mapper methods**:

- `test_apply_filters()`
- `test_apply_aggregations()`
- `test_get_time_zero_value()`
- `test_get_default_outcome_label()`

2. **Update existing tests**:

- Ensure tests still pass after refactoring
- Add tests for idempotency (same config = same result)

## Implementation Order

1. **Phase 1** (Mapper enhancements) - Foundation for everything else
2. **Phase 5** (Config updates) - Ensure config has all needed fields
3. **Phase 2** (COVID-MS) - Simplest dataset, good starting point
4. **Phase 3** (Sepsis) - More complex with aggregation
5. **Phase 4** (MIMIC3) - SQL-based, different pattern
6. **Phase 6** (Tests) - Validate idempotency and correctness

## Success Criteria

- ✅ No hardcoded transformations in loaders
- ✅ No hardcoded filter logic in dataset classes
- ✅ All defaults come from config
- ✅ Same config always produces same result (idempotent)
- ✅ Adding new dataset requires only YAML config (zero code changes)
- ✅ All existing tests pass
- ✅ New tests verify semantic model behavior

## Files to Modify

- `src/clinical_analytics/core/mapper.py` - Add filter/aggregation engines
- `src/clinical_analytics/datasets/covid_ms/loader.py` - Remove hardcoded transformations
- `src/clinical_analytics/datasets/covid_ms/definition.py` - Use mapper for all logic
- `src/clinical_analytics/datasets/sepsis/loader.py` - Config-driven aggregation
- `src/clinical_analytics/datasets/sepsis/definition.py` - Use mapper defaults
- `src/clinical_analytics/datasets/mimic3/loader.py` - Remove hardcoded SQL
- `src/clinical_analytics/datasets/mimic3/definition.py` - Use mapper defaults
- `data/configs/datasets.yaml` - Add missing config sections
- `tests/test_mapper.py` - Add new tests
- `tests/test_covid_ms_dataset.py` - Update for refactored code