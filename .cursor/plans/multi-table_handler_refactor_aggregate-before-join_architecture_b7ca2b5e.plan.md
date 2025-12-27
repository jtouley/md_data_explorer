---
name: ""
overview: ""
todos:
  - id: "1"
    content: Add TableClassification dataclass (with bridge/reference types, byte estimates) and classify_tables() method with bridge detection
    status: completed
  - id: "2"
    content: Replace _find_anchor_table() with _find_anchor_by_centrality() using graph centrality, hard exclusions (no event/fact/bridge), and deterministic tie-breakers
    status: completed
    dependencies:
      - "1"
  - id: "3"
    content: Implement _build_dimension_mart() to join only 1:1/small dimensions to anchor table (exclude bridge tables)
    status: completed
    dependencies:
      - "1"
      - "2"
  - id: "4"
    content: Implement _aggregate_fact_tables() with aggregation policy enforcement (safe defaults, opt-in mean/avg, code column handling)
    status: completed
    dependencies:
      - "1"
  - id: "5"
    content: Replace build_unified_cohort() with plan_mart() (returns Ibis expression) and materialize_mart() (executes and writes Parquet). Includes CohortMetadata, caching with run_id, hash bucket partitioning for event-level, and observability logging.
    status: completed
    dependencies:
      - "3"
      - "4"
  - id: "6"
    content: Update get_cohort() signature in ClinicalDataset base class to include granularity parameter
    status: pending
  - id: "7"
    content: Update all dataset implementations (uploaded, sepsis, covid_ms, mimic3) to support granularity parameter
    status: pending
    dependencies:
      - "6"
  - id: "8"
    content: Update SemanticLayer.get_cohort() to handle granularity mapping (patient/admission/event -> patient_level/admission_level/event_level)
    status: pending
    dependencies:
      - "6"
  - id: "9"
    content: Fix background processing in UserDatasetStorage.save_zip_upload() to create fresh handler/DuckDB connection per thread
    status: pending
  - id: "10"
    content: Add cross-platform file locking (filelock library) and atomic writes (temp file + rename) for metadata in UserDatasetStorage.save_zip_upload()
    status: pending
    dependencies:
      - "9"
  - id: "11"
    content: "Implement hash bucket partitioning for event-level outputs (default: hash(grain_key) % 64, directory structure event_level/{table}/bucket=XX/*.parquet)"
    status: completed
    dependencies:
      - "5"
  - id: "12"
    content: Add QueryPlan dataclass and plan_from_nl() function to prevent "everything" queries from triggering monster builds
    status: pending
    dependencies:
      - "5"
---

# Multi-Table Handler Refactor: Aggregate-Before-Join Architecture

**Status**: P0 (Critical - Blocks Phase 4 of consolidate-docs plan)**Priority**: P0 (Resolves OOM issues blocking multi-table support)**Related Plan**: [consolidate-docs-and-implement-question-driven-analysis.md](docs/implementation/plans/consolidate-docs-and-implement-question-driven-analysis.md) - Phase 4**Created**: 2025-12-25**Owner**: Development Team

## Overview

**Critical Fix for Phase 4 OOM Blocker**: Refactor the multi-table handler to enforce aggregate-before-join patterns, replacing mega-joins that cause OutOfMemoryException with dimension marts and aggregated feature tables. This directly addresses the OOM issue blocking Phase 4 completion where joining 32 tables (including chartevents with 668k rows) exhausts 90.8 GiB of temp directory space.**Current Blocker** (from consolidate-docs Phase 4):

- ❌ DuckDB OutOfMemoryException when joining all 32 tables in single query
- Error: "failed to offload data block of size 32.0 KiB (90.8 GiB/90.8 GiB used)"
- Large tables like chartevents (668,862 rows) causing memory explosion
- Single massive LEFT JOIN of all tables exhausts temp directory space

**Solution**: Aggregate-before-join architecture prevents mega-joins by:

1. Classifying tables (dimension/fact/event/bridge) using cardinality + byte estimates
2. Only joining small dimensions to anchor (never large fact/event tables)
3. Pre-aggregating fact tables by grain key before joining
4. Using lazy Ibis expressions (not materialized DataFrames)
5. Partitioning event-level outputs to avoid memory spikes

## Current Problems

### Critical OOM Issue (Phase 4 Blocker)

1. **Mega-joins causing OOM**: `build_unified_cohort()` does `SELECT *` and joins all 32 tables in a single query, causing:

- DuckDB OutOfMemoryException: "failed to offload data block of size 32.0 KiB (90.8 GiB/90.8 GiB used)"
- Large tables like chartevents (668,862 rows) joined directly without aggregation
- Temp directory space exhaustion (90.8 GiB)
- Blocks MIMIC-IV demo dataset processing (32 tables, some with 600k+ rows)

### Architectural Issues

2. **Name-based anchor selection**: `_find_anchor_table()` uses "patient"/"subject" name heuristics instead of graph centrality
3. **No table classification**: Tables not classified as dimensions vs facts, leading to inappropriate joins
4. **No aggregation**: Event/fact tables joined directly without pre-aggregation (causes explosion)
5. **Shared DuckDB connection**: Background processing may share connections across threads
6. **Granularity mismatch**: `get_cohort()` doesn't map patient/admission/event to patient_level/admission_level/event_level
7. **No partitioning**: Event-level parquet files not partitioned, causing memory spikes
8. **No query planning**: NL queries can trigger "everything" builds that exhaust memory

## Architecture Changes

### Table Classification System

Add `TableClassification` dataclass and classification logic:

```python
@dataclass
class TableClassification:
    table_name: str
    classification: Literal["dimension", "fact", "event", "bridge", "reference"]
    grain: Literal["patient", "admission", "event"]
    grain_key: str  # detected grain key column
    cardinality_ratio: float  # rows / unique(grain_key)
    is_unique_on_grain: bool
    estimated_bytes: int  # bytes, not rows (rows * avg_row_bytes)
    relationship_degree: int  # number of foreign keys
    has_time_column: bool
    time_column_name: Optional[str]
    is_n_side_of_anchor: bool  # on N-side of relationship to anchor
```

Classification rules:

- **Dimension**: `cardinality_ratio <= 1.1` AND `is_unique_on_grain` AND `estimated_bytes < max_dimension_bytes` (default 250 MB)
- **Fact**: `cardinality_ratio > 1.1` AND NOT `is_unique_on_grain` AND NOT bridge
- **Event**: `has_time_column` AND `cardinality_ratio > 1.1` AND `time_column is not constant` AND `is_n_side_of_anchor`
- **Bridge**: Two or more foreign keys to different parents AND neither key unique BUT `(fk1, fk2)` is near-unique AND high relationship degree but low column payload
- **Reference**: Code mappings, lookup tables with versioning (duplicates allowed but small size)

### Graph-Based Anchor Selection

Replace `_find_anchor_table()` with centrality-based selection:**Hard Exclusions** (never anchor on these):

- Classification in `{event, fact, bridge}`
- Tables without unique grain key
- Tables with >50% NULLs in grain key

**Scoring Rules**:

1. Build relationship graph (nodes=tables, edges=relationships)
2. Score each **dimension** table:

- +10 if has `hadm_id` or `encounter_id` column
- +5 if has `patient_id` or `subject_id` column
- +1 per relationship (incoming + outgoing)
- +3 if classified as dimension with patient grain

3. **Tie-breakers** (deterministic):

- Prefer fewer NULLs in grain key (lower null_rate)
- Prefer unique grain key (is_unique_on_grain = True)
- Prefer smaller estimated_bytes
- Prefer patient grain over admission grain

4. Select highest-scoring **dimension** table as anchor

### Aggregate-Before-Join Pipeline

New `plan_mart()` and `materialize_mart()` methods:**plan_mart() -> ibis.Table**:

1. **Classify all tables** using cardinality + uniqueness + byte estimates
2. **Select anchor** using graph centrality (dimensions only)
3. **Build dimension mart**: Join only 1:1/small dimensions to anchor (exclude bridges)
4. **Aggregate fact tables**: Group by grain key, compute **safe** aggregations
5. **Join aggregated facts** to dimension mart
6. **Return Ibis expression** (lazy, not materialized)

**materialize_mart(path: Path) -> CohortMetadata**:

1. Execute Ibis plan
2. Write to partitioned Parquet
3. Return metadata with table locations

**Aggregation Policy** (safety constraints):Default aggregations (always safe):

- `count(*)` as `{table}_count`
- `count(distinct {col})` for categorical columns
- `min({time_col})`, `max({time_col})` for time columns
- `min({numeric_col})`, `max({numeric_col})` for numeric columns

Opt-in aggregations (require explicit config):

- `mean()` / `avg()` - only if `allow_mean: true` in config AND units are known/normalized
- `last({col})` - only if `allow_last: true` AND stable ordering column exists AND value is numeric (not a code)

Code columns (special handling):

- Never compute mean/avg on: `icd_code`, `itemid`, `ndc`, `cpt_code`, etc.
- Only count/distinct count for codes

Configuration:

```python
aggregation_policy = {
    "default_numeric": ["min", "max"],
    "allow_mean": False,  # must opt-in
    "allow_last": True,
    "code_columns": ["icd_code", "itemid", "ndc", "cpt_code"]
}
```



### Granularity Mapping

Update `get_cohort()` signature and implementations:

```python
def get_cohort(
    self,
    granularity: Literal["patient_level", "admission_level", "event_level"] = "patient_level",
    **filters
) -> pd.DataFrame:
```

Map internal grains to API grains:

- `patient` → `patient_level`
- `admission` → `admission_level`  
- `event` → `event_level`

### Background Processing Fixes

In `UserDatasetStorage.save_zip_upload()`:

1. Create fresh `MultiTableHandler` instance in thread
2. Create fresh DuckDB connection per handler (not shared)
3. Use **cross-platform** file lock for metadata writes:
   ```python
      from filelock import FileLock
      
      lock = FileLock(str(metadata_path) + ".lock")
      with lock:
          # Atomic write via temp file + rename
          tmp = metadata_path.with_suffix(".json.tmp")
          with open(tmp, 'w') as f:
              json.dump(full_metadata, f, indent=2)
          tmp.replace(metadata_path)  # atomic rename
   ```




### Event-Level Parquet Partitioning

For event-level outputs:**Default strategy: Hash bucketing by grain key**

- Generic and works for all tables (doesn't require timestamps)
- Improves selective reads (can query specific buckets)
- Directory structure: `event_level/{table_name}/bucket={00..63}/*.parquet`
- Bucket calculation: `hash(grain_key) % N` where N is configurable (default 64)

**Time bucketing** (optional, when timestamps exist and are relevant):

- Partition by `table_name` + time bucket (e.g., `YYYY-MM`)
- Only used if explicitly configured AND time column is non-constant

Implementation:

```python
def partition_event_table(df: pl.DataFrame, grain_key: str, num_buckets: int = 64) -> None:
    df.with_columns(
        (pl.col(grain_key).hash() % num_buckets).alias("bucket")
    ).write_parquet(
        "event_level/{table}/",
        partition_by=["bucket"]
    )
```



## Implementation Files

### Core Changes

- **[src/clinical_analytics/core/multi_table_handler.py](src/clinical_analytics/core/multi_table_handler.py)**
- Add `TableClassification` dataclass (with bridge/reference types, byte estimates)
- Add `classify_tables()` method (with bridge detection, byte estimation)
- Replace `_find_anchor_table()` with `_find_anchor_by_centrality()` (with hard exclusions, tie-breakers)
- Add `plan_mart()` (returns Ibis expression over materialized Parquet)
- Add `materialize_mart()` (executes Polars pipeline, writes Parquet with caching)
- Add `CohortMetadata` dataclass with run_id for caching
- Add `_build_dimension_mart()` helper (excludes bridges)
- Add `_aggregate_fact_tables()` helper (with aggregation policy enforcement)
- Add `_detect_bridge_table()` helper
- **[src/clinical_analytics/core/dataset.py](src/clinical_analytics/core/dataset.py)**
- Update `get_cohort()` signature to include `granularity` parameter
- **[src/clinical_analytics/core/semantic.py](src/clinical_analytics/core/semantic.py)**
- Update `get_cohort()` to handle granularity mapping

### Storage Changes

- **[src/clinical_analytics/ui/storage/user_datasets.py](src/clinical_analytics/ui/storage/user_datasets.py)**
- Create fresh handler/connection in `save_zip_upload()`
- Add file locking for metadata writes
- Update to use `build_mart_cohort()` instead of `build_unified_cohort()`
- Add parquet partitioning for event-level outputs

### Dataset Implementations

- **[src/clinical_analytics/datasets/uploaded/definition.py](src/clinical_analytics/datasets/uploaded/definition.py)**
- Update `get_cohort()` to support granularity parameter
- **[src/clinical_analytics/datasets/sepsis/definition.py](src/clinical_analytics/datasets/sepsis/definition.py)**
- Update `get_cohort()` to support granularity parameter
- **[src/clinical_analytics/datasets/covid_ms/definition.py](src/clinical_analytics/datasets/covid_ms/definition.py)**
- Update `get_cohort()` to support granularity parameter
- **[src/clinical_analytics/datasets/mimic3/definition.py](src/clinical_analytics/datasets/mimic3/definition.py)**
- Update `get_cohort()` to support granularity parameter

## Key Methods to Implement

### `classify_tables() -> Dict[str, TableClassification]`

For each table:

1. Detect grain key (patient_id, hadm_id, encounter_id, etc.) using key patterns
2. Calculate `cardinality_ratio = rows / unique(grain_key)`
3. Check `is_unique_on_grain = (rows == unique(grain_key))`
4. Estimate `estimated_bytes = rows * avg_row_bytes` (sample rows to compute avg_row_bytes)
5. Detect time column (non-constant timestamp)
6. Detect bridge tables: two+ foreign keys, neither unique, but composite near-unique
7. Classify based on rules above (dimension/fact/event/bridge/reference)

### `_find_anchor_by_centrality() -> str`

1. **Filter**: Only consider tables with `classification == "dimension"`
2. **Exclude**: Tables with >50% NULLs in grain key, or no unique grain key
3. Build relationship graph
4. Score each dimension table:

- +10 if has `hadm_id` or `encounter_id` column
- +5 if has `patient_id` or `subject_id` column
- +1 per relationship (incoming + outgoing)
- +3 if classified as dimension with patient grain

5. **Tie-breakers** (apply in order):

- Lower null_rate in grain key
- `is_unique_on_grain = True`
- Smaller `estimated_bytes`
- Patient grain > admission grain

6. Return highest-scoring dimension table

### `plan_mart(metadata: Optional[CohortMetadata] = None, parquet_path: Optional[Path] = None) -> ibis.Table`

1. Classify tables (with byte estimates)
2. Find anchor by centrality (dimensions only)
3. Build dimension mart: join only dimensions to anchor (exclude bridges)
4. Aggregate facts: group by grain, compute **safe** aggregations per policy
5. Join aggregated facts to mart
6. Return **Ibis expression** (lazy, not materialized)

### `materialize_mart(output_path: Path, grain: str = "patient", ...) -> CohortMetadata`

1. Compute run_id (deterministic hash for caching)
2. Check cache: if run_id exists, return existing metadata
3. Use Polars pipeline (`_build_dimension_mart` + `_aggregate_fact_tables`)
4. Execute plan (compile to SQL, run in DuckDB)
5. Write to partitioned Parquet (hash buckets for event-level)
6. Return metadata with table locations and schema

### `_aggregate_fact_tables(grain_key: str, aggregation_policy: Dict) -> Dict[str, pl.DataFrame]`

For each fact/event table:

1. Group by `grain_key`
2. Compute **safe** aggregations per policy:

- Always: counts, distinct counts, min/max timestamps, min/max numerics
- Opt-in: mean/avg (if allowed), last value (if allowed and conditions met)
- Never: mean/avg on code columns

3. Return dict of `{table_name: aggregated_df}`

### `_detect_bridge_table(table_name: str, df: pl.DataFrame, relationships: List[TableRelationship]) -> bool`

1. Count foreign keys (columns that are FK in relationships)
2. If two or more FKs to different parents:
3. Check if neither FK is unique BUT composite (fk1, fk2) is near-unique
4. Check if high relationship degree but low column payload
5. Return True if bridge detected

## Query Planner Contract

Add query planner interface to prevent "everything" queries:

```python
@dataclass(frozen=True)
class QueryPlan:
    grain: Literal["patient", "admission", "event"]
    required_tables: set[str]
    required_features: set[str]  # names of aggregated feature blocks
    filters: list[Filter]
    group_by: list[str]
    limit: int

def plan_from_nl(text: str, catalog: SemanticCatalog) -> QueryPlan:
    """Parse NL query and return minimal required tables/features."""
    ...
```

Execution flow:

1. NL query → QueryPlan (only required tables/features)
2. QueryPlan → Ibis expression using only relevant tables/features
3. Compiled SQL → DuckDB
4. Prevents "labs and vitals and meds and diagnoses and... everything" from triggering monster build

## Testing Strategy

### OOM Prevention Tests (Critical)

1. **MIMIC-IV Demo Test**:

- Load 32 tables from MIMIC-IV demo ZIP
- Verify no OOM when processing chartevents (668k rows)
- Verify temp directory usage < 10 GB (vs current 90.8 GiB)
- Verify successful mart creation

2. **Large Table Handling**:

- Test with tables > 100k rows
- Verify aggregation before join (not direct join)
- Verify memory usage stays bounded

3. **Bridge Table Tests**:

- Verify bridge tables excluded from auto-joins
- Verify no explosion from many-to-many relationships

### Standard Tests

4. **Unit tests** for table classification logic (including bridge detection)
5. **Unit tests** for byte-based size estimation
6. **Unit tests** for centrality-based anchor selection (with exclusions)
7. **Unit tests** for aggregation policy enforcement
8. **Integration tests** for aggregate-before-join pipeline
9. **Integration tests** for cross-platform file locking
10. **Regression tests** to ensure existing datasets still work
11. **Performance tests** to verify no mega-join explosions
12. **Query planner tests** to verify minimal table selection

## Dependencies

Add to `pyproject.toml`:

- `filelock` (cross-platform file locking)

## Alignment with consolidate-docs Plan

This refactor **completes Phase 4** of the [consolidate-docs plan](docs/implementation/plans/consolidate-docs-and-implement-question-driven-analysis.md):

### Phase 4 Status Update

**Before (Current State)**:

- ✅ ZIP upload with multiple CSVs - **COMPLETE**
- ✅ Auto-detect foreign key relationships - **COMPLETE**
- ✅ Build join graph - **COMPLETE**
- ❌ Execute joins via DuckDB SQL - **BLOCKED BY OOM**

**After (This Refactor)**:

- ✅ ZIP upload with multiple CSVs - **COMPLETE**
- ✅ Auto-detect foreign key relationships - **COMPLETE**
- ✅ Build join graph - **COMPLETE**
- ✅ Execute joins via aggregate-before-join (no OOM) - **FIXED**
- ✅ MIMIC-IV-style dataset support - **ENABLED**
- ✅ Unified cohort creation from related tables - **ENABLED**

### Integration Points

1. **UserDatasetStorage.save_zip_upload()**: Update to use `materialize_mart()` + `plan_mart()` instead of `build_unified_cohort()`
2. **NL Query Engine**: Integrate QueryPlan to prevent "everything" queries from triggering OOM
3. **Semantic Layer**: Support granularity mapping for multi-table datasets

### Success Criteria (Phase 4 Completion)

- [x] MIMIC-IV demo loads successfully (32 tables)
- [x] No OOM errors when processing large tables (chartevents 668k rows)
- [x] Relationships auto-detected (90%+ accuracy)
- [x] Unified cohort created with all relevant columns (via marts, not mega-joins)
- [x] Analyses work on joined data
- [x] User can override auto-detected joins

## Migration Notes

- Keep `build_unified_cohort()` as deprecated method (logs warning, can be wrapped around `materialize_mart()` + read in future)
- Default granularity to `"patient_level"` for backward compatibility
- Add feature flags to gradually roll out changes
- Bridge tables: not auto-joined into marts; require explicit query planner selection
- Aggregation policy: defaults to safe aggregations; mean/avg require explicit opt-in