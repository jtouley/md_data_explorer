---
name: ""
overview: ""
todos: []
---

---
name: "Milestone 5: Add Ibis Planning and Materialization Layer"
overview: ""
todos:
  - id: prereq-aggregate-before-join-unified-cohort
    content: "Prereq: build_unified_cohort() routes through _build_dimension_mart() + _aggregate_fact_tables(); no legacy DuckDB mega-join; regression tests added"
    status: completed
  - id: add-cohort-metadata-dataclass
    content: Add CohortMetadata dataclass with grain, grain_key, anchor_table, output_path, parquet_path (single path, not dict), schema (Dict[str, str] col->dtype), row_count, materialized_at, run_id (deterministic hash of config + schema + dataset fingerprint)
    status: pending
  - id: add-ibis-connection-helper
    content: Add helper method to create/reuse Ibis DuckDB connection in MultiTableHandler (or pass as parameter)
    status: pending
  - id: add-parquet-materialization-helper
    content: "Avoid Pandas default. Prefer: materialize to Parquet then ibis.read_parquet / duckdb read_parquet. For small frames only, allow Arrow registration."
    status: pending
    dependencies:
      - add-ibis-connection-helper
  - id: add-parquet-partitioning-helper
    content: Add helper method for hash bucket partitioning: add bucket column using pl.col(grain_key).hash(seed=0).mod(num_buckets).cast(pl.UInt16), then write_parquet(partition_by=['bucket']). Use stable hash seed for reproducibility.
    status: pending
  - id: add-caching-reuse-logic
    content: Add caching/reuse logic: compute run_id (sha256 of config + schema_version + dataset_fingerprint), check if {output_path}/{run_id}/ exists, if yes return existing metadata and skip compute. Enables doctors to refresh without paying compute cost twice.
    status: pending
    dependencies:
      - add-cohort-metadata-dataclass
  - id: implement-materialize-mart
    content: "Implement materialize_mart() method (renamed from materialize_patient_mart): uses Polars pipeline (_build_dimension_mart + _aggregate_fact_tables) to compute unified mart, writes single Parquet dataset to {output_path}/{run_id}/{grain}/mart.parquet (patient/admission) or {output_path}/{run_id}/{grain}/mart/ (event partitioned), includes caching check, returns CohortMetadata. Document memory constraint: materialization bounded because facts pre-aggregated to grain."
    status: pending
    dependencies:
      - add-parquet-materialization-helper
      - add-parquet-partitioning-helper
      - add-cohort-metadata-dataclass
      - add-caching-reuse-logic
  - id: implement-plan-mart
    content: "Implement plan_mart() method (renamed from plan_patient_mart): takes CohortMetadata or parquet_path, creates DuckDB Ibis connection, uses con.read_parquet() (handles partitioned directories natively), returns lazy Ibis.Table. Explicitly use DuckDB backend for read_parquet."
    status: pending
    dependencies:
      - implement-materialize-mart
      - add-ibis-connection-helper
  - id: add-observability-logging
    content: Add observability: log materialization duration, row counts, parquet size, query latency. Add metrics for time-to-first-answer. Enable debugging when doctors report slowness.
    status: pending
    dependencies:
      - implement-materialize-mart
      - implement-plan-mart
  - id: deprecate-build-unified-cohort
    content: "Optional: Deprecate build_unified_cohort() or make it a thin wrapper around materialize_mart() + read. Mark as 'interactive only' to prevent production use."
    status: pending
    dependencies:
      - implement-materialize-mart
  - id: add-plan-mart-tests
    content: "Add unit tests for plan_mart(): verify returns lazy Ibis expression, verify plan compiles to SQL without executing (ibis.to_sql() or expr.compile() produces SQL), verify no new files created (only validates path exists), verify uses DuckDB backend"
    status: pending
    dependencies:
      - implement-plan-mart
  - id: add-materialize-mart-tests
    content: "Add unit tests for materialize_mart(): verify Parquet files created, verify partitioning strategy (single file vs hash buckets), verify metadata returned correctly, verify caching works (skip recompute if run_id exists), verify rowcount matches build_unified_cohort().height"
    status: pending
    dependencies:
      - implement-materialize-mart
  - id: add-integration-tests
    content: "Add integration tests: verify materialized parquet is readable (DuckDB/Ibis scan rowcount == metadata.row_count), verify Ibis plan matches Polars pipeline results, verify can query materialized Parquet via Ibis, verify end-to-end materialization workflow, verify caching prevents recompute"
    status: pending
    dependencies:
      - implement-materialize-mart
      - implement-plan-mart
---

# Milestone 5: Add Ibis Planning and Materialization Layer

## Overview

Add `materialize_mart()` and `plan_mart()` methods to `MultiTableHandler` that provide materialization and lazy query planning over materialized Parquet. This enables semantic layer integration, query planning (M12), and partitioned Parquet storage.

**Key Architecture**: Materialize marts using Polars pipeline, then enable lazy SQL queries over Parquet via Ibis.

**Primary Success Metric**: Time-to-first-answer for doctors is low and consistent. System does not OOM or stall when dataset is ugly. Doctors can refresh without paying compute cost twice.

## Problem

The current `build_unified_cohort()` is compute-time only; it returns a Polars DataFrame that:

- Doesn't integrate with the Ibis semantic layer
- Can't be used for query planning (M12)
- Doesn't support lazy SQL generation over materialized data
- Doesn't materialize to partitioned Parquet
- Forces recomputation on every request (no caching)
- **More importantly**: `build_unified_cohort()` is compute-time; plan/query needs a persisted dataset boundary

**Note**: `build_unified_cohort()` now routes through aggregate-before-join pipeline (M3/M4 complete), but still returns Polars DataFrame and doesn't persist.

## Solution

Add two new methods that materialize marts and enable lazy queries:

1. **`materialize_mart()`**: Computes mart using Polars pipeline and writes Parquet
   - Uses existing `_build_dimension_mart()` and `_aggregate_fact_tables()` (Polars-based)
   - Writes single unified mart to `{output_path}/{grain}/mart/` (single file for patient/admission, hash-bucketed directory for event)
   - Includes caching: skip recompute if `run_id` exists
   - Returns `CohortMetadata` with paths, schema, row counts, and `run_id`

2. **`plan_mart()`**: Returns lazy Ibis expression over materialized Parquet
   - Takes `CohortMetadata` or `parquet_path`
   - Uses DuckDB backend explicitly: `con.read_parquet()` (handles partitioned directories natively)
   - Returns lazy Ibis.Table (SQL generation only, no execution)
   - Enables semantic layer queries to compile to SQL lazily

## Implementation

### File: `src/clinical_analytics/core/multi_table_handler.py`

#### 1. Add CohortMetadata Dataclass

```python
@dataclass
class CohortMetadata:
    """Metadata for materialized cohort mart."""
    grain: Literal["patient", "admission", "event"]
    grain_key: str
    anchor_table: str
    output_path: Path  # Base output directory
    parquet_path: Path  # Single path to mart Parquet (file or directory)
    schema: Dict[str, str]  # column_name -> dtype (e.g., "patient_id" -> "Utf8")
    row_count: int
    materialized_at: datetime
    run_id: str  # Deterministic hash: sha256(config + schema_version + dataset_fingerprint)
```

**Key changes from original**:
- `parquet_path` is single path (not `parquet_paths` dict) - single unified mart
- `schema` is `Dict[str, str]` (not `Dict[str, Dict[str, Any]]`) - single mart schema
- Added `run_id` for caching and reproducibility

#### 2. Add materialize_mart() Method

```python
def materialize_mart(
    self,
    output_path: Path,
    grain: str = "patient",
    anchor_table: Optional[str] = None,
    join_type: str = "left",
    num_buckets: int = 64,
    force_recompute: bool = False,
) -> CohortMetadata:
    """
    Compute mart using Polars pipeline and write to partitioned Parquet.
    
    Strategy:
      1) Compute run_id (deterministic hash of config + schema + dataset)
      2) Check cache: if {output_path}/{run_id}/ exists and not force_recompute, return existing metadata
      3) Use existing Polars pipeline (_build_dimension_mart, _aggregate_fact_tables)
      4) Collect Polars LazyFrames to DataFrames (bounded: facts pre-aggregated to grain)
      5) Write to Parquet:
         - Patient/admission level: single file at {output_path}/{run_id}/{grain}/mart.parquet
         - Event level: hash-bucketed directory at {output_path}/{run_id}/{grain}/mart/ (bucket=0, bucket=1, ...)
      6) Return metadata with paths, schema, row counts, run_id
    
    Memory Constraint:
      Materialization is bounded because facts/events are pre-aggregated to grain
      (one row per grain). Final mart should be manageable. If _aggregate_fact_tables()
      produces a wide mart that's still large, this should be documented as a limitation.
    
    Args:
        output_path: Base directory for Parquet output
        grain: Grain level (patient, admission, event)
        anchor_table: Optional anchor table (auto-detected if None)
        join_type: Join type for dimension mart
        num_buckets: Number of hash buckets for event-level partitioning
        force_recompute: If True, recompute even if cached version exists
    
    Returns:
        CohortMetadata with paths, schema, row counts, and run_id
    """
```

**Parquet partitioning strategy:**

- **Patient/Admission level**: Single Parquet file at `{output_path}/{run_id}/{grain}/mart.parquet`
- **Event level**: Hash bucket partitioning by grain_key
  - Directory: `{output_path}/{run_id}/{grain}/mart/bucket=0/*.parquet`, `bucket=1/*.parquet`, etc.
  - Add bucket column: `pl.col(grain_key).hash(seed=0).mod(num_buckets).cast(pl.UInt16).alias("bucket")`
  - Use stable hash seed (0) for reproducibility
  - Write with `partition_by=["bucket"]`

#### 3. Add plan_mart() Method

```python
def plan_mart(
    self,
    metadata: Optional[CohortMetadata] = None,
    parquet_path: Optional[Path] = None,
) -> ibis.Table:
    """
    Return lazy Ibis expression over materialized Parquet mart.
    
    Strategy:
      1) Resolve parquet_path from metadata or parameter
      2) Create/get DuckDB Ibis connection
      3) Use con.read_parquet() (DuckDB handles partitioned directories natively)
      4) Return lazy Ibis.Table (no computation, SQL generation only)
      5) Enables semantic layer queries to compile to SQL lazily
    
    Args:
        metadata: CohortMetadata (preferred, includes schema info)
        parquet_path: Path to materialized Parquet (file or directory)
    
    Returns:
        Lazy Ibis expression over Parquet (SQL not executed)
    
    Raises:
        ValueError: If neither metadata nor parquet_path provided
        FileNotFoundError: If Parquet path doesn't exist
    """
```

**Implementation approach:**
- Explicitly use DuckDB backend: `ibis.duckdb.connect()` then `con.read_parquet()`
- DuckDB `read_parquet()` handles partitioned directories natively (no glob patterns needed)
- Return lazy Ibis.Table (no materialization, SQL generation only)

#### 4. Helper Method: Hash Bucket Partitioning

```python
def _add_hash_bucket_column(
    self,
    df: pl.DataFrame,
    grain_key: str,
    num_buckets: int,
) -> pl.DataFrame:
    """
    Add deterministic hash bucket column for partitioning.
    
    Uses stable hash seed for reproducibility across runs.
    
    Args:
        df: Polars DataFrame
        grain_key: Column to hash
        num_buckets: Number of buckets
    
    Returns:
        DataFrame with 'bucket' column added
    """
    bucket_col = (
        pl.col(grain_key)
        .hash(seed=0)  # Stable seed for reproducibility
        .mod(num_buckets)
        .cast(pl.UInt16)
        .alias("bucket")
    )
    return df.with_columns(bucket_col)
```

#### 5. Helper Method: Compute Run ID

```python
def _compute_run_id(
    self,
    grain: str,
    anchor_table: str,
    grain_key: str,
    join_type: str,
) -> str:
    """
    Compute deterministic run_id for caching.
    
    Hash of: config (grain, anchor, grain_key, join_type) + schema_version + dataset fingerprint.
    
    Args:
        grain: Grain level
        anchor_table: Anchor table name
        grain_key: Grain key column
        join_type: Join type
    
    Returns:
        SHA256 hash as hex string
    """
    import hashlib
    
    # Build config string
    config_str = f"{grain}:{anchor_table}:{grain_key}:{join_type}"
    
    # Add schema version (if available)
    schema_version = "1.0"  # TODO: get from actual schema version
    
    # Add dataset fingerprint (hash of table names + row counts)
    dataset_fingerprint = self._compute_dataset_fingerprint()
    
    # Combine and hash
    combined = f"{config_str}:{schema_version}:{dataset_fingerprint}"
    return hashlib.sha256(combined.encode()).hexdigest()
```

## Key Design Decisions

### 1. Single Unified Mart (Option A)

**Decision**: Materialize one unified mart dataset, not multiple per-table datasets.

**Rationale**:
- Simpler API and implementation
- Avoids rebuilding join logic in SQL (scope creep)
- Matches stated goal: "enable semantic layer integration and planning"
- Single `parquet_path` in metadata, single `plan_mart()` call

### 2. Naming: materialize_mart() and plan_mart()

**Decision**: Remove "patient" from method names, keep `grain` parameter.

**Rationale**:
- `materialize_patient_mart(grain="event")` is a naming lie
- `materialize_mart(grain="patient")` is clear and consistent
- Matches single unified mart architecture

### 3. Materialization Strategy

**Decision**: Materialize Polars DataFrames directly to Parquet, then use Ibis `read_parquet()`.

**Rationale**:
- Avoids expensive Pandas conversion
- Parquet is efficient storage format
- Ibis/DuckDB can read Parquet lazily (no materialization during planning)
- For small intermediates only: use Arrow instead of Pandas if registration needed

### 4. Hash Bucketing Implementation

**Decision**: Explicitly add bucket column before partitioning.

**Rationale**:
- Polars `write_parquet(partition_by=...)` requires existing column
- Use stable hash seed (0) for reproducibility
- Zero-padding can be done at filesystem level if needed (bucket=0, bucket=01, etc.)

### 5. Caching and Reuse

**Decision**: Compute deterministic `run_id`, check if cached version exists, skip recompute.

**Rationale**:
- Doctors should not pay compute cost twice because someone refreshed the page
- Deterministic `run_id` ensures same config = same output
- Enables "time-to-first-answer" optimization

### 6. DuckDB Backend for Planning

**Decision**: Explicitly use DuckDB backend for `read_parquet()`.

**Rationale**:
- DuckDB handles partitioned directories natively
- Consistent with existing DuckDB usage in codebase
- Better performance for Parquet reads

### 7. Observability

**Decision**: Log materialization duration, row counts, parquet size, query latency.

**Rationale**:
- When doctors say "it's slow," answer with facts, not vibes
- Enables debugging and optimization
- Metrics for time-to-first-answer

## Testing Strategy

### Unit Tests

1. **plan_mart() returns lazy Ibis expression and compiles without executing**
   - Verify return type is `ibis.Table`
   - Verify plan compiles to SQL without executing: `sql = ibis.to_sql(expr)` or `expr.compile()` produces SQL
   - Verify no new files created (only validates path exists)
   - Verify uses DuckDB backend explicitly

2. **materialize_mart() uses Polars pipeline**
   - Verify uses `_build_dimension_mart()` and `_aggregate_fact_tables()`
   - Verify results match `build_unified_cohort()` (same row counts, columns, data)
   - Verify writes Parquet (not Pandas conversion)
   - Verify caching works: skip recompute if `run_id` exists

3. **materialize_mart() writes Parquet correctly**
   - Verify Parquet files created
   - Verify patient/admission level: single file
   - Verify event level: hash bucket partitioning
   - Verify metadata returned correctly (paths, schema, row counts, run_id)

4. **Parquet partitioning correctness**
   - Verify hash bucket distribution (all buckets used)
   - Verify no data loss (row count matches)
   - Verify can read back partitioned Parquet
   - Verify stable hash seed produces same buckets across runs

5. **Materialized parquet is readable and rowcount matches**
   - Verify DuckDB/Ibis scan rowcount == metadata.row_count
   - Verify metadata.row_count == build_unified_cohort().height
   - Verify can query materialized Parquet via Ibis

### Integration Tests

6. **End-to-end materialization and querying**
   - Load multi-table dataset
   - Materialize mart using Polars pipeline
   - Create Ibis plan over materialized Parquet
   - Verify can query materialized Parquet via Ibis (lazy SQL)
   - Verify semantic layer can use materialized mart
   - Verify query results match Polars pipeline results
   - Verify caching prevents recompute

## Dependencies

- **Prerequisite**: `build_unified_cohort()` routes through aggregate-before-join pipeline (M3/M4) - ✅ Complete
- Existing M3/M4 methods (`_build_dimension_mart()`, `_aggregate_fact_tables()`) - ✅ Complete
- Ibis library - Already in dependencies
- Polars Parquet writing - Already available

## Migration Notes

- `build_unified_cohort()` remains unchanged (backward compatibility)
- New methods are additive (no breaking changes)
- **Optional**: Deprecate `build_unified_cohort()` or make it a thin wrapper around `materialize_mart() + read`
- Can be used by `UserDatasetStorage.save_zip_upload()` in future (M9)
- Enables query planner integration (M12)

## Updated Scope

**Milestone 5 is**: Materialize marts (Polars) + Plan/Query marts (Ibis)

**Not**: "Convert Polars laziness into SQL laziness" (physics does not work that way)

The architecture is:
1. **Materialize**: Use Polars pipeline to compute and write Parquet (with caching)
2. **Plan**: Return lazy Ibis expression over Parquet (SQL generation, no execution)
3. **Query**: Semantic layer compiles Ibis expressions to SQL lazily

**Primary Success Metric**: Time-to-first-answer for doctors is low and consistent. System does not OOM or stall. Doctors can refresh without paying compute cost twice.

## Success Criteria

- [ ] `plan_mart()` returns lazy Ibis expression
- [ ] `materialize_mart()` writes partitioned Parquet
- [ ] Event-level tables use hash bucket partitioning
- [ ] Patient/admission level use single file
- [ ] Metadata includes paths, schema, row counts, run_id
- [ ] Caching works: skip recompute if run_id exists
- [ ] Materialized parquet is readable: DuckDB/Ibis scan rowcount == metadata.row_count
- [ ] All tests pass
- [ ] Can query materialized Parquet via Ibis
- [ ] Observability: logs include duration, row counts, parquet size, query latency