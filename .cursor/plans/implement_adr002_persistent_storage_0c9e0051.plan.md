---
name: Implement ADR002 Persistent Storage
overview: Implement persistent DuckDB storage, Parquet export, lazy evaluation, and conversation history for ADR002, following test-first workflow with phase commits and quality gates.
todos:
  - id: phase0-verify-feature-parity
    content: "Phase 0: Verify ADR007 feature parity completion (prerequisite check)"
    status: pending
  - id: phase1-dataset-versioning
    content: "Phase 1: Implement dataset versioning and metadata schema updates"
    status: pending
    dependencies:
      - phase0-verify-feature-parity
  - id: phase2-persistent-duckdb
    content: "Phase 2: Create DataStore class for persistent DuckDB storage"
    status: pending
    dependencies:
      - phase1-dataset-versioning
  - id: phase3-parquet-export
    content: "Phase 3: Add Parquet export and verify lazy evaluation"
    status: pending
    dependencies:
      - phase2-persistent-duckdb
  - id: phase4-session-recovery
    content: "Phase 4: Implement session recovery on app startup"
    status: pending
    dependencies:
      - phase3-parquet-export
  - id: phase5-conversation-history
    content: "Phase 5: Implement JSONL conversation history logger"
    status: pending
    dependencies:
      - phase4-session-recovery
  - id: phase6-integration-validation
    content: "Phase 6: End-to-end integration validation and success metrics"
    status: pending
    dependencies:
      - phase5-conversation-history
---

# Implement ADR002: Persistent Storage Layer with DuckDB

## Overview

This plan implements ADR002's persistent storage architecture: DuckDB-backed persistence, Parquet export for lazy Polars evaluation, dataset versioning, and JSONL conversation history. All phases follow test-first workflow with mandatory quality gates before commits.

## Prerequisites

- ADR007 (Feature Parity) must be completed or verified as complete
- Current state: `normalize_upload_to_table_list()` exists, lazy frames partially supported
- DuckDB tables currently in-memory only (`:memory:` connections)

## Architecture

```javascript
Upload → Polars Validation → DuckDB Table (persistent) → Parquet Export → Lazy Polars Queries
                                      ↓
                                Metadata JSON (with dataset_version)
                                      ↓
                          Conversation History (JSONL)
```

**Key Components:**

- `DataStore` class: Manages persistent DuckDB at `data/analytics.duckdb`
- Dataset versioning: Content hash for idempotent queries
- Parquet export: Columnar format for lazy scanning
- JSONL logger: Query/result audit trail

## Core Invariants

### Persistence Invariant

**"Given the same upload hash + semantic config, results are immutable and reused."**

This means:

- Same content hash → same `dataset_version` → same DuckDB table → same query results
- Re-uploading identical data reuses existing storage (no duplication)
- Query execution uses `(upload_id, dataset_version)` as idempotent run key
- Semantic config changes create new version (different results)

### Boundary Rules

**IO is eager, transforms are lazy, semantics are declarative.**

- **IO Boundary**: File reads (`pl.read_csv()`, `pl.read_parquet()`) are eager. Use `pl.scan_csv()`/`pl.scan_parquet()` for lazy IO when possible.
- **Transform Boundary**: All data transformations use Polars lazy frames (`pl.LazyFrame`). Materialize only at query execution or UI render.
- **Semantic Boundary**: Semantic layer config is declarative (JSON/YAML). No imperative logic in semantic layer initialization.
- **Documentation**: Any code that violates these boundaries must have explicit comment explaining why (e.g., Excel files require eager read due to Polars limitations).

## Phase 0: Verify Feature Parity (Prerequisite Check)

**Goal**: Verify ADR007 completion before proceeding. If incomplete, complete it first.**Files to Check:**

- [`src/clinical_analytics/ui/storage/user_datasets.py`](src/clinical_analytics/ui/storage/user_datasets.py) - Verify `normalize_upload_to_table_list()` usage
- [`src/clinical_analytics/datasets/uploaded/definition.py`](src/clinical_analytics/datasets/uploaded/definition.py) - Verify unified semantic layer registration

**Tasks:**

1. Verify both upload types use `normalize_upload_to_table_list()`
2. Verify both upload types save to `{upload_id}_tables/` directory
3. Verify both upload types register tables in DuckDB semantic layer
4. Verify `get_upload_data()` returns lazy frames (already implemented)

**Success Criteria:**

- Single-table and multi-table uploads have identical persistence structure
- Both upload types register all tables in DuckDB (in-memory is OK for now)
- No conditional logic based on upload type in save/load paths

**Quality Gate:**

- Run `make test-ui` to verify existing tests pass
- If ADR007 incomplete, implement it first (separate plan)

## Phase 1: Dataset Versioning & Metadata Schema (MVP)

**Goal**: Implement core dataset versioning for persistence invariant. Defer edge cases to Phase 5+.

**Scope**: MVP versioning only. Perfect dedup, re-upload handling, and edge cases deferred.

**Files to Create:**

- [`src/clinical_analytics/storage/__init__.py`](src/clinical_analytics/storage/__init__.py) - New storage module
- [`src/clinical_analytics/storage/versioning.py`](src/clinical_analytics/storage/versioning.py) - Version computation logic

**Files to Modify:**

- [`src/clinical_analytics/ui/storage/user_datasets.py`](src/clinical_analytics/ui/storage/user_datasets.py) - Add version computation to `save_table_list()`

**Test Files to Create:**

- [`tests/storage/__init__.py`](tests/storage/__init__.py)
- [`tests/storage/test_versioning.py`](tests/storage/test_versioning.py)

**Implementation Tasks:**

1. **Create `compute_dataset_version()` function** (MVP):
   ```python
   def compute_dataset_version(tables: list[pl.DataFrame]) -> str:
       """
       Compute content hash of canonicalized tables.
       
       MVP: Simple hash of sorted table data. Perfect dedup deferred to Phase 5+.
       """
       # Canonicalize: sort rows by first column, normalize column order
       # Return 16-char hex hash
       # TODO (Phase 5+): Handle re-uploads, detect duplicates, reuse storage
   ```

2. **Update metadata JSON schema** (MVP):

- Add `dataset_version` field to metadata
- Ensure `inferred_schema` format is used (not `variable_mapping`)
- Add `provenance.tables` with basic fingerprints
- **Deferred**: Complex provenance tracking, duplicate detection

3. **Update `save_table_list()`** (MVP):

- Compute `dataset_version` after table normalization
- Store version in metadata JSON
- Store basic table fingerprints in `provenance.tables`
- **Deferred**: Re-upload detection, storage reuse logic

**Test Requirements (AAA Pattern - MVP Only):**

```python
def test_compute_dataset_version_identical_tables_same_version():
    # Arrange: Two identical DataFrames
    # Act: Compute versions
    # Assert: Versions match

def test_compute_dataset_version_different_tables_different_version():
    # Arrange: Two different DataFrames
    # Act: Compute versions
    # Assert: Versions differ

def test_compute_dataset_version_canonicalization_order_independent():
    # Arrange: Same data, different row/column order
    # Act: Compute versions
    # Assert: Versions match (canonicalization works)

def test_save_table_list_stores_dataset_version():
    # Arrange: Storage, tables, metadata
    # Act: Save tables
    # Assert: Metadata contains dataset_version

# Deferred to Phase 5+: Re-upload detection, duplicate storage prevention
```

**Quality Gate:**

- Write tests first (Red)
- Run `make test-fast` (should fail)
- Implement versioning logic (Green)
- Run `make test-fast` (should pass)
- Run `make format` and `make lint-fix`
- Run `make type-check`
- Run `make check` (all gates pass)
- **Commit**: `feat: Phase 1 - Dataset versioning and metadata schema (MVP)`

## Phase 2: Persistent DuckDB Storage (MVP)

**Goal**: Create `DataStore` class managing persistent DuckDB. Focus on core persistence, defer optimization.

**Files to Create:**

- [`src/clinical_analytics/storage/datastore.py`](src/clinical_analytics/storage/datastore.py) - DataStore class

**Files to Modify:**

- [`src/clinical_analytics/ui/storage/user_datasets.py`](src/clinical_analytics/ui/storage/user_datasets.py) - Integrate DataStore into `save_table_list()`

**Test Files to Create:**

- [`tests/storage/test_datastore.py`](tests/storage/test_datastore.py)

**Implementation Tasks:**

1. **Create `DataStore` class** (MVP):
   ```python
   class DataStore:
       """
       Manages persistent DuckDB storage.
       
       Boundary: IO is eager (DuckDB writes), transforms are lazy (return LazyFrame).
       """
       def __init__(self, db_path: Path):
           """Initialize persistent DuckDB connection."""
           self.db_path = db_path
           self.conn = duckdb.connect(str(db_path))
       
       def save_table(
           self,
           table_name: str,
           data: pl.DataFrame,  # Eager IO boundary: materialize for DuckDB write
           upload_id: str,
           dataset_version: str,
       ) -> None:
           """
           Save table to DuckDB with versioning.
           
           Enforces persistence invariant: same (upload_id, dataset_version) → same table.
           """
           # CREATE TABLE IF NOT EXISTS with version suffix
           # Table name: {upload_id}_{table_name}_{dataset_version}
       
       def load_table(
           self,
           upload_id: str,
           table_name: str,
           dataset_version: str,
       ) -> pl.LazyFrame:  # Transform boundary: return lazy for downstream
           """
           Load table as Polars lazy frame.
           
           Boundary: Returns LazyFrame (lazy transform), not eager DataFrame.
           """
           # SELECT * FROM table, return pl.scan_duckdb()
       
       def list_datasets(self) -> list[dict]:
           """List all datasets in DuckDB."""
           # Query information_schema for table names
   ```

2. **Integrate into `save_table_list()`** (MVP):

- After saving CSV files, also save to DuckDB via DataStore
- Use `upload_id` and `dataset_version` for table naming (enforces persistence invariant)
- Keep CSV as export format (backward compatibility)
- **Deferred**: Table deduplication, storage optimization

3. **Update semantic layer initialization** (MVP):

- Modify `_maybe_init_semantic()` to load from persistent DuckDB
- Use DataStore to query tables instead of CSV files
- Fallback to CSV if DuckDB table missing (migration path)
- **Document boundary**: Semantic layer config is declarative, initialization uses DataStore (IO boundary)

**Test Requirements:**

```python
@pytest.fixture
def datastore(tmp_path):
    """Create DataStore with temporary database."""
    db_path = tmp_path / "test.duckdb"
    return DataStore(db_path)

def test_datastore_save_table_persists_data(datastore, sample_table):
    # Arrange: DataStore, sample table
    # Act: Save table
    # Assert: Table exists in DuckDB, data matches

def test_datastore_load_table_returns_lazy_frame(datastore, sample_table):
    # Arrange: Save table first
    # Act: Load table
    # Assert: Returns LazyFrame, data matches

def test_datastore_list_datasets_returns_all_uploads(datastore):
    # Arrange: Save multiple tables from different uploads
    # Act: List datasets
    # Assert: Returns all upload_ids

def test_datastore_table_survives_restart(tmp_path, sample_table):
    # Arrange: Save table, close connection
    # Act: Create new DataStore, load table
    # Assert: Data persists across connections
```

**Quality Gate:**

- Write tests first (Red)
- Run `make test-fast` (should fail)
- Implement DataStore (Green)
- Run `make test-fast` (should pass)
- Run `make format` and `make lint-fix`
- Run `make type-check`
- Run `make check` (all gates pass)
- **Commit**: `feat: Phase 2 - Persistent DuckDB storage (MVP)`

## Phase 3: Parquet Export & Lazy Evaluation

**Goal**: Export DuckDB tables to Parquet for lazy Polars scanning.**Files to Modify:**

- [`src/clinical_analytics/storage/datastore.py`](src/clinical_analytics/storage/datastore.py) - Add Parquet export
- [`src/clinical_analytics/ui/storage/user_datasets.py`](src/clinical_analytics/ui/storage/user_datasets.py) - Update `get_upload_data()` to use Parquet

**Test Files to Modify:**

- [`tests/storage/test_datastore.py`](tests/storage/test_datastore.py) - Add Parquet tests

**Implementation Tasks:**

1. **Add Parquet export to DataStore**:
   ```python
         def export_to_parquet(
             self,
             upload_id: str,
             table_name: str,
             dataset_version: str,
             parquet_dir: Path,
         ) -> Path:
             """Export DuckDB table to Parquet file."""
             # COPY table TO 'path.parquet' (FORMAT PARQUET)
             # Return path to Parquet file
   ```

2. **Update `save_table_list()`**:

- After saving to DuckDB, export to Parquet
- Store Parquet path in metadata
- Verify compression ratio (target: ≥40% smaller than CSV)

3. **Update `get_upload_data()`**:

- Check for Parquet file first (if exists, use `pl.scan_parquet()`)
- Fallback to CSV if Parquet missing (migration path)
- Return lazy frame (already implemented, verify it works)

4. **Update query execution paths**:

- Verify `query_describe.py` uses lazy frames
- Verify `query_compare.py` uses lazy frames
- Add predicate pushdown tests

**Test Requirements:**

```python
def test_export_to_parquet_creates_file(datastore, sample_table, tmp_path):
    # Arrange: Save table to DuckDB
    # Act: Export to Parquet
    # Assert: Parquet file exists, can be scanned

def test_parquet_compression_smaller_than_csv(datastore, sample_table, tmp_path):
    # Arrange: Save table, export to Parquet and CSV
    # Act: Compare file sizes
    # Assert: Parquet ≥40% smaller than CSV

def test_get_upload_data_uses_parquet_when_available(upload_storage, create_upload):
    # Arrange: Upload with Parquet export
    # Act: Load data
    # Assert: Uses pl.scan_parquet() (verify via query plan)

def test_lazy_evaluation_predicate_pushdown(upload_storage, create_upload):
    # Arrange: Large dataset (1000+ rows)
    # Act: Filter before collect
    # Assert: Only filtered rows materialized (verify via logging)
```

**Quality Gate:**

- Write tests first (Red)
- Run `make test-fast` (should fail)
- Implement Parquet export (Green)
- Run `make test-fast` (should pass)
- Run `make format` and `make lint-fix`
- Run `make type-check`
- Run `make check` (all gates pass)
- **Commit**: `feat: Phase 3 - Parquet export and lazy evaluation`

## Phase 4: Session Recovery

**Goal**: Restore datasets on app startup from persistent DuckDB.**Files to Modify:**

- [`src/clinical_analytics/ui/app.py`](src/clinical_analytics/ui/app.py) - Add session recovery logic

**Test Files to Create:**

- [`tests/ui/test_session_recovery.py`](tests/ui/test_session_recovery.py)

**Implementation Tasks:**

1. **Add `restore_datasets()` function**:
   ```python
         def restore_datasets(storage: UserDatasetStorage, datastore: DataStore) -> list[dict]:
             """Detect existing datasets and return metadata."""
             # Query DataStore for all upload_ids
             # Load metadata JSON for each
             # Return list of dataset metadata
   ```

2. **Update app startup**:

- Call `restore_datasets()` on app initialization
- Populate session state with available datasets
- Display "Previous datasets" UI if datasets exist

3. **Add UI for dataset selection**:

- Show list of available datasets
- Allow user to select dataset to restore
- Load selected dataset into session

**Test Requirements:**

```python
def test_restore_datasets_detects_existing_uploads(upload_storage, datastore):
    # Arrange: Save multiple uploads
    # Act: Restore datasets
    # Assert: Returns all upload metadata

def test_restore_datasets_handles_missing_metadata(upload_storage, datastore):
    # Arrange: DuckDB table exists but metadata JSON missing
    # Act: Restore datasets
    # Assert: Skips dataset with warning, continues

def test_session_recovery_loads_dataset_on_startup(upload_storage, datastore):
    # Arrange: Save upload, simulate app restart
    # Act: Initialize app
    # Assert: Dataset available in session state
```

**Quality Gate:**

- Write tests first (Red)
- Run `make test-fast` (should fail)
- Implement session recovery (Green)
- Run `make test-fast` (should pass)
- Run `make format` and `make lint-fix`
- Run `make type-check`
- Run `make check` (all gates pass)
- **Commit**: `feat: Phase 4 - Session recovery`

## Phase 5: Conversation History (JSONL Logger)

**Goal**: Implement JSONL conversation history for query/result audit trail.**Files to Create:**

- [`src/clinical_analytics/storage/query_logger.py`](src/clinical_analytics/storage/query_logger.py) - QueryLogger class

**Files to Modify:**

- [`src/clinical_analytics/core/nl_query_engine.py`](src/clinical_analytics/core/nl_query_engine.py) - Hook into query parsing
- [`src/clinical_analytics/analysis/compute.py`](src/clinical_analytics/analysis/compute.py) - Hook into execution
- [`src/clinical_analytics/ui/pages/3_💬_Ask_Questions.py`](src/clinical_analytics/ui/pages/3_💬_Ask_Questions.py) - Hook into UI interactions

**Test Files to Create:**

- [`tests/storage/test_query_logger.py`](tests/storage/test_query_logger.py)

**Implementation Tasks:**

1. **Create `QueryLogger` class**:
   ```python
         class QueryLogger:
             def __init__(self, log_dir: Path):
                 """Initialize JSONL logger."""
                 self.log_dir = log_dir
             
             def log_query(
                 self,
                 upload_id: str,
                 query_text: str,
                 query_plan: dict,
                 parsing_attempts: list[dict],
             ) -> None:
                 """Log query parsing event."""
                 # Append JSONL entry with query metadata
             
             def log_execution(
                 self,
                 upload_id: str,
                 run_key: str,
                 execution_time_ms: int,
                 cohort_shape: tuple[int, int],
             ) -> None:
                 """Log query execution event."""
             
             def log_result(
                 self,
                 upload_id: str,
                 result_type: str,
                 result_metadata: dict,
             ) -> None:
                 """Log query result."""
             
             def log_follow_up(
                 self,
                 upload_id: str,
                 suggestions_shown: list[str],
                 suggestion_clicked: str | None,
             ) -> None:
                 """Log follow-up suggestion interactions."""
   ```

2. **Integrate logging hooks**:

- Hook into `NLQueryEngine.parse_query()` for parsing events
- Hook into `execute_analysis_with_idempotency()` for execution events
- Hook into result rendering for result metadata
- Hook into follow-up suggestions for UI interactions

3. **Add conversation history UI**:

- Display query history in "Ask Questions" page
- Show query text, result summary, timestamp
- Allow user to re-run queries from history

**Test Requirements:**

```python
@pytest.fixture
def query_logger(tmp_path):
    """Create QueryLogger with temporary log directory."""
    return QueryLogger(tmp_path / "logs")

def test_log_query_appends_jsonl_entry(query_logger, sample_query_plan):
    # Arrange: QueryLogger, query plan
    # Act: Log query
    # Assert: JSONL file contains entry, valid JSON

def test_log_execution_appends_to_same_file(query_logger):
    # Arrange: Log query first
    # Act: Log execution
    # Assert: Both entries in same file, correct order

def test_load_conversation_history_restores_queries(query_logger):
    # Arrange: Log multiple queries
    # Act: Load history
    # Assert: Returns all queries in order

def test_query_logger_handles_file_rotation(query_logger):
    # Arrange: Log many entries (>1000)
    # Act: Check file size
    # Assert: File rotation occurs (if implemented)
```

**Quality Gate:**

- Write tests first (Red)
- Run `make test-fast` (should fail)
- Implement QueryLogger (Green)
- Run `make test-fast` (should pass)
- Run `make format` and `make lint-fix`
- Run `make type-check`
- Run `make check` (all gates pass)
- **Commit**: `feat: Phase 5 - Conversation history JSONL logger`

## Phase 6: Integration & Validation

**Goal**: End-to-end validation of all phases.**Test Files to Create:**

- [`tests/storage/test_integration.py`](tests/storage/test_integration.py)

**Validation Tasks:**

1. **Test upload → DuckDB → Parquet → Query flow**:

- Upload dataset
- Verify DuckDB table created
- Verify Parquet file created
- Verify query uses lazy frame
- Verify data persists across app restart

2. **Test conversation history persistence**:

- Run 5 queries
- Restart app
- Verify all 5 queries in history

3. **Test dataset versioning (MVP)**:

- Upload dataset, verify `dataset_version` stored
- Upload different data, verify different `dataset_version`
- **Deferred to Phase 5+**: Re-upload same data, verify storage reuse (perfect dedup)

4. **Test Parquet compression**:

- Upload 100MB dataset
- Verify Parquet ≥40% smaller than CSV

5. **Test lazy evaluation**:

- Upload 1M row dataset
- Run query with filters
- Verify no OOM errors
- Verify predicate pushdown works

**Test Requirements:**

```python
@pytest.mark.slow
@pytest.mark.integration
def test_end_to_end_upload_persistence_query(upload_storage, datastore):
    # Arrange: Fresh storage
    # Act: Upload, query, restart, query again
    # Assert: Data persists, queries work

@pytest.mark.slow
@pytest.mark.integration
def test_conversation_history_survives_restart(query_logger, upload_storage):
    # Arrange: Upload, run queries
    # Act: Restart app, load history
    # Assert: All queries in history

@pytest.mark.slow
@pytest.mark.integration
def test_lazy_evaluation_1m_rows_no_oom(upload_storage):
    # Arrange: 1M row dataset
    # Act: Run query with filters
    # Assert: No OOM, predicate pushdown works
```

**Quality Gate:**

- Run `make test` (all tests including slow/integration)
- Run `make check` (all quality gates)
- Verify success metrics from ADR002:
- [ ] Upload 100MB dataset, refresh → data reloads in <2 seconds
- [ ] Run 5 queries, restart app → conversation history shows all 5
- [ ] Test with 1M rows → queries execute without OOM errors
- [ ] Parquet files are ≥40% smaller than source CSV
- **Commit**: `feat: Phase 6 - Integration validation complete`

## Success Metrics (Go/No-Go)

From ADR002, must verify:

- [ ] Upload 100MB dataset, refresh page → data reloads in <2 seconds
- [ ] Run 5 queries, restart app → conversation history shows all 5
- [ ] Test with 1M rows → queries execute without OOM errors
- [ ] Parquet files are ≥40% smaller than source CSV

## Quality Gate Checklist (Per Phase)

Before every commit:

- [ ] Tests written and passing (`make test-fast` or module-specific)
- [ ] Code formatted (`make format`)
- [ ] Linting fixed (`make lint-fix`)
- [ ] Type checking passes (`make type-check`)
- [ ] Full quality gate passes (`make check`)
- [ ] No duplicate imports
- [ ] AAA pattern in tests
- [ ] Polars-first (no pandas except at boundaries)
- [ ] DRY principles followed

## Notes

- **ADR007 Dependency**: Phase 0 verifies ADR007 completion. If incomplete, implement ADR007 first.
- **Backward Compatibility**: Keep CSV export for migration path. Parquet is preferred but CSV fallback exists.
- **Session State**: Store only references (upload_id) in session state, not full data.
- **Error Handling**: All storage operations must handle failures gracefully (log, don't crash UI).
- **Privacy**: Conversation history contains no PHI (only metadata, counts, column names).

## Deferred to Phase 5+ (Explicit TODOs)

**Staff feedback**: Over-optimizing for theoretical reuse in V1. These are correct long-term but Phase-5 polish.

1. **Perfect Deduplication**:

   - Re-upload detection (same content hash)
   - Storage reuse (don't duplicate tables/artifacts)
   - TODO: Add to Phase 5+ plan

2. **Edge Case Handling**:

   - Multi-table re-uploads with partial changes
   - Schema evolution handling
   - TODO: Add to Phase 5+ plan

3. **Storage Optimization**:

   - Table compression strategies
   - Archive old versions
   - TODO: Add to Phase 5+ plan

**MVP Focus**: Ship core persistence invariant first. Move unresolved elegance into explicit follow-up plans.

## References

- [ADR002: Persistent Storage Layer](docs/implementation/ADR/ADR002.md)
- [ADR007: Feature Parity Architecture](docs/implementation/ADR/ADR007.md)
- [Plan Execution Hygiene](.cursor/rules/104-plan-execution-hygiene.mdc)