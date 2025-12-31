# ADR008 Implementation Progress

## Status: Phase 4.1 Complete (4/9 phases, 44%)

### Completed Phases ✅

**Phase 0: File Locking Helper**
- Added `file_lock()` context manager (fcntl/msvcrt)
- FileLockTimeoutError exception
- 2/2 tests passing
- Commit: 4214a39

**Phase 1: Cross-Dataset Deduplication**
- Added `find_datasets_by_content_hash()`
- Warns about duplicate content with different names
- Allows upload to proceed (UX-focused)
- 3/3 tests passing
- Commit: 61ebd44

**Phase 2: Version History Metadata**
- Added `compute_schema_fingerprint()`
- Canonical `tables` map structure
- Initialized `version_history` array
- Preserved stable internal table identifiers
- 3/3 tests passing
- Commit: 651ad8e

**Phase 3: Schema Drift Detection**
- Added `SchemaDriftPolicy` enum (REJECT, WARN, ALLOW)
- Added `detect_schema_drift()` function
- Detects added/removed columns, type changes
- 5/5 tests passing
- Commit: 53f3ae1

**Phase 4.1: Metadata Invariants**
- Added `assert_metadata_invariants()`
- Validates version_history structure
- Ensures exactly one active version
- 3/3 tests passing
- Commit: 484cc9e

### Remaining Phases

**Phase 4.2: Overwrite Behavior** (NEXT)
- Add `overwrite` parameter to save_upload()
- Preserve version history on overwrite
- Add new version entry with is_active=True
- Mark old version as is_active=False
- Handle schema drift per policy

**Phase 5: Event Logging**
- Add event_id and timestamp to version entries
- Log upload/overwrite/rollback events

**Phase 6: Rollback Mechanism**
- Implement `rollback_to_version()` method
- Switch active version
- Update metadata atomically with file_lock()

**Phase 7: Active Version Resolution**
- Implement `get_active_version()` algorithm
- Return active version entry from version_history

**Phase 8: Query Validation**
- Validate queries against active schema
- Prevent queries on rolled-back schemas

**Phase 9: UI Integration**
- Add overwrite checkbox to upload UI
- Display version history
- Add rollback button
- Manual testing

## Technical Debt / Notes
- Type checker has 246 pre-existing errors (not from ADR008 code)
- All ADR008 code passes lint, format, and tests
- Using test-first development methodology
- Commits after each phase with quality gates
