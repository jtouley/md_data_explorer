# Phase 1: Critical Security & Data Integrity Fixes

**Timeline:** Week 1 (3-5 days)
**Priority:** P1 - CRITICAL
**Owner:** TBD
**Status:** Not Started

---

## Overview

This phase addresses **CRITICAL security vulnerabilities and data integrity issues** that must be fixed before ANY production deployment with real PHI/clinical data.

**Goals:**
- Fix SQL injection vulnerability
- Fix path traversal vulnerability
- Add strict data validation
- Implement transaction boundaries
- Fix class attribute mutation bug

---

## Tasks

### Task 1: Fix SQL Injection in Semantic Layer

**File:** `src/clinical_analytics/core/semantic.py:202-205`
**Severity:** 🔴 CRITICAL
**Effort:** 1 hour
**Owner:** TBD

**Current Code:**
```python
duckdb_con.execute(
    f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM read_csv_auto(?)",
    [abs_path]
)
```

**Issue:** `table_name` is f-string interpolated, allowing SQL injection

**Fix:**
```python
# Use quoted identifiers to prevent SQL injection
quoted_table = f'"{table_name}"'  # DuckDB identifier quoting
duckdb_con.execute(
    f'CREATE OR REPLACE TABLE {quoted_table} AS SELECT * FROM read_csv_auto(?)',
    [abs_path]
)
```

**Testing:**
```python
def test_sql_injection_prevention():
    """Test that malicious table names don't inject SQL"""
    malicious_name = '"; DROP TABLE patients; --'
    semantic = SemanticLayer(dataset_name=malicious_name, ...)
    # Should safely quote, not execute DROP
    assert '"' + malicious_name + '"' in semantic._get_table_name()
```

**Acceptance Criteria:**
- [ ] All table name interpolations use quoted identifiers
- [ ] Test with malicious input (SQL injection attempts)
- [ ] No SQL execution outside parameterized queries

---

### Task 2: Fix Path Traversal in ZIP Upload

**File:** `src/clinical_analytics/ui/storage/user_datasets.py:391-431`
**Severity:** 🔴 CRITICAL
**Effort:** 4 hours
**Owner:** TBD

**Current Code:**
```python
csv_files = [f for f in zip_file.namelist()
             if (f.endswith('.csv') or f.endswith('.csv.gz'))]
# No path validation before extraction!
```

**Issue:** Malicious ZIP can contain `../../../etc/passwd.csv`

**Fix:**
```python
def safe_extract(zip_file: zipfile.ZipFile, member: str, extract_to: Path) -> Path:
    """
    Safely extract ZIP member, preventing path traversal.

    Raises:
        SecurityError: If path traversal detected
    """
    target_path = (Path(extract_to) / member).resolve()

    # Check 1: Must be within extract directory
    if not target_path.is_relative_to(extract_to):
        raise SecurityError(f"Path traversal detected: {member}")

    # Check 2: No symlinks
    if target_path.is_symlink():
        raise SecurityError(f"Symlinks not allowed: {member}")

    # Check 3: Validate filename
    if '..' in member or member.startswith('/'):
        raise SecurityError(f"Invalid path: {member}")

    # Safe to extract
    return Path(zip_file.extract(member, extract_to))

# Usage:
for csv_file in csv_files:
    try:
        extracted_path = safe_extract(zip_file, csv_file, temp_dir)
    except SecurityError as e:
        logger.error(f"Security violation: {e}")
        raise
```

**Testing:**
```python
def test_path_traversal_prevention():
    """Test that malicious ZIP paths are blocked"""
    # Create malicious ZIP
    malicious_zip = BytesIO()
    with zipfile.ZipFile(malicious_zip, 'w') as zf:
        zf.writestr('../../../etc/passwd.csv', 'malicious data')

    # Should raise SecurityError
    with pytest.raises(SecurityError, match="Path traversal"):
        storage.save_zip_upload(malicious_zip.getvalue(), 'evil.zip', {})
```

**Acceptance Criteria:**
- [ ] All ZIP extractions use `safe_extract()` function
- [ ] Test with malicious paths (`../`, absolute paths, symlinks)
- [ ] No files written outside designated upload directory

---

### Task 3: Add Strict Value Mapping Validation

**File:** `src/clinical_analytics/core/semantic.py:240-256`
**Severity:** 🔴 CRITICAL
**Effort:** 2 hours
**Owner:** TBD

**Current Code:**
```python
result = ibis.null()  # Default for unmapped values
for key, value in reversed(list(mapping.items())):
    condition = _[source_col] == key
    result = condition.ifelse(value, result)

mutations[outcome_name] = result.cast('int64')  # Unmapped → NULL → 0
```

**Issue:** Unmapped values silently become NULL/0, corrupting outcome variable

**Fix:**
```python
# Track unmapped values
def apply_mapping_strict(
    table: ibis.Table,
    source_col: str,
    mapping: Dict[Any, int],
    outcome_name: str
) -> ibis.Table:
    """
    Apply value mapping with STRICT validation.

    Raises:
        DataQualityError: If unmapped values found
    """
    # Build expression
    result = ibis.null()
    for key, value in reversed(list(mapping.items())):
        condition = table[source_col] == key
        result = condition.ifelse(value, result)

    # Check for unmapped values BEFORE casting
    unmapped_count = table.filter(result.isnull() & table[source_col].notnull()).count().execute()

    if unmapped_count > 0:
        # Get examples of unmapped values
        unmapped_examples = (
            table.filter(result.isnull() & table[source_col].notnull())
            .select(source_col)
            .distinct()
            .limit(10)
            .execute()[source_col]
            .tolist()
        )

        raise DataQualityError(
            f"Found {unmapped_count} unmapped values in '{source_col}' → '{outcome_name}'. "
            f"All values must be explicitly mapped. "
            f"Unmapped examples: {unmapped_examples}. "
            f"Valid mappings: {list(mapping.keys())}"
        )

    return table.mutate({outcome_name: result.cast('int64')})
```

**Testing:**
```python
def test_unmapped_values_rejected():
    """Test that unmapped outcome values raise error"""
    mapping = {"Yes": 1, "No": 0}
    df = pl.DataFrame({"outcome_str": ["Yes", "No", "Unknown", "N/A"]})

    with pytest.raises(DataQualityError, match="unmapped values"):
        apply_mapping_strict(df, "outcome_str", mapping, "outcome")
```

**Acceptance Criteria:**
- [ ] All value mappings use strict validation
- [ ] Error raised if ANY unmapped values exist
- [ ] Error message shows examples of unmapped values
- [ ] No silent NULL/0 conversions

---

### Task 4: Implement Transaction Boundaries for ZIP Upload

**File:** `src/clinical_analytics/ui/storage/user_datasets.py:515-607`
**Severity:** 🔴 CRITICAL
**Effort:** 4 hours
**Owner:** TBD

**Current Code:**
```python
# Multi-step operation with no rollback:
handler = MultiTableHandler(tables)
relationships = handler.detect_relationships()
unified_df = handler.build_unified_cohort()
csv_path = self.raw_dir / f"{upload_id}.csv"
unified_df.write_csv(csv_path)  # Could fail here

# Separate file write (orphaned if above fails)
metadata_path = self.metadata_dir / f"{upload_id}.json"
with open(metadata_path, 'w') as f:
    json.dump(full_metadata, f, indent=2)
```

**Issue:** If any step fails, partial state left on disk with no cleanup

**Fix:**
```python
def save_zip_upload_atomic(
    self,
    file_bytes: bytes,
    original_filename: str,
    metadata: dict
) -> Tuple[bool, str, Optional[str]]:
    """
    Save ZIP upload with atomic write guarantees.

    Either ALL files are written or NONE (rollback on error).
    """
    upload_id = self._generate_upload_id(original_filename)
    temp_upload_id = f"{upload_id}_temp"

    # Temporary paths
    temp_csv = self.raw_dir / f"{temp_upload_id}.csv"
    temp_metadata = self.metadata_dir / f"{temp_upload_id}.json"
    temp_tables_dir = self.raw_dir / f"{temp_upload_id}_tables"

    # Final paths
    final_csv = self.raw_dir / f"{upload_id}.csv"
    final_metadata = self.metadata_dir / f"{upload_id}.json"
    final_tables_dir = self.raw_dir / f"{upload_id}_tables"

    try:
        # Step 1: Extract to temp directory
        tables = self._extract_zip_tables(file_bytes, temp_tables_dir)

        # Step 2: Build cohort
        handler = MultiTableHandler(tables)
        relationships = handler.detect_relationships()
        unified_df = handler.build_unified_cohort()

        # Step 3: Write to temp files
        unified_df.write_csv(temp_csv)

        with open(temp_metadata, 'w') as f:
            json.dump(metadata, f, indent=2)

        # Step 4: Atomic rename (all-or-nothing)
        temp_csv.rename(final_csv)
        temp_metadata.rename(final_metadata)
        if temp_tables_dir.exists():
            temp_tables_dir.rename(final_tables_dir)

        return True, f"Upload successful: {upload_id}", upload_id

    except Exception as e:
        # Rollback: cleanup ALL temp files
        logger.error(f"Upload failed, rolling back: {e}")
        self._cleanup_temp_files(temp_upload_id)
        return False, f"Upload failed: {e}", None

def _cleanup_temp_files(self, temp_upload_id: str):
    """Remove all temporary files for failed upload."""
    temp_csv = self.raw_dir / f"{temp_upload_id}.csv"
    temp_metadata = self.metadata_dir / f"{temp_upload_id}.json"
    temp_tables_dir = self.raw_dir / f"{temp_upload_id}_tables"

    for path in [temp_csv, temp_metadata]:
        if path.exists():
            path.unlink()

    if temp_tables_dir.exists():
        import shutil
        shutil.rmtree(temp_tables_dir)
```

**Testing:**
```python
def test_upload_rollback_on_failure(tmp_path, monkeypatch):
    """Test that failed uploads are fully rolled back"""
    storage = UserDatasetStorage(upload_dir=tmp_path)

    # Mock failure during cohort building
    def mock_build_cohort():
        raise RuntimeError("Simulated failure")

    monkeypatch.setattr(MultiTableHandler, "build_unified_cohort", mock_build_cohort)

    # Upload should fail
    success, msg, upload_id = storage.save_zip_upload(b"...", "test.zip", {})
    assert not success

    # No orphaned files should remain
    assert len(list(tmp_path.glob("*_temp*"))) == 0
    assert len(list(tmp_path.glob("*.csv"))) == 0
```

**Acceptance Criteria:**
- [ ] All multi-step operations use atomic writes
- [ ] Temp files cleaned up on any failure
- [ ] Test rollback scenarios (disk full, permission denied, etc.)
- [ ] No orphaned files after failed uploads

---

### Task 5: Fix Class Attribute Mutation Bug

**File:** `src/clinical_analytics/core/dataset.py:31`
**Severity:** 🔴 CRITICAL
**Effort:** 1 hour
**Owner:** TBD

**Current Code:**
```python
class ClinicalDataset(ABC):
    semantic: Optional["SemanticLayer"] = None  # ❌ CLASS ATTRIBUTE (shared!)
```

**Issue:** All instances share the same `semantic` attribute

**Fix:**
```python
class ClinicalDataset(ABC):
    def __init__(self, ...):
        self._semantic: Optional[SemanticLayer] = None  # Instance attribute

    @property
    def semantic(self) -> SemanticLayer:
        """Lazy-initialized semantic layer."""
        if self._semantic is None:
            raise ValueError(
                "Semantic layer not initialized. Call load() first."
            )
        return self._semantic

    def _init_semantic(self, ...):
        """Initialize semantic layer (called during load())."""
        if self._semantic is None:
            self._semantic = SemanticLayer(...)
```

**Testing:**
```python
def test_semantic_layer_isolation():
    """Test that each dataset has independent semantic layer."""
    ds1 = UploadedDataset("upload1")
    ds2 = UploadedDataset("upload2")

    ds1.load()
    ds2.load()

    # Should have different semantic layer instances
    assert ds1.semantic is not ds2.semantic

    # Modifying one should not affect the other
    ds1.semantic.con.execute("CREATE TABLE test1 ...")
    # ds2 should not have test1 table
```

**Acceptance Criteria:**
- [ ] All class attributes converted to instance attributes
- [ ] Property decorator for lazy initialization
- [ ] Test independence between instances
- [ ] No shared state between datasets

---

## Summary

### Effort Breakdown
- Task 1 (SQL Injection): 1 hour
- Task 2 (Path Traversal): 4 hours
- Task 3 (Value Mapping): 2 hours
- Task 4 (Transactions): 4 hours
- Task 5 (Class Attribute): 1 hour

**Total Estimated Effort:** 12 hours (1.5 days)

### Dependencies
- None - all tasks can be done in parallel

### Success Criteria
- [ ] All 5 tasks completed
- [ ] All tests passing
- [ ] Code reviewed by senior engineer
- [ ] Security review by external auditor
- [ ] No regressions in existing functionality

### Next Phase
After Phase 1 completion, proceed to **Phase 2: Data Integrity** (referential integrity, duplicate detection, error handling)
