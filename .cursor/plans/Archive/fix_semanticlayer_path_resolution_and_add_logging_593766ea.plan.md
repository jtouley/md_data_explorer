---
name: Fix SemanticLayer Path Resolution and Add Logging
overview: Fix path resolution bug in SemanticLayer._register_source() to resolve relative paths relative to workspace root, add comprehensive logging throughout initialization, and create tests to verify the fix works correctly.
todos:
  - id: "1"
    content: Add logging import and logger setup to semantic.py module
    status: completed
  - id: "2"
    content: Add workspace_root parameter to __init__() and _detect_workspace_root() helper method (config -> parameter -> marker detection -> cwd fallback)
    status: completed
    dependencies:
      - "1"
  - id: 2a
    content: Update _register_source() to use self.workspace_root for resolving relative paths
    status: completed
    dependencies:
      - "2"
  - id: "3"
    content: Add logging statements throughout _register_source() with appropriate levels (debug/info for path resolution, warning/error for failures)
    status: completed
    dependencies:
      - 2a
  - id: "4"
    content: Add logging to __init__() method for initialization tracking
    status: completed
    dependencies:
      - "1"
  - id: "5"
    content: Create test file tests/core/test_semantic_layer.py with fixtures for temporary test data
    status: completed
  - id: "6"
    content: Add test for relative path resolution (should resolve to workspace root)
    status: completed
    dependencies:
      - "5"
  - id: "7"
    content: Add test for absolute path handling (should work unchanged)
    status: completed
    dependencies:
      - "5"
  - id: "8"
    content: Add test for missing file error (should show resolved path in error)
    status: completed
    dependencies:
      - "5"
  - id: "9"
    content: Add test for directory source (should still raise NotImplementedError)
    status: completed
    dependencies:
      - "5"
  - id: "10"
    content: Add test for database table source (should work with db_table param)
    status: completed
    dependencies:
      - "5"
  - id: "11"
    content: Add test for workspace root detection (marker-based detection and fallback to cwd)
    status: completed
    dependencies:
      - "5"
  - id: 11a
    content: Add minimal logging test (verify stable substrings like 'resolved path' in logs, avoid fragile detailed assertions)
    status: completed
    dependencies:
      - "5"
      - "3"
  - id: "12"
    content: Run all tests and verify they pass
    status: completed
    dependencies:
      - "6"
      - "7"
      - "8"
      - "9"
      - "10"
      - "11"
      - 11a
---

# Fix SemanticLayer Path Resolution and Add Logging

## Problem

The `SemanticLayer._register_source()` method doesn't resolve relative paths relative to the workspace root, causing `FileNotFoundError` when config contains paths like `data/raw/covid_ms/GDSI_OpenDataset_Final.csv`. Additionally, there's no logging in the semantic layer, making debugging difficult.

## Solution

1. **Fix path resolution**: Resolve relative paths using robust workspace root detection (not hardcoded depth)
2. **Add structured logging**: Log path resolution at debug/info level, failures at warning/error level
3. **Add tests**: Verify relative/absolute path handling, error cases with resolved paths in messages

## Implementation

### 1. Fix Path Resolution in `semantic.py`

Update `SemanticLayer` class in [src/clinical_analytics/core/semantic.py](src/clinical_analytics/core/semantic.py):**Add workspace_root parameter to `__init__()`:**

- Accept optional `workspace_root: Optional[Path]` parameter
- Store as instance variable `self.workspace_root`

**Add `_detect_workspace_root()` helper method:**

- Priority order:

1. Use `workspace_root` parameter if provided
2. Check `config.get("workspace_root")` if present
3. Walk up from `__file__` parent until finding marker file (`.git`, `pyproject.toml`, `README.md`)
4. Fallback to `Path.cwd()` and log warning

**Update `_register_source()` method:**

- Add logging import at module level
- Use `self.workspace_root` to resolve relative paths
- Check if path is absolute before resolving (absolute paths unchanged)
- Log at debug level: original path, resolved path, workspace root used
- Log at info level: successful table registration
- Log at warning/error level: file not found, path resolution fallback

Key changes:

- No hardcoded directory depth
- Workspace root detection via markers (`.git`, `pyproject.toml`)
- Fallback to `Path.cwd()` with warning log
- Appropriate log levels (debug/info for happy path, warning/error for failures)

### 2. Add Logging Throughout SemanticLayer

Add logging with appropriate levels:

- `__init__()`: Log at info level for dataset name, workspace root detection result
- `_register_source()`: 
- Debug level: path resolution steps (original path, resolved path, workspace root)
- Info level: successful table registration
- Warning level: workspace root fallback to cwd()
- Error level: file not found (with resolved path in message)
- Avoid noisy logs on happy-path for every request

Use structured logging pattern consistent with other modules (see `covid_ms/definition.py`).

### 3. Create Tests

Create [tests/core/test_semantic_layer.py](tests/core/test_semantic_layer.py) with:

- **Test relative path resolution**: Verify relative paths resolve correctly when workspace_root provided
- **Test workspace root detection**: Verify marker-based detection (finds .git or pyproject.toml)
- **Test workspace root fallback**: Verify falls back to cwd() when no markers found (with warning log)
- **Test absolute path handling**: Verify absolute paths work unchanged
- **Test missing file error**: Verify exception message includes resolved path
- **Test directory source**: Verify NotImplementedError still raised (for Sepsis)
- **Test database table source**: Verify db_table path works
- **Test logging behavior**: Verify stable log substrings (e.g., "resolved path", "workspace root") - avoid fragile detailed assertions

Focus on behavior verification: correct resolved paths and exception messages include resolved paths.Use fixtures from `conftest.py` for project_root and create temporary test files.

## Files to Modify

1. **[src/clinical_analytics/core/semantic.py](src/clinical_analytics/core/semantic.py)**

- Add logging import and logger setup
- Add `workspace_root` parameter to `__init__()`
- Add `_detect_workspace_root()` helper method (marker-based detection with fallback)
- Update `_register_source()` to use `self.workspace_root` for path resolution
- Add logging statements with appropriate levels (debug/info/warning/error)

2. **[tests/core/test_semantic_layer.py](tests/core/test_semantic_layer.py)** (new file)

- Test workspace root detection (marker-based and fallback)
- Test relative path resolution behavior
- Test absolute path handling
- Test error cases (missing files, exception messages)
- Test minimal logging assertions (stable substrings only)

## Testing Strategy

1. **Unit tests**: Test workspace root detection logic in isolation (marker detection, fallback)
2. **Integration tests**: Test with actual config files and temporary data files
3. **Error case tests**: Verify exception messages include resolved paths
4. **Behavior verification**: Focus on correct path resolution behavior, not detailed log parsing
5. **Minimal logging tests**: Verify stable log substrings only (avoid fragile assertions)

## Success Criteria

- Relative paths in config resolve correctly using workspace root detection (not hardcoded depth)
- Workspace root detection works via markers (.git, pyproject.toml) or falls back to cwd()
- Absolute paths work unchanged
- Clear error messages when files don't exist (show resolved path in exception)
- Logging at appropriate levels (debug/info for path resolution, warning/error for failures)
- All tests pass