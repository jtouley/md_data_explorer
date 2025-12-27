---
name: Fix Registry Param Filtering and SemanticLayer Table Naming
overview: "Implement two systemic architecture fixes: (1) Registry filters init params by constructor signature to prevent \"unexpected keyword argument\" errors, (2) SemanticLayer generates SQL-safe table identifiers from dataset names and uses parameter binding to prevent SQL injection and special character issues."
todos:
  - id: "1"
    content: Add _filter_kwargs_for_ctor() helper function to registry.py with inspect.signature logic
    status: pending
  - id: "2"
    content: Update DatasetRegistry.get_dataset() to filter params using _filter_kwargs_for_ctor() before instantiation
    status: pending
    dependencies:
      - "1"
  - id: "3"
    content: Add _safe_identifier() helper function to semantic.py with sanitization and hashing logic
    status: pending
  - id: "4"
    content: Update SemanticLayer._register_source() to use _safe_identifier() for table names
    status: pending
    dependencies:
      - "3"
  - id: "5"
    content: Update SemanticLayer._register_source() to use parameter binding (?) for SQL queries instead of string interpolation
    status: pending
    dependencies:
      - "4"
  - id: "6"
    content: "Test: verify registry filtering works with configs containing extra params (e.g., db_connection for Mimic3Dataset)"
    status: pending
    dependencies:
      - "2"
  - id: "7"
    content: "Test: verify safe identifiers work with dataset names containing hyphens, dots, spaces, emojis"
    status: pending
    dependencies:
      - "4"
      - "5"
---

# Fix Registry Param Filtering and SemanticLayer Table Naming

## Problem

Two systemic architecture bugs cause failures with user-provided datasets:

1. **Registry blindly passes all config params**: `DatasetRegistry.get_dataset()` does `dataset_class(**params)` without checking if the constructor accepts those params. A stray `db_connection: null` in config explodes any dataset that doesn't accept it.
2. **SemanticLayer uses dataset names as SQL identifiers**: Table names like `mimic-iv-clinical-database-demo-2.2_raw` break SQL syntax. User-provided names with hyphens, dots, spaces, or emojis cause failures.

## Solution

Two targeted fixes at the architecture level (no changes to individual dataset classes):

1. **Registry param filtering**: Filter kwargs by constructor signature before instantiation
2. **SemanticLayer safe table naming**: Generate SQL-safe identifiers (sanitize + hash) and use parameter binding

## Implementation

### 1. Add Param Filtering to Registry

**File**: `src/clinical_analytics/core/registry.py`Add helper function and update `get_dataset()`:

```python
import inspect
import logging

logger = logging.getLogger(__name__)

def _filter_kwargs_for_ctor(cls, kwargs: dict) -> dict:
    """
    Filter kwargs to only include parameters accepted by the class constructor.
    
    Prevents "unexpected keyword argument" errors when configs contain params
    that a dataset class doesn't accept (e.g., db_connection for Mimic3Dataset).
    
    Args:
        cls: Dataset class to instantiate
        kwargs: Dictionary of parameters to filter
        
    Returns:
        Filtered dictionary with only accepted parameters
    """
    sig = inspect.signature(cls.__init__)
    params = sig.parameters
    
    # If constructor accepts **kwargs, pass everything through
    accepts_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
    if accepts_kwargs:
        return kwargs
    
    # Filter to only accepted parameters
    allowed = {k: v for k, v in kwargs.items() if k in params}
    dropped = sorted(set(kwargs) - set(allowed))
    
    if dropped:
        logger.info(
            f"Dropping unsupported init params for {cls.__name__}: {dropped}. "
            f"These parameters were ignored. Check dataset constructor signature."
        )
    
    return allowed
```

Update `get_dataset()` method:

```python
@classmethod
def get_dataset(cls, name: str, **override_params) -> ClinicalDataset:
    # ... existing discovery and validation code ...
    
    dataset_class = cls._datasets[name]
    config = cls._configs.get(name, {})
    params = {**config.get('init_params', {}), **override_params}
    
    # Filter params by constructor signature
    params = _filter_kwargs_for_ctor(dataset_class, params)
    
    # Instantiate and return
    return dataset_class(**params)
```



### 2. Add Safe Table Naming to SemanticLayer

**File**: `src/clinical_analytics/core/semantic.py`Add helper function at module level or as static method:

```python
import hashlib
import re

def _safe_identifier(name: str, max_len: int = 50) -> str:
    """
    Generate a SQL-safe identifier from a dataset name.
    
    Handles any user-provided name (hyphens, dots, spaces, emojis, etc.)
    by sanitizing and adding a hash for uniqueness.
    
    Args:
        name: Original dataset name (can contain any characters)
        max_len: Maximum length for base name (before hash)
        
    Returns:
        SQL-safe identifier (e.g., "mimic_iv_clinical_demo_2_2_a1b2c3d4")
    """
    # Replace non-alphanumeric (except underscore) with underscore
    base = re.sub(r"[^0-9a-zA-Z_]+", "_", name).strip("_").lower()
    
    # Ensure it starts with letter or underscore (not a number)
    if not base or base[0].isdigit():
        base = f"t_{base}" if base else "t"
    
    # Add hash suffix for uniqueness and collision prevention
    h = hashlib.sha256(name.encode("utf-8")).hexdigest()[:8]
    
    # Limit base length
    base = base[:max_len]
    
    return f"{base}_{h}"
```

Update `_register_source()` method:

```python
def _register_source(self):
    """Register the raw data source (CSV, table, etc.) with DuckDB."""
    init_params = self.config.get('init_params', {})
    
    if 'source_path' in init_params:
        # ... existing path resolution code ...
        
        # Generate SQL-safe table name from dataset name
        safe_name = _safe_identifier(self.dataset_name)
        table_name = f"{safe_name}_raw"
        
        # ... existing directory check ...
        
        else:
            # Single file (CSV)
            abs_path = str(source_path.resolve())
            logger.debug(f"Registering source file with DuckDB: {abs_path}")
            
            # Get underlying DuckDB connection
            duckdb_con = self.con.con
            
            # Use parameter binding to prevent SQL injection
            # DuckDB supports ? placeholders for parameterized queries
            duckdb_con.execute(
                f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM read_csv_auto(?)",
                [abs_path]
            )
            
            # Now reference it via Ibis
            self.raw = self.con.table(table_name)
            logger.info(f"Successfully registered source table '{table_name}' from {abs_path}")
    
    elif 'db_table' in init_params:
        # Database table source (already registered)
        # Note: db_table must be a valid DuckDB identifier (no special chars, or quoted)
        # If user provides a table name with special characters, they must quote it
        # or use a sanitized identifier. Ibis con.table() handles quoted identifiers.
        table_name = init_params['db_table']
        logger.debug(f"Using database table source: {table_name}")
        self.raw = self.con.table(table_name)
        logger.info(f"Successfully registered database table '{table_name}'")
    else:
        raise ValueError(f"No valid source found in config for {self.dataset_name}")
```



## Files to Modify

1. **`src/clinical_analytics/core/registry.py`**

- Add `import inspect` and `import logging` at top
- Add `_filter_kwargs_for_ctor()` helper function
- Update `get_dataset()` to filter params before instantiation

2. **`src/clinical_analytics/core/semantic.py`**

- Add `import hashlib` and `import re` at top
- Add `_safe_identifier()` helper function
- Update `_register_source()` to use safe identifiers and parameter binding

## Testing Considerations

- **Registry filtering**: Test with configs containing extra params (e.g., `db_connection: null` for Mimic3Dataset)
- **Safe identifiers**: Test with dataset names containing hyphens, dots, spaces, emojis, numbers at start
- **Parameter binding**: Verify SQL injection attempts are safely handled
- **Backward compatibility**: Existing datasets with simple names should continue working

## Success Criteria

- No more "unexpected keyword argument" errors from registry param mismatches
- No more SQL syntax errors from special characters in dataset names
- User-provided dataset names work regardless of characters used