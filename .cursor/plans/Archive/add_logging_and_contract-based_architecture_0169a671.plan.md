---
name: Add Logging and Contract-Based Architecture
overview: "Add comprehensive logging to app/UI entry point AND refactor to contract-based architecture. Fixes: (1) semantic layer as class attribute with TYPE_CHECKING forward refs, (2) get_semantic_layer() contract method, (3) lazy semantic init in UploadedDataset with absolute paths, (4) centralized require_outcome() helper, (5) logging configured once in app.py (not per page)."
todos:
  - id: "1"
    content: Create shared logging configuration module (logging_config.py) with truly idempotent configure_logging()
    status: completed
  - id: "2"
    content: Create UI helpers module (helpers.py) with require_outcome() function
    status: completed
  - id: "3"
    content: "Add semantic contract to ClinicalDataset: class attribute semantic: Optional[SemanticLayer] with TYPE_CHECKING forward refs"
    status: completed
  - id: "4"
    content: Add get_semantic_layer() method to ClinicalDataset base class
    status: completed
    dependencies:
      - "3"
  - id: "5"
    content: Update UploadedDataset to override get_semantic_layer() with lazy initialization using absolute paths
    status: completed
    dependencies:
      - "3"
      - "4"
  - id: "6"
    content: Add logging configuration to app.py (ONLY entry point - not in pages)
    status: completed
    dependencies:
      - "1"
  - id: "7"
    content: "Update app.py: replace hasattr(dataset, semantic) with try/except get_semantic_layer() pattern"
    status: completed
    dependencies:
      - "4"
      - "6"
  - id: "8"
    content: "Update app.py: replace scattered OUTCOME checks with require_outcome() helper where appropriate"
    status: completed
    dependencies:
      - "2"
      - "6"
  - id: "9"
    content: "Update 7_🔬_Analyze.py: replace hasattr with try/except get_semantic_layer() (NO logging config)"
    status: completed
    dependencies:
      - "4"
  - id: "10"
    content: "Update 4_🎯_Risk_Factors.py: use require_outcome() helper (NO logging config)"
    status: completed
    dependencies:
      - "2"
  - id: "11"
    content: Remove existing logging config from 1_📤_Upload_Data.py (logging done in app.py)
    status: in_progress
    dependencies:
      - "6"
  - id: "12"
    content: "Test: verify logs appear in terminal, semantic layer works for uploaded datasets, outcome checks work correctly"
    status: completed
    dependencies:
      - "5"
      - "6"
      - "7"
      - "8"
      - "9"
      - "10"
---

# Add Logging and Contract-Based Architecture

## Problem

Two related issues:

1. **No logging**: Only one page has logging, making debugging impossible
2. **Spaghetti code**: Capabilities encoded as random `hasattr()` checks and ad-hoc branching scattered across UI files

Current problems:

- `hasattr(dataset, 'semantic')` checks in multiple places
- `if UnifiedCohort.OUTCOME in cohort.columns` scattered everywhere
- Attribute exists but is None traps
- Same bugs rediscovered in multiple places
- Import cycle risks
- Logging configured per-page causing Streamlit handler issues

## Solution

Three-part refactor with proper safeguards:

1. **Semantic layer as contract**: Class attribute with TYPE_CHECKING forward refs, `get_semantic_layer()` method
2. **Centralized outcome checks**: Helper function `require_outcome()` for analyses that need outcomes
3. **Comprehensive logging**: Configured once in app.py entry point (not per page)

## Implementation

### 1. Add Semantic Layer Contract to ClinicalDataset

Update `src/clinical_analytics/core/dataset.py`:

```python
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from clinical_analytics.core.semantic import SemanticLayer

class ClinicalDataset(ABC):
    # Class attribute - exists regardless of __init__() being called
    semantic: Optional["SemanticLayer"] = None
    
    def __init__(self, name: str, source_path: Union[str, Path, None] = None, db_connection: Any = None):
        self.name = name
        self.source_path = Path(source_path) if source_path else None
        self.db_connection = db_connection
        # Note: semantic is a class attribute, not set here
    
    def get_semantic_layer(self) -> "SemanticLayer":
        """
        Get semantic layer for this dataset.
        
        Returns:
            SemanticLayer instance
            
        Raises:
            ValueError: If semantic layer is not available for this dataset
        """
        if self.semantic is None:
            raise ValueError(
                f"Dataset '{self.name}' does not support semantic layer features "
                "(natural language queries, SQL preview, query builder). "
                "This is typically only available for registry datasets with config-driven semantic layers."
            )
        return self.semantic
```



### 2. Update UploadedDataset to Support Lazy Semantic Layer

Update `src/clinical_analytics/datasets/uploaded/definition.py`:

```python
class UploadedDataset(ClinicalDataset):
    def __init__(self, upload_id: str, storage: Optional[UserDatasetStorage] = None):
        # ... existing init code ...
        self._semantic_initialized = False  # Track lazy init
    
    def get_semantic_layer(self) -> "SemanticLayer":
        """
        Get semantic layer, lazy-initializing from unified cohort if available.
        
        Overrides base class to support multi-table uploads.
        """
        # Lazy initialize if not already done
        if not self._semantic_initialized:
            self._maybe_init_semantic()
            self._semantic_initialized = True
        
        # Call parent to get semantic (or raise if None)
        return super().get_semantic_layer()
    
    def _maybe_init_semantic(self) -> None:
        """Lazy initialization of semantic layer for multi-table uploads."""
        # Only for multi-table uploads with inferred_schema
        inferred_schema = self.metadata.get('inferred_schema')
        if not inferred_schema:
            return
        
        csv_path = self.storage.raw_dir / f"{self.upload_id}.csv"
        if not csv_path.exists():
            logger.warning(f"Unified cohort CSV not found for upload {self.upload_id}")
            return
        
        try:
            config = self._build_config_from_inferred_schema(inferred_schema)
            workspace_root = self.storage.upload_dir.parent.parent
            
            # Use absolute path - SemanticLayer handles absolute paths correctly
            config['init_params'] = {'source_path': str(csv_path.resolve())}
            
            self.semantic = SemanticLayer(
                dataset_name=self.name,
                config=config,
                workspace_root=workspace_root
            )
            logger.info(f"Created semantic layer for uploaded dataset '{self.name}'")
        except Exception as e:
            logger.warning(f"Failed to create semantic layer: {e}")
            # Leave self.semantic as None (class attribute default)
    
    def _build_config_from_inferred_schema(self, inferred_schema: Dict[str, Any]) -> Dict[str, Any]:
        """Build semantic layer config from inferred schema."""
        outcomes = inferred_schema.get('outcomes', {})
        default_outcome = list(outcomes.keys())[0] if outcomes else None
        
        config = {
            'name': self.name,
            'display_name': self.metadata.get('original_filename', self.name),
            'status': 'available',
            'init_params': {},  # Will be set to absolute CSV path
            'column_mapping': inferred_schema.get('column_mapping', {}),
            'outcomes': outcomes,
            'time_zero': inferred_schema.get('time_zero', {}),
            'analysis': {
                'default_outcome': default_outcome,
                'default_predictors': inferred_schema.get('predictors', []),
                'categorical_variables': inferred_schema.get('categorical_columns', [])
            }
        }
        return config
```



### 3. Create Centralized Outcome Helper

Create `src/clinical_analytics/ui/helpers.py`:

```python
"""
UI helper functions for common checks and validations.
"""

import streamlit as st
import pandas as pd
from clinical_analytics.core.schema import UnifiedCohort


def require_outcome(cohort: pd.DataFrame, analysis_name: str = "This analysis") -> None:
    """
    Require outcome column in cohort, show error and stop execution if missing.
    
    Args:
        cohort: Cohort DataFrame
        analysis_name: Name of analysis requiring outcome (for error message)
        
    Raises:
        SystemExit: If outcome is missing (stops page execution via st.stop())
    """
    if UnifiedCohort.OUTCOME not in cohort.columns:
        st.error(
            f"{analysis_name} requires an outcome variable, but none was found in the dataset. "
            "Please upload data with outcome mapping or select a dataset that includes outcomes."
        )
        st.stop()
```



### 4. Create Shared Logging Configuration

Create `src/clinical_analytics/ui/logging_config.py`:

```python
"""
Centralized logging configuration for Streamlit UI.

Configure once in app.py entry point, not per page.
"""

import logging
import sys


def configure_logging(level: int = logging.INFO) -> None:
    """
    Configure Python logging for the entire application.
    
    Truly idempotent: safe to call multiple times without side effects.
    Checks if root logger already has handlers before configuring.
    No force=True to avoid Streamlit handler issues.
    """
    root_logger = logging.getLogger()
    
    # Only configure if no handlers exist (truly idempotent)
    if root_logger.handlers:
        return
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
        # No force=True - rely on idempotency check above
    )
    
    # Set specific loggers
    logging.getLogger('clinical_analytics.core.semantic').setLevel(logging.INFO)
    logging.getLogger('clinical_analytics.core.registry').setLevel(logging.INFO)
    logging.getLogger('clinical_analytics.core.multi_table_handler').setLevel(logging.INFO)
    logging.getLogger('clinical_analytics.ui.storage.user_datasets').setLevel(logging.INFO)
    logging.getLogger('clinical_analytics.datasets').setLevel(logging.INFO)
    
    # Reduce noise
    logging.getLogger('streamlit').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
```



### 5. Update UI Files to Use New Patterns

**app.py** (ONLY place logging is configured):

- Add logging configuration at top (before other imports)
- Replace `hasattr(dataset, 'semantic')` with `try/except get_semantic_layer()`
- Replace `if UnifiedCohort.OUTCOME in cohort.columns` with conditional logic using helper

**7_🔬_Analyze.py**:

- NO logging config (already configured in app.py)
- Replace `hasattr(dataset, 'semantic')` with `try/except get_semantic_layer()`

**app.py display_query_builder()**:

- Replace `hasattr(dataset, 'semantic')` with `try/except get_semantic_layer()`

**4_🎯_Risk_Factors.py**:

- NO logging config (already configured in app.py)
- Use `require_outcome()` helper

**All other pages**:

- NO logging configuration (configured once in app.py)
- Use `require_outcome()` where needed

## Files to Create/Modify

1. **NEW**: `src/clinical_analytics/ui/logging_config.py` - Shared logging configuration
2. **NEW**: `src/clinical_analytics/ui/helpers.py` - Centralized UI helpers (require_outcome)
3. **MODIFY**: `src/clinical_analytics/core/dataset.py` - Add semantic contract (class attribute + method)
4. **MODIFY**: `src/clinical_analytics/datasets/uploaded/definition.py` - Override get_semantic_layer() with lazy init
5. **MODIFY**: `src/clinical_analytics/ui/app.py` - Add logging (ONCE), use contract pattern, use helper
6. **MODIFY**: `src/clinical_analytics/ui/pages/7_🔬_Analyze.py` - Use contract pattern (NO logging)
7. **MODIFY**: `src/clinical_analytics/ui/pages/4_🎯_Risk_Factors.py` - Use require_outcome() (NO logging)
8. **MODIFY**: Other page files as needed - Use helpers (NO logging)

## Success Criteria

- No more `hasattr()` checks for semantic layer
- Single pattern for semantic layer access (try/except get_semantic_layer())
- Centralized outcome requirement checks (require_outcome())
- Logging configured once in app.py (not per page)