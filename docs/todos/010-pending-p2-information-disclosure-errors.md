---
status: pending
priority: p2
issue_id: "010"
tags: [code-review, security, information-disclosure]
dependencies: []
estimated_effort: small
created_date: 2025-12-24
---

# Information Disclosure via Verbose Error Messages

## Problem Statement

Error messages throughout the application **expose sensitive system information** including file paths, database structure, internal configuration, and stack traces. Attackers can use this information to plan targeted attacks.

**Why it matters:**
- Reveals internal system architecture
- Exposes file system paths
- Shows technology stack versions
- Aids in attack reconnaissance
- OWASP Top 10: Security Misconfiguration

**Impact:** Information leakage aids attackers, security through obscurity violated

## Findings

**Locations:**

1. **`src/clinical_analytics/core/semantic.py:78`**
```python
try:
    duckdb_con.execute(f"CREATE TABLE {table_name} AS SELECT * FROM read_csv_auto('{abs_path}')")
except Exception as e:
    # Exposes full file system path and SQL query
    raise ValueError(f"Failed to load {abs_path}: {e}")
    # User sees: "Failed to load /Users/admin/clinical_analytics/data/secrets/file.csv: Permission denied"
```

2. **`src/clinical_analytics/ui/app.py:multiple locations`**
```python
# Unhandled exceptions show full stack traces in Streamlit
try:
    result = logistic_regression(df, outcome, predictors)
except Exception as e:
    st.error(f"Error: {e}")  # Shows full exception with stack trace
    # Reveals: library versions, file paths, internal function names
```

3. **`src/clinical_analytics/datasets/*/definition.py`**
```python
# Configuration errors expose YAML structure
if 'required_field' not in config:
    raise KeyError(f"Missing field in config: {config}")
    # Exposes entire configuration dictionary
```

**Example Information Disclosure:**
```
Error loading dataset:
  File "/usr/local/lib/python3.11/site-packages/clinical_analytics/core/semantic.py", line 78
    duckdb_con.execute(f"CREATE TABLE {table_name} AS SELECT * FROM read_csv_auto('{abs_path}')")
  duckdb.Error: Could not open file '/var/clinical_data/phi/patients_sensitive.csv': Permission denied

Attacker learns:
- Python version: 3.11
- Framework: clinical_analytics
- Database: DuckDB
- File path: /var/clinical_data/phi/
- File name pattern: *_sensitive.csv
- Permission model exists
```

## Proposed Solutions

### Solution 1: User-Friendly Error Messages with Internal Logging (Recommended)
**Pros:**
- No information disclosure to users
- Full details logged for debugging
- Good user experience
- Security best practice

**Cons:**
- Requires error message mapping
- Need centralized error handling

**Effort:** Small (3 hours)
**Risk:** Low

**Implementation:**
```python
# src/clinical_analytics/errors/handler.py
import logging
from typing import Any

logger = logging.getLogger(__name__)

class SafeErrorHandler:
    """
    Handles errors safely without information disclosure.

    - Shows user-friendly messages to users
    - Logs full details internally for debugging
    """

    @staticmethod
    def handle_error(
        error: Exception,
        user_message: str,
        context: dict[str, Any] | None = None
    ) -> str:
        """
        Handle error safely.

        Args:
            error: The exception that occurred
            user_message: Safe message to show user
            context: Additional context for logging

        Returns:
            User-friendly error message
        """
        # Log full error details internally (not shown to user)
        logger.error(
            f"Error occurred: {user_message}",
            exc_info=error,
            extra={"context": context}
        )

        # Return safe message to user (no internal details)
        return user_message


# Usage in semantic.py
from clinical_analytics.errors.handler import SafeErrorHandler

def _register_source(self):
    try:
        source_path = Path(init_params['source_path'])
        validated_path = PathValidator.validate_path(source_path)
        self.raw = self.con.read_csv(str(validated_path))

    except FileNotFoundError as e:
        # DON'T expose path
        safe_message = SafeErrorHandler.handle_error(
            error=e,
            user_message="Dataset file not found. Please contact administrator.",
            context={"dataset": self.dataset_name}
        )
        raise ValueError(safe_message)

    except PermissionError as e:
        # DON'T expose path or permissions
        safe_message = SafeErrorHandler.handle_error(
            error=e,
            user_message="Access denied. Please check your permissions.",
            context={"dataset": self.dataset_name}
        )
        raise ValueError(safe_message)

    except Exception as e:
        # DON'T expose any internal details
        safe_message = SafeErrorHandler.handle_error(
            error=e,
            user_message="Failed to load dataset. Please contact administrator.",
            context={"dataset": self.dataset_name}
        )
        raise ValueError(safe_message)


# Usage in app.py
import streamlit as st
from clinical_analytics.errors.handler import SafeErrorHandler

def main():
    try:
        dataset = load_dataset(dataset_name)
    except Exception as e:
        # Show safe message to user
        safe_message = SafeErrorHandler.handle_error(
            error=e,
            user_message="Unable to load dataset. Please try again or contact support.",
            context={"action": "load_dataset", "dataset": dataset_name}
        )

        # Don't show exception details in UI
        st.error(safe_message)
        st.stop()
```

**Error Message Mapping:**
```python
# Safe error messages (shown to users)
ERROR_MESSAGES = {
    "file_not_found": "Dataset not found. Please contact administrator.",
    "permission_denied": "Access denied. Please check your permissions.",
    "invalid_config": "Configuration error. Please contact administrator.",
    "database_error": "Database error occurred. Please try again.",
    "analysis_failed": "Analysis could not be completed. Please check your inputs.",
    "export_failed": "Export failed. Please try again.",
}

# Internal error details logged but NOT shown to users
```

### Solution 2: Custom Error Pages
**Pros:**
- Branded error experience
- Consistent messaging
- Can include support contact info

**Cons:**
- Requires Streamlit customization
- More frontend work

**Effort:** Medium (4 hours)
**Risk:** Low

### Solution 3: Error Code System
**Pros:**
- Terse, no information disclosure
- Easy to track in logs
- Support can look up codes

**Cons:**
- Less user-friendly
- Requires error code registry
- Users need to communicate codes

**Effort:** Medium (5 hours)
**Risk:** Low

## Recommended Action

**Implement Solution 1** with:

1. **SafeErrorHandler class** for centralized error handling
2. **User-friendly error messages** (no internal details)
3. **Detailed internal logging** for debugging
4. **Apply to all error paths** throughout application
5. **Streamlit error handling** without stack traces

**Error Handling Checklist:**
- [ ] Never expose file paths
- [ ] Never expose database queries
- [ ] Never expose configuration details
- [ ] Never expose library versions
- [ ] Never expose stack traces to users
- [ ] Always log full details internally
- [ ] Provide helpful user messages
- [ ] Include support contact info

## Technical Details

**Affected Files:**
- `src/clinical_analytics/core/semantic.py` - Add safe error handling
- `src/clinical_analytics/core/mapper.py` - Add safe error handling
- `src/clinical_analytics/ui/app.py` - Catch and sanitize all errors
- All dataset implementations - Safe error messages

**New Files:**
```
src/clinical_analytics/errors/
├── __init__.py
├── handler.py         # SafeErrorHandler class
└── messages.py        # Error message constants
```

**Logging Configuration:**
```python
# config/logging.yaml
version: 1
formatters:
  detailed:
    format: '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

handlers:
  file:
    class: logging.handlers.RotatingFileHandler
    filename: logs/application.log
    maxBytes: 10485760  # 10MB
    backupCount: 5
    formatter: detailed

loggers:
  clinical_analytics:
    level: DEBUG  # Log everything internally
    handlers: [file]
```

## Acceptance Criteria

- [ ] SafeErrorHandler implemented
- [ ] All error paths use SafeErrorHandler
- [ ] No file paths exposed in error messages
- [ ] No database queries exposed in error messages
- [ ] No configuration details exposed
- [ ] No stack traces shown to users
- [ ] Full error details logged internally
- [ ] Error messages are user-friendly
- [ ] Support contact info included in errors
- [ ] Security testing confirms no information disclosure

## Work Log

### 2025-12-24
- **Action:** Security review identified information disclosure
- **Learning:** Error messages should be user-friendly, not debug-friendly
- **Next:** Implement SafeErrorHandler and apply throughout application

## Resources

- **OWASP Information Exposure:** https://owasp.org/www-community/Improper_Error_Handling
- **CWE-209:** Information Exposure Through an Error Message
- **Error Handling Best Practices:** OWASP Cheat Sheet
- **Related Finding:** Security hardening recommendations
