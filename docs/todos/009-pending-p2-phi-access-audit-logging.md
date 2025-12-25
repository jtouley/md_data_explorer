---
status: pending
priority: p2
issue_id: "009"
tags: [code-review, security, compliance, hipaa, audit-logging]
dependencies: ["003"]
estimated_effort: medium
created_date: 2025-12-24
---

# Missing PHI Access Audit Logging

## Problem Statement

**No audit logging** for Protected Health Information (PHI) access. The system does not track who accessed what patient data, when, or what operations they performed. This violates **HIPAA Audit Controls** requirements and makes breach investigation impossible.

**Why it matters:**
- **HIPAA Security Rule § 164.312(b)** requires audit controls
- Cannot investigate data breaches or unauthorized access
- No accountability for data access
- Regulatory fines up to $1.5M per violation category
- Required for HIPAA compliance certification

**Impact:** HIPAA compliance FAIL, breach investigation impossible

## Findings

**Location:** Entire application - no audit logging infrastructure

**HIPAA Requirements (Not Met):**

| Requirement | Status | Evidence |
|------------|--------|----------|
| § 164.312(b) Audit Controls | ❌ MISSING | No logging infrastructure |
| Log access to PHI | ❌ MISSING | No data access logs |
| Log who accessed data | ❌ MISSING | No authentication (see #003) |
| Log when accessed | ❌ MISSING | No timestamps |
| Log what was accessed | ❌ MISSING | No operation tracking |
| Log data exports | ❌ MISSING | CSV downloads not logged |
| Log failed access attempts | ❌ MISSING | No failure tracking |
| Tamper-resistant logs | ❌ MISSING | No log protection |

**Current State:**
```python
# app.py - NO AUDIT LOGGING
def load_dataset(dataset_name: str):
    dataset = get_dataset(dataset_name)
    dataset.load()  # NO LOG: Who loaded? When? Which records?
    return dataset

def export_to_csv(df: pd.DataFrame, filename: str):
    df.to_csv(filename)  # NO LOG: Who exported? What data?
    st.success(f"Exported {len(df)} records")
```

**What Should Be Logged:**
1. User authentication events (login/logout/failures)
2. Dataset access (which dataset, how many records)
3. Cohort queries (what filters, result counts)
4. Data exports (what data, how many records, format)
5. Statistical analyses (what analysis, which variables)
6. Data profiling (which columns profiled)
7. Failed access attempts
8. Configuration changes

## Proposed Solutions

### Solution 1: Structured Audit Log with Tamper Protection (Recommended)
**Pros:**
- HIPAA-compliant audit trail
- Tamper detection via cryptographic hashing
- Queryable structured logs
- Integration with SIEM systems

**Cons:**
- Additional storage requirements
- Slight performance overhead
- Need log rotation strategy

**Effort:** Medium (5 hours)
**Risk:** Low

**Implementation:**
```python
# src/clinical_analytics/audit/logger.py
import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Any
import logging

class AuditLogger:
    """
    HIPAA-compliant audit logger with tamper detection.

    Logs all PHI access events with:
    - Who: User identity
    - What: Operation and data accessed
    - When: Timestamp (ISO 8601)
    - Where: IP address, session ID
    - Result: Success/failure
    """

    def __init__(self, log_dir: Path):
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = log_dir / f"audit_{datetime.now():%Y%m%d}.log"
        self.previous_hash = self._load_previous_hash()

    def log_event(
        self,
        user: str,
        action: str,
        resource: str,
        details: dict[str, Any],
        result: str = "SUCCESS",
        ip_address: str | None = None
    ) -> None:
        """
        Log audit event with tamper detection.

        Args:
            user: Username or user ID
            action: Action performed (LOGIN, ACCESS_DATASET, EXPORT, etc.)
            resource: Resource accessed (dataset name, table name, etc.)
            details: Additional context (filter parameters, record counts, etc.)
            result: SUCCESS or FAILURE
            ip_address: Client IP address
        """
        event = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "user": user,
            "action": action,
            "resource": resource,
            "details": details,
            "result": result,
            "ip_address": ip_address,
            "session_id": self._get_session_id(),
            "previous_hash": self.previous_hash
        }

        # Compute hash for tamper detection
        event_json = json.dumps(event, sort_keys=True)
        current_hash = hashlib.sha256(event_json.encode()).hexdigest()
        event["event_hash"] = current_hash

        # Write to log file (append-only)
        with open(self.log_file, "a") as f:
            f.write(event_json + "\n")

        # Update chain hash
        self.previous_hash = current_hash

    def verify_log_integrity(self) -> bool:
        """Verify audit log has not been tampered with."""
        previous_hash = None

        with open(self.log_file, "r") as f:
            for line in f:
                event = json.loads(line)

                # Check hash chain
                if event["previous_hash"] != previous_hash:
                    return False  # Chain broken - tampering detected

                # Verify event hash
                event_copy = event.copy()
                stored_hash = event_copy.pop("event_hash")
                computed_hash = hashlib.sha256(
                    json.dumps(event_copy, sort_keys=True).encode()
                ).hexdigest()

                if stored_hash != computed_hash:
                    return False  # Event modified - tampering detected

                previous_hash = stored_hash

        return True  # Integrity verified

    def _get_session_id(self) -> str:
        """Get current Streamlit session ID."""
        import streamlit as st
        return st.runtime.scriptrunner.get_script_run_ctx().session_id

    def _load_previous_hash(self) -> str | None:
        """Load last hash from previous log file."""
        if self.log_file.exists():
            with open(self.log_file, "r") as f:
                lines = f.readlines()
                if lines:
                    last_event = json.loads(lines[-1])
                    return last_event.get("event_hash")
        return None


# Integration with app.py
from clinical_analytics.audit.logger import AuditLogger

# Initialize audit logger
audit_logger = AuditLogger(Path("logs/audit"))

def load_dataset(dataset_name: str, user: str):
    """Load dataset with audit logging."""
    try:
        dataset = get_dataset(dataset_name)
        dataset.load()

        # LOG SUCCESS
        audit_logger.log_event(
            user=user,
            action="ACCESS_DATASET",
            resource=dataset_name,
            details={
                "record_count": len(dataset.df),
                "columns": list(dataset.df.columns)
            },
            result="SUCCESS",
            ip_address=st.session_state.get('client_ip')
        )

        return dataset

    except Exception as e:
        # LOG FAILURE
        audit_logger.log_event(
            user=user,
            action="ACCESS_DATASET",
            resource=dataset_name,
            details={"error": str(e)},
            result="FAILURE"
        )
        raise

def export_to_csv(df: pd.DataFrame, filename: str, user: str):
    """Export data with audit logging."""
    # LOG EXPORT
    audit_logger.log_event(
        user=user,
        action="EXPORT_DATA",
        resource=filename,
        details={
            "record_count": len(df),
            "columns": list(df.columns),
            "format": "CSV"
        },
        result="SUCCESS"
    )

    df.to_csv(filename, index=False)
    st.success(f"Exported {len(df)} records")

def run_analysis(analysis_type: str, params: dict, user: str):
    """Run analysis with audit logging."""
    audit_logger.log_event(
        user=user,
        action="RUN_ANALYSIS",
        resource=analysis_type,
        details=params,
        result="SUCCESS"
    )
```

### Solution 2: Cloud-Based Audit Service
**Pros:**
- Managed service (AWS CloudTrail, Azure Monitor)
- Automatic tamper protection
- Integration with security tools
- No infrastructure to maintain

**Cons:**
- Requires cloud deployment
- Additional cost
- Data leaves local environment

**Effort:** Medium (6 hours)
**Risk:** Low

### Solution 3: Database Audit Log
**Pros:**
- Queryable via SQL
- Easy reporting
- Transactional consistency

**Cons:**
- Logs can be deleted via SQL
- Less tamper-resistant
- Performance impact on database

**Effort:** Small (3 hours)
**Risk:** Medium (tampering)

## Recommended Action

**Implement Solution 1** with:

1. **Structured audit logger** with tamper detection
2. **Log all PHI access events** (access, export, analysis)
3. **Daily log rotation** with retention policy
4. **Log integrity verification** on application startup
5. **Integration with authentication** (requires todo #003)

**Log Retention Policy:**
- Retain audit logs for **6 years** (HIPAA requirement)
- Daily rotation to new files
- Compress logs older than 90 days
- Archive to secure storage monthly
- Never delete audit logs

## Technical Details

**New Files:**
```
src/clinical_analytics/audit/
├── __init__.py
├── logger.py          # AuditLogger class
├── events.py          # Event type definitions
└── verifier.py        # Log integrity verification

logs/audit/
├── audit_20251224.log
├── audit_20251223.log
└── ...
```

**Log Format:**
```json
{
  "timestamp": "2025-12-24T15:30:45.123456Z",
  "user": "researcher1",
  "action": "ACCESS_DATASET",
  "resource": "covid_ms",
  "details": {
    "record_count": 1247,
    "filters": {"age_min": 18, "age_max": 65},
    "columns": ["age", "sex", "outcome"]
  },
  "result": "SUCCESS",
  "ip_address": "10.0.1.42",
  "session_id": "abc123def456",
  "previous_hash": "d4e5f6a7b8c9...",
  "event_hash": "1a2b3c4d5e6f..."
}
```

**Actions to Log:**
| Action | Details to Capture |
|--------|-------------------|
| LOGIN | user, timestamp, IP, result |
| LOGOUT | user, timestamp, session_duration |
| ACCESS_DATASET | dataset, record_count, columns |
| QUERY_COHORT | filters, result_count |
| RUN_ANALYSIS | analysis_type, variables, result_count |
| EXPORT_DATA | format, record_count, columns |
| FAILED_ACCESS | user, resource, reason |
| CONFIG_CHANGE | what_changed, old_value, new_value |

## Acceptance Criteria

- [ ] AuditLogger class implemented with tamper detection
- [ ] All PHI access events logged
- [ ] Log format follows HIPAA requirements (who/what/when)
- [ ] Log integrity verification on startup
- [ ] Failed access attempts logged
- [ ] Daily log rotation implemented
- [ ] 6-year retention policy documented
- [ ] Logs stored in append-only location
- [ ] No user can delete audit logs
- [ ] Log query tools for compliance reporting
- [ ] Documentation for compliance officers

## Work Log

### 2025-12-24
- **Action:** Compliance review identified missing audit controls
- **Learning:** HIPAA requires comprehensive audit trail for PHI access
- **Next:** Implement AuditLogger with tamper detection, integrate with authentication

## Resources

- **HIPAA Security Rule:** https://www.hhs.gov/hipaa/for-professionals/security/
- **§ 164.312(b) Audit Controls:** https://www.hhs.gov/hipaa/for-professionals/security/laws-regulations/
- **Audit Log Best Practices:** NIST SP 800-92
- **Log Integrity:** Cryptographic hash chains
- **Related Finding:** Missing authentication (todo #003)
