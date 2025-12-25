---
status: pending
priority: p1
issue_id: "003"
tags: [code-review, security, critical, hipaa, authentication]
dependencies: []
estimated_effort: large
created_date: 2025-12-24
---

# No Authentication or Authorization System

## Problem Statement

The Streamlit application has **NO authentication mechanism**. Anyone with network access can view and export Protected Health Information (PHI). This is a **HIPAA violation** and creates severe data breach risk for a medical data platform.

**Why it matters:**
- Unauthorized access to patient health information
- HIPAA Security Rule § 164.312(d) requires person authentication
- Data breach liability
- No audit trail of who accessed what data

**Impact:** HIPAA compliance FAIL, potential legal liability

## Findings

**Location:** `src/clinical_analytics/ui/app.py` - entire application

**Current State:**
- No login page or authentication widget
- No user sessions or identity management
- No role-based access control (RBAC)
- Anyone can access any dataset
- No restrictions on data export

**Evidence:**
```python
# Current app.py - NO AUTH CHECK
def main():
    st.set_page_config(...)
    # Direct access to all features - NO LOGIN REQUIRED
    dataset_choice = st.sidebar.selectbox(...)
    dataset = load_dataset(dataset_choice)  # No permission check
```

**Compliance Gap:**
| HIPAA Requirement | Status |
|------------------|--------|
| Access Control § 164.312(a)(1) | ❌ MISSING |
| Person Authentication § 164.312(d) | ❌ MISSING |
| Audit Controls § 164.312(b) | ❌ MISSING |

## Proposed Solutions

### Solution 1: Streamlit-Authenticator (Recommended)
**Pros:**
- Purpose-built for Streamlit
- Cookie-based sessions
- Role-based access control
- Password hashing built-in
- Active maintenance

**Cons:**
- Additional dependency
- Requires user management file

**Effort:** Large (10 hours including RBAC)
**Risk:** Low

**Implementation:**
```python
import streamlit_authenticator as stauth
import yaml

# Load credentials from secure config
with open('config/users.yaml') as file:
    config = yaml.safe_load(file)

authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days']
)

# Add login widget
name, authentication_status, username = authenticator.login('Login', 'main')

if authentication_status:
    authenticator.logout('Logout', 'main')
    st.write(f'Welcome *{name}*')

    # Check dataset permissions
    if not has_dataset_access(username, dataset_choice):
        st.error("Access denied")
        audit_log(f"{username} attempted unauthorized access to {dataset_choice}")
        st.stop()

    # Main app logic here
elif authentication_status == False:
    st.error('Username/password is incorrect')
    audit_log(f"Failed login: {username}")
elif authentication_status == None:
    st.warning('Please enter credentials')
    st.stop()
```

### Solution 2: Custom OAuth2/SAML Integration
**Pros:**
- Enterprise SSO integration
- Centralized identity management
- MFA support

**Cons:**
- Complex implementation
- Requires OAuth provider
- More maintenance

**Effort:** Very Large (20-30 hours)
**Risk:** Medium

### Solution 3: Basic HTTP Auth via Reverse Proxy
**Pros:**
- Simple to implement
- Handled outside application
- Standard approach

**Cons:**
- No fine-grained control
- All-or-nothing access
- No role-based permissions

**Effort:** Small (4 hours)
**Risk:** Medium (limited functionality)

## Recommended Action

**Implement Solution 1** (Streamlit-Authenticator) with:

1. User credential management
2. Role-based dataset access control
3. Session timeout (30 minutes)
4. Audit logging integration
5. Password policy enforcement

**User Configuration:**
```yaml
credentials:
  usernames:
    researcher1:
      email: researcher1@hospital.org
      name: Dr. Jane Smith
      password: $2b$12$... # bcrypt hashed
      roles: [researcher]
      datasets: [covid_ms, sepsis]

    admin1:
      email: admin@hospital.org
      name: Admin User
      password: $2b$12$...
      roles: [admin, researcher]
      datasets: [covid_ms, sepsis, mimic3]

roles:
  researcher:
    can_export: false
    max_export_rows: 1000
  admin:
    can_export: true
    max_export_rows: 10000

cookie:
  name: clinical_analytics_auth
  key: [generate with secrets.token_hex(32)]
  expiry_days: 1
```

## Technical Details

**New Files:**
- `config/users.yaml` - User credentials (encrypted at rest)
- `src/clinical_analytics/auth/rbac.py` - Role-based access control
- `src/clinical_analytics/auth/audit.py` - Authentication audit logging

**Modified Files:**
- `src/clinical_analytics/ui/app.py` - Add authentication wrapper

**Dependencies to Add:**
```toml
streamlit-authenticator = ">=0.2.3"
pyyaml = ">=6.0"
bcrypt = ">=4.0.0"
```

## Acceptance Criteria

- [ ] Login page displayed before any data access
- [ ] User credentials stored securely (bcrypt hashed passwords)
- [ ] Session management with configurable timeout
- [ ] Role-based access control (researcher, admin roles)
- [ ] Per-dataset access permissions
- [ ] Failed login attempts logged
- [ ] Successful logins logged with timestamp
- [ ] Logout functionality
- [ ] Session expiration after inactivity
- [ ] Password complexity requirements
- [ ] Account lockout after failed attempts (5 tries)

## Work Log

### 2025-12-24
- **Action:** Security review identified missing authentication
- **Learning:** Medical data platforms require authentication before data access
- **Next:** Install streamlit-authenticator and create users.yaml template

## Resources

- **Streamlit-Authenticator:** https://github.com/mkhorasani/Streamlit-Authenticator
- **HIPAA Security Rule:** https://www.hhs.gov/hipaa/for-professionals/security/
- **Authentication Best Practices:** OWASP Authentication Cheat Sheet
- **Related Finding:** PHI access audit logging (todo #010)
