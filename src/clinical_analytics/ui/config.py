"""
UI Configuration Module.

Feature flags and configuration settings for the Clinical Analytics UI.
Single source of truth for feature toggles.
"""

import os

# Feature flags - single source of truth
# Set MULTI_TABLE_ENABLED=true in environment to enable multi-table (ZIP) uploads
MULTI_TABLE_ENABLED = os.getenv("MULTI_TABLE_ENABLED", "false").lower() == "true"

# V1 MVP: Gate legacy analysis pages (2-6)
# Set V1_MVP_MODE=false to show all pages (for development/testing)
V1_MVP_MODE = os.getenv("V1_MVP_MODE", "true").lower() == "true"

# Logging configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Upload settings
MAX_UPLOAD_SIZE_MB = int(os.getenv("MAX_UPLOAD_SIZE_MB", "100"))
