"""
UI Configuration Module.

Feature flags and configuration settings for the Clinical Analytics UI.
Single source of truth for feature toggles.

Configuration is loaded from config/ui.yaml via config_loader.
Environment variables take precedence over YAML values.
"""

from clinical_analytics.core.config_loader import load_ui_config

# Load configuration from YAML via config_loader
# This loads from config/ui.yaml with environment variable overrides
_config = load_ui_config()

# Feature flags - single source of truth
# Set MULTI_TABLE_ENABLED=true in environment to enable multi-table (ZIP) uploads
MULTI_TABLE_ENABLED: bool = _config["multi_table_enabled"]

# V1 MVP: Gate legacy analysis pages (2-6)
# Set V1_MVP_MODE=false to show all pages (for development/testing)
V1_MVP_MODE: bool = _config["v1_mvp_mode"]

# Logging configuration
LOG_LEVEL: str = _config["log_level"]

# Upload settings
MAX_UPLOAD_SIZE_MB: int = _config["max_upload_size_mb"]

# Page names (centralized to avoid brittle string references)
# These are the actual Streamlit page file paths
ASK_QUESTIONS_PAGE: str = _config["ask_questions_page"]
