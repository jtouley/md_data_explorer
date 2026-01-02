"""
Centralized logging configuration for Streamlit UI.

Configure once in app.py entry point, not per page.

Configuration values are loaded from config/logging.yaml via config_loader.
The configure_logging() function applies these settings.
"""

import logging
import sys

from clinical_analytics.core.config_loader import load_logging_config

# Load configuration from YAML via config_loader
# This loads from config/logging.yaml
_logging_config = load_logging_config()


def configure_logging(level: int | None = None) -> None:
    """
    Configure Python logging for the entire application.

    Truly idempotent: safe to call multiple times without side effects.
    Checks if root logger already has handlers before configuring.
    No force=True to avoid Streamlit handler issues.

    Args:
        level: Optional log level (int). If None, uses root_level from YAML config.
    """
    root_logger = logging.getLogger()

    # Only configure if no handlers exist (truly idempotent)
    if root_logger.handlers:
        return

    # Use provided level or YAML config root_level
    if level is None:
        level_str = _logging_config["root_level"]
        level = getattr(logging, level_str.upper(), logging.INFO)

    log_format = _logging_config["format"]

    logging.basicConfig(
        level=level,
        format=log_format,
        handlers=[logging.StreamHandler(sys.stdout)],
        # No force=True - rely on idempotency check above
    )

    # Set specific loggers from YAML config
    module_levels = _logging_config.get("module_levels", {})
    for logger_name, level_str in module_levels.items():
        level_value = getattr(logging, level_str.upper(), logging.INFO)
        logging.getLogger(logger_name).setLevel(level_value)

    # Reduce noise from YAML config
    reduce_noise = _logging_config.get("reduce_noise", {})
    for logger_name, level_str in reduce_noise.items():
        level_value = getattr(logging, level_str.upper(), logging.WARNING)
        logging.getLogger(logger_name).setLevel(level_value)
