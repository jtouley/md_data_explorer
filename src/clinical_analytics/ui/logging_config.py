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
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        # No force=True - rely on idempotency check above
    )

    # Set specific loggers
    logging.getLogger("clinical_analytics.core.semantic").setLevel(logging.INFO)
    logging.getLogger("clinical_analytics.core.registry").setLevel(logging.INFO)
    logging.getLogger("clinical_analytics.core.multi_table_handler").setLevel(logging.INFO)
    logging.getLogger("clinical_analytics.ui.storage.user_datasets").setLevel(logging.INFO)
    logging.getLogger("clinical_analytics.datasets").setLevel(logging.INFO)

    # Reduce noise
    logging.getLogger("streamlit").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
