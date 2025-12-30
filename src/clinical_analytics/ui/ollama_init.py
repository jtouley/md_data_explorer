"""
Ollama Initialization for App Startup.

Self-contained initialization of Ollama LLM service, similar to DuckDB.
Checks availability and provides setup instructions if needed.
"""

import subprocess

import structlog

logger = structlog.get_logger()


def initialize_ollama() -> dict[str, bool | str]:
    """
    Initialize and verify Ollama service at app startup.

    Returns:
        Dictionary with status information:
        - installed: bool - Whether Ollama binary is installed
        - running: bool - Whether Ollama service is running
        - ready: bool - Whether Ollama is ready to use (running + models available)
        - message: str - Status message for display
    """
    from clinical_analytics.core.ollama_manager import get_ollama_manager

    manager = get_ollama_manager()
    status = manager.get_status()

    result = {
        "installed": status["installed"],
        "running": status["running"],
        "ready": status["ready"],
        "message": "",
    }

    if status["ready"]:
        models_count = len(status["available_models"])
        result["message"] = f"✓ Ollama LLM ready ({models_count} model(s) available)"
        logger.info("ollama_initialized", models=status["available_models"])
        return result

    # Not ready - build helpful message
    if not status["installed"]:
        result["message"] = (
            "⚠ Ollama not installed - Natural language queries will use pattern matching only. "
            "Install at https://ollama.ai for advanced query parsing."
        )
        logger.info("ollama_not_installed")
    elif not status["running"]:
        result["message"] = (
            "⚠ Ollama service not running - Natural language queries will use pattern matching only. "
            "Start with: ollama serve"
        )
        logger.info("ollama_not_running")
    elif not status["available_models"]:
        result["message"] = (
            "⚠ Ollama running but no models available - Natural language queries will use pattern matching only. "
            f"Download a model with: ollama pull {status['default_model']}"
        )
        logger.info("ollama_no_models", default_model=status["default_model"])

    return result


def try_start_ollama_service() -> bool:
    """
    Attempt to start Ollama service in background.

    Returns:
        True if service was started or already running, False otherwise
    """
    from clinical_analytics.core.ollama_manager import get_ollama_manager

    manager = get_ollama_manager()

    # Check if already running
    if manager.is_service_running():
        return True

    # Check if installed
    if not manager.is_ollama_installed():
        return False

    # Try to start service
    try:
        # Start Ollama in background (non-blocking)
        # Note: This will start a background process
        subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )

        # Wait a moment for service to start
        import time

        time.sleep(2)

        # Check if it's now running
        if manager.is_service_running():
            logger.info("ollama_service_started")
            return True

    except Exception as e:
        logger.warning("ollama_start_failed", error=str(e))

    return False
