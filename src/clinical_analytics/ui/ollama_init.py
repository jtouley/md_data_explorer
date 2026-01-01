"""
Ollama Initialization for App Startup.

Self-contained initialization of Ollama LLM service, similar to DuckDB.
Checks availability and provides setup instructions if needed.

Features:
- Auto-download missing models on first startup
- Progress feedback during download
- Graceful degradation if download fails
"""

import subprocess

import structlog

from clinical_analytics.core.ollama_manager import get_ollama_manager

logger = structlog.get_logger()


def ensure_models_downloaded(show_progress: bool = False) -> dict[str, bool | str]:
    """
    Ensure required Ollama models are downloaded, auto-downloading if missing.

    This prevents the "silent failure" mode where users think NL queries work
    but they're actually using degraded pattern matching.

    Args:
        show_progress: If True, show progress feedback (for UI integration)

    Returns:
        Status dictionary with ready state and message
    """
    manager = get_ollama_manager()
    status = manager.get_status()

    # If already ready, nothing to do
    if status["ready"]:
        logger.info("ollama_models_ready", models=status["available_models"])
        return status

    # If service not running, can't download
    if not status["running"]:
        logger.warning("ollama_service_not_running_cannot_download")
        return status

    # Service is running but no models - auto-download
    if not status["available_models"]:
        logger.info(
            "ollama_auto_download_starting",
            default_model=manager.default_model,
            fallback_model=manager.fallback_model,
        )

        # Try default model first
        try:
            logger.info("downloading_default_model", model=manager.default_model)
            result = subprocess.run(
                ["ollama", "pull", manager.default_model],
                capture_output=not show_progress,  # Show output if requested
                check=True,
                timeout=600,  # 10 minute timeout for download
            )

            if result.returncode == 0:
                logger.info("ollama_default_model_downloaded", model=manager.default_model)
                # Refresh status
                manager._available_models = None  # Clear cache
                return manager.get_status()

        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            logger.warning(
                "ollama_default_model_download_failed",
                model=manager.default_model,
                error=str(e),
            )

            # Try fallback model
            try:
                logger.info("downloading_fallback_model", model=manager.fallback_model)
                result = subprocess.run(
                    ["ollama", "pull", manager.fallback_model],
                    capture_output=not show_progress,
                    check=True,
                    timeout=600,
                )

                if result.returncode == 0:
                    logger.info("ollama_fallback_model_downloaded", model=manager.fallback_model)
                    # Refresh status
                    manager._available_models = None  # Clear cache
                    return manager.get_status()

            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e2:
                logger.error(
                    "ollama_fallback_model_download_failed",
                    model=manager.fallback_model,
                    error=str(e2),
                )

    # Return current status
    return manager.get_status()


def initialize_ollama(auto_download: bool = True) -> dict[str, bool | str]:
    """
    Initialize and verify Ollama service at app startup.

    Args:
        auto_download: If True, automatically download models if missing (default: True)

    Returns:
        Dictionary with status information:
        - installed: bool - Whether Ollama binary is installed
        - running: bool - Whether Ollama service is running
        - ready: bool - Whether Ollama is ready to use (running + models available)
        - message: str - Status message for display
        - auto_downloaded: bool - Whether models were auto-downloaded
    """
    manager = get_ollama_manager()
    status = manager.get_status()

    # Try auto-download if enabled and service is running but no models
    auto_downloaded = False
    if auto_download and status["running"] and not status["available_models"]:
        logger.info("attempting_auto_download")
        download_status = ensure_models_downloaded(show_progress=False)
        if download_status["ready"]:
            status = download_status
            auto_downloaded = True
            logger.info("auto_download_successful")

    result = {
        "installed": status["installed"],
        "running": status["running"],
        "ready": status["ready"],
        "message": "",
        "auto_downloaded": auto_downloaded,
    }

    if status["ready"]:
        models_count = len(status["available_models"])
        if auto_downloaded:
            result["message"] = f"✓ Ollama LLM ready ({models_count} model(s) downloaded and available)"
        else:
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
            "⚠ Model download failed - Natural language queries will use pattern matching only. "
            f"Try manually: ollama pull {status['default_model']}"
        )
        logger.info("ollama_auto_download_failed", default_model=status["default_model"])

    return result


def try_start_ollama_service() -> bool:
    """
    Attempt to start Ollama service in background.

    Returns:
        True if service was started or already running, False otherwise
    """
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
