"""Integration tests for run_app.sh script cleanup functionality."""

import re
import subprocess
from pathlib import Path

import pytest


class TestRunAppScriptCleanup:
    """Test suite for run_app.sh graceful shutdown functionality."""

    @pytest.fixture
    def script_path(self) -> Path:
        """Return path to run_app.sh script."""
        return Path(__file__).parent.parent.parent / "scripts" / "run_app.sh"

    @pytest.fixture
    def script_content(self, script_path: Path) -> str:
        """Read script content."""
        return script_path.read_text()

    def test_unit_script_exists_is_executable(self, script_path: Path):
        """Test that run_app.sh exists and is executable."""
        # Arrange & Act & Assert
        assert script_path.exists(), "run_app.sh script not found"
        assert script_path.stat().st_mode & 0o111, "run_app.sh is not executable"

    def test_unit_cleanup_function_defined(self, script_content: str):
        """Test that cleanup function is defined in the script."""
        # Arrange
        cleanup_pattern = r"^\s*cleanup\(\)\s*\{"

        # Act
        match = re.search(cleanup_pattern, script_content, re.MULTILINE)

        # Assert
        assert match is not None, "cleanup() function not defined in script"

    def test_unit_trap_registered_handles_signals(self, script_content: str):
        """Test that trap is registered for EXIT, INT, and TERM signals."""
        # Arrange
        # Match entire trap line with multiple signals
        trap_pattern = r"trap\s+cleanup\s+((?:EXIT|INT|TERM)(?:\s+(?:EXIT|INT|TERM))*)"

        # Act
        match = re.search(trap_pattern, script_content)

        # Assert
        assert match is not None, "trap cleanup not registered for signals"

        # Extract all signals from the matched line
        signals_text = match.group(1)
        signals_found = set(re.findall(r"EXIT|INT|TERM", signals_text))

        # Should handle at least EXIT and INT (Ctrl+C)
        required_signals = {"EXIT", "INT"}
        assert required_signals.issubset(
            signals_found
        ), f"trap must handle EXIT and INT signals, found: {signals_found}"

    def test_unit_cleanup_checks_ollama_pid(self, script_content: str):
        """Test that cleanup function checks OLLAMA_PID variable."""
        # Arrange
        # Extract cleanup function body
        cleanup_match = re.search(r"cleanup\(\)\s*\{(.*?)\n\}", script_content, re.DOTALL | re.MULTILINE)

        # Act & Assert
        assert cleanup_match is not None, "Could not extract cleanup function"
        cleanup_body = cleanup_match.group(1)

        # Check for OLLAMA_PID reference
        assert "OLLAMA_PID" in cleanup_body, "cleanup() must check OLLAMA_PID variable"

    def test_unit_cleanup_supports_stop_ollama_flag(self, script_content: str):
        """Test that cleanup function respects STOP_OLLAMA_ON_EXIT flag."""
        # Arrange
        cleanup_match = re.search(r"cleanup\(\)\s*\{(.*?)\n\}", script_content, re.DOTALL | re.MULTILINE)

        # Act & Assert
        assert cleanup_match is not None, "Could not extract cleanup function"
        cleanup_body = cleanup_match.group(1)

        # Check for STOP_OLLAMA_ON_EXIT flag
        assert "STOP_OLLAMA_ON_EXIT" in cleanup_body, "cleanup() must check STOP_OLLAMA_ON_EXIT flag"

    def test_unit_cleanup_kills_ollama_process(self, script_content: str):
        """Test that cleanup function kills Ollama process."""
        # Arrange
        cleanup_match = re.search(r"cleanup\(\)\s*\{(.*?)\n\}", script_content, re.DOTALL | re.MULTILINE)

        # Act & Assert
        assert cleanup_match is not None, "Could not extract cleanup function"
        cleanup_body = cleanup_match.group(1)

        # Check for kill command
        assert "kill" in cleanup_body, "cleanup() must use kill command"

    @pytest.mark.slow
    @pytest.mark.integration
    def test_integration_script_syntax_valid(self, script_path: Path):
        """Test that script has valid bash syntax."""
        # Arrange & Act
        result = subprocess.run(["bash", "-n", str(script_path)], capture_output=True, text=True)

        # Assert
        assert result.returncode == 0, f"Script has syntax errors:\n{result.stderr}"

    def test_unit_ollama_pid_initialized_globally(self, script_content: str):
        """Test that OLLAMA_PID is initialized as global variable."""
        # Arrange
        # Look for OLLAMA_PID initialization before cleanup function
        cleanup_pos = script_content.find("cleanup()")
        if cleanup_pos == -1:
            pytest.skip("cleanup() function not found")

        script_before_cleanup = script_content[:cleanup_pos]

        # Act
        # Check for OLLAMA_PID="" initialization
        pid_init_pattern = r'^\s*OLLAMA_PID\s*=\s*""'
        match = re.search(pid_init_pattern, script_before_cleanup, re.MULTILINE)

        # Assert
        assert match is not None, "OLLAMA_PID must be initialized as empty string before cleanup function"
