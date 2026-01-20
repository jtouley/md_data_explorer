"""Tests for checkpoint logging Makefile commands."""

from pathlib import Path


class TestCheckpointCommands:
    """Test suite for checkpoint Makefile commands."""

    def test_makefile_contains_checkpoint_targets(self):
        """Test that Makefile contains required checkpoint command targets."""
        # Arrange: Read Makefile
        makefile_path = Path("Makefile")
        assert makefile_path.exists(), "Makefile not found"

        makefile_content = makefile_path.read_text()

        # Assert: Required targets exist (matching actual Makefile implementation)
        assert "checkpoint-create:" in makefile_content, "checkpoint-create target missing"
        assert "checkpoint-resume:" in makefile_content, "checkpoint-resume target missing"
        assert "git-log-export:" in makefile_content, "git-log-export target missing"
        assert "git-log-latest:" in makefile_content, "git-log-latest target missing"
