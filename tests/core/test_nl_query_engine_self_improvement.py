"""
Tests for NL Query Engine self-improvement features.

Tests cover:
- Prompt overlay path resolution (env var support)
- Overlay caching with mtime-based hot reload
- Stable hashing for metrics
- Integration with _build_llm_prompt()
"""

import os
from pathlib import Path
from unittest.mock import patch


class TestPromptOverlayPath:
    """Test overlay path resolution with env var support."""

    def test_overlay_path_defaults_to_tmp_directory(self, mock_semantic_layer):
        """Test that overlay path defaults to /tmp/nl_query_learning/ when no env var set."""
        # Arrange: Create engine without NL_PROMPT_OVERLAY_PATH env var
        from clinical_analytics.core.nl_query_engine import NLQueryEngine

        with patch.dict(os.environ, {}, clear=False):
            # Remove env var if it exists
            os.environ.pop("NL_PROMPT_OVERLAY_PATH", None)

            engine = NLQueryEngine(mock_semantic_layer)

            # Act: Get overlay path
            overlay_path = engine._prompt_overlay_path()

            # Assert: Path is /tmp/nl_query_learning/prompt_overlay.txt
            assert overlay_path == Path("/tmp/nl_query_learning/prompt_overlay.txt")
            assert str(overlay_path).startswith("/tmp/")
            assert "prompt_overlay.txt" in str(overlay_path)

    def test_overlay_path_respects_env_var_override(self, mock_semantic_layer):
        """Test that NL_PROMPT_OVERLAY_PATH env var overrides default path."""
        # Arrange: Set custom env var path
        from clinical_analytics.core.nl_query_engine import NLQueryEngine

        custom_path = "/custom/location/my_overlay.txt"

        with patch.dict(os.environ, {"NL_PROMPT_OVERLAY_PATH": custom_path}):
            engine = NLQueryEngine(mock_semantic_layer)

            # Act: Get overlay path
            overlay_path = engine._prompt_overlay_path()

            # Assert: Path matches env var
            assert overlay_path == Path(custom_path)
            assert str(overlay_path) == custom_path


class TestPromptOverlayCaching:
    """Test overlay caching with mtime-based hot reload."""

    def test_overlay_cache_initializes_to_empty(self, mock_semantic_layer):
        """Test that overlay cache fields initialize to empty on engine creation."""
        # Arrange & Act: Create engine
        from clinical_analytics.core.nl_query_engine import NLQueryEngine

        engine = NLQueryEngine(mock_semantic_layer)

        # Assert: Cache fields exist and are empty
        assert hasattr(engine, "_overlay_cache_text")
        assert hasattr(engine, "_overlay_cache_mtime_ns")
        assert engine._overlay_cache_text == ""
        assert engine._overlay_cache_mtime_ns == 0

    def test_load_overlay_returns_empty_when_file_missing(self, mock_semantic_layer, tmp_path):
        """Test that _load_prompt_overlay() returns empty string when file doesn't exist."""
        # Arrange: Create engine with overlay path pointing to nonexistent file
        from clinical_analytics.core.nl_query_engine import NLQueryEngine

        nonexistent_path = tmp_path / "nonexistent_overlay.txt"

        with patch.dict(os.environ, {"NL_PROMPT_OVERLAY_PATH": str(nonexistent_path)}):
            engine = NLQueryEngine(mock_semantic_layer)

            # Act: Load overlay
            result = engine._load_prompt_overlay()

            # Assert: Returns empty string
            assert result == ""
            assert engine._overlay_cache_text == ""
            assert engine._overlay_cache_mtime_ns == 0

    def test_load_overlay_reads_file_content(self, mock_semantic_layer, tmp_path):
        """Test that _load_prompt_overlay() reads and caches file content."""
        # Arrange: Create overlay file with test content
        from clinical_analytics.core.nl_query_engine import NLQueryEngine

        overlay_file = tmp_path / "test_overlay.txt"
        test_content = "=== AUTO-GENERATED FIXES ===\nTest fix content"
        overlay_file.write_text(test_content, encoding="utf-8")

        with patch.dict(os.environ, {"NL_PROMPT_OVERLAY_PATH": str(overlay_file)}):
            engine = NLQueryEngine(mock_semantic_layer)

            # Act: Load overlay
            result = engine._load_prompt_overlay()

            # Assert: Returns file content
            assert result == test_content.strip()
            assert engine._overlay_cache_text == test_content.strip()
            assert engine._overlay_cache_mtime_ns > 0

    def test_load_overlay_uses_cache_when_file_unchanged(self, mock_semantic_layer, tmp_path):
        """Test that _load_prompt_overlay() returns cached content when file mtime unchanged."""
        # Arrange: Create overlay file and load it once
        from clinical_analytics.core.nl_query_engine import NLQueryEngine

        overlay_file = tmp_path / "cached_overlay.txt"
        original_content = "Original content"
        overlay_file.write_text(original_content, encoding="utf-8")

        with patch.dict(os.environ, {"NL_PROMPT_OVERLAY_PATH": str(overlay_file)}):
            engine = NLQueryEngine(mock_semantic_layer)

            # First load: populates cache
            first_result = engine._load_prompt_overlay()
            first_mtime = engine._overlay_cache_mtime_ns

            # Act: Load again without changing file
            second_result = engine._load_prompt_overlay()

            # Assert: Returns cached content (mtime unchanged)
            assert second_result == first_result
            assert engine._overlay_cache_mtime_ns == first_mtime

    def test_load_overlay_reloads_when_file_modified(self, mock_semantic_layer, tmp_path):
        """Test that _load_prompt_overlay() reloads when file mtime changes."""
        # Arrange: Create overlay file and load it once
        from clinical_analytics.core.nl_query_engine import NLQueryEngine

        overlay_file = tmp_path / "modified_overlay.txt"
        original_content = "Original content"
        overlay_file.write_text(original_content, encoding="utf-8")

        with patch.dict(os.environ, {"NL_PROMPT_OVERLAY_PATH": str(overlay_file)}):
            engine = NLQueryEngine(mock_semantic_layer)

            # First load
            first_result = engine._load_prompt_overlay()

            # Modify file (content + mtime change)
            import time

            time.sleep(0.01)  # Ensure mtime changes
            new_content = "Updated content"
            overlay_file.write_text(new_content, encoding="utf-8")

            # Act: Load again after modification
            second_result = engine._load_prompt_overlay()

            # Assert: Returns new content (cache invalidated)
            assert first_result == original_content.strip()
            assert second_result == new_content.strip()
            assert engine._overlay_cache_text == new_content.strip()


class TestStableHash:
    """Test stable hashing for metrics."""

    def test_stable_hash_returns_deterministic_output(self):
        """Test that _stable_hash() returns same hash for same input across runs."""
        # Arrange: Import helper
        from clinical_analytics.core.nl_query_engine import _stable_hash

        test_query = "compare mortality by treatment"

        # Act: Hash same input multiple times
        hash1 = _stable_hash(test_query)
        hash2 = _stable_hash(test_query)
        hash3 = _stable_hash(test_query)

        # Assert: All hashes are identical (deterministic)
        assert hash1 == hash2 == hash3
        assert len(hash1) == 12  # First 12 chars of SHA256

    def test_stable_hash_differs_for_different_inputs(self):
        """Test that _stable_hash() produces different hashes for different inputs."""
        # Arrange: Import helper
        from clinical_analytics.core.nl_query_engine import _stable_hash

        query1 = "compare mortality by treatment"
        query2 = "compare survival by age"

        # Act: Hash different inputs
        hash1 = _stable_hash(query1)
        hash2 = _stable_hash(query2)

        # Assert: Hashes are different
        assert hash1 != hash2

    def test_stable_hash_is_not_pythons_builtin_hash(self):
        """Test that _stable_hash() is NOT Python's randomized hash()."""
        # Arrange: Import helper
        from clinical_analytics.core.nl_query_engine import _stable_hash

        test_query = "test query"

        # Act: Get stable hash vs Python hash
        stable = _stable_hash(test_query)
        builtin = str(hash(test_query))

        # Assert: Stable hash is hex string, not random int
        assert stable != builtin
        assert len(stable) == 12
        # Hex characters only
        assert all(c in "0123456789abcdef" for c in stable)
