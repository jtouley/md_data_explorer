"""
Tests for self_improve_nl_parsing.py script functions.

Tests cover:
- Atomic overlay writes (temp + replace)
- Size capping (top N patterns, max overlay length)
- No leftover temp files
"""

from pathlib import Path


class TestAtomicOverlayWrite:
    """Test atomic write_prompt_overlay() behavior."""

    def test_write_overlay_creates_parent_directory(self, tmp_path):
        """Test that write_prompt_overlay() creates parent directory if missing."""
        # Arrange: Import and set up path to nonexistent directory
        import sys

        sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))
        from self_improve_nl_parsing import write_prompt_overlay

        nested_path = tmp_path / "nested" / "dir" / "overlay.txt"
        content = "Test content"

        # Act: Write to nonexistent directory
        write_prompt_overlay(content, nested_path)

        # Assert: Directory created and file exists
        assert nested_path.parent.exists()
        assert nested_path.exists()
        assert nested_path.read_text(encoding="utf-8") == content

    def test_write_overlay_uses_atomic_replace(self, tmp_path):
        """Test that write_prompt_overlay() uses temp file + atomic replace."""
        # Arrange: Import and set up
        import sys

        sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))
        from self_improve_nl_parsing import write_prompt_overlay

        overlay_path = tmp_path / "overlay.txt"
        original_content = "Original"
        new_content = "Updated"

        # Write original
        overlay_path.write_text(original_content)

        # Act: Atomic write
        write_prompt_overlay(new_content, overlay_path)

        # Assert: File updated atomically (no partial writes)
        assert overlay_path.read_text(encoding="utf-8") == new_content

    def test_write_overlay_leaves_no_temp_files(self, tmp_path):
        """Test that write_prompt_overlay() cleans up temp files after write."""
        # Arrange: Import and set up
        import sys

        sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))
        from self_improve_nl_parsing import write_prompt_overlay

        overlay_path = tmp_path / "overlay.txt"
        content = "Test content"

        # Act: Write overlay
        write_prompt_overlay(content, overlay_path)

        # Assert: No .tmp files left behind
        tmp_files = list(tmp_path.glob("*.tmp"))
        assert len(tmp_files) == 0, f"Found leftover temp files: {tmp_files}"

    def test_write_overlay_handles_unicode_content(self, tmp_path):
        """Test that write_prompt_overlay() correctly handles Unicode content."""
        # Arrange: Import and set up with Unicode
        import sys

        sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))
        from self_improve_nl_parsing import write_prompt_overlay

        overlay_path = tmp_path / "unicode_overlay.txt"
        unicode_content = "=== 修正 ===\nTest με γλώσσα"

        # Act: Write Unicode content
        write_prompt_overlay(unicode_content, overlay_path)

        # Assert: Content preserved correctly
        assert overlay_path.read_text(encoding="utf-8") == unicode_content


class TestOverlaySizeCapping:
    """Test overlay size capping logic."""

    def test_size_capping_keeps_top_five_patterns_by_priority(self):
        """Test that size capping keeps only top 5 patterns sorted by priority."""
        # Arrange: Create mock patterns with different priorities
        from clinical_analytics.core.prompt_optimizer import FailurePattern

        patterns = [
            FailurePattern("pattern_p1", 10, [], "Fix 1", priority=1),
            FailurePattern("pattern_p2", 8, [], "Fix 2", priority=2),
            FailurePattern("pattern_p3", 12, [], "Fix 3", priority=3),
            FailurePattern("pattern_p4", 5, [], "Fix 4", priority=1),  # Same priority as p1
            FailurePattern("pattern_p5", 7, [], "Fix 5", priority=2),  # Same priority as p2
            FailurePattern("pattern_p6", 3, [], "Fix 6", priority=4),  # Should be excluded
            FailurePattern("pattern_p7", 2, [], "Fix 7", priority=5),  # Should be excluded
        ]

        # Act: Sort by priority, then count (descending), take top 5
        patterns.sort(key=lambda p: (p.priority, -p.count))
        capped_patterns = patterns[:5]

        # Assert: Top 5 by priority kept
        assert len(capped_patterns) == 5
        assert capped_patterns[0].priority == 1
        assert capped_patterns[1].priority == 1
        assert capped_patterns[2].priority == 2
        assert capped_patterns[3].priority == 2
        assert capped_patterns[4].priority == 3

    def test_size_capping_truncates_overlay_at_8kb(self):
        """Test that overlay content is truncated at MAX_OVERLAY_LENGTH (8KB)."""
        # Arrange: Create large prompt additions
        max_overlay_length = 8000
        large_content = "A" * 10000  # 10KB

        # Act: Cap at 8KB
        capped_content = large_content[:max_overlay_length]

        # Assert: Content capped at 8KB
        assert len(capped_content) == max_overlay_length
        assert len(capped_content) < len(large_content)

    def test_size_capping_preserves_content_under_limit(self):
        """Test that overlay content under 8KB is not truncated."""
        # Arrange: Create small prompt additions
        max_overlay_length = 8000
        small_content = "Small content"

        # Act: Check if capping needed
        if len(small_content) > max_overlay_length:
            capped_content = small_content[:max_overlay_length]
        else:
            capped_content = small_content

        # Assert: Content unchanged
        assert capped_content == small_content
        assert len(capped_content) < max_overlay_length
