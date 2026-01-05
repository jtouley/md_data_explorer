"""
Tests for analyze_parse_outcomes.py script.

Tests cover:
- Script parses structlog JSON correctly
- Script handles empty log files gracefully
- Script validates parse_outcome events exist before processing
- Metrics aggregation (tier distribution, tier3 checkpoints)
"""

import json
import sys
from pathlib import Path
from unittest.mock import patch

# Add scripts directory to path for import
_scripts_dir = Path(__file__).parent.parent.parent / "scripts"
if str(_scripts_dir) not in sys.path:
    sys.path.insert(0, str(_scripts_dir))

from analyze_parse_outcomes import analyze_logs


class TestAnalyzeParseOutcomes:
    """Test analyze_parse_outcomes.py script functionality."""

    def test_script_handles_missing_log_file(self):
        """Test that script handles missing log file gracefully."""
        # Arrange

        nonexistent_file = Path("/tmp/nonexistent_log_file.log")

        # Act & Assert: Should not crash
        analyze_logs(nonexistent_file)
        # No exception should be raised

    def test_script_handles_empty_log_file(self, tmp_path):
        """Test that script handles empty log file gracefully."""
        # Arrange

        empty_log = tmp_path / "empty.log"
        empty_log.write_text("")

        # Act
        with patch("builtins.print") as mock_print:
            analyze_logs(empty_log)

            # Assert: Should print warning about no parse_outcome events
            print_calls = [str(call) for call in mock_print.call_args_list]
            assert any(
                "No 'parse_outcome' events found" in str(call) for call in print_calls
            ), "Should warn about missing parse_outcome events"

    def test_script_validates_parse_outcome_events_exist(self, tmp_path):
        """Test that script validates parse_outcome events exist before processing."""
        # Arrange

        log_file = tmp_path / "test.log"
        # Write log file with other events but no parse_outcome
        log_file.write_text('{"event": "query_parse_start", "query": "test"}\n')

        # Act
        with patch("builtins.print") as mock_print:
            analyze_logs(log_file)

            # Assert: Should warn about missing parse_outcome events
            print_calls = [str(call) for call in mock_print.call_args_list]
            assert any(
                "No 'parse_outcome' events found" in str(call) for call in print_calls
            ), "Should warn when parse_outcome events are missing"

    def test_script_parses_structlog_json_correctly(self, tmp_path):
        """Test that script parses structlog JSON lines correctly."""
        # Arrange

        log_file = tmp_path / "test.log"
        # Write valid parse_outcome events
        events = [
            {"event": "parse_outcome", "tier": "tier1", "success": True, "query_hash": "abc123"},
            {"event": "parse_outcome", "tier": "tier2", "success": True, "query_hash": "def456"},
            {"event": "parse_outcome", "tier": "tier3", "llm_called": True, "query_hash": "ghi789"},
        ]
        log_file.write_text("\n".join(json.dumps(e) for e in events))

        # Act
        with patch("builtins.print") as mock_print:
            analyze_logs(log_file)

            # Assert: Should process events without errors
            print_calls = [str(call) for call in mock_print.call_args_list]
            assert any("Total parses:" in str(call) for call in print_calls), "Should print total parses"

    def test_script_handles_json_parse_errors_gracefully(self, tmp_path):
        """Test that script handles JSON parse errors without crashing."""
        # Arrange

        log_file = tmp_path / "test.log"
        # Write log file with mix of valid and invalid JSON
        log_file.write_text(
            '{"event": "parse_outcome", "tier": "tier1"}\n'
            "invalid json line\n"
            '{"event": "parse_outcome", "tier": "tier2"}\n'
        )

        # Act & Assert: Should not crash, should process valid lines
        with patch("builtins.print") as mock_print:
            analyze_logs(log_file)

            # Should still process valid events
            print_calls = [str(call) for call in mock_print.call_args_list]
            assert any("Total parses:" in str(call) for call in print_calls), "Should process valid events"

    def test_script_computes_tier_distribution(self, tmp_path):
        """Test that script computes tier distribution correctly."""
        # Arrange

        log_file = tmp_path / "test.log"
        # Write events with known tier distribution
        events = [
            {"event": "parse_outcome", "tier": "tier1", "success": True, "query_hash": "a1"},
            {"event": "parse_outcome", "tier": "tier1", "success": True, "query_hash": "a2"},
            {"event": "parse_outcome", "tier": "tier2", "success": True, "query_hash": "b1"},
            {"event": "parse_outcome", "tier": "tier3", "llm_called": True, "query_hash": "c1"},
        ]
        log_file.write_text("\n".join(json.dumps(e) for e in events))

        # Act
        with patch("builtins.print") as mock_print:
            analyze_logs(log_file)

            # Assert: Should show tier distribution
            print_calls = [str(call) for call in mock_print.call_args_list]
            output = "\n".join(str(call) for call in print_calls)

            assert "Tier 1 (pattern): 2" in output or "tier1" in output.lower(), "Should show tier1 count"
            assert "Tier 2 (semantic): 1" in output or "tier2" in output.lower(), "Should show tier2 count"
            assert "Tier 3 (LLM): 1" in output or "tier3" in output.lower(), "Should show tier3 count"

    def test_script_computes_tier3_checkpoints(self, tmp_path):
        """Test that script computes tier3 granular checkpoints correctly."""
        # Arrange

        log_file = tmp_path / "test.log"
        # Write tier3 events with checkpoints
        events = [
            {"event": "parse_outcome", "tier": "tier3", "llm_called": True, "query_hash": "c1"},
            {"event": "parse_outcome", "tier": "tier3", "llm_http_success": True, "query_hash": "c1"},
            {"event": "parse_outcome", "tier": "tier3", "json_parse_success": True, "query_hash": "c1"},
            {
                "event": "parse_outcome",
                "tier": "tier3",
                "schema_validate_success": True,
                "final_returned_from_tier3": True,
                "query_hash": "c1",
            },
        ]
        log_file.write_text("\n".join(json.dumps(e) for e in events))

        # Act
        with patch("builtins.print") as mock_print:
            analyze_logs(log_file)

            # Assert: Should show tier3 checkpoints
            print_calls = [str(call) for call in mock_print.call_args_list]
            output = "\n".join(str(call) for call in print_calls)

            assert "LLM called:" in output or "llm_called" in output.lower(), "Should show llm_called"
            assert "LLM HTTP success:" in output or "llm_http_success" in output.lower(), "Should show llm_http_success"
            assert (
                "JSON parse success:" in output or "json_parse_success" in output.lower()
            ), "Should show json_parse_success"
            assert (
                "Schema validate success:" in output or "schema_validate_success" in output.lower()
            ), "Should show schema_validate_success"
