"""Tests for repo-context MCP path/slug helpers."""

from __future__ import annotations

import pytest

from repo_context_mcp._lib import diagnostic_file, parse_initiative_slug


def test_parse_initiative_slug_empty_returns_none():
    assert parse_initiative_slug(None) is None
    assert parse_initiative_slug("") is None
    assert parse_initiative_slug("   ") is None


def test_parse_initiative_slug_accepts_safe_token():
    assert parse_initiative_slug("electron_ui_migration") == "electron_ui_migration"
    assert parse_initiative_slug("p01-upload") == "p01-upload"


def test_parse_initiative_slug_rejects_path_traversal():
    with pytest.raises(ValueError, match="initiative must match"):
        parse_initiative_slug("../etc/passwd")
    with pytest.raises(ValueError, match="initiative must match"):
        parse_initiative_slug("foo/bar")


def test_diagnostic_file_rolling_vs_initiative(tmp_path):
    r = tmp_path / "repo"
    assert diagnostic_file(r, None) == r / ".context" / "diagnostics" / "repo_context.md"
    assert diagnostic_file(r, "foo") == r / ".context" / "diagnostics" / "foo_context.md"
