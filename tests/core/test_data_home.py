"""Tests for core/data_home.py: runtime data never lives in the source tree."""

from pathlib import Path

import pytest

from clinical_analytics.core import data_home

REPO_ROOT = Path(__file__).resolve().parents[2]


class TestDataHome:
    def test_data_home_envOverride_usedForAllRuntimePaths(self, tmp_path, monkeypatch):
        # Arrange
        monkeypatch.setenv(data_home.DATA_HOME_ENV, str(tmp_path))

        # Act / Assert
        assert data_home.data_home() == tmp_path.resolve()
        assert data_home.uploads_dir() == tmp_path.resolve() / "data" / "uploads"
        assert data_home.session_db_path() == tmp_path.resolve() / "data" / "clinical_analytics.db"
        assert data_home.nl_learning_dir() == tmp_path.resolve() / "data" / "nl_query_learning"

    @pytest.mark.parametrize(
        ("platform", "env", "expected_suffix"),
        [
            ("darwin", {}, ("Library", "Application Support", "ClinicalAnalytics")),
            ("linux", {}, (".local", "share", "clinical-analytics")),
            ("linux", {"XDG_DATA_HOME": "/xdg"}, ("clinical-analytics",)),
            ("win32", {"LOCALAPPDATA": "/appdata"}, ("ClinicalAnalytics",)),
        ],
        ids=["macos", "linux_default", "linux_xdg", "windows"],
    )
    def test_data_home_noOverride_usesPerUserAppDataOutsideRepo(self, monkeypatch, platform, env, expected_suffix):
        # Arrange
        monkeypatch.delenv(data_home.DATA_HOME_ENV, raising=False)
        monkeypatch.delenv("XDG_DATA_HOME", raising=False)
        monkeypatch.delenv("LOCALAPPDATA", raising=False)
        for key, value in env.items():
            monkeypatch.setenv(key, value)
        monkeypatch.setattr(data_home.sys, "platform", platform)

        # Act
        resolved = data_home.data_home()

        # Assert
        assert resolved.parts[-len(expected_suffix) :] == expected_suffix
        assert not resolved.is_relative_to(REPO_ROOT)

    def test_user_dataset_storage_default_writesUnderDataHome(self, tmp_path, monkeypatch):
        # Arrange
        from clinical_analytics.ui.storage.user_datasets import UserDatasetStorage

        monkeypatch.setenv(data_home.DATA_HOME_ENV, str(tmp_path))

        # Act
        storage = UserDatasetStorage()

        # Assert
        assert storage.upload_dir == tmp_path.resolve() / "data" / "uploads"
        assert storage.raw_dir.is_dir()
        assert not storage.upload_dir.is_relative_to(REPO_ROOT)
