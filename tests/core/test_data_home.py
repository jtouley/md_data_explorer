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


def _make_storage() -> object:
    from clinical_analytics.ui.storage.user_datasets import UserDatasetStorage

    return UserDatasetStorage()


def _make_db_url() -> object:
    from clinical_analytics.api.db.database import _default_database_url

    return _default_database_url()


def _make_log_key() -> object:
    from clinical_analytics.core import log_redaction

    log_redaction._log_keys.clear()
    return log_redaction.log_hash_key()


@pytest.mark.skipif(not hasattr(__import__("os"), "getuid"), reason="POSIX permissions")
@pytest.mark.parametrize(
    ("creator", "existing_mode"),
    [
        (_make_storage, None),
        (_make_db_url, None),
        (_make_log_key, None),
        (_make_storage, 0o755),
        (_make_storage, 0o700),
    ],
    ids=["uploads_fresh", "session_db_fresh", "log_key_fresh", "uploads_existing_open", "uploads_existing_private"],
)
def test_dataHome_firstWriter_createsOwnerOnlyDirectory(tmp_path, monkeypatch, creator, existing_mode):
    # Arrange
    home = tmp_path / "nested" / "home"
    if existing_mode is not None:
        home.mkdir(parents=True)
        home.chmod(existing_mode)
    monkeypatch.setenv(data_home.DATA_HOME_ENV, str(home))

    # Act
    creator()

    # Assert
    assert home.stat().st_mode & 0o777 == 0o700


_POSIX_ROOT = hasattr(__import__("os"), "geteuid") and __import__("os").geteuid() == 0


@pytest.mark.skipif(not _POSIX_ROOT, reason="needs chown to another uid")
@pytest.mark.parametrize("mode", [0o777, 0o700], ids=["world_open", "owner_only"])
def test_ensureDataHome_ownedByAnotherUser_refusesWithClearMessage(tmp_path, monkeypatch, mode):
    # Arrange
    import os

    home = tmp_path / "theirs"
    home.mkdir()
    home.chmod(mode)
    os.chown(home, 12345, 12345)
    monkeypatch.setenv(data_home.DATA_HOME_ENV, str(home))

    # Act / Assert
    with pytest.raises(PermissionError, match="owned by another user"):
        data_home.ensure_data_home()
    assert home.stat().st_mode & 0o777 == mode


@pytest.mark.skipif(not hasattr(__import__("os"), "getuid"), reason="POSIX permissions")
@pytest.mark.parametrize(
    ("parent_mode", "expect_warning"),
    [(0o777, True), (0o1777, False), (0o755, False)],
    ids=["world_writable", "sticky_like_tmp", "normal"],
)
def test_ensureDataHome_parentPermissions_warnsOnlyWhenWorldWritableWithoutSticky(
    tmp_path, monkeypatch, parent_mode, expect_warning
):
    # Arrange
    from structlog.testing import capture_logs

    parent = tmp_path / "shared"
    parent.mkdir()
    parent.chmod(parent_mode)
    monkeypatch.setenv(data_home.DATA_HOME_ENV, str(parent / "home"))

    # Act
    with capture_logs() as logs:
        home = data_home.ensure_data_home()

    # Assert
    warned = [e for e in logs if e["event"] == "data_home_parent_world_writable"]
    assert bool(warned) is expect_warning
    assert home.stat().st_mode & 0o777 == 0o700
