"""The backend runs as an installed wheel, from an unrelated cwd, with no network.

Builds the wheel, installs it into a fresh venv outside the repo, starts the API with HOME and
XDG_DATA_HOME pointing at a temp data home, then uploads the synthetic cohort and runs a query.
Run with ``make smoke-installed``.
"""

from __future__ import annotations

import shutil

import pytest

from scripts.smoke_installed import _unshare_available, run_smoke


@pytest.mark.slow
@pytest.mark.integration
def test_installed_wheel_unrelatedCwdOffline_servesHealthUploadAndQuery(tmp_path):
    # Arrange
    if shutil.which("uv") is None:
        pytest.fail("uv is required to build and install the wheel")

    # Act
    result = run_smoke(tmp_path)

    # Assert
    assert result.health_status == 200, result.server_log_tail
    assert result.upload_status == 200, result.server_log_tail
    assert result.query_status == "completed", result.server_log_tail
    assert result.new_files_in_install == [], "the app wrote into its install location"
    assert result.data_home_files > 0, "runtime data did not go to the temp data home"
    if _unshare_available():
        assert result.offline, "expected the smoke to run in a network namespace without internet"
