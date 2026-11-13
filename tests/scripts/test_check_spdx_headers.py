"""Tests for scripts/check_spdx_headers.py (SPDX licence identifier guard)."""

import importlib.util
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "check_spdx_headers.py"
_spec = importlib.util.spec_from_file_location("check_spdx_headers", _SCRIPT)
check_spdx_headers = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(check_spdx_headers)

PY = "# SPDX-License-Identifier: MIT"
JS = "// SPDX-License-Identifier: MIT"


@pytest.mark.parametrize(
    ("path", "prefix"),
    [
        ("src/clinical_analytics/__init__.py", "#"),
        ("src/clinical_analytics/core/deep/module.py", "#"),
        ("electron/src/main.js", "//"),
        ("electron/src/main.test.js", "//"),
        ("electron/forge.config.js", None),
        ("electron/src/index.css", None),
        ("electron/src/fonts/SourceSans3-OFL.md", None),
        ("tests/core/test_x.py", None),
        ("scripts/run_app.sh", None),
    ],
    ids=[
        "init",
        "nested_py",
        "electron_js",
        "electron_test_js",
        "electron_config",
        "css",
        "font_licence",
        "tests",
        "shell",
    ],
)
def test_comment_prefix_path_coveredOnlyForFirstPartySource(path, prefix):
    # Act
    result = check_spdx_headers.comment_prefix(path)

    # Assert
    assert result == prefix


@pytest.mark.parametrize(
    ("text", "prefix", "expected"),
    [
        ("", "#", False),
        ('"""Doc."""\n', "#", False),
        (f"{PY}\n", "#", True),
        (f"#!/usr/bin/env python3\n{PY}\n", "#", True),
        (f"{JS}\n/** doc */\n", "//", True),
        (f"{PY}\n", "//", False),
        ("# SPDX-License-Identifier: Apache-2.0\n", "#", False),
        ("\n" * 10 + f"{PY}\n", "#", False),
    ],
    ids=[
        "empty",
        "docstring_only",
        "py_header",
        "after_shebang",
        "js_header",
        "wrong_comment_style",
        "other_licence",
        "too_far_down",
    ],
)
def test_has_header_text_detectsIdentifierNearTop(text, prefix, expected):
    # Act
    result = check_spdx_headers.has_header(text, prefix)

    # Assert
    assert result is expected


@pytest.mark.parametrize(
    ("text", "prefix", "expected"),
    [
        ("", "#", f"{PY}\n"),
        ('"""Doc."""\n', "#", f'{PY}\n"""Doc."""\n'),
        ("#!/usr/bin/env python3\nx = 1\n", "#", f"#!/usr/bin/env python3\n{PY}\nx = 1\n"),
        (
            "#!/usr/bin/env python3\n# -*- coding: utf-8 -*-\n",
            "#",
            f"#!/usr/bin/env python3\n# -*- coding: utf-8 -*-\n{PY}\n",
        ),
        ("/** doc */\r\n", "//", f"{JS}\r\n/** doc */\r\n"),
        (f"{PY}\nx = 1\n", "#", f"{PY}\nx = 1\n"),
    ],
    ids=["empty_file", "docstring", "after_shebang", "after_shebang_and_coding", "crlf_matched", "idempotent"],
)
def test_add_header_text_insertsOnceAfterPreamble(text, prefix, expected):
    # Act
    result = check_spdx_headers.add_header(text, prefix)

    # Assert
    assert result == expected
    assert check_spdx_headers.has_header(result, prefix)


def test_missing_mixedPaths_returnsOnlyCoveredFilesWithoutHeader():
    # Arrange
    contents = {
        "src/a.py": f"{PY}\n",
        "src/b.py": "x = 1\n",
        "electron/src/c.js": "const c = 1;\n",
        "tests/d.py": "x = 1\n",
    }

    # Act
    result = check_spdx_headers.missing(list(contents), contents.__getitem__)

    # Assert
    assert result == ["src/b.py", "electron/src/c.js"]
