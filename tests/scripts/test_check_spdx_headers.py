"""Tests for scripts/check_spdx_headers.py (SPDX licence identifier guard)."""

import pytest

from scripts import check_spdx_headers

PY = "# SPDX-License-Identifier: MIT"
JS = "// SPDX-License-Identifier: MIT"
BOM = "\ufeff"


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


def test_scan_mixedPaths_returnsOnlyCoveredFilesWithoutHeader():
    # Arrange
    contents = {
        "src/a.py": f"{PY}\n",
        "src/b.py": "x = 1\n",
        "electron/src/c.js": "const c = 1;\n",
        "tests/d.py": "x = 1\n",
    }

    # Act
    result = check_spdx_headers.scan(list(contents), contents.__getitem__)

    # Assert
    assert result.missing == ["src/b.py", "electron/src/c.js"]
    assert result.undecodable == []


# --- PEP 263 encoding declarations -----------------------------------------------------------


@pytest.mark.parametrize(
    ("line", "matches"),
    [
        ("# -*- coding: latin-1 -*-", True),
        ("# coding=utf-8", True),
        ("\t# coding: utf-8", True),
        ("# vim: set fileencoding=utf-8 :", True),
        ("# This Python file uses the following encoding: utf-8", True),
        ("# decoding: base64 helpers", False),
        ("# TODO recoding=later", False),
        ("# coding: ", False),
        ("x = 1  # coding: utf-8", False),
        ("", False),
    ],
    ids=[
        "emacs_style",
        "equals_form",
        "leading_tab",
        "vim_fileencoding",
        "plain_encoding",
        "decoding_prose",
        "recoding_prose",
        "no_charset",
        "trailing_comment_on_code",
        "empty",
    ],
)
def test_encoding_line_comment_matchesOnlyRealDeclarations(line, matches):
    # Act
    result = check_spdx_headers.ENCODING_LINE.match(line)

    # Assert
    assert (result is not None) is matches


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            "# Copyright X\n# -*- coding: latin-1 -*-\nx = 1\n",
            f"# Copyright X\n# -*- coding: latin-1 -*-\n{PY}\nx = 1\n",
        ),
        ("# -*- coding: latin-1 -*-\nx = 1\n", f"# -*- coding: latin-1 -*-\n{PY}\nx = 1\n"),
        ("# vim: set fileencoding=latin-1 :\nx = 1\n", f"# vim: set fileencoding=latin-1 :\n{PY}\nx = 1\n"),
        ("#!/usr/bin/env python3\n# coding=utf-8\nx = 1\n", f"#!/usr/bin/env python3\n# coding=utf-8\n{PY}\nx = 1\n"),
        ("# decoding: notes\nx = 1\n", f"{PY}\n# decoding: notes\nx = 1\n"),
        ("x = 1\ny = 2\n# coding: latin-1\n", f"{PY}\nx = 1\ny = 2\n# coding: latin-1\n"),
    ],
    ids=[
        "coding_on_line_2_after_comment",
        "coding_on_line_1",
        "vim_on_line_1",
        "shebang_then_coding",
        "prose_is_not_preamble",
        "coding_past_line_2_ignored",
    ],
)
def test_add_header_encodingLine_staysWithinFirstTwoLines(text, expected):
    # Act
    result = check_spdx_headers.add_header(text, "#")

    # Assert
    assert result == expected
    first_two = result.splitlines()[:2]
    if any(check_spdx_headers.ENCODING_LINE.match(line) for line in text.splitlines()[:2]):
        assert any(check_spdx_headers.ENCODING_LINE.match(line) for line in first_two)


# --- UTF-8 BOM ----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "prefix", "expected"),
    [
        (f"{BOM}{PY}\n", "#", True),
        (f"{BOM}#!/usr/bin/env python3\n{PY}\n", "#", True),
        (f"{BOM}{JS}\n", "//", True),
        (f"{BOM}x = 1\n", "#", False),
    ],
    ids=["py_header", "shebang_then_header", "js_header", "no_header"],
)
def test_has_header_bomPrefixedText_seesHeaderAfterBom(text, prefix, expected):
    # Act
    result = check_spdx_headers.has_header(text, prefix)

    # Assert
    assert result is expected


@pytest.mark.parametrize(
    ("text", "prefix", "expected"),
    [
        (f'{BOM}"""Doc."""\n', "#", f'{BOM}{PY}\n"""Doc."""\n'),
        (f"{BOM}# -*- coding: utf-8 -*-\nx = 1\n", "#", f"{BOM}# -*- coding: utf-8 -*-\n{PY}\nx = 1\n"),
        (f"{BOM}const a = 1;\r\n", "//", f"{BOM}{JS}\r\nconst a = 1;\r\n"),
        (f"{BOM}{PY}\nx = 1\n", "#", f"{BOM}{PY}\nx = 1\n"),
        (BOM, "#", f"{BOM}{PY}\n"),
    ],
    ids=["docstring", "coding_line", "crlf_js", "idempotent", "bom_only"],
)
def test_add_header_bomPrefixedText_keepsBomFirst(text, prefix, expected):
    # Act
    result = check_spdx_headers.add_header(text, prefix)

    # Assert
    assert result == expected
    assert result.count(BOM) == 1
    assert check_spdx_headers.has_header(result, prefix)


def test_add_header_bomFile_resultCompiles():
    # Arrange
    text = f'{BOM}"""Doc."""\nx = "é"\n'

    # Act
    result = check_spdx_headers.add_header(text, "#")

    # Assert: Python accepts a BOM only as the very first character.
    compile(result.encode("utf-8"), "bom.py", "exec")


# --- main(): run location and undecodable files ----------------------------------------------

_HEADED = f"{PY}\nx = 1\n".encode()
_UNHEADED = b"x = 1\n"


@pytest.mark.parametrize("cwd", [".", "src", "src/pkg", "docs"], ids=["root", "src", "nested", "sibling_dir"])
def test_main_runFromAnyDirectory_reportsRepoRelativeMissingFile(make_git_repo, monkeypatch, capsys, cwd):
    # Arrange
    root = make_git_repo({"src/pkg/a.py": _UNHEADED, "src/pkg/b.py": _HEADED, "docs/index.md": b"# Docs\n"})
    monkeypatch.chdir(root / cwd)

    # Act
    code = check_spdx_headers.main([])

    # Assert
    assert code == 1
    assert "  src/pkg/a.py\n" in capsys.readouterr().err


@pytest.mark.parametrize("cwd", [".", "src/pkg"], ids=["root", "nested"])
def test_main_fixFromAnyDirectory_writesFileAtRepoPath(make_git_repo, monkeypatch, cwd):
    # Arrange
    root = make_git_repo({"src/pkg/a.py": _UNHEADED})
    monkeypatch.chdir(root / cwd)

    # Act
    code = check_spdx_headers.main(["--fix"])

    # Assert
    assert code == 0
    assert (root / "src/pkg/a.py").read_bytes() == f"{PY}\nx = 1\n".encode()
    assert check_spdx_headers.main([]) == 0


@pytest.mark.parametrize(
    "content",
    [
        "# café\nx = 1\n".encode("latin-1"),
        "x = '\u201cq\u201d'\n".encode("cp1252"),
        f"{PY}\nx = 1\n".encode("utf-16"),
        b"\xff\xfe\x00",
    ],
    ids=["latin1", "cp1252", "utf16_with_bom", "garbage_bytes"],
)
@pytest.mark.parametrize("fix", [False, True], ids=["check", "fix"])
def test_main_nonUtf8File_failsNamingPath(make_git_repo, monkeypatch, capsys, content, fix):
    # Arrange
    root = make_git_repo({"src/bad.py": content, "src/ok.py": _HEADED, "src/todo.py": _UNHEADED})
    monkeypatch.chdir(root)

    # Act
    code = check_spdx_headers.main(["--fix"] if fix else [])

    # Assert
    err = capsys.readouterr().err
    assert code == 1
    assert "src/bad.py" in err
    assert "not valid UTF-8" in err
    assert (root / "src/bad.py").read_bytes() == content
    if fix:
        assert (root / "src/todo.py").read_bytes().startswith(PY.encode())
    else:
        assert "  src/todo.py\n" in err


def test_scan_undecodableFile_listedSeparatelyFromMissing():
    # Arrange
    def read(path: str) -> str:
        if path == "src/bad.py":
            raise UnicodeDecodeError("utf-8", b"\xe9", 0, 1, "invalid continuation byte")
        return "x = 1\n"

    # Act
    result = check_spdx_headers.scan(["src/bad.py", "src/b.py"], read)

    # Assert
    assert result.missing == ["src/b.py"]
    assert [path for path, _reason in result.undecodable] == ["src/bad.py"]
