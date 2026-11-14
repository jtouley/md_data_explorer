# SPDX-License-Identifier: MIT
"""Exceptions of the governed SQL path. Messages are shown to the user: they never quote a cell
value or a path."""

from __future__ import annotations


class SqlRefusedError(Exception):
    """The query is not run (or its result is not shown), for the plain-language ``reason``."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason
