# SPDX-License-Identifier: MIT
"""Exceptions for data contracts and data-quality runs. Messages never contain cell values."""

from __future__ import annotations


class ContractError(Exception):
    """Base class: building, storing or checking a data contract failed."""


class ContractValidationError(ContractError):
    """A contract does not conform to the ODCS model."""


class ContractDamagedError(ContractValidationError):
    """The saved contract for a version is empty, truncated or not a usable contract.

    It is never rewritten (contracts are immutable), so every later check reports this until the
    cohort is deleted and uploaded again.
    """


class ContractStoreError(ContractError):
    """A contract or quality report could not be written or read in the data home."""


class InvalidDeclarationKeyError(ContractStoreError):
    """The dataset id under which declarations would be stored is not a plain identifier."""


class QualityCheckError(ContractError):
    """A contract rule could not be compiled to a Polars check."""
