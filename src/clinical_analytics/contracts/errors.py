"""Exceptions for data contracts and data-quality runs. Messages never contain cell values."""

from __future__ import annotations


class ContractError(Exception):
    """Base class: building, storing or checking a data contract failed."""


class ContractValidationError(ContractError):
    """A contract does not conform to the ODCS model."""


class ContractStoreError(ContractError):
    """A contract or quality report could not be written or read in the data home."""


class QualityCheckError(ContractError):
    """A contract rule could not be compiled to a Polars check."""
