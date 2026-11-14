# SPDX-License-Identifier: MIT
"""Governed SQL on a leased semantic layer: the one entry point the API calls.

The layer (held through its lease, ``api/layer_cache.py``) says which dataset version is being
asked about and where that version's stored Parquet is; the contract and the confirmed level
merges of that same version are read from the data home. A legacy upload served from memory, or
a version without a contract, is refused rather than guessed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import structlog

from clinical_analytics.contracts.declarations import DeclarationStore
from clinical_analytics.contracts.errors import ContractError
from clinical_analytics.contracts.service import read_saved_contract
from clinical_analytics.contracts.store import contract_path
from clinical_analytics.sql.catalog import SqlCatalog
from clinical_analytics.sql.errors import SqlRefusedError
from clinical_analytics.sql.privacy import SqlAnswer, answer_query
from clinical_analytics.sql.sandbox import SandboxConnection
from clinical_analytics.sql.validate import validate_sql

logger = structlog.get_logger()

NO_CONTRACT = (
    "This dataset has no data contract yet, so SQL cannot be checked against it. "
    "Open the dataset's quality report once to create it."
)
NOT_STORED = "This dataset is not stored as Parquet yet, so SQL cannot run on it. Upload it again."


def run_governed_sql(
    sql: str, layer: Any, *, dataset_id: str, data_dir: Path, declarations: DeclarationStore
) -> SqlAnswer:
    """Validate ``sql`` against the contract of the layer's dataset version and answer it.

    Raises:
        SqlRefusedError: the reason the query was not run or its result not shown.
    """
    log = logger.bind(dataset_id=dataset_id)
    version = getattr(layer, "loaded_version", None)
    source = ((getattr(layer, "config", None) or {}).get("init_params") or {}).get("source_path")
    if not isinstance(version, str) or not version:
        raise SqlRefusedError(NO_CONTRACT)
    if not isinstance(source, str) or not source:
        raise SqlRefusedError(NOT_STORED)
    path = contract_path(data_dir, dataset_id, version)
    if not path.is_file():
        raise SqlRefusedError(NO_CONTRACT)
    try:
        contract = read_saved_contract(path)
    except ContractError as e:
        log.warning("governed_sql_contract_unreadable", error_type=type(e).__name__)
        raise SqlRefusedError("This dataset's data contract can't be read, so SQL cannot be checked against it.") from e
    levels = declarations.load(dataset_id).levels_for(version)
    catalog = SqlCatalog.from_contract(contract, Path(source), levels)
    query = validate_sql(sql, catalog)
    with SandboxConnection(catalog) as sandbox:
        return answer_query(query, sandbox, catalog)
