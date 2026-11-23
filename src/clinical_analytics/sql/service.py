# SPDX-License-Identifier: MIT
"""Governed SQL on a leased semantic layer: the one entry point the API calls.

The layer (held through its lease, ``api/layer_cache.py``) says which dataset version is being
asked about and where that version's stored Parquet is; the contract and the confirmed level
merges of that same version are read from the data home. A legacy upload served from memory, or
a version without a contract, is refused rather than guessed.

A dataset whose rows can't be shown to be patients by their IDs (no ID column, or rows without
one) is refused until the user confirms one row per patient. The confirmation is the dataset's
grain declaration (``contracts/declarations.py``, grain ``patient``): kept per dataset version and
row count, so a re-upload asks again. The refusal carries a confirmation token bound to the dataset
version, ID column and row count it asked about; only that token is accepted (ADR013 section 6.3), so
a confirmation never lands on data the user was not asked about (a re-upload in between, or a
client that always sends one). The same grain declaration is stored when the user answers "Is
each row one patient?" at upload (``core.patient_rows``), so either answer serves both paths.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import sqlglot
import structlog
from sqlglot import exp

from clinical_analytics.contracts.declarations import (
    Declarations,
    DeclarationStore,
    GrainDeclaration,
    MissingCodeDeclaration,
    confirmation_token,
)
from clinical_analytics.contracts.errors import ContractError, ContractStoreError
from clinical_analytics.contracts.holds import held_message, open_question_holds
from clinical_analytics.contracts.readings import declared_reads
from clinical_analytics.contracts.semantics import layer_semantics
from clinical_analytics.contracts.service import read_saved_contract
from clinical_analytics.contracts.store import contract_path
from clinical_analytics.sql.answer import PatientRowsUnconfirmedError, SqlAnswer, answer_query
from clinical_analytics.sql.catalog import SqlCatalog
from clinical_analytics.sql.errors import SqlHeldError, SqlRefusedError
from clinical_analytics.sql.sandbox import SandboxConnection
from clinical_analytics.sql.validate import validate_sql

logger = structlog.get_logger()

NO_CONTRACT = (
    "This dataset has no data contract yet, so SQL cannot be checked against it. "
    "Open the dataset's quality report once to create it."
)
NOT_STORED = "This dataset is not stored as Parquet yet, so SQL cannot run on it. Upload it again."
DECLARATIONS_UNREADABLE = "This dataset's saved confirmations can't be read or saved, so SQL can't run on it now."
AGGREGATE_ROWS = (
    "Each line of this table counts cases for a group (in {count}), so a count of lines is a count of groups, "
    "not of patients, and it can't be confirmed as one patient per line. Add up {count} with SUM({count}) instead."
)
AGGREGATE_NOTE = "Each line counts cases for a group: sums of {count} are cases, never patients."
CONFIRMATION_STALE = (
    "The dataset changed after you were asked to confirm one row per patient. "
    "Check that each row is still one patient, then confirm again."
)


def run_governed_sql(
    sql: str,
    layer: Any,
    *,
    dataset_id: str,
    data_dir: Path,
    declarations: DeclarationStore,
    confirm_one_row_per_patient: str | None = None,
) -> SqlAnswer:
    """Validate ``sql`` against the contract of the layer's dataset version and answer it.

    ``confirm_one_row_per_patient``: the confirmation token from an earlier refusal; the user
    confirms that each row is one patient. Stored (and used) only when the IDs can't show it, the
    query is valid and the token matches this dataset version, ID column and row count.

    Raises:
        PatientRowsUnconfirmedError: rows can't be shown to be patients and are not confirmed.
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
    try:
        loaded = declarations.load(dataset_id)
    except ContractStoreError as e:
        log.warning("governed_sql_declarations_unreadable", error_type=type(e).__name__)
        raise SqlRefusedError(DECLARATIONS_UNREADABLE) from e
    declared = loaded.for_version(version)
    # The leased layer's own semantics, with the answers it was built with (contracts.holds).
    semantics = layer_semantics(layer, declared=True)
    reads = declared_reads(loaded, version, semantics)
    missing = [
        MissingCodeDeclaration(column=column, value=value, missing=True, dataset_version=version)
        for column, values in reads.missing.items()
        for value in values
    ]
    catalog = SqlCatalog.from_contract(
        contract, Path(source), declared.levels, missing, declared.date_readings_for(version)
    )
    query = validate_sql(sql, catalog)
    read = {c.name for c in sqlglot.parse_one(query.shown_sql, read="duckdb").find_all(exp.Column)}
    # Counting lines is gated below by the one-row-per-patient confirmation, SQL's own answer to it.
    holds = open_question_holds(semantics, columns=read, counts_lines=False)
    if holds:
        log.info("governed_sql_held", questions=len(holds))
        raise SqlHeldError(held_message(holds), questions=tuple(h.question_id for h in holds))
    aggregate = semantics.aggregate if semantics is not None and semantics.aggregate is not None else None
    if aggregate is not None and aggregate.source == "declared":
        # A declared table of groups: a count of lines counts groups, never patients; a sum of its
        # count column is cases, and is answered.
        if any(a.kind == "count" for a in query.aggregates):
            log.info("governed_sql_aggregate_rows_refused", kind=aggregate.kind)
            raise SqlRefusedError(AGGREGATE_ROWS.format(count=aggregate.count_column or "the count"))
        with SandboxConnection(catalog) as sandbox:
            return answer_query(
                query, sandbox, catalog, groups_note=AGGREGATE_NOTE.format(count=aggregate.count_column)
            )
    with SandboxConnection(catalog) as sandbox:
        try:
            return answer_query(query, sandbox, catalog, confirmed_patient_rows=_confirmed_rows(declared, catalog))
        except PatientRowsUnconfirmedError as e:
            if aggregate is not None:
                # Lines of a table of groups are never patients: no confirmation is offered or taken.
                log.info("governed_sql_aggregate_rows_refused", kind=aggregate.kind)
                raise SqlRefusedError(AGGREGATE_ROWS.format(count=aggregate.count_column or "the count")) from e
            token = confirmation_token(version, e.key, e.rows)
            if confirm_one_row_per_patient is None:
                raise PatientRowsUnconfirmedError(e.reason, key=e.key, rows=e.rows, confirmation=token) from e
            if confirm_one_row_per_patient != token:
                log.info("governed_sql_patient_grain_confirmation_stale")
                raise PatientRowsUnconfirmedError(CONFIRMATION_STALE, key=e.key, rows=e.rows, confirmation=token) from e
            grain = GrainDeclaration(grain="patient", key=e.key, rows=e.rows, patients=e.rows, dataset_version=version)
            try:
                declarations.update(dataset_id, lambda current: current.with_grain(grain))
            except ContractStoreError as store_error:
                log.warning("governed_sql_declarations_unsaved", error_type=type(store_error).__name__)
                raise SqlRefusedError(DECLARATIONS_UNREADABLE) from store_error
            log.info("governed_sql_patient_grain_confirmed", dataset_version=version)
            return answer_query(query, sandbox, catalog, confirmed_patient_rows=e.rows)


def _confirmed_rows(declared: Declarations, catalog: SqlCatalog) -> int | None:
    """The row count of this version's one-row-per-patient confirmation, if it names the same ID
    column as the contract (None: no ID column) and is not a visit declaration."""
    if not declared.grains:
        return None
    grain = declared.grains[0]
    key = catalog.id_column.name if catalog.id_column is not None else None
    return grain.rows if grain.grain == "patient" and grain.key == key else None
