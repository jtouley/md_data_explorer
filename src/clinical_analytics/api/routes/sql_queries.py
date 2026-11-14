# SPDX-License-Identifier: MIT
"""Governed SQL (behind ``ENABLE_SQL_QUERIES``, off by default).

POST /api/queries/sql runs one user-typed SELECT on the dataset's current version: checked
against its contract, run read-only in a sandbox, small cells suppressed (``clinical_analytics.sql``).
"""

from __future__ import annotations

import structlog
from fastapi import APIRouter, HTTPException, status

from clinical_analytics.api.dependencies import lease_semantic_layer
from clinical_analytics.api.models.schemas import SqlQueryRequest, SqlQueryResult
from clinical_analytics.contracts.declarations import FileDeclarationStore
from clinical_analytics.core import nl_query_config
from clinical_analytics.core.data_home import uploads_dir
from clinical_analytics.sql.errors import SqlRefusedError
from clinical_analytics.sql.privacy import PatientRowsUnconfirmedError
from clinical_analytics.sql.sandbox import SqlBusyError
from clinical_analytics.sql.service import run_governed_sql
from clinical_analytics.storage.dataset_deletion import InvalidDatasetIdError, validate_dataset_id

router = APIRouter()
logger = structlog.get_logger()


@router.post("/queries/sql", response_model=SqlQueryResult)
def run_sql_query(request: SqlQueryRequest) -> SqlQueryResult:
    """Answer one SELECT, or say why not. Sync: FastAPI runs it in a worker thread, and the lease
    is released in ``finally`` whatever happens.

    Raises:
        HTTPException: 404 when the flag is off or the dataset does not exist, 400 for a bad id,
            409/503 as ``lease_semantic_layer``, 503 when every SQL sandbox stays busy.
    """
    if not nl_query_config.ENABLE_SQL_QUERIES:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Not Found")
    try:
        validate_dataset_id(request.dataset_id)
    except InvalidDatasetIdError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)) from e
    log = logger.bind(dataset_id=request.dataset_id)
    data_dir = uploads_dir().parent
    with lease_semantic_layer(request.dataset_id) as lease:
        try:
            answer = run_governed_sql(
                request.sql,
                lease.layer,
                dataset_id=request.dataset_id,
                data_dir=data_dir,
                declarations=FileDeclarationStore(data_dir),
                confirm_one_row_per_patient=request.confirm_one_row_per_patient,
            )
        except SqlBusyError as e:
            log.info("governed_sql_busy")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=e.reason, headers={"Retry-After": "5"}
            ) from e
        except PatientRowsUnconfirmedError as e:
            log.info("governed_sql_needs_patient_grain")
            return SqlQueryResult(status="refused", sql=None, reason=e.reason, needs_confirmation="one_row_per_patient")
        except SqlRefusedError as e:
            log.info("governed_sql_refused")
            return SqlQueryResult(status="refused", sql=None, reason=e.reason, needs_confirmation=None)
    return SqlQueryResult(
        status="answered",
        sql=answer.sql,
        columns=answer.columns,
        rows=answer.rows,
        notes=answer.notes,
        reason=None,
        needs_confirmation=None,
    )
