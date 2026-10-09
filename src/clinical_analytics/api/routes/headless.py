"""Headless natural-language ask endpoint.

Same execution path as `scripts/headless_ask.py`. Dataset ids are the
golden-suite ids (gdsi, dexa, statin, mimic_patients), not upload ids.
"""

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from clinical_analytics.eval.catalog import DATASET_TABLES
from clinical_analytics.eval.headless import _result_frame, ask

router = APIRouter()


class HeadlessAskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_id: str = Field(..., description="Golden dataset id")
    query: str = Field(..., min_length=1)


class HeadlessAskResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_id: str
    query: str
    intent: str | None
    metric: str | None
    group_by: str | None
    success: bool
    issues: list[dict[str, Any]]
    rows: list[dict[str, Any]]


@router.post("/headless/ask", response_model=HeadlessAskResponse)
def headless_ask(body: HeadlessAskRequest) -> HeadlessAskResponse:
    if body.dataset_id not in DATASET_TABLES:
        known = ", ".join(sorted(DATASET_TABLES))
        raise HTTPException(status_code=404, detail=f"Unknown dataset '{body.dataset_id}'. Known: {known}")
    result = ask(body.dataset_id, body.query)
    frame = _result_frame(result.get("payload"))
    rows: list[dict[str, Any]] = []
    if frame is not None:
        rows = frame.head(100).to_dicts()
    return HeadlessAskResponse(
        dataset_id=result["dataset_id"],
        query=result["query"],
        intent=result["intent"],
        metric=result["metric"],
        group_by=result["group_by"],
        success=bool(result["success"]),
        issues=result["issues"],
        rows=rows,
    )
