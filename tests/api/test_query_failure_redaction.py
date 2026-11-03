"""An analysis exception's text (which can quote patient values) never reaches the stream, the
stored result or a log line (S2 follow-up, finding 4)."""

from __future__ import annotations

import asyncio
import logging
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import structlog
from structlog.testing import LogCapture

from clinical_analytics.api.services.query_service import UNEXPECTED_FAILURE_MESSAGE, AsyncQueryService

SECRET = "MRN 4471 has HbA1c 9.1"


@pytest.mark.asyncio
async def test_executeQuery_coreRaisesWithValues_streamStoreAndLogsCarryGenericMessage(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Arrange
    previous = structlog.get_config()
    capture = LogCapture()
    structlog.configure(
        processors=[capture],
        wrapper_class=structlog.make_filtering_bound_logger(logging.DEBUG),
        cache_logger_on_first_use=False,
    )
    try:
        with patch("clinical_analytics.api.services.query_service.CoreQueryService") as core_cls:
            core_cls.return_value.ask.side_effect = ValueError(SECRET)
            service = AsyncQueryService(MagicMock())

            # Act
            with caplog.at_level(logging.DEBUG):
                query_id = await service.submit_query(query="how many", dataset_id="ds")
                for _ in range(200):
                    if service._queries[query_id].status == "failed":
                        break
                    await asyncio.sleep(0.01)
    finally:
        structlog.configure(**previous)

    # Assert
    events: list[dict[str, Any]] = [e.data for e in service._events[query_id]]
    failed = [e for e in events if e.get("error")]
    assert failed and failed[-1]["error"] == UNEXPECTED_FAILURE_MESSAGE
    assert failed[-1]["error_type"] == "ValueError"
    assert service._queries[query_id].error == UNEXPECTED_FAILURE_MESSAGE
    assert SECRET not in repr(events)
    assert SECRET not in repr(capture.entries)
    assert SECRET not in caplog.text
    assert any(e.get("error_type") == "ValueError" and e.get("error_hash") for e in capture.entries)
