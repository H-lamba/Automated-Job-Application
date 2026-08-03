"""
api/routes/applications.py — Application record endpoints.
"""

from __future__ import annotations

import math
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func, select

from api.schemas import (
    ApplicationListResponse,
    ApplicationRecordResponse,
    ApplicationTriggerResponse,
    PaginationMeta,
)
from core.config import Settings, settings_dep
from core.database import get_session
from core.logger import logger
from models.application import ApplicationRecord

router = APIRouter(prefix="/applications", tags=["Applications"])

SettingsDep = Annotated[Settings, Depends(settings_dep)]


def _to_response(r: ApplicationRecord) -> ApplicationRecordResponse:
    return ApplicationRecordResponse(
        id=r.id,
        job_id=r.job_id,
        status=r.status,
        started_at=r.started_at,
        completed_at=r.completed_at,
        duration_seconds=r.duration_seconds,
        ats_detected=r.ats_detected,
        resume_path=r.resume_path,
        error_message=r.error_message,
        dry_run=r.dry_run,
        screenshot_count=len(r.get_screenshot_list()),
    )


@router.get("", response_model=ApplicationListResponse)
async def list_applications(
    settings: SettingsDep,
    status: str | None = Query(None),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
):
    """List all application records with pagination."""
    async with get_session(settings.storage.database_url) as db:
        query = select(ApplicationRecord)

        if status:
            query = query.where(ApplicationRecord.status == status)

        count_query = select(func.count()).select_from(query.subquery())
        total = (await db.execute(count_query)).scalar() or 0

        offset = (page - 1) * page_size
        query = (
            query.order_by(ApplicationRecord.started_at.desc())
            .offset(offset)
            .limit(page_size)
        )
        result = await db.execute(query)
        records = result.scalars().all()

    return ApplicationListResponse(
        applications=[_to_response(r) for r in records],
        meta=PaginationMeta(
            total=total,
            page=page,
            page_size=page_size,
            pages=math.ceil(total / page_size) if total else 0,
        ),
    )


@router.get("/{application_id}", response_model=ApplicationRecordResponse)
async def get_application(application_id: str, settings: SettingsDep):
    """Get a single application record."""
    async with get_session(settings.storage.database_url) as db:
        result = await db.execute(
            select(ApplicationRecord).where(ApplicationRecord.id == application_id)
        )
        record = result.scalar_one_or_none()

    if not record:
        raise HTTPException(status_code=404, detail="Application not found")

    return _to_response(record)


@router.post("/trigger", response_model=ApplicationTriggerResponse)
async def trigger_applications(settings: SettingsDep) -> ApplicationTriggerResponse:
    """
    Trigger an application run.

    Wraps ApplicationAgent.process_queue() so the FastAPI server can be
    the single entry point (alongside the standalone `run_application.py`).
    Respects `application.dry_run` from config — set `dry_run: false` to
    actually submit.
    """
    from agents.application_agent import ApplicationAgent

    try:
        agent = ApplicationAgent()
        processed = await agent.process_queue()
    except Exception as e:
        logger.exception("Trigger-applications run failed: {}", e)
        raise HTTPException(status_code=500, detail=str(e)) from e

    return ApplicationTriggerResponse(
        success=True,
        dry_run=settings.application.dry_run,
        jobs_processed=processed,
    )
