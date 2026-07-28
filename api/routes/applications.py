"""
api/routes/applications.py — Application record endpoints.
"""

from __future__ import annotations

import math

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from api.schemas import (
    ApplicationListResponse,
    ApplicationRecordResponse,
    PaginationMeta,
    TriggerApplicationRequest,
)
from core.config import Settings, settings_dep
from core.database import get_session
from core.logger import logger
from models.application import ApplicationRecord

router = APIRouter(prefix="/applications", tags=["Applications"])


@router.get("", response_model=ApplicationListResponse)
async def list_applications(
    status: str | None = Query(None),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    settings: Settings = Depends(settings_dep),
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
async def get_application(
    application_id: str,
    settings: Settings = Depends(settings_dep),
):
    """Get a single application record."""
    async with get_session(settings.storage.database_url) as db:
        result = await db.execute(
            select(ApplicationRecord).where(ApplicationRecord.id == application_id)
        )
        record = result.scalar_one_or_none()

    if not record:
        raise HTTPException(status_code=404, detail="Application not found")

    return ApplicationRecordResponse(
        id=record.id,
        job_id=record.job_id,
        status=record.status,
        started_at=record.started_at,
        completed_at=record.completed_at,
        duration_seconds=record.duration_seconds,
        ats_detected=record.ats_detected,
        resume_path=record.resume_path,
        error_message=record.error_message,
        dry_run=record.dry_run,
        screenshot_count=len(record.get_screenshot_list()),
    )
