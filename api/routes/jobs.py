"""
api/routes/jobs.py — Job listing endpoints.
"""

from __future__ import annotations

import math
from profile.loader import load_profile
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func, select

from agents.discovery_agent import DiscoveryAgent
from api.schemas import (
    JobDiscoverRequest,
    JobDiscoverResponse,
    JobListingResponse,
    JobListResponse,
    JobStatusUpdateRequest,
    PaginationMeta,
)
from core.config import Settings, settings_dep
from core.database import get_session
from core.logger import logger
from llm.client import OllamaClient
from models.job import JobListing, JobStatus

router = APIRouter(prefix="/jobs", tags=["Jobs"])

SettingsDep = Annotated[Settings, Depends(settings_dep)]


@router.get("", response_model=JobListResponse)
async def list_jobs(
    settings: SettingsDep,
    status: str | None = Query(None, description="Filter by status"),
    min_score: float = Query(0.0, ge=0, le=100),
    company: str | None = Query(None),
    remote_only: bool = Query(False),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
):
    """List job listings with optional filters and pagination."""
    async with get_session(settings.storage.database_url) as db:
        query = select(JobListing)

        if status:
            query = query.where(JobListing.status == status)
        if min_score > 0:
            query = query.where(JobListing.relevance_score >= min_score)
        if company:
            query = query.where(JobListing.company.ilike(f"%{company}%"))
        if remote_only:
            query = query.where(JobListing.remote.is_(True))

        # Total count
        count_query = select(func.count()).select_from(query.subquery())
        total_result = await db.execute(count_query)
        total = total_result.scalar() or 0

        # Paginated results
        offset = (page - 1) * page_size
        query = (
            query.order_by(JobListing.relevance_score.desc())
            .offset(offset)
            .limit(page_size)
        )
        result = await db.execute(query)
        jobs = result.scalars().all()

    return JobListResponse(
        jobs=[JobListingResponse.model_validate(j) for j in jobs],
        meta=PaginationMeta(
            total=total,
            page=page,
            page_size=page_size,
            pages=math.ceil(total / page_size) if total else 0,
        ),
    )


@router.get("/{job_id}", response_model=JobListingResponse)
async def get_job(job_id: str, settings: SettingsDep):
    """Get a single job listing by ID."""
    async with get_session(settings.storage.database_url) as db:
        result = await db.execute(select(JobListing).where(JobListing.id == job_id))
        job = result.scalar_one_or_none()

    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    return JobListingResponse.model_validate(job)


@router.patch("/{job_id}/status", response_model=JobListingResponse)
async def update_job_status(
    job_id: str,
    body: JobStatusUpdateRequest,
    settings: SettingsDep,
):
    """Manually update a job's status."""
    valid_statuses = {s.value for s in JobStatus}
    if body.status not in valid_statuses:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid status. Must be one of: {valid_statuses}",
        )

    async with get_session(settings.storage.database_url) as db:
        result = await db.execute(select(JobListing).where(JobListing.id == job_id))
        job = result.scalar_one_or_none()
        if not job:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

        job.status = body.status
        await db.commit()
        await db.refresh(job)

    logger.info("Job {} status updated to '{}'", job_id, body.status)
    return JobListingResponse.model_validate(job)


@router.post("/discover", response_model=JobDiscoverResponse)
async def trigger_discovery(
    body: JobDiscoverRequest,
    settings: SettingsDep,
):
    """
    Trigger a full job discovery run.

    This is an async operation that runs the DiscoveryAgent pipeline:
    fetch → normalize → score → save.
    """
    try:
        profile = load_profile(settings.profile.path)
        llm_client = OllamaClient.from_settings(settings)

        async with get_session(settings.storage.database_url) as db:
            agent = DiscoveryAgent(
                settings=settings,
                db=db,
                llm_client=llm_client,
                profile=profile,
            )
            result = await agent.run_discovery()

        return JobDiscoverResponse(**result)

    except Exception as e:
        logger.exception("Discovery run failed: {}", e)
        raise HTTPException(status_code=500, detail=str(e)) from e
