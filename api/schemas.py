"""
api/schemas.py — Pydantic schemas for all API request/response bodies.

Separate from ORM models — we control the API contract independently
from the database schema. Never expose internal ORM models directly.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

# ──────────────────────────────────────────────────────────────────────────────
# Shared
# ──────────────────────────────────────────────────────────────────────────────


class PaginationMeta(BaseModel):
    total: int
    page: int
    page_size: int
    pages: int


class APIResponse(BaseModel):
    """Standard API envelope."""
    success: bool
    message: str = ""
    data: Any = None


# ──────────────────────────────────────────────────────────────────────────────
# Jobs
# ──────────────────────────────────────────────────────────────────────────────


class JobListingResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    title: str
    company: str
    location: str | None = None
    remote: bool
    application_url: str
    ats_type: str
    relevance_score: float
    score_reasoning: str | None = None
    status: str
    source: str
    posted_at: datetime | None = None
    discovered_at: datetime


class JobListResponse(BaseModel):
    jobs: list[JobListingResponse]
    meta: PaginationMeta


class JobDiscoverRequest(BaseModel):
    """Trigger a discovery run with optional overrides."""
    sources: list[str] = Field(
        default=[],
        description="Specific sources to use. Empty = use all configured sources.",
        examples=[["greenhouse_api", "lever_api"]],
    )
    max_jobs: int = Field(default=50, ge=1, le=500)


class JobDiscoverResponse(BaseModel):
    success: bool
    summary: str
    raw_fetched: int
    new_saved: int
    skipped: int
    scored: int
    duration_seconds: float
    errors: list[str] = []


class JobStatusUpdateRequest(BaseModel):
    status: str


# ──────────────────────────────────────────────────────────────────────────────
# Applications
# ──────────────────────────────────────────────────────────────────────────────


class ApplicationRecordResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    job_id: str
    status: str
    started_at: datetime
    completed_at: datetime | None = None
    duration_seconds: float | None = None
    ats_detected: str | None = None
    resume_path: str | None = None
    error_message: str | None = None
    dry_run: bool
    screenshot_count: int = 0


class ApplicationListResponse(BaseModel):
    applications: list[ApplicationRecordResponse]
    meta: PaginationMeta


class TriggerApplicationRequest(BaseModel):
    job_id: str
    dry_run: bool = True


class ApplicationTriggerResponse(BaseModel):
    """Result of POST /applications/trigger."""

    success: bool
    dry_run: bool
    jobs_processed: int = 0


# ──────────────────────────────────────────────────────────────────────────────
# Documents
# ──────────────────────────────────────────────────────────────────────────────


class DocumentInfo(BaseModel):
    filename: str
    size_bytes: int
    is_default: bool


class DocumentsResponse(BaseModel):
    resume: list[DocumentInfo] = []
    cover_letter: list[DocumentInfo] = []
    certificate: list[DocumentInfo] = []
    other: list[DocumentInfo] = []


# ──────────────────────────────────────────────────────────────────────────────
# Health & Stats
# ──────────────────────────────────────────────────────────────────────────────


class HealthResponse(BaseModel):
    status: str                    # "healthy" | "degraded" | "unhealthy"
    ollama_connected: bool
    database_connected: bool
    reasoning_model: str
    vision_model: str
    profile_loaded: bool


class StatsResponse(BaseModel):
    total_discovered: int
    applied: int
    queued: int
    failed: int
    skipped: int
    by_status: dict[str, int]
