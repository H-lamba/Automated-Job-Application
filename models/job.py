"""
models/job.py — JobListing ORM model and Pydantic schemas.

The SQLAlchemy model defines the database table.
The Pydantic schemas are used for API serialization and internal data transfer.
"""

from __future__ import annotations

import enum
import uuid
from datetime import datetime

from sqlalchemy import DateTime, Float, Index, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from core.database import Base


# ──────────────────────────────────────────────────────────────────────────────
# Enumerations
# ──────────────────────────────────────────────────────────────────────────────


class JobStatus(str, enum.Enum):
    """Lifecycle state of a discovered job."""

    DISCOVERED = "discovered"   # Just found, not yet scored
    SCORED = "scored"           # Scored, waiting for queue decision
    QUEUED = "queued"           # Approved for application
    APPLYING = "applying"       # Application in progress
    APPLIED = "applied"         # Successfully submitted
    FAILED = "failed"           # Application attempt failed
    SKIPPED = "skipped"         # Below relevance threshold or already applied
    EXPIRED = "expired"         # Job listing no longer active


class ATSType(str, enum.Enum):
    """Known ATS platforms."""

    GREENHOUSE = "greenhouse"
    LEVER = "lever"
    ASHBY = "ashby"
    WORKDAY = "workday"
    SAP = "sap"
    ICIMS = "icims"
    TALEO = "taleo"
    SMARTRECRUITERS = "smartrecruiters"
    GENERIC = "generic"
    UNKNOWN = "unknown"


class JobSource(str, enum.Enum):
    """Where the job was discovered."""

    GREENHOUSE_API = "greenhouse_api"
    LEVER_API = "lever_api"
    ASHBY_API = "ashby_api"
    MANUAL_URL = "manual_url"


# ──────────────────────────────────────────────────────────────────────────────
# SQLAlchemy ORM Model
# ──────────────────────────────────────────────────────────────────────────────


class JobListing(Base):
    """
    Persisted job listing record.

    url_hash is the primary deduplication key — it's a hash of the
    application URL so we never attempt to apply to the same job twice.
    """

    __tablename__ = "jobs"

    # Primary key
    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )

    # Deduplication key (hash of application_url)
    url_hash: Mapped[str] = mapped_column(String(16), unique=True, nullable=False)

    # Source metadata
    source: Mapped[str] = mapped_column(String(32), nullable=False)
    source_job_id: Mapped[str | None] = mapped_column(String(256), nullable=True)
    company_slug: Mapped[str | None] = mapped_column(String(128), nullable=True)

    # Job details
    title: Mapped[str] = mapped_column(String(512), nullable=False)
    company: Mapped[str] = mapped_column(String(256), nullable=False)
    location: Mapped[str | None] = mapped_column(String(256), nullable=True)
    remote: Mapped[bool] = mapped_column(default=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    requirements: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Compensation
    salary_min: Mapped[int | None] = mapped_column(nullable=True)
    salary_max: Mapped[int | None] = mapped_column(nullable=True)
    salary_currency: Mapped[str | None] = mapped_column(String(8), nullable=True)

    # URLs
    application_url: Mapped[str] = mapped_column(String(1024), nullable=False)
    job_post_url: Mapped[str | None] = mapped_column(String(1024), nullable=True)

    # ATS classification
    ats_type: Mapped[str] = mapped_column(String(32), default=ATSType.UNKNOWN.value)

    # Scoring
    relevance_score: Mapped[float] = mapped_column(Float, default=0.0)
    score_reasoning: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Lifecycle
    status: Mapped[str] = mapped_column(String(16), default=JobStatus.DISCOVERED.value)
    posted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    discovered_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    __table_args__ = (
        Index("ix_jobs_status", "status"),
        Index("ix_jobs_company", "company"),
        Index("ix_jobs_relevance_score", "relevance_score"),
        Index("ix_jobs_source", "source"),
    )

    def __repr__(self) -> str:
        return f"<JobListing id={self.id} title='{self.title}' company='{self.company}' score={self.relevance_score:.1f}>"
