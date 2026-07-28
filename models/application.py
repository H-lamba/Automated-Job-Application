"""
models/application.py — ApplicationRecord ORM model.

Tracks every application attempt — successful or not.
One JobListing can have multiple ApplicationRecords (retry attempts).
"""

from __future__ import annotations

import enum
import uuid
from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from core.database import Base


class ApplicationStatus(str, enum.Enum):
    """Lifecycle state of a single application attempt."""

    STARTED = "started"
    FORM_FILLING = "form_filling"
    UPLOADING = "uploading"
    REVIEWING = "reviewing"
    SUBMITTED = "submitted"
    FAILED = "failed"
    SKIPPED = "skipped"   # dry_run=True


class ApplicationRecord(Base):
    __tablename__ = "applications"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )

    # Foreign key to JobListing
    job_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("jobs.id", ondelete="CASCADE"), nullable=False
    )

    # Timing
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    duration_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Outcome
    status: Mapped[str] = mapped_column(
        String(16), default=ApplicationStatus.STARTED.value, nullable=False
    )
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    error_step: Mapped[str | None] = mapped_column(String(64), nullable=True)

    # Documents used
    resume_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    cover_letter_path: Mapped[str | None] = mapped_column(String(512), nullable=True)

    # ATS information
    ats_detected: Mapped[str | None] = mapped_column(String(32), nullable=True)

    # Screenshots — stored as pipe-delimited paths (avoids JSON column issues with SQLite)
    screenshot_paths: Mapped[str | None] = mapped_column(Text, nullable=True)

    # LLM-generated answers — stored as JSON string
    answers_json: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Dry run flag (was this a real submission?)
    dry_run: Mapped[bool] = mapped_column(default=True)

    # ORM relationship (lazy — don't auto-load job for every query)
    job = relationship("JobListing", lazy="select")

    def get_screenshot_list(self) -> list[str]:
        """Return screenshot paths as a Python list."""
        if not self.screenshot_paths:
            return []
        return [p for p in self.screenshot_paths.split("|") if p]

    def set_screenshot_list(self, paths: list[str]) -> None:
        self.screenshot_paths = "|".join(paths)

    def __repr__(self) -> str:
        return (
            f"<ApplicationRecord id={self.id} job_id={self.job_id} "
            f"status={self.status}>"
        )
