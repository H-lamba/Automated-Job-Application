"""
discovery/base_source.py — Abstract interface for all job sources.

Every source (Greenhouse, Lever, Ashby, manual URLs) implements this
interface. The DiscoveryAgent doesn't know or care which source it's
talking to — it calls fetch() and gets back a list of RawJob objects.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class RawJob:
    """
    Minimally-processed job data from a source.

    Fields are kept loose here — normalisation happens in normalizer.py.
    Every field except title, company, and application_url is optional.
    """

    # Required
    title: str
    company: str
    application_url: str

    # Source metadata
    source: str = ""                    # e.g. "greenhouse_api"
    source_job_id: str = ""             # ID from the source system
    company_slug: str = ""              # e.g. "anthropic"

    # Job details (may be None if source doesn't provide)
    location: str = ""
    remote: bool = False
    description: str = ""
    requirements: str = ""
    salary_min: int | None = None
    salary_max: int | None = None
    salary_currency: str = ""
    posted_at: datetime | None = None
    job_post_url: str = ""              # Separate from application_url
    department: str = ""
    employment_type: str = ""           # full-time, part-time, contract

    # Raw payload (for debugging / future enrichment)
    raw: dict[str, Any] = field(default_factory=dict)


class JobSource(ABC):
    """
    Abstract base for all job data sources.

    Implementations must be safe to call concurrently — they should
    not share mutable state at the instance level.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this source (e.g. 'greenhouse_api')."""
        ...

    @abstractmethod
    async def fetch(self) -> list[RawJob]:
        """
        Fetch all available jobs from this source.

        Returns:
            List of RawJob objects (may be empty if source is down).

        Never raises — log and return empty list on failure.
        """
        ...

    @abstractmethod
    async def health_check(self) -> bool:
        """
        Verify the source is reachable and returning data.

        Returns True if healthy, False otherwise. Never raises.
        """
        ...

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.name}>"
