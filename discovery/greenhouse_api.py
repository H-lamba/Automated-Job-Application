"""
discovery/greenhouse_api.py — Greenhouse public job board API source.

Greenhouse exposes a free, unauthenticated JSON API for all companies
that use their hosted boards:
  GET https://boards-api.greenhouse.io/v1/boards/{slug}/jobs?content=true

No API key required. Rate limits are generous for read operations.
"""

from __future__ import annotations

import contextlib
from datetime import datetime

import httpx

from core.logger import logger
from discovery.base_source import JobSource, RawJob

# Greenhouse public API base URL
_BASE_URL = "https://boards-api.greenhouse.io/v1/boards"

# Request timeout (generous — some boards are slow)
_TIMEOUT = 20.0


class GreenhouseAPISource(JobSource):
    """
    Fetches jobs from Greenhouse-hosted boards using the public JSON API.

    One instance covers all configured company slugs. Failures for a
    single company do not abort the others.
    """

    def __init__(
        self,
        company_slugs: list[str],
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._slugs = company_slugs
        # Reuse an existing client if provided (better for connection pooling)
        self._client = http_client or httpx.AsyncClient(
            timeout=_TIMEOUT,
            headers={"User-Agent": "CareerAgent/1.0 (job-discovery-bot)"},
            follow_redirects=True,
        )

    @property
    def name(self) -> str:
        return "greenhouse_api"

    async def fetch(self) -> list[RawJob]:
        """Fetch jobs for all configured company slugs."""
        all_jobs: list[RawJob] = []
        for slug in self._slugs:
            try:
                jobs = await self._fetch_company(slug)
                logger.info(
                    "Greenhouse [{}] — fetched {} jobs", slug, len(jobs)
                )
                all_jobs.extend(jobs)
            except Exception as e:
                logger.warning("Greenhouse [{}] — fetch failed: {}", slug, e)
        return all_jobs

    async def _fetch_company(self, slug: str) -> list[RawJob]:
        """Fetch and parse jobs for a single company slug."""
        url = f"{_BASE_URL}/{slug}/jobs"
        params = {"content": "true"}   # Include full job description

        response = await self._client.get(url, params=params)

        if response.status_code == 404:
            logger.warning("Greenhouse board not found for slug '{}'", slug)
            return []

        response.raise_for_status()
        data = response.json()

        jobs_data = data.get("jobs", [])
        logger.debug("Greenhouse [{}] — raw jobs count: {}", slug, len(jobs_data))

        return [self._parse_job(job, slug) for job in jobs_data]

    def _parse_job(self, data: dict, slug: str) -> RawJob:
        """Convert a single Greenhouse job dict into a RawJob."""
        # Location: Greenhouse may have multiple offices
        offices = data.get("offices", [])
        if offices:
            location = offices[0].get("name", "")
        else:
            location = data.get("location", {}).get("name", "")

        # Salary — Greenhouse doesn't always include this
        salary_min = salary_max = None
        salary_data = data.get("salary", {})
        if salary_data:
            salary_min = salary_data.get("min_value")
            salary_max = salary_data.get("max_value")

        # Posted date
        posted_at = None
        updated_at_str = data.get("updated_at") or data.get("created_at")
        if updated_at_str:
            with contextlib.suppress(ValueError, AttributeError):
                posted_at = datetime.fromisoformat(
                    updated_at_str.replace("Z", "+00:00")
                )

        # Build application URL
        # Greenhouse jobs link directly to the apply page
        absolute_url = data.get("absolute_url", "")

        # Get content (full description)
        content = data.get("content", "")

        departments = data.get("departments") or []
        department = departments[0].get("name", "") if departments else ""

        title_lower = (data.get("title") or "").lower()
        remote = "remote" in location.lower() or "remote" in title_lower

        return RawJob(
            title=data.get("title", "Unknown Title"),
            company=slug.replace("-", " ").title(),
            application_url=absolute_url,
            source="greenhouse_api",
            source_job_id=str(data.get("id", "")),
            company_slug=slug,
            location=location,
            remote=remote,
            description=content,
            requirements="",
            salary_min=salary_min,
            salary_max=salary_max,
            posted_at=posted_at,
            job_post_url=absolute_url,
            department=department,
            raw=data,
        )

    async def health_check(self) -> bool:
        """Check if Greenhouse API is reachable using the first configured slug."""
        if not self._slugs:
            return False
        try:
            slug = self._slugs[0]
            response = await self._client.get(
                f"{_BASE_URL}/{slug}/jobs", timeout=10
            )
            return response.status_code in (200, 404)  # 404 = valid response, wrong slug
        except Exception:
            return False

    async def aclose(self) -> None:
        """Close the HTTP client. Call this when done."""
        await self._client.aclose()
