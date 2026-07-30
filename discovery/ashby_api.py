"""
discovery/ashby_api.py — Ashby public job board API source.

Ashby exposes a public GET API (no auth required):
  GET https://api.ashbyhq.com/posting-api/job-board/{slug}

Returns all published job postings for a company.
"""

from __future__ import annotations

from datetime import datetime, timezone

import httpx

from core.logger import logger
from discovery.base_source import JobSource, RawJob

_BASE_URL = "https://api.ashbyhq.com/posting-api/job-board"
_TIMEOUT = 20.0


class AshbyAPISource(JobSource):
    """
    Fetches jobs from Ashby using the public job board API.

    Note: Uses GET (not POST). The old POST endpoint returns 401.
    Response schema: {"jobs": [...]} (not "jobPostings").
    """

    def __init__(
        self,
        company_slugs: list[str],
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._slugs = company_slugs
        self._client = http_client or httpx.AsyncClient(
            timeout=_TIMEOUT,
            headers={
                "User-Agent": "CareerAgent/1.0 (job-discovery-bot)",
                "Accept": "application/json",
            },
            follow_redirects=True,
        )

    @property
    def name(self) -> str:
        return "ashby_api"

    async def fetch(self) -> list[RawJob]:
        all_jobs: list[RawJob] = []
        for slug in self._slugs:
            try:
                jobs = await self._fetch_company(slug)
                logger.info("Ashby [{}] — fetched {} jobs", slug, len(jobs))
                all_jobs.extend(jobs)
            except Exception as e:
                logger.warning("Ashby [{}] — fetch failed: {}", slug, e)
        return all_jobs

    async def _fetch_company(self, slug: str) -> list[RawJob]:
        url = f"{_BASE_URL}/{slug}"

        # Ashby public board uses GET (POST returns 401)
        response = await self._client.get(url)

        if response.status_code == 404:
            logger.warning("Ashby board not found for slug '{}'", slug)
            return []

        response.raise_for_status()
        data = response.json()

        # Response schema: {"jobs": [...]}  (NOT "jobPostings")
        job_postings = data.get("jobs", data.get("jobPostings", []))
        logger.debug("Ashby [{}] — raw postings: {}", slug, len(job_postings))

        return [self._parse_posting(p, slug) for p in job_postings]

    def _parse_posting(self, data: dict, slug: str) -> RawJob:
        # Location
        location_info = data.get("location") or data.get("locationName") or ""
        if not location_info and data.get("isRemote"):
            location_info = "Remote"

        secondary_location = data.get("secondaryLocation", "")
        if secondary_location and "remote" in secondary_location.lower():
            location_info = location_info or secondary_location

        is_remote = (
            bool(data.get("isRemote", False))
            or "remote" in location_info.lower()
        )

        # Timestamps
        posted_at = None
        published_date = data.get("publishedDate") or data.get("updatedAt")
        if published_date:
            try:
                posted_at = datetime.fromisoformat(
                    published_date.replace("Z", "+00:00")
                )
            except (ValueError, AttributeError):
                pass

        # Application URL
        job_id = data.get("id", "")
        apply_url = data.get("applyUrl") or f"https://jobs.ashbyhq.com/{slug}/{job_id}"
        hosted_url = data.get("jobUrl") or apply_url

        # Description (may be absent in listing, fetched separately if needed)
        description = data.get("descriptionHtml") or data.get("description", "")

        company_name = (
            data.get("organizationName")
            or data.get("company")
            or slug.replace("-", " ").title()
        )

        return RawJob(
            title=data.get("title", "Unknown Title"),
            company=company_name,
            application_url=apply_url,
            source="ashby_api",
            source_job_id=job_id,
            company_slug=slug,
            location=location_info,
            remote=is_remote,
            description=description,
            requirements="",
            posted_at=posted_at,
            job_post_url=hosted_url,
            department=data.get("department", ""),
            employment_type=data.get("employmentType", ""),
            raw=data,
        )

    async def health_check(self) -> bool:
        if not self._slugs:
            return False
        try:
            response = await self._client.get(
                f"{_BASE_URL}/{self._slugs[0]}", timeout=10
            )
            return response.status_code in (200, 404)
        except Exception:
            return False

    async def aclose(self) -> None:
        await self._client.aclose()
