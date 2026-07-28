"""
discovery/lever_api.py — Lever public job postings API source.

Lever exposes a free, unauthenticated API:
  GET https://api.lever.co/v0/postings/{slug}?mode=json

Returns all published job postings for a company.
"""

from __future__ import annotations

from datetime import datetime, timezone

import httpx

from core.logger import logger
from discovery.base_source import JobSource, RawJob

_BASE_URL = "https://api.lever.co/v0/postings"
_TIMEOUT = 20.0


class LeverAPISource(JobSource):
    """
    Fetches jobs from Lever using the public postings API.
    """

    def __init__(
        self,
        company_slugs: list[str],
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._slugs = company_slugs
        self._client = http_client or httpx.AsyncClient(
            timeout=_TIMEOUT,
            headers={"User-Agent": "CareerAgent/1.0 (job-discovery-bot)"},
            follow_redirects=True,
        )

    @property
    def name(self) -> str:
        return "lever_api"

    async def fetch(self) -> list[RawJob]:
        all_jobs: list[RawJob] = []
        for slug in self._slugs:
            try:
                jobs = await self._fetch_company(slug)
                logger.info("Lever [{}] — fetched {} jobs", slug, len(jobs))
                all_jobs.extend(jobs)
            except Exception as e:
                logger.warning("Lever [{}] — fetch failed: {}", slug, e)
        return all_jobs

    async def _fetch_company(self, slug: str) -> list[RawJob]:
        url = f"{_BASE_URL}/{slug}"
        params = {"mode": "json", "limit": "250"}

        response = await self._client.get(url, params=params)

        if response.status_code == 404:
            logger.warning("Lever board not found for slug '{}'", slug)
            return []

        response.raise_for_status()
        postings = response.json()

        if not isinstance(postings, list):
            logger.warning("Lever [{}] — unexpected response format", slug)
            return []

        return [self._parse_posting(p, slug) for p in postings]

    def _parse_posting(self, data: dict, slug: str) -> RawJob:
        # Lever's location is nested
        categories = data.get("categories", {})
        location = categories.get("location", "") or categories.get("allLocations", [""])[0] if isinstance(categories.get("allLocations"), list) else ""

        # Remote detection from commitment and location
        commitment = categories.get("commitment", "").lower()
        is_remote = "remote" in location.lower() or "remote" in commitment

        # Posted timestamp (Lever returns milliseconds)
        posted_at = None
        created_at_ms = data.get("createdAt")
        if created_at_ms:
            try:
                posted_at = datetime.fromtimestamp(
                    created_at_ms / 1000, tz=timezone.utc
                )
            except (ValueError, OSError):
                pass

        # Build description from Lever's list and description fields
        description_parts = []
        description_body = data.get("descriptionBody") or data.get("description", "")
        if description_body:
            description_parts.append(description_body)
        for section in data.get("lists", []):
            section_text = section.get("text", "")
            items = section.get("content", "")
            if section_text:
                description_parts.append(f"\n{section_text}:\n{items}")
        description = "\n".join(description_parts)

        apply_url = data.get("applyUrl") or data.get("hostedUrl", "")
        hosted_url = data.get("hostedUrl", "")

        company_name = data.get("company") or slug.replace("-", " ").title()

        return RawJob(
            title=data.get("text", "Unknown Title"),
            company=company_name,
            application_url=apply_url or hosted_url,
            source="lever_api",
            source_job_id=data.get("id", ""),
            company_slug=slug,
            location=location,
            remote=is_remote,
            description=description,
            requirements="",
            posted_at=posted_at,
            job_post_url=hosted_url,
            department=categories.get("department", ""),
            employment_type=commitment,
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
