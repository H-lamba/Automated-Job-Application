"""
discovery/normalizer.py — Convert RawJob objects into JobListing ORM instances.

Responsibilities:
1. Map RawJob fields → JobListing columns
2. Generate url_hash for deduplication
3. Detect ATS type from the application URL
4. Strip HTML from descriptions
5. Keyword pre-score before LLM scoring
"""

from __future__ import annotations

import html
import re
from urllib.parse import urlparse

import xxhash

from core.logger import logger
from discovery.base_source import RawJob
from models.job import ATSType, JobListing, JobSource, JobStatus

# ──────────────────────────────────────────────────────────────────────────────
# ATS Detection
# ──────────────────────────────────────────────────────────────────────────────

_ATS_PATTERNS: list[tuple[str, ATSType]] = [
    (r"greenhouse\.io", ATSType.GREENHOUSE),
    (r"lever\.co", ATSType.LEVER),
    (r"ashbyhq\.com", ATSType.ASHBY),
    (r"myworkdayjobs\.com|workday\.com", ATSType.WORKDAY),
    (r"jobs\.sap\.com|successfactors\.com", ATSType.SAP),
    (r"icims\.com", ATSType.ICIMS),
    (r"taleo\.net", ATSType.TALEO),
    (r"smartrecruiters\.com", ATSType.SMARTRECRUITERS),
]


def detect_ats(url: str) -> ATSType:
    """Detect the ATS platform from an application URL."""
    for pattern, ats_type in _ATS_PATTERNS:
        if re.search(pattern, url, re.IGNORECASE):
            return ats_type
    return ATSType.UNKNOWN


# ──────────────────────────────────────────────────────────────────────────────
# Text cleaning
# ──────────────────────────────────────────────────────────────────────────────

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_WHITESPACE_RE = re.compile(r"\s{3,}")


def strip_html(text: str) -> str:
    """Remove HTML tags and normalise whitespace."""
    if not text:
        return ""
    # Decode HTML entities first
    text = html.unescape(text)
    # Remove tags
    text = _HTML_TAG_RE.sub(" ", text)
    # Normalise whitespace
    text = _WHITESPACE_RE.sub("\n\n", text)
    return text.strip()


# ──────────────────────────────────────────────────────────────────────────────
# Deduplication hash
# ──────────────────────────────────────────────────────────────────────────────


def compute_url_hash(url: str) -> str:
    """
    Generate a short, stable hash for a job application URL.

    Uses xxhash for speed (not a security hash — just for dedup).
    Returns an 8-character hex string.
    """
    # Normalise the URL before hashing (strip trailing slashes, lowercase scheme)
    parsed = urlparse(url.strip().lower())
    normalised = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
    if parsed.query:
        normalised += f"?{parsed.query}"
    return xxhash.xxh64(normalised).hexdigest()[:16]


# ──────────────────────────────────────────────────────────────────────────────
# Remote detection
# ──────────────────────────────────────────────────────────────────────────────

# Strong, low-ambiguity remote phrases only — dropped "distributed" and
# "anywhere" (bare) because they false-positive on things like "distributed
# systems experience" or "work with people anywhere in the org".
_REMOTE_KEYWORDS = {
    "remote", "work from home", "wfh",
    "fully remote", "100% remote", "remote-first", "remote first",
    "work from anywhere",
}


def detect_remote(raw: RawJob) -> bool:
    """Infer whether a job is remote from its fields."""
    if raw.remote:
        return True

    # Location and title are high-signal; description is noisy, so only
    # check a short prefix and require a strong phrase, not bare words.
    location_and_title = (raw.location + " " + raw.title).lower()
    if any(kw in location_and_title for kw in _REMOTE_KEYWORDS):
        return True

    description_snippet = raw.description[:300].lower()
    strong_phrases = {
        "fully remote",
        "100% remote",
        "remote-first",
        "remote first",
        "work from anywhere",
    }
    return any(kw in description_snippet for kw in strong_phrases)


# ──────────────────────────────────────────────────────────────────────────────
# Location filtering
# ──────────────────────────────────────────────────────────────────────────────

_INDIA_KEYWORDS = {
    "india", "bengaluru", "bangalore", "hyderabad", "pune", "chennai",
    "mumbai", "delhi", "gurugram", "gurgaon", "noida", "kolkata",
    "ahmedabad", "ncr",
}


def matches_target_locations(
    job: JobListing,
    locations_ok: list[str],
    require_india_or_remote: bool = True,
) -> bool:
    """
    Return True if the job's location is acceptable.
    Strictly enforces that jobs must be based in India or truly global remote.
    Blocks any job mentioning foreign regions in the title or location, even
    if flagged as remote by the ATS.
    """
    location_and_title = f"{job.location or ''} {job.title}".lower()

    # 1. Strict blocklist for non-India regions in BOTH title and location
    geo_restrictions = [
        "us", "united states", "usa", "uk", "united kingdom", "eu", "europe", 
        "canada", "north america", "emea", "latam", "americas", "apac", 
        "australia", "singapore", "germany", "france", "ireland", "dublin",
        "london", "new york", "san francisco", "seattle", "remote - us", "remote us"
    ]
    
    for restriction in geo_restrictions:
        if re.search(r'\b' + re.escape(restriction) + r'\b', location_and_title):
            # Block unless this specific restriction was explicitly allowed by the user
            allowed = any(
                restriction == loc.lower() or restriction in loc.lower()
                for loc in locations_ok
            )
            if not allowed:
                return False

    # 2. Check for India keywords (pass automatically)
    has_india = any(kw in location_and_title for kw in _INDIA_KEYWORDS)
    for loc in locations_ok:
        if loc.lower() != "remote" and loc.lower() in location_and_title:
            has_india = True

    if has_india:
        return True

    # 3. If it's remote, allow it only because it survived the strict blocklist above
    return bool(job.remote)


# ──────────────────────────────────────────────────────────────────────────────
# Main normalizer
# ──────────────────────────────────────────────────────────────────────────────


class JobNormalizer:
    """
    Converts RawJob objects into JobListing ORM instances.

    Stateless — safe to call concurrently.
    """

    def normalize(self, raw: RawJob) -> JobListing | None:
        """
        Normalize a single RawJob.

        Returns None if the job is invalid (missing required fields).
        """
        if not raw.title or not raw.company or not raw.application_url:
            logger.warning(
                "Skipping malformed job — missing required fields: title={} company={} url={}",
                bool(raw.title),
                bool(raw.company),
                bool(raw.application_url),
            )
            return None

        url_hash = compute_url_hash(raw.application_url)
        ats_type = detect_ats(raw.application_url)
        is_remote = detect_remote(raw)

        description = strip_html(raw.description)
        requirements = strip_html(raw.requirements)

        job = JobListing(
            url_hash=url_hash,
            source=raw.source or JobSource.GREENHOUSE_API.value,
            source_job_id=raw.source_job_id,
            company_slug=raw.company_slug,
            title=raw.title.strip(),
            company=raw.company.strip(),
            location=raw.location.strip() if raw.location else None,
            remote=is_remote,
            description=description or None,
            requirements=requirements or None,
            salary_min=raw.salary_min,
            salary_max=raw.salary_max,
            salary_currency=raw.salary_currency or None,
            application_url=raw.application_url.strip(),
            job_post_url=raw.job_post_url.strip() if raw.job_post_url else None,
            ats_type=ats_type.value,
            relevance_score=0.0,
            status=JobStatus.DISCOVERED.value,
            posted_at=raw.posted_at,
        )

        logger.debug(
            "Normalized job — title='{}' company='{}' ats={} remote={}",
            job.title,
            job.company,
            ats_type.value,
            is_remote,
        )
        return job

    def normalize_batch(self, raw_jobs: list[RawJob]) -> list[JobListing]:
        """Normalize a list of RawJobs, silently dropping invalid ones."""
        results = []
        for raw in raw_jobs:
            try:
                job = self.normalize(raw)
                if job:
                    results.append(job)
            except Exception as e:
                logger.warning("Failed to normalize job '{}': {}", raw.title, e)
        return results
