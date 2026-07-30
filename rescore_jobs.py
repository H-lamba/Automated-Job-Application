"""
rescore_jobs.py — Score unscored jobs already in the database.

Picks up all jobs with no LLM score (relevance_score IS NULL or == 0)
and runs them through the scorer sequentially with prompt v2 + blocklist.

Usage:
    source .venv/bin/activate
    python rescore_jobs.py [--limit N]
"""

from __future__ import annotations

import asyncio
import sys
from datetime import datetime, timezone

from sqlalchemy import select, or_, func

from core.config import get_settings
from core.database import init_db, get_session
from core.exceptions import LLMParseError
from core.logger import logger, setup_logging
from llm.client import OllamaClient
from llm.prompts import SCORE_JOB_PROMPT, keyword_pre_score
from llm.response_parser import parse_job_score
from models.job import JobListing, JobStatus
from profile.loader import load_profile


async def score_job(job: JobListing, llm: OllamaClient, profile, settings) -> bool:
    """Score a single job in-place. Returns True if scored successfully."""
    try:
        # Keyword pre-filter with blocklist (fast, no LLM)
        tech_skills = profile.skills_summary().split(", ") if hasattr(profile, "skills_summary") else []
        pre = keyword_pre_score(
            job_title=job.title or "",
            job_description=(job.description or "")[:500],
            target_roles=profile.preferences.target_roles,
            technical_skills=tech_skills,
        )
        if pre == 0.0:
            # Blocklisted title — skip without LLM
            job.relevance_score = 5.0
            job.status = JobStatus.SKIPPED.value
            job.score_reasoning = "Blocked by non-engineering title filter"
            return True

        system, user_msg = SCORE_JOB_PROMPT.format(
            candidate_name=profile.personal.name,
            current_title=profile.most_recent_title(),
            years_experience=profile.years_of_experience(),
            target_roles=", ".join(profile.preferences.target_roles),
            technical_skills=profile.skills_summary(),
            remote_preference=profile.preferences.remote,
            preferred_locations=", ".join(profile.preferences.locations_ok),
            job_title=job.title,
            company=job.company,
            location=job.location or "Not specified",
            remote=str(job.remote),
            description=(job.description or "")[:1500],
        )

        raw = await llm.chat_json(
            messages=[{"role": "user", "content": user_msg}],
            system=system,
            temperature=settings.llm.scoring_temperature,
            think=False,
        )
        score = parse_job_score(raw)

        job.relevance_score = float(score.overall)
        job.score_reasoning = score.reasoning
        job.status = (
            JobStatus.QUEUED.value if score.apply else JobStatus.SKIPPED.value
        )
        return True

    except LLMParseError as e:
        logger.warning("Parse error for '{}': {}", job.title, e)
        return False
    except Exception as e:
        logger.warning("Score failed for '{}': {}", job.title, e)
        return False


async def main() -> None:
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 500

    settings = get_settings()
    setup_logging(settings)

    logger.info("=" * 60)
    logger.info("  Rescore Run — scoring unscored jobs from DB")
    logger.info("  Limit: {} jobs | Model: {}", limit, settings.llm.reasoning_model)
    logger.info("=" * 60)

    await init_db(settings.storage.database_url)

    llm = OllamaClient.from_settings(settings)
    ok = await llm.health_check()
    if not ok:
        logger.error("❌ Ollama not reachable. Run: ollama serve")
        sys.exit(1)

    profile = load_profile(settings.profile.path)
    logger.info("✓ Profile: {} | {} target roles", profile.personal.name, len(profile.preferences.target_roles))

    async with get_session(settings.storage.database_url) as db:
        # Count unscored
        total_unscored = (await db.execute(
            select(func.count()).select_from(JobListing)
            .where(or_(JobListing.relevance_score.is_(None), JobListing.relevance_score == 0))
        )).scalar()
        logger.info("Unscored jobs in DB: {} | Will process: {}", total_unscored, min(total_unscored, limit))

        # Fetch batch — prioritise jobs from companies likely to have engineering roles
        res = await db.execute(
            select(JobListing)
            .where(or_(JobListing.relevance_score.is_(None), JobListing.relevance_score == 0))
            .limit(limit)
        )
        jobs = res.scalars().all()

    if not jobs:
        logger.info("✅ All jobs already scored!")
        return

    scored = 0
    skipped = 0
    failed = 0
    relevant = 0
    save_interval = 50  # Commit every 50 jobs

    start = datetime.now(timezone.utc)

    async with get_session(settings.storage.database_url) as db:
        # Re-attach jobs to this session
        job_ids = [j.id for j in jobs]
        res = await db.execute(
            select(JobListing).where(JobListing.id.in_(job_ids))
        )
        db_jobs = res.scalars().all()

        for i, job in enumerate(db_jobs, 1):
            # Fast blocklist check before LLM
            tech_skills = profile.skills_summary().split(", ")
            pre = keyword_pre_score(
                job_title=job.title or "",
                job_description=(job.description or "")[:200],
                target_roles=profile.preferences.target_roles,
                technical_skills=tech_skills,
            )

            if pre == 0.0:
                job.relevance_score = 5.0
                job.status = JobStatus.SKIPPED.value
                job.score_reasoning = "Blocked: non-engineering role"
                skipped += 1
                if i % 100 == 0:
                    logger.debug("Blocklisted {}/{}", i, len(db_jobs))
            else:
                ok = await score_job(job, llm, profile, settings)
                if ok:
                    scored += 1
                    if (job.relevance_score or 0) >= settings.discovery.min_relevance_score:
                        relevant += 1
                        logger.info(
                            "[{}/{}] ✅ {} @ {} — score={:.0f} apply={}",
                            i, len(db_jobs),
                            job.title, job.company,
                            job.relevance_score, job.status == JobStatus.QUEUED.value,
                        )
                    else:
                        logger.debug(
                            "[{}/{}] {} @ {} — score={:.0f}",
                            i, len(db_jobs), job.title, job.company,
                            job.relevance_score or 0,
                        )
                else:
                    failed += 1

            # Periodic commit
            if i % save_interval == 0:
                await db.commit()
                elapsed = (datetime.now(timezone.utc) - start).total_seconds()
                rate = i / elapsed * 60
                logger.info(
                    "Progress {}/{} | scored={} skipped={} relevant={} | {:.0f} jobs/min",
                    i, len(db_jobs), scored, skipped, relevant, rate,
                )

        await db.commit()

    elapsed = (datetime.now(timezone.utc) - start).total_seconds()
    logger.info("=" * 60)
    logger.info("✅ Rescore complete!")
    logger.info("   LLM scored  : {}", scored)
    logger.info("   Blocklisted : {}", skipped)
    logger.info("   Failed      : {}", failed)
    logger.info("   Relevant    : {}", relevant)
    logger.info("   Duration    : {:.1f}s ({:.1f} min)", elapsed, elapsed / 60)
    logger.info("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
