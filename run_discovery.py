"""
run_discovery.py — One-shot discovery runner.

Runs the DiscoveryAgent directly (no FastAPI/uvicorn needed).
Usage:
    source .venv/bin/activate
    python run_discovery.py
"""

from __future__ import annotations

import asyncio
import sys

from sqlalchemy import select, func

from core.config import get_settings
from core.database import init_db, get_session
from core.logger import logger, setup_logging
from llm.client import OllamaClient
from models.job import JobListing, JobStatus
from profile.loader import load_profile
from agents.discovery_agent import DiscoveryAgent


async def main() -> None:
    settings = get_settings()
    setup_logging(settings)

    logger.info("=" * 60)
    logger.info("  Discovery Run — standalone mode")
    logger.info("  Reasoning model : {}", settings.llm.reasoning_model)
    logger.info("=" * 60)

    # Init DB
    await init_db(settings.storage.database_url)

    # Check Ollama
    llm_client = OllamaClient.from_settings(settings)
    ok = await llm_client.health_check()
    if not ok:
        logger.error("❌ Ollama is not reachable. Start it with: ollama serve")
        sys.exit(1)
    logger.info("✓ Ollama healthy — {}", settings.llm.reasoning_model)

    # Load profile
    profile = load_profile(settings.profile.path)
    logger.info("✓ Profile loaded — {}", profile.personal.name)

    # Run discovery using run_discovery() for proper stats
    async with get_session(settings.storage.database_url) as db:
        agent = DiscoveryAgent(settings=settings, db=db, llm_client=llm_client, profile=profile)
        summary = await agent.run_discovery()

    logger.info("=" * 60)
    if summary["success"]:
        logger.info("✅ Discovery complete!")
        logger.info("   Fetched   : {}", summary.get("raw_fetched", "?"))
        logger.info("   Scored    : {}", summary.get("scored", "?"))
        logger.info("   Saved     : {}", summary.get("new_saved", "?"))
        logger.info("   Skipped   : {}", summary.get("skipped", "?"))
        logger.info("   Duration  : {:.1f}s", summary.get("duration_seconds", 0))
    else:
        logger.error("❌ Discovery failed")
        for err in summary.get("errors", []):
            logger.error("   {}", err)
    logger.info("=" * 60)

    # Query DB for relevant jobs (score >= threshold)
    async with get_session(settings.storage.database_url) as db:
        min_score = settings.discovery.min_relevance_score
        result = await db.execute(
            select(JobListing)
            .where(JobListing.relevance_score >= min_score)
            .order_by(JobListing.relevance_score.desc())
            .limit(20)
        )
        relevant = result.scalars().all()

        total_result = await db.execute(select(func.count()).select_from(JobListing))
        total = total_result.scalar()

    logger.info("")
    logger.info("📊 Database Summary")
    logger.info("   Total jobs in DB : {}", total)
    logger.info("   Relevant (≥{})  : {}", int(min_score), len(relevant))

    if relevant:
        logger.info("")
        logger.info("🎯 Top Relevant Jobs:")
        for job in relevant:
            logger.info(
                "   [{:3.0f}] {} @ {} | {}",
                job.relevance_score or 0,
                job.title,
                job.company,
                job.location or "Remote",
            )
            logger.info("         {}", job.application_url or "")
    else:
        logger.info("")
        logger.info("   No jobs above threshold yet — scoring more companies will help.")
        logger.info("   (The current run only fetched Anthropic jobs; most were non-engineering roles.)")


if __name__ == "__main__":
    asyncio.run(main())
