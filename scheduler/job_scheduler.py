"""
scheduler/job_scheduler.py — APScheduler configuration.

Schedules:
1. discovery_job  — runs every N hours to fetch new jobs
2. cleanup_job    — weekly cleanup of old screenshots

Jobs are registered with the AsyncIOScheduler so they integrate
cleanly with FastAPI's async event loop.
"""

from __future__ import annotations

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from core.config import Settings
from core.logger import logger

_scheduler: AsyncIOScheduler | None = None


async def _run_discovery(settings: Settings) -> None:
    """Scheduled discovery task."""
    from agents.discovery_agent import DiscoveryAgent
    from core.database import get_db
    from llm.client import OllamaClient
    from profile.loader import load_profile

    logger.info("Scheduler: starting discovery run")
    try:
        profile = load_profile(settings.profile.path)
        llm_client = OllamaClient.from_settings(settings)

        async with get_db(settings.storage.database_url) as db:
            agent = DiscoveryAgent(
                settings=settings,
                db=db,
                llm_client=llm_client,
                profile=profile,
            )
            result = await agent.run_discovery()
            logger.info("Scheduler: discovery complete — {}", result)
    except Exception as e:
        logger.error("Scheduler: discovery run failed: {}", e)


async def _run_cleanup(settings: Settings) -> None:
    """Clean up old screenshots."""
    import shutil
    from datetime import datetime, timedelta, timezone
    from pathlib import Path

    logger.info("Scheduler: running screenshot cleanup")
    screenshots_dir = Path(settings.storage.screenshots_dir)
    retention_days = settings.scheduler.screenshot_retention_days
    cutoff = datetime.now(timezone.utc) - timedelta(days=retention_days)

    removed = 0
    if screenshots_dir.exists():
        for item in screenshots_dir.iterdir():
            if item.is_dir():
                mtime = datetime.fromtimestamp(item.stat().st_mtime, tz=timezone.utc)
                if mtime < cutoff:
                    shutil.rmtree(item)
                    removed += 1

    logger.info("Cleanup: removed {} screenshot directories older than {} days", removed, retention_days)


def create_scheduler(settings: Settings) -> AsyncIOScheduler:
    """Create and configure the APScheduler instance."""
    global _scheduler

    _scheduler = AsyncIOScheduler(timezone="UTC")

    # Discovery job — run every N hours
    _scheduler.add_job(
        _run_discovery,
        trigger=IntervalTrigger(hours=settings.scheduler.discovery_interval_hours),
        args=[settings],
        id="discovery_job",
        name="Job Discovery",
        replace_existing=True,
        misfire_grace_time=300,  # 5 min grace period
    )

    # Cleanup job — weekly on Sunday midnight
    _scheduler.add_job(
        _run_cleanup,
        trigger=CronTrigger(day_of_week="sun", hour=0, minute=0),
        args=[settings],
        id="cleanup_job",
        name="Screenshot Cleanup",
        replace_existing=True,
    )

    logger.info(
        "Scheduler configured — discovery every {}h",
        settings.scheduler.discovery_interval_hours,
    )
    return _scheduler


def get_scheduler() -> AsyncIOScheduler | None:
    return _scheduler
