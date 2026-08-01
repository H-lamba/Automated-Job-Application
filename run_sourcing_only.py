"""
run_sourcing_only.py — Run ONLY the job-sourcing step.

Fetches raw jobs from Greenhouse / Lever / Ashby using the configured
company slugs in config.yaml, and prints/saves them — no DB writes,
no normalization, no LLM scoring.

Usage:
    source .venv/bin/activate
    python run_sourcing_only.py                  # all configured sources
    python run_sourcing_only.py --source greenhouse
    python run_sourcing_only.py --out raw_jobs.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import asdict

import httpx

from core.config import get_settings
from core.logger import logger, setup_logging
from discovery.ashby_api import AshbyAPISource
from discovery.greenhouse_api import GreenhouseAPISource
from discovery.lever_api import LeverAPISource


async def main(source_filter: str | None, out_path: str | None) -> None:
    settings = get_settings()
    setup_logging(settings)
    cfg = settings.discovery

    http_client = httpx.AsyncClient(
        timeout=25.0,
        headers={"User-Agent": "CareerAgent/1.0"},
        follow_redirects=True,
    )

    sources = []
    try:
        if cfg.greenhouse_companies and source_filter in (None, "greenhouse"):
            sources.append(GreenhouseAPISource(cfg.greenhouse_companies, http_client))
        if cfg.lever_companies and source_filter in (None, "lever"):
            sources.append(LeverAPISource(cfg.lever_companies, http_client))
        if cfg.ashby_companies and source_filter in (None, "ashby"):
            sources.append(AshbyAPISource(cfg.ashby_companies, http_client))

        if not sources:
            logger.error("No matching sources configured for filter='{}'", source_filter)
            return

        fetch_tasks = [s.fetch() for s in sources]
        results = await asyncio.gather(*fetch_tasks, return_exceptions=True)

        all_jobs = []
        for source, result in zip(sources, results):
            if isinstance(result, Exception):
                logger.error("Source [{}] raised: {}", source.name, result)
                continue
            logger.info("Source [{}] — {} jobs", source.name, len(result))
            all_jobs.extend(result)

        logger.info("=" * 50)
        logger.info("Total raw jobs fetched: {}", len(all_jobs))
        logger.info("=" * 50)

        # Show a quick sample in the console
        for job in all_jobs[:10]:
            logger.info("  • {} @ {} ({})", job.title, job.company, job.source)

        if out_path:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(
                    [
                        {**asdict(j), "posted_at": j.posted_at.isoformat() if j.posted_at else None}
                        for j in all_jobs
                    ],
                    f,
                    indent=2,
                    default=str,
                )
            logger.info("Saved {} raw jobs to {}", len(all_jobs), out_path)

    finally:
        await http_client.aclose()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run only the job-sourcing step.")
    parser.add_argument(
        "--source",
        choices=["greenhouse", "lever", "ashby"],
        default=None,
        help="Limit to a single source (default: all configured sources)",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Optional path to save raw jobs as JSON (e.g. raw_jobs.json)",
    )
    args = parser.parse_args()
    asyncio.run(main(args.source, args.out))