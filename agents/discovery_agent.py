"""
agents/discovery_agent.py — Job Discovery Agent.

Orchestrates all job sources, normalizes results, deduplicates against
the database, scores jobs with the LLM, and persists new ones.

Flow:
1. Fetch from all sources in parallel (asyncio.gather)
2. Normalize each raw job → JobListing
3. Deduplicate against existing url_hashes in the database
4. Keyword pre-score → discard obviously irrelevant jobs
5. LLM-score remaining jobs in batches
6. Persist to database
7. Return summary
"""

from __future__ import annotations

import asyncio

import httpx
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from agents.base_agent import (
    ActionResult,
    BaseAgent,
    Observation,
    Plan,
)
from core.config import Settings
from core.exceptions import LLMParseError
from core.logger import logger
from discovery.ashby_api import AshbyAPISource
from discovery.base_source import JobSource, RawJob
from discovery.greenhouse_api import GreenhouseAPISource
from discovery.lever_api import LeverAPISource
from discovery.normalizer import JobNormalizer, matches_target_locations
from llm.client import OllamaClient
from llm.prompts import SCORE_JOB_PROMPT, keyword_pre_score
from llm.response_parser import JobScoreResponse, parse_job_score
from models.job import JobListing, JobStatus
from models.profile import UserProfile


class DiscoveryAgent(BaseAgent):
    """
    Discovers, scores, and persists new job listings.

    Designed to run as a scheduled background task or triggered via API.
    """

    max_iterations = 10  # Discovery completes in one logical pass

    def __init__(
        self,
        settings: Settings,
        db: AsyncSession,
        llm_client: OllamaClient,
        profile: UserProfile,
    ) -> None:
        super().__init__(name="DiscoveryAgent")
        self.settings = settings
        self.db = db
        self.llm = llm_client
        self.profile = profile
        self.normalizer = JobNormalizer()

        # State tracked across the single-pass run
        self._raw_jobs: list[RawJob] = []
        self._new_jobs: list[JobListing] = []
        self._scored_count = 0
        self._skipped_count = 0
        self._saved_count = 0

    # ──────────────────────────────────────────────────────────────────────────
    # BaseAgent interface
    # ──────────────────────────────────────────────────────────────────────────

    async def observe(self) -> Observation:
        """
        Observation: how many existing jobs are in the DB (for context).
        """
        result = await self.db.execute(select(JobListing.url_hash))
        existing_hashes = set(result.scalars().all())
        return Observation(
            data={"existing_hashes": existing_hashes, "count": len(existing_hashes)},
            source="database",
        )

    async def think(self, observation: Observation) -> Plan:
        """
        Given the observation, decide the next action.

        Since discovery is a linear pipeline, we step through phases:
        fetch → normalize → score → save → done
        """
        existing_hashes = observation.data.get("existing_hashes", set())

        if not self._raw_jobs:
            return Plan(
                action="fetch",
                args={"existing_hashes": existing_hashes},
                reasoning="No jobs fetched yet — start with fetch phase",
            )
        if not self._new_jobs:
            return Plan(
                action="normalize_and_deduplicate",
                args={"existing_hashes": existing_hashes},
                reasoning="Raw jobs available — normalize and deduplicate",
            )
        if self._scored_count == 0:
            return Plan(
                action="score",
                args={},
                reasoning="New jobs normalized — score for relevance",
            )
        return Plan(
            action="save",
            args={},
            reasoning="Jobs scored — persist to database",
        )

    async def act(self, plan: Plan) -> ActionResult:
        """Execute the plan action."""
        action = plan.action

        if action == "fetch":
            return await self._action_fetch(plan.args["existing_hashes"])
        elif action == "normalize_and_deduplicate":
            return await self._action_normalize(plan.args["existing_hashes"])
        elif action == "score":
            return await self._action_score()
        elif action == "save":
            return await self._action_save()
        else:
            return ActionResult(success=False, error=f"Unknown action: {action}", action=action)

    def should_continue(self, result: ActionResult, iteration: int) -> bool:
        return result.action != "save"

    # ──────────────────────────────────────────────────────────────────────────
    # Action implementations
    # ──────────────────────────────────────────────────────────────────────────

    async def _action_fetch(self, existing_hashes: set[str]) -> ActionResult:
        """Fetch all jobs from all configured sources in parallel."""
        cfg = self.settings.discovery
        sources: list[JobSource] = []

        # Share a single HTTP client across all sources for connection pooling
        http_client = httpx.AsyncClient(
            timeout=25.0,
            headers={"User-Agent": "CareerAgent/1.0"},
            follow_redirects=True,
        )

        try:
            if cfg.greenhouse_companies:
                sources.append(GreenhouseAPISource(cfg.greenhouse_companies, http_client))
            if cfg.lever_companies:
                sources.append(LeverAPISource(cfg.lever_companies, http_client))
            if cfg.ashby_companies:
                sources.append(AshbyAPISource(cfg.ashby_companies, http_client))

            if not sources:
                return ActionResult(
                    success=False,
                    error="No job sources configured",
                    action="fetch",
                )

            # Fetch all sources concurrently
            fetch_tasks = [source.fetch() for source in sources]
            results = await asyncio.gather(*fetch_tasks, return_exceptions=True)

            total_raw = 0
            for source, result in zip(sources, results, strict=True):
                if isinstance(result, BaseException):
                    logger.error("Source [{}] raised: {}", source.name, result)
                else:
                    self._raw_jobs.extend(result)
                    total_raw += len(result)

            logger.info(
                "Fetch complete — {} raw jobs from {} sources",
                total_raw,
                len(sources),
            )
            return ActionResult(
                success=True,
                data={"raw_count": total_raw},
                action="fetch",
            )
        finally:
            await http_client.aclose()

    async def _action_normalize(self, existing_hashes: set[str]) -> ActionResult:
        """Normalize raw jobs and filter duplicates."""
        normalized = self.normalizer.normalize_batch(self._raw_jobs)

        # Deduplicate against database
        new_jobs = [j for j in normalized if j.url_hash not in existing_hashes]
        self._skipped_count = len(normalized) - len(new_jobs)

        # ── Location filter — India or Remote only ──────────────────────────
        location_filtered: list[JobListing] = []
        location_rejected = 0
        for job in new_jobs:
            passed_loc = matches_target_locations(
                job, self.profile.preferences.locations_ok
            )
            logger.info(
                "LOCFILTER job='{}' loc='{}' remote={} passed={}",
                job.title,
                job.location,
                job.remote,
                passed_loc,
            )

            if passed_loc:
                location_filtered.append(job)
            else:
                job.status = JobStatus.SKIPPED.value
                job.score_reasoning = (
                    f"Location filter: '{job.location}' not in India/Remote"
                )
                location_rejected += 1

        self._skipped_count += location_rejected
        logger.info(
            "Location filter — {} passed, {} rejected (not India/Remote)",
            len(location_filtered),
            location_rejected,
        )

        # Keyword pre-score — cheap filter before LLM
        target_roles = self.profile.preferences.target_roles
        skill_names = self.profile.skills.technical_names()
        min_score = self.settings.discovery.min_relevance_score

        pre_scored: list[JobListing] = []
        for job in location_filtered:
            score = keyword_pre_score(
                job_title=job.title,
                job_description=job.description or "",
                target_roles=target_roles,
                technical_skills=skill_names,
            )
            if score >= (min_score * 0.4):  # 40% of threshold → still worth LLM scoring
                job.relevance_score = score
                pre_scored.append(job)
            else:
                job.status = JobStatus.SKIPPED.value
                self._skipped_count += 1

        self._new_jobs = pre_scored

        logger.info(
            "Normalize complete — {} normalized, {} deduplicated, {} passed pre-filter",
            len(normalized),
            self._skipped_count,
            len(self._new_jobs),
        )
        return ActionResult(
            success=True,
            data={
                "normalized": len(normalized),
                "new": len(self._new_jobs),
                "skipped": self._skipped_count,
            },
            action="normalize_and_deduplicate",
        )

    async def _action_score(self) -> ActionResult:
        """LLM-score new jobs in batches."""
        if not self._new_jobs:
            self._scored_count = 0
            return ActionResult(success=True, data={"scored": 0}, action="score")

        max_jobs = self.settings.discovery.max_jobs_per_run
        jobs_to_score = self._new_jobs[:max_jobs]

        logger.info(
            "Scoring {} jobs sequentially with LLM "
            "(Ollama n_slots=1 — parallel is counterproductive)",
            len(jobs_to_score),
        )

        # Process SEQUENTIALLY — Ollama has n_slots=1, so parallel gather
        # just queues requests and causes timeout on later jobs in the batch.
        for i, job in enumerate(jobs_to_score, 1):
            await self._score_single_job(job)
            logger.debug("Scored job {}/{}: '{}'", i, len(jobs_to_score), job.title)

        self._scored_count = len(jobs_to_score)
        logger.info("Scoring complete — {} jobs scored", self._scored_count)
        return ActionResult(success=True, data={"scored": self._scored_count}, action="score")

    async def _score_single_job(self, job: JobListing) -> None:
        """Score a single job with the LLM and update its fields in-place."""
        try:
            system, user_msg = SCORE_JOB_PROMPT.format(
                candidate_name=self.profile.personal.name,
                current_title=self.profile.most_recent_title(),
                years_experience=self.profile.years_of_experience(),
                target_roles=", ".join(self.profile.preferences.target_roles),
                technical_skills=self.profile.skills_summary(),
                remote_preference=self.profile.preferences.remote,
                preferred_locations=", ".join(self.profile.preferences.locations_ok),
                job_title=job.title,
                company=job.company,
                location=job.location or "Not specified",
                remote=str(job.remote),
                description=(job.description or "")[:1500],  # Short = fast
            )

            raw = await self.llm.chat_json(
                messages=[{"role": "user", "content": user_msg}],
                system=system,
                temperature=self.settings.llm.scoring_temperature,
                think=False,   # Disable qwen3 chain-of-thought for speed
                schema=JobScoreResponse.model_json_schema()
            )
            score = parse_job_score(raw)

            job.relevance_score = float(score.overall)
            job.score_reasoning = score.reasoning
            job.status = (
                JobStatus.QUEUED.value if score.apply else JobStatus.SKIPPED.value
            )

            logger.info(
                "Scored '{}' @ {} — score={} apply={}",
                job.title,
                job.company,
                score.overall,
                score.apply,
            )

        except LLMParseError as e:
            logger.warning("LLM parse error for '{}': {}", job.title, e)
            # Keep keyword score, don't change status
        except Exception as e:
            logger.warning("Score failed for '{}': {}", job.title, e)

    async def _action_save(self) -> ActionResult:
        """Persist all new jobs to the database."""
        if not self._new_jobs:
            return ActionResult(success=True, data={"saved": 0}, action="save")

        saved = 0
        for job in self._new_jobs:
            try:
                self.db.add(job)
                saved += 1
            except Exception as e:
                logger.error("Failed to queue job '{}' for save: {}", job.title, e)

        try:
            await self.db.commit()
            self._saved_count = saved
            logger.info("Saved {} new jobs to database", saved)
        except Exception as e:
            await self.db.rollback()
            logger.error("Database commit failed: {}", e)
            return ActionResult(
                success=False,
                error=f"Database commit failed: {e}",
                action="save",
            )

        return ActionResult(
            success=True,
            data={"saved": saved},
            action="save",
        )

    # ──────────────────────────────────────────────────────────────────────────
    # High-level convenience method
    # ──────────────────────────────────────────────────────────────────────────

    async def run_discovery(self) -> dict:
        """
        High-level entry point: run a full discovery cycle and return a summary dict.

        Used by the scheduler and API endpoints.
        """
        result = await self.run()
        return {
            "success": result.success,
            "summary": result.summary,
            "raw_fetched": len(self._raw_jobs),
            "new_saved": self._saved_count,
            "skipped": self._skipped_count,
            "scored": self._scored_count,
            "duration_seconds": result.duration_seconds,
            "errors": result.errors,
        }
