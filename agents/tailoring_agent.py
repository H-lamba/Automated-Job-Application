"""
agents/tailoring_agent.py — Drafts a tailored cover letter per job.

Design notes:
- Reuses the existing COVER_LETTER_PROMPT template from llm/prompts.py
  rather than building a new prompt from scratch.
- Does NOT own its own LLM client. One must be injected, so the agent
  reuses the single shared client created once at startup — same
  pattern DiscoveryAgent already follows for scoring.
- Output is plain .txt (not PDF) — Greenhouse, Lever, and Ashby all
  accept plain text uploads, and this avoids adding a PDF-generation
  dependency for now.
- Failure to draft a letter is non-fatal: callers should treat a
  None return as "skip the cover letter for this job", not abort
  the whole application.
"""

from __future__ import annotations

from pathlib import Path

from core.logger import logger
from llm.prompts import COVER_LETTER_PROMPT
from models.job import JobListing
from models.profile import UserProfile


class TailoringAgent:
    """
    Generates a tailored cover letter (plain text) for a specific job.
    """

    def __init__(self, llm_client, staging_dir: str = "./data/tailored") -> None:
        self.llm = llm_client
        self.staging_dir = Path(staging_dir)
        self.staging_dir.mkdir(parents=True, exist_ok=True)

    async def draft_cover_letter(self, job: JobListing, profile: UserProfile) -> Path | None:
        """
        Generate a tailored cover letter for `job` and save it as a .txt
        file in the staging directory.

        Returns the Path to the saved file, or None if generation failed.
        """
        try:
            top_skills = ", ".join(profile.skills.technical_names()[:8]) or "N/A"

            achievements: list[str] = []
            for exp in profile.experience:
                achievements.extend(exp.achievements)
            achievements_text = "; ".join(achievements[:5]) if achievements else "N/A"

            job_highlights = (job.description or "")[:1000]  # keep prompt short = fast

            system, user_msg = COVER_LETTER_PROMPT.format(
                candidate_name=profile.personal.name,
                current_title=profile.most_recent_title(),
                years_experience=profile.years_of_experience(),
                top_skills=top_skills,
                achievements=achievements_text,
                job_title=job.title,
                company=job.company,
                job_highlights=job_highlights,
            )

            letter_text = await self.llm.chat(
                messages=[{"role": "user", "content": user_msg}],
                system=system,
                temperature=0.4,   # a bit of natural variation, still controlled
            )
            letter_text = (letter_text or "").strip()

            if not letter_text:
                logger.warning(
                    "Tailoring: empty cover letter returned for '{}' @ {}",
                    job.title,
                    job.company,
                )
                return None

            out_path = self.staging_dir / f"cover_letter_{job.id}.txt"
            out_path.write_text(letter_text, encoding="utf-8")

            logger.info(
                "Tailoring: drafted cover letter for '{}' @ {} -> {}",
                job.title,
                job.company,
                out_path,
            )
            return out_path

        except Exception as e:
            logger.warning(
                "Tailoring: failed to draft cover letter for '{}' @ {}: {}",
                job.title,
                job.company,
                e,
            )
            return None
