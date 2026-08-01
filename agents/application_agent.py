import asyncio
from sqlalchemy import select

from core.logger import logger
from core.database import get_session
from core.config import get_settings
from core.exceptions import DocumentNotFoundError
from models.job import JobListing
from browser.browser_agent import BrowserAgent
from vision.vision_module import VisionModule
from profile.loader import load_profile
from llm.client import OllamaClient
from agents.tailoring_agent import TailoringAgent
from documents.document_manager import DocumentManager


class ApplicationAgent:
    def __init__(self):
        self.settings = get_settings()
        self.browser = BrowserAgent()
        self.vision = VisionModule()
        self.profile = load_profile(self.settings.profile.path)

        # Shared LLM client — created once, reused for tailoring.
        # Same pattern DiscoveryAgent already uses for scoring; avoids
        # spinning up a new Ollama/Gemini client per job.
        self.llm = OllamaClient.from_settings(self.settings)
        self.tailoring = TailoringAgent(llm_client=self.llm)
        self.documents = DocumentManager(
            documents_dir=self.settings.storage.documents_dir,
            default_resume=self.settings.profile.default_resume,
        )

    async def process_queue(self):
        """Finds jobs in DB with score >= min_relevance_score and status 'discovered'."""
        async with get_session(self.settings.storage.database_url) as db:
            result = await db.execute(
                select(JobListing)
                .where(
                    JobListing.status == 'discovered',
                    JobListing.relevance_score >= self.settings.discovery.min_relevance_score
                )
                .limit(self.settings.application.daily_limit)
            )
            jobs = result.scalars().all()

            if not jobs:
                logger.info("No queued jobs meeting score threshold.")
                return

            logger.info(f"Found {len(jobs)} jobs to apply for. Starting Application Agent.")
            await self.browser.start()

            for job in jobs:
                try:
                    await self.apply_to_job(db, job)
                except Exception as e:
                    logger.error(f"Application failed for job {job.id}: {e}")
                    job.status = 'failed'
                    await db.commit()

                await asyncio.sleep(self.settings.application.inter_application_delay)

            await self.browser.close()

    async def apply_to_job(self, db, job: JobListing):
        logger.info(f"Applying to job: {job.title} @ {job.company}")
        job.status = 'applying'
        await db.commit()

        # 1. Navigate
        # FIX: JobListing has no `source_url` field. Fall back to
        # job_post_url instead — application_url remains the primary target.
        url = job.application_url or job.job_post_url
        if not url:
            logger.error(f"No application URL available for job {job.id}")
            job.status = 'failed'
            await db.commit()
            return

        success = await self.browser.navigate(url)
        if not success:
            job.status = 'failed'
            await db.commit()
            return

        # 2. Take initial screenshot
        screenshot_path = await self.browser.take_screenshot(job.id, "1_initial_load")

        # 3. Vision check: Is it an application form?
        vision_prompt = "Is this page an online job application form? Does it have fields to fill? Answer 'YES' or 'NO'."
        vision_res = await self.vision.analyze_screenshot(screenshot_path, vision_prompt)

        if "YES" not in vision_res.upper():
            logger.warning(f"Vision model says this isn't an application form. (Response: {vision_res})")
            if not self.settings.application.dry_run:
                job.status = 'failed'
                await db.commit()
                return

        # 4. Draft a tailored cover letter (if enabled) before touching the
        # form, so it's ready to attach/paste if the form has that field.
        cover_letter_path = None
        if self.profile.application_preferences.auto_generate_cover_letter:
            cover_letter_path = await self.tailoring.draft_cover_letter(job, self.profile)

        # 5. Fill standard fields, then attach documents
        await self._fill_common_fields()
        await self._upload_documents(job, cover_letter_path)

        screenshot_path_filled = await self.browser.take_screenshot(job.id, "2_form_filled")

        if self.settings.application.dry_run:
            logger.info(f"DRY RUN: Skipping submit for {job.title}")
            job.status = 'skipped'
        else:
            logger.info("Clicking Submit...")
            submit_btn = self.browser.page.locator(
                "button:has-text('Submit application'), button:has-text('Submit Application')"
            ).first
            if await submit_btn.count() > 0:
                await submit_btn.click()
                await asyncio.sleep(3)  # Wait for submission to process
            else:
                logger.warning("Submit button not found!")
            job.status = 'applied'

        await db.commit()

    async def _fill_common_fields(self):
        """Naive DOM-based filling for Phase 2 proof of concept."""
        page = self.browser.page

        if await page.locator("input[name*='name' i]").count() > 0:
            logger.info("Filling name field")
            await page.locator("input[name*='name' i]").first.fill(self.profile.personal.name)
        elif await page.locator("input[name*='first' i]").count() > 0:
            logger.info("Filling first/last name fields")
            await page.locator("input[name*='first' i]").first.fill(self.profile.personal.first_name)
            if await page.locator("input[name*='last' i]").count() > 0:
                await page.locator("input[name*='last' i]").first.fill(self.profile.personal.last_name)

        if await page.locator("input[name*='email' i]").count() > 0:
            logger.info("Filling email field")
            await page.locator("input[name*='email' i]").first.fill(self.profile.personal.email)

        if await page.locator("input[name*='phone' i]").count() > 0:
            logger.info("Filling phone field")
            await page.locator("input[name*='phone' i]").first.fill(self.profile.personal.phone)

        if await page.locator("input[name*='linkedin' i]").count() > 0 and getattr(self.profile.personal, 'linkedin', None):
            logger.info("Filling LinkedIn field")
            await page.locator("input[name*='linkedin' i]").first.fill(self.profile.personal.linkedin)

        if await page.locator("input[name*='github' i]").count() > 0 and getattr(self.profile.personal, 'github', None):
            logger.info("Filling GitHub field")
            await page.locator("input[name*='github' i]").first.fill(self.profile.personal.github)

    async def _upload_documents(self, job: JobListing, cover_letter_path) -> None:
        """
        Upload the resume and (optionally) the tailored cover letter.

        Every locator is existence-checked before use, consistent with
        _fill_common_fields — a form without a matching field is skipped
        with a log line instead of raising.
        """
        page = self.browser.page

        # ── Resume ────────────────────────────────────────────────────────
        try:
            resume_doc = self.documents.get_resume()
        except DocumentNotFoundError as e:
            logger.warning(f"No resume available to upload for job {job.id}: {e}")
            resume_doc = None

        if resume_doc:
            resume_input = page.locator(
                "input[type='file'][name*='resume' i], "
                "input[type='file'][id*='resume' i], "
                "input[type='file'][name*='cv' i]"
            ).first
            if await resume_input.count() > 0:
                logger.info(f"Uploading resume: {resume_doc.filename}")
                await resume_input.set_input_files(str(resume_doc.path))
            else:
                # Fallback: some ATS forms expose only one generic file
                # input for the resume (no resume-specific name/id).
                generic_input = page.locator("input[type='file']").first
                if await generic_input.count() > 0:
                    logger.info(f"Uploading resume via generic file input: {resume_doc.filename}")
                    await generic_input.set_input_files(str(resume_doc.path))
                else:
                    logger.warning("No file input found for resume upload — form may need manual review.")

        # ── Cover letter ─────────────────────────────────────────────────
        if cover_letter_path:
            cover_input = page.locator(
                "input[type='file'][name*='cover' i], "
                "input[type='file'][id*='cover' i]"
            ).first
            if await cover_input.count() > 0:
                logger.info(f"Uploading cover letter: {cover_letter_path.name}")
                await cover_input.set_input_files(str(cover_letter_path))
            else:
                # Many ATS forms (Greenhouse especially) use a textarea
                # instead of a file input for the cover letter.
                cover_textarea = page.locator(
                    "textarea[name*='cover' i], textarea[id*='cover' i]"
                ).first
                if await cover_textarea.count() > 0:
                    logger.info("Pasting cover letter into textarea field")
                    # Fix: cover_letter_path is either a string or Path. 
                    # If it's a string, we need to wrap it in Path().
                    # Given the original TailoringAgent returns Path, we assume Path or string.
                    from pathlib import Path
                    path_obj = Path(cover_letter_path)
                    await cover_textarea.fill(path_obj.read_text(encoding="utf-8"))
                else:
                    logger.info("No cover letter field found on this form — skipping.")
