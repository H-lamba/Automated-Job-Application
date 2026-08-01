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
from vision.form_extractor import FormExtractor
from documents.fabrication_store import FabricationStore


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
        self.form_extractor = FormExtractor(self.llm, self.profile)
        self.fabrication_store = FabricationStore(
            getattr(self.settings.application, "fabrication_dir", "./data/fabricated")
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

        # 5. Fill standard fields and fabricated fields using LLM
        await self._fill_form_with_llm(screenshot_path, job)
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

    async def _fill_form_with_llm(self, screenshot_path: str, job: JobListing):
        """Uses DOM scraping + LLM to dynamically fill form fields and answer custom questions."""
        page = self.browser.page
        
        # 1. Scrape form inputs from the DOM
        js_script = """
        () => {
            const inputs = [];
            document.querySelectorAll('input:not([type="hidden"]):not([type="file"]), textarea, select').forEach(el => {
                let label = '';
                if (el.id) {
                    const l = document.querySelector(`label[for="${el.id}"]`);
                    if (l) label = l.innerText;
                }
                const options = [];
                if (el.tagName.toLowerCase() === 'select') {
                    for (let opt of el.options) {
                        options.push(opt.text || opt.value);
                    }
                }
                inputs.push({
                    tag: el.tagName.toLowerCase(),
                    name: el.name,
                    id: el.id,
                    type: el.type,
                    label: label || el.placeholder || '',
                    options: options
                });
            });
            return inputs;
        }
        """
        form_inputs = await page.evaluate(js_script)
        if not form_inputs:
            logger.info("No fillable form inputs found on page.")
            return

        import json
        
        previous_answers = self.fabrication_store.load_previous_answers()
        
        form = await self.form_extractor.extract(
            screenshot_path=screenshot_path,
            job=job,
            form_inputs_json=json.dumps(form_inputs, indent=2),
            previous_answers=previous_answers
        )
        
        # Save fabricated answers for audit trail
        self.fabrication_store.save(job, form)

        # 5. Execute Fills
        for field in form.fields:
            if not field.field_id or field.answer is None or field.source == "skipped":
                continue
                
            selector_attr = field.field_id
            value = field.answer
                
            # Try to match by name first, then id
            loc = page.locator(f"[name='{selector_attr}']").first
            if await loc.count() == 0:
                loc = page.locator(f"[id='{selector_attr}']").first
                
            if await loc.count() > 0:
                try:
                    tag_name = await loc.evaluate("el => el.tagName.toLowerCase()")
                    if tag_name == "select":
                        # For select, we try to select by label
                        try:
                            await loc.select_option(label=str(value))
                        except Exception:
                            # Fallback to value if label fails
                            await loc.select_option(value=str(value))
                    else:
                        type_attr = await loc.evaluate("el => el.type")
                        if type_attr in ["checkbox", "radio"]:
                            # We assume value is true/false or string indicating to check it
                            if str(value).lower() in ["true", "yes", "1", "on"]:
                                await loc.check()
                        else:
                            await loc.fill(str(value))
                    logger.info(f"Filled {selector_attr} with '{value}' (source: {field.source})")
                except Exception as e:
                    logger.warning(f"Failed to fill {selector_attr}: {e}")
            else:
                logger.warning(f"Could not find element for selector: {selector_attr}")

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
