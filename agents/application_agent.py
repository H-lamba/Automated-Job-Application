import asyncio
import json
from datetime import datetime
from sqlalchemy import select

from core.logger import logger
from core.database import get_session
from core.config import get_settings
from models.job import JobListing
from models.application import ApplicationRecord
from browser.browser_agent import BrowserAgent
from vision.vision_module import VisionModule
from profile.loader import load_profile

class ApplicationAgent:
    def __init__(self):
        self.settings = get_settings()
        self.browser = BrowserAgent()
        self.vision = VisionModule()
        self.profile = load_profile(self.settings.profile.path)
        
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
        url = job.application_url or job.source_url
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
                
        # 4. Fill form (This is where ATS adapters / DOM parsing comes in)
        # For Phase 2 sandbox, we will just use Playwright to locate common fields
        await self._fill_common_fields()
        
        screenshot_path_filled = await self.browser.take_screenshot(job.id, "2_form_filled")
        
        if self.settings.application.dry_run:
            logger.info(f"DRY RUN: Skipping submit for {job.title}")
            job.status = 'skipped'
        else:
            # Click submit
            logger.info("Clicking Submit...")
            # Ashby submit button is typically "Submit Application"
            submit_btn = self.browser.page.locator("button:has-text('Submit application'), button:has-text('Submit Application')").first
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
        
        # Handle simple name input
        if await page.locator("input[name*='name' i]").count() > 0:
            logger.info("Filling name field")
            await page.locator("input[name*='name' i]").first.fill(self.profile.personal.name)
        elif await page.locator("input[name*='first' i]").count() > 0:
            logger.info("Filling first/last name fields")
            # Handle first/last name split
            await page.locator("input[name*='first' i]").first.fill(self.profile.personal.first_name)
            if await page.locator("input[name*='last' i]").count() > 0:
                await page.locator("input[name*='last' i]").first.fill(self.profile.personal.last_name)
            
        # Email
        if await page.locator("input[name*='email' i]").count() > 0:
            logger.info("Filling email field")
            await page.locator("input[name*='email' i]").first.fill(self.profile.personal.email)
            
        # Phone
        if await page.locator("input[name*='phone' i]").count() > 0:
            logger.info("Filling phone field")
            await page.locator("input[name*='phone' i]").first.fill(self.profile.personal.phone)

        # LinkedIn
        if await page.locator("input[name*='linkedin' i]").count() > 0 and getattr(self.profile.personal, 'linkedin', None):
            logger.info("Filling LinkedIn field")
            await page.locator("input[name*='linkedin' i]").first.fill(self.profile.personal.linkedin)
            
        # GitHub
        if await page.locator("input[name*='github' i]").count() > 0 and getattr(self.profile.personal, 'github', None):
            logger.info("Filling GitHub field")
            await page.locator("input[name*='github' i]").first.fill(self.profile.personal.github)
