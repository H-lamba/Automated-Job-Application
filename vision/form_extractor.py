from __future__ import annotations

import json
from core.logger import logger
from models.profile import UserProfile
from models.job import JobListing
from llm.client import OllamaClient
from llm.prompts import FORM_EXTRACTION_PROMPT
from llm.response_parser import parse_extracted_form, ExtractedForm
from core.config import get_settings

# Kept short and cheap — this isn't the primary field-extraction step,
# just enough context to classify page_type reliably before the DOM-based
# extraction runs. Falls back to "" on any vision failure so the pipeline
# never blocks on this.
_VISION_PAGE_CLASSIFY_PROMPT = (
    "Briefly describe this webpage in 2-3 sentences: is it a job application "
    "form, a confirmation/success page, an error page, or a login page? "
    "Mention roughly how many form fields are visible, if any."
)


class FormExtractor:
    def __init__(self, llm_client: OllamaClient, profile: UserProfile):
        self.llm = llm_client
        self.profile = profile
        self.settings = get_settings()

    async def _get_vision_context(self, screenshot_path: str) -> str:
        """Get a short vision-model summary of the screenshot. Never raises."""
        try:
            summary = await self.llm.describe_image(
                screenshot_path,
                _VISION_PAGE_CLASSIFY_PROMPT,
            )
            return (summary or "").strip()[:800]  # keep it short — it's context, not the main payload
        except Exception as e:
            logger.warning(f"Vision context unavailable, continuing DOM-only: {e}")
            return "(vision analysis unavailable)"

    async def extract(
        self,
        screenshot_path: str,
        job: JobListing,
        form_inputs_json: str,
        previous_answers: dict
    ) -> ExtractedForm:
        """
        Calls the vision model for page classification, then the reasoning
        model to parse form inputs and generate structured answers.
        """
        logger.info(f"Extracting form data for job '{job.title}'...")

        vision_context = await self._get_vision_context(screenshot_path)

        experience_summary = "\\n".join([f"{e.title} at {e.company}" for e in self.profile.experience])

        system_prompt, user_msg = FORM_EXTRACTION_PROMPT.format(
            vision_context=vision_context,
            name=self.profile.personal.name,
            email=self.profile.personal.email,
            phone=self.profile.personal.phone,
            linkedin=self.profile.personal.linkedin or "",
            github=self.profile.personal.github or "",
            salary=f"{self.profile.preferences.desired_salary_min} - {self.profile.preferences.desired_salary_max}",
            locations=", ".join(self.profile.preferences.locations_ok),
            skills=self.profile.skills_summary(),
            experience=experience_summary,
            previous_answers=json.dumps(previous_answers, indent=2),
            job_title=job.title,
            company=job.company,
            form_inputs=form_inputs_json
        )

        response_json = await self.llm.chat_json(
            messages=[{"role": "user", "content": user_msg}],
            system=system_prompt,
            model=self.settings.llm.reasoning_model
        )

        form = parse_extracted_form(response_json)

        # Enforce the fabrication policy at the source, not just in storage —
        # see config enforcement fix below.
        if not getattr(self.settings.application, "allow_fabrication", True):
            for field in form.fields:
                if field.source == "fabricated":
                    field.source = "skipped"
                    field.answer = None
                    field.reasoning = "Fabrication disabled by config (allow_fabrication=false)"

        return form
