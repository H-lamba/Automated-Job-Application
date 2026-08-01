from __future__ import annotations

import json
from core.logger import logger
from models.profile import UserProfile
from models.job import JobListing
from llm.client import OllamaClient
from llm.prompts import FORM_EXTRACTION_PROMPT
from llm.response_parser import parse_extracted_form, ExtractedForm
from core.config import get_settings

class FormExtractor:
    def __init__(self, llm_client: OllamaClient, profile: UserProfile):
        self.llm = llm_client
        self.profile = profile
        self.settings = get_settings()

    async def extract(
        self,
        screenshot_path: str,
        job: JobListing,
        form_inputs_json: str,
        previous_answers: dict
    ) -> ExtractedForm:
        """
        Calls the LLM to parse the form inputs and generate structured answers.
        """
        logger.info(f"Extracting form data for job '{job.title}'...")
        
        experience_summary = "\\n".join([f"{e.title} at {e.company}" for e in self.profile.experience])
        
        system_prompt, user_msg = FORM_EXTRACTION_PROMPT.format(
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
        
        return parse_extracted_form(response_json)
