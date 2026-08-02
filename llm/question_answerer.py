from core.logger import logger
from core.config import get_settings
from llm.client import OllamaClient
from llm.prompts import ANSWER_QUESTION_PROMPT
from profile.loader import load_profile


class ApplicationQA:
    def __init__(self):
        self.settings = get_settings()
        self.profile = load_profile(self.settings.profile.path)
        self.client = OllamaClient.from_settings(self.settings)

    async def answer_question(self, question: str) -> str:
        """Uses the configured LLM to answer a job application question based on the user's profile."""

        profile_context = f"Name: {self.profile.personal.name}\n"
        profile_context += f"Email: {self.profile.personal.email}\n"
        profile_context += f"Current Title: {self.profile.most_recent_title()}\n"
        profile_context += f"Years of Experience: {self.profile.years_of_experience()}\n"
        profile_context += f"Experience: {len(self.profile.experience)} roles\n"
        profile_context += f"Skills: {', '.join(self.profile.skills.technical_names())}\n"

        system, user_msg = ANSWER_QUESTION_PROMPT.format(
            question=question,
            profile=profile_context,
        )

        logger.debug(f"Asking LLM question: {question}")
        answer = await self.client.chat(
            [{"role": "user", "content": user_msg}],
            system=system,
            model=self.settings.llm.reasoning_model,
            temperature=0.1,
        )

        logger.info(f"LLM Answer for '{question[:30]}...': {answer}")
        return answer.strip()
