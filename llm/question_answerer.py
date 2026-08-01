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
        
        # We need a text representation of the profile for the LLM context.
        profile_context = f"Name: {self.profile.name}\n"
        profile_context += f"Email: {self.profile.contact.email}\n"
        profile_context += f"Experience: {len(self.profile.experience)} roles\n"
        profile_context += f"Skills: {', '.join([s.name for s in self.profile.skills])}\n"
        
        prompt = ANSWER_QUESTION_PROMPT.format(
            question=question,
            profile=profile_context
        )
        
        logger.debug(f"Asking LLM question: {question}")
        answer = await self.client.chat([{"role": "user", "content": prompt}], model=self.settings.llm.reasoning_model, temperature=0.1)
        
        logger.info(f"LLM Answer for '{question[:30]}...': {answer}")
        return answer.strip()
