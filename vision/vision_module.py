from core.logger import logger
from core.config import get_settings
from llm.client import OllamaClient

class VisionModule:
    def __init__(self):
        self.settings = get_settings()
        # The from_settings factory automatically returns GeminiClient if configured
        self.client = OllamaClient.from_settings(self.settings)
        
    async def analyze_screenshot(self, image_path: str, prompt: str) -> str:
        """Analyzes a screenshot and answers the prompt using the configured vision backend."""
        logger.debug(f"Analyzing {image_path} with vision model...")
        return await self.client.describe_image(image_path, prompt)
