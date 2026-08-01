import json
import base64
import os
from pathlib import Path
from typing import Any, AsyncGenerator
from dotenv import load_dotenv

from google import genai
from google.genai import types

from core.logger import logger
from core.exceptions import LLMConnectionError

# Load environment variables
load_dotenv()

class GeminiClient:
    """
    Unified async LLM client for Google Gemini.
    Implements the same interface as OllamaClient.
    """
    def __init__(
        self,
        reasoning_model: str = "gemini-2.5-flash",
        vision_model: str = "gemini-2.5-flash",
        timeout: int = 120,
        max_retries: int = 3,
    ) -> None:
        self.reasoning_model = reasoning_model
        self.vision_model = vision_model
        self.timeout = timeout
        self.max_retries = max_retries
        
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            logger.error("GEMINI_API_KEY is not set in environment or .env file.")
        
        self.client = genai.Client(api_key=api_key)
        self.async_client = self.client.aio

    async def _async_retry(self, coro_func, *args, **kwargs):
        import asyncio
        import re
        for attempt in range(self.max_retries):
            try:
                return await coro_func(*args, **kwargs)
            except Exception as e:
                err_str = str(e)
                if "429" in err_str:
                    if attempt < self.max_retries - 1:
                        match = re.search(r"retry in (\d+\.?\d*)s", err_str.lower())
                        delay = float(match.group(1)) + 1.0 if match else 30.0
                        logger.warning(f"Gemini API rate limit (429). Retrying in {delay:.1f}s (attempt {attempt+1}/{self.max_retries})")
                        await asyncio.sleep(delay)
                        continue
                raise

    def _convert_messages_to_gemini(self, messages: list[dict[str, Any]]) -> list[types.Content]:
        """Converts OpenAI-style messages to Gemini Content objects."""
        contents = []
        for msg in messages:
            role = "user" if msg["role"] == "user" else "model"
            content = msg["content"]
            
            parts = []
            if isinstance(content, str):
                parts.append(types.Part.from_text(text=content))
            elif isinstance(content, list):
                # Handle multimodal content dicts
                pass # Simplified for now, vision uses describe_image directly
                
            contents.append(types.Content(role=role, parts=parts))
        return contents

    async def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str | None = None,
        temperature: float = 0.1,
        system: str | None = None,
        format: str | None = None,
        think: bool = False,
        timeout: int | None = None,
    ) -> str:
        target_model = model or self.reasoning_model
        
        config_kwargs = {"temperature": temperature}
        if system:
            config_kwargs["system_instruction"] = system
        if format == "json":
            config_kwargs["response_mime_type"] = "application/json"
            
        config = types.GenerateContentConfig(**config_kwargs)
        
        contents = self._convert_messages_to_gemini(messages)
        
        try:
            logger.debug(f"Gemini chat - model={target_model}")
            response = await self._async_retry(
                self.async_client.models.generate_content,
                model=target_model,
                contents=contents,
                config=config
            )
            return response.text
        except Exception as e:
            raise LLMConnectionError(f"Gemini error: {e}", context={"model": target_model}) from e

    async def chat_json(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str | None = None,
        temperature: float = 0.1,
        system: str | None = None,
        think: bool = False,
        timeout: int | None = None,
    ) -> dict[str, Any]:
        raw = await self.chat(
            messages,
            model=model,
            temperature=temperature,
            system=system,
            format="json",
            think=think,
            timeout=timeout
        )
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            import re
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if match:
                return json.loads(match.group())
            raise

    async def chat_stream(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str | None = None,
        temperature: float = 0.4,
        system: str | None = None,
    ) -> AsyncGenerator[str, None]:
        target_model = model or self.reasoning_model
        config_kwargs = {"temperature": temperature}
        if system:
            config_kwargs["system_instruction"] = system
            
        config = types.GenerateContentConfig(**config_kwargs)
        contents = self._convert_messages_to_gemini(messages)
        
        try:
            async for chunk in await self.async_client.models.generate_content_stream(
                model=target_model,
                contents=contents,
                config=config
            ):
                if chunk.text:
                    yield chunk.text
        except Exception as e:
            raise LLMConnectionError(f"Gemini stream error: {e}")

    async def describe_image(
        self,
        image_path: str,
        prompt: str,
        *,
        temperature: float = 0.1,
    ) -> str:
        target_model = self.vision_model
        logger.debug(f"Gemini vision - model={target_model} image={image_path}")
        
        # Read the file
        path = Path(image_path)
        if not path.exists():
            return "ERROR: Image not found"
            
        try:
            with open(path, "rb") as f:
                image_bytes = f.read()
            
            # Figure out MIME type from extension
            ext = path.suffix.lower()
            mime_type = "image/jpeg"
            if ext == ".png":
                mime_type = "image/png"
            elif ext == ".webp":
                mime_type = "image/webp"
                
            part = types.Part.from_bytes(data=image_bytes, mime_type=mime_type)
            
            response = await self._async_retry(
                self.async_client.models.generate_content,
                model=target_model,
                contents=[part, prompt],
                config=types.GenerateContentConfig(temperature=temperature)
            )
            return response.text
        except Exception as e:
            logger.error(f"Gemini vision error: {e}")
            return ""

    async def health_check(self) -> bool:
        if not os.getenv("GEMINI_API_KEY"):
            logger.error("Health Check Failed: GEMINI_API_KEY is not set.")
            return False
        return True
