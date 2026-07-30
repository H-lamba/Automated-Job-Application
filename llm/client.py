"""
llm/client.py — Async LLM client with dual-backend support.

Backends:
  - Ollama (ollama.AsyncClient): used for text reasoning → qwen3:8b
  - MLX  (mlx_lm): used for vision/screenshot understanding → gemma-4-12b-it-4bit
    runs locally on Apple Silicon via the mlx-community HuggingFace model.

All agents use this client — never import ollama or mlx_lm directly elsewhere.
"""

from __future__ import annotations

import asyncio
import base64
import json
from pathlib import Path
from typing import Any, AsyncGenerator

import ollama
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from core.exceptions import LLMConnectionError, LLMTimeoutError
from core.logger import logger


# ──────────────────────────────────────────────────────────────────────────────
# MLX Vision helper (lazy import — only on Apple Silicon when needed)
# ──────────────────────────────────────────────────────────────────────────────


def _mlx_generate(model_path: str, prompt: str, image_path: str | None = None) -> str:
    """
    Run a synchronous MLX generation call.

    Runs in a thread pool so it doesn't block the event loop.
    MLX is not async-native, so we keep it in a thread.
    """
    try:
        from mlx_lm import load, generate
        from mlx_lm.utils import load as load_model

        model, tokenizer = load(model_path)
        response = generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=512,
            verbose=False,
        )
        return response
    except ImportError:
        raise LLMConnectionError(
            "mlx_lm is not installed. Install with: pip install mlx-lm",
            context={"model": model_path},
        )
    except Exception as e:
        raise LLMConnectionError(
            f"MLX generation failed: {e}",
            context={"model": model_path},
        ) from e


def _mlx_vision_generate(model_path: str, prompt: str, image_path: str) -> str:
    """
    Run a vision-capable MLX generation using the chat template.

    Gemma 4 supports interleaved image+text via the chat format.
    """
    try:
        from mlx_lm import load, generate
        import mlx.core as mx

        model, tokenizer = load(model_path)

        # Build prompt with image tag if tokenizer supports it
        if hasattr(tokenizer, "apply_chat_template"):
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image_path},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            try:
                formatted = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                # Fallback: text-only prompt if image tokenization fails
                formatted = f"<image>{prompt}"
        else:
            formatted = prompt

        response = generate(
            model,
            tokenizer,
            prompt=formatted,
            max_tokens=1024,
            verbose=False,
        )
        return response
    except ImportError:
        raise LLMConnectionError(
            "mlx_lm is not installed. Install with: pip install mlx-lm",
            context={"model": model_path},
        )
    except Exception as e:
        raise LLMConnectionError(
            f"MLX vision generation failed: {e}",
            context={"model": model_path},
        ) from e


class OllamaClient:
    """
    Unified async LLM client.

    - Text reasoning → Ollama (qwen3:8b) via ollama.AsyncClient
    - Vision/screenshot analysis → MLX (gemma-4-12b-it-4bit) or Ollama vision model

    Designed for dependency injection — create once, share everywhere.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        reasoning_model: str = "qwen3:8b",
        vision_model: str = "mlx-community/gemma-4-12b-it-4bit",
        vision_backend: str = "mlx",   # "ollama" | "mlx"
        timeout: int = 120,
        scoring_timeout: int = 300,    # Per-job scoring timeout (higher than chat)
        max_retries: int = 3,
    ) -> None:
        self.base_url = base_url
        self.reasoning_model = reasoning_model
        self.vision_model = vision_model
        self.vision_backend = vision_backend
        self.timeout = timeout
        self.scoring_timeout = scoring_timeout
        self.max_retries = max_retries

        # Ollama async client (always initialised; used for reasoning)
        self._client = ollama.AsyncClient(host=base_url)

    # ──────────────────────────────────────────────────────────────────────────
    # Core chat interface (Ollama reasoning)
    # ──────────────────────────────────────────────────────────────────────────

    async def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str | None = None,
        temperature: float = 0.1,
        system: str | None = None,
        format: str | None = None,   # "json" for structured output
        think: bool = False,         # Disable qwen3 chain-of-thought for speed
        timeout: int | None = None,  # Override per-call timeout
    ) -> str:
        """
        Send a chat request to Ollama and return the response as a string.

        Args:
            messages: OpenAI-compatible message list
            model:    Override the default reasoning model
            temperature: Sampling temperature (0.0 = deterministic)
            system:   Optional system message prepended
            format:   Pass "json" to request JSON-structured output
        """
        target_model = model or self.reasoning_model
        full_messages = []

        if system:
            full_messages.append({"role": "system", "content": system})
        full_messages.extend(messages)

        logger.debug(
            "LLM chat — model={} msgs={} temperature={} think={}",
            target_model,
            len(full_messages),
            temperature,
            think,
        )

        effective_timeout = timeout or self.timeout
        try:
            response = await asyncio.wait_for(
                self._client.chat(
                    model=target_model,
                    messages=full_messages,
                    options={
                        "temperature": temperature,
                        "num_ctx": 4096,      # Explicit context window — prevents OOM/empty responses
                        "num_predict": 512,   # Cap output tokens for scoring (we only need ~100)
                    },
                    think=think,
                    format=format or "",
                ),
                timeout=effective_timeout,
            )
            content: str = response.message.content or ""
            logger.debug("LLM response received — length={}", len(content))
            return content

        except asyncio.TimeoutError as e:
            raise LLMTimeoutError(
                f"Ollama request timed out after {effective_timeout}s",
                context={"model": target_model},
            ) from e
        except ollama.ResponseError as e:
            raise LLMConnectionError(
                f"Ollama response error: {e}",
                context={"model": target_model},
            ) from e
        except Exception as e:
            raise LLMConnectionError(
                f"Unexpected Ollama error: {e}",
                context={"model": target_model},
            ) from e

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
        """
        Like chat(), but parses the response as JSON.

        Passes format="json" to Ollama so qwen3:8b is constrained to
        produce valid JSON. Uses scoring_timeout by default (longer wait).
        """
        raw = await self.chat(
            messages,
            model=model,
            temperature=temperature,
            system=system,
            format="json",
            think=think,
            timeout=timeout or self.scoring_timeout,
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
        """Stream the assistant's response token by token (Ollama only)."""
        target_model = model or self.reasoning_model
        full_messages = []
        if system:
            full_messages.append({"role": "system", "content": system})
        full_messages.extend(messages)

        async for chunk in await self._client.chat(
            model=target_model,
            messages=full_messages,
            options={"temperature": temperature},
            stream=True,
        ):
            if chunk.message and chunk.message.content:
                yield chunk.message.content

    # ──────────────────────────────────────────────────────────────────────────
    # Vision interface — routes to MLX or Ollama based on vision_backend
    # ──────────────────────────────────────────────────────────────────────────

    async def describe_image(
        self,
        image_path: str,
        prompt: str,
        *,
        temperature: float = 0.1,
    ) -> str:
        """
        Ask the vision model to describe or analyse a screenshot.

        Routes to MLX (Gemma4) or Ollama vision depending on vision_backend.

        Args:
            image_path: Absolute path to the screenshot file
            prompt:     Task-specific prompt (e.g. "identify all form fields")
        """
        if self.vision_backend == "mlx":
            return await self._describe_image_mlx(image_path, prompt)
        else:
            return await self._describe_image_ollama(image_path, prompt, temperature)

    async def _describe_image_mlx(self, image_path: str, prompt: str) -> str:
        """Use MLX Gemma4 for vision inference (runs on Apple Silicon GPU)."""
        logger.debug(
            "MLX vision — model={} image={}",
            self.vision_model,
            Path(image_path).name,
        )
        loop = asyncio.get_event_loop()
        try:
            result = await loop.run_in_executor(
                None,
                _mlx_vision_generate,
                self.vision_model,
                prompt,
                image_path,
            )
            logger.debug("MLX vision response — length={}", len(result))
            return result
        except Exception as e:
            logger.error("MLX vision failed: {}. Falling back to text-only.", e)
            # Graceful fallback: skip vision, return empty so the agent can continue
            return ""

    async def _describe_image_ollama(
        self,
        image_path: str,
        prompt: str,
        temperature: float,
    ) -> str:
        """Use an Ollama vision model (e.g. llava:13b) for image understanding."""
        image_bytes = Path(image_path).read_bytes()
        b64 = base64.b64encode(image_bytes).decode()

        messages = [
            {
                "role": "user",
                "content": prompt,
                "images": [b64],
            }
        ]
        return await self.chat(
            messages,
            model=self.vision_model,
            temperature=temperature,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Health check
    # ──────────────────────────────────────────────────────────────────────────

    async def health_check(self) -> bool:
        """
        Verify that Ollama is running and the reasoning model is available.

        Vision health (MLX) is checked separately — it doesn't go through Ollama.
        Returns True if the reasoning model is healthy, False otherwise (never raises).
        """
        try:
            models_response = await asyncio.wait_for(
                self._client.list(), timeout=10
            )
            available = [m.model for m in models_response.models]
            reasoning_ok = any(self.reasoning_model in m for m in available)

            if not reasoning_ok:
                logger.warning(
                    "Reasoning model '{}' not found in Ollama. Available: {}",
                    self.reasoning_model,
                    available,
                )
            else:
                logger.info(
                    "Ollama healthy — reasoning={} available_models={}",
                    self.reasoning_model,
                    available,
                )

            # Log vision backend status
            if self.vision_backend == "mlx":
                logger.info(
                    "Vision backend: MLX — model={}",
                    self.vision_model,
                )
            else:
                vision_ok = any(self.vision_model in m for m in available)
                if not vision_ok:
                    logger.warning(
                        "Vision model '{}' not found in Ollama. Available: {}",
                        self.vision_model,
                        available,
                    )

            return reasoning_ok
        except Exception as e:
            logger.error("Ollama health check failed: {}", e)
            return False

    # ──────────────────────────────────────────────────────────────────────────
    # Convenience factory
    # ──────────────────────────────────────────────────────────────────────────

    @classmethod
    def from_settings(cls, settings: Any) -> "OllamaClient":
        """Build an OllamaClient from the application Settings object."""
        return cls(
            base_url=settings.llm.ollama_base_url,
            reasoning_model=settings.llm.reasoning_model,
            vision_model=settings.llm.vision_model,
            vision_backend=getattr(settings.llm, "vision_backend", "mlx"),
            timeout=settings.llm.timeout,
            max_retries=settings.llm.max_retries,
        )
