"""
llm/response_parser.py — Parse LLM responses into typed Pydantic models.

Every LLM call returns raw text. This module converts that text into
structured, validated Python objects that agents can rely on.

If parsing fails, we raise LLMParseError with the raw response attached
so the caller can decide whether to retry or log and continue.
"""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import BaseModel, ValidationError

from core.exceptions import LLMParseError
from core.logger import logger


# ──────────────────────────────────────────────────────────────────────────────
# Typed response models
# ──────────────────────────────────────────────────────────────────────────────


class JobScoreResponse(BaseModel):
    """Structured output from the SCORE_JOB_PROMPT."""

    title_match: int
    skills_match: int
    experience_match: int
    location_match: int
    overall: int
    apply: bool
    reasoning: str
    missing_skills: list[str] = []

    def clamp(self) -> "JobScoreResponse":
        """Clamp all score fields to [0, 100]."""
        return self.model_copy(
            update={
                "title_match": max(0, min(100, self.title_match)),
                "skills_match": max(0, min(100, self.skills_match)),
                "experience_match": max(0, min(100, self.experience_match)),
                "location_match": max(0, min(100, self.location_match)),
                "overall": max(0, min(100, self.overall)),
            }
        )


class PlannerResponse(BaseModel):
    """Structured output from the PLANNER_SYSTEM_PROMPT."""

    thought: str
    tool: str
    args: dict[str, Any] = {}
    final_answer: str | None = None

from typing import Literal

class ExtractedFormField(BaseModel):
    field_id: str            # a stable synthetic id (label+index) so we can map back to DOM
    label: str
    field_type: str          # text | textarea | email | phone | dropdown | checkbox | radio | file
    required: bool = False
    options: list[str] = []  # for dropdown/radio
    answer: str | None = None
    source: Literal["profile", "resume", "fabricated", "skipped"] = "profile"
    confidence: float = 1.0  # LLM self-reported confidence, mainly meaningful for "fabricated"
    reasoning: str = ""      # why it answered this way — useful for audit

class ExtractedForm(BaseModel):
    page_type: str
    fields: list[ExtractedFormField] = []
    submit_button_label: str | None = None
    notes: str = ""



class VisionResponse(BaseModel):
    """Structured output from the VISION_UNDERSTAND_PROMPT."""

    class FormField(BaseModel):
        label: str
        type: str
        required: bool = False

    class SubmitButton(BaseModel):
        visible: bool
        label: str = ""

    page_type: str
    form_fields: list[FormField] = []
    submit_button: SubmitButton = SubmitButton(visible=False)
    errors: list[str] = []
    is_complete: bool = False
    notes: str = ""


# ──────────────────────────────────────────────────────────────────────────────
# Core parsing utilities
# ──────────────────────────────────────────────────────────────────────────────


def extract_json(text: str) -> dict[str, Any]:
    """
    Extract the first JSON object from a string.

    Handles:
    - Pure JSON responses
    - JSON wrapped in ```json ... ``` blocks
    - JSON embedded in prose (extracts first {...} block)
    """
    text = text.strip()

    # Try 1: parse as-is
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try 2: strip markdown code fences
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence_match:
        try:
            return json.loads(fence_match.group(1))
        except json.JSONDecodeError:
            pass

    # Try 3: find the first {...} block
    brace_match = re.search(r"\{.*\}", text, re.DOTALL)
    if brace_match:
        try:
            return json.loads(brace_match.group())
        except json.JSONDecodeError:
            pass

    raise LLMParseError(
        "Could not extract JSON from LLM response",
        raw_response=text,
    )


def parse_job_score(raw: str | dict) -> JobScoreResponse:
    """Parse the LLM's job scoring response into a JobScoreResponse."""
    try:
        data = raw if isinstance(raw, dict) else extract_json(raw)
        score = JobScoreResponse(**data).clamp()
        logger.debug("Parsed job score — overall={}", score.overall)
        return score
    except (ValidationError, KeyError, TypeError) as e:
        raw_str = json.dumps(raw) if isinstance(raw, dict) else str(raw)
        raise LLMParseError(
            f"Failed to parse job score response: {e}",
            raw_response=raw_str,
        ) from e


def parse_planner_response(raw: str | dict) -> PlannerResponse:
    """Parse the planner's tool-selection response."""
    try:
        data = raw if isinstance(raw, dict) else extract_json(raw)
        return PlannerResponse(**data)
    except (ValidationError, KeyError, TypeError) as e:
        raw_str = json.dumps(raw) if isinstance(raw, dict) else str(raw)
        raise LLMParseError(
            f"Failed to parse planner response: {e}",
            raw_response=raw_str,
        ) from e


def parse_vision_response(raw: str | dict) -> VisionResponse:
    """Parse the vision module's screen analysis response."""
    try:
        data = raw if isinstance(raw, dict) else extract_json(raw)
        return VisionResponse(**data)
    except (ValidationError, KeyError, TypeError) as e:
        raw_str = json.dumps(raw) if isinstance(raw, dict) else str(raw)
        raise LLMParseError(
            f"Failed to parse vision response: {e}",
            raw_response=raw_str,
        ) from e

def parse_extracted_form(raw: str | dict) -> ExtractedForm:
    """Parse the form extractor response."""
    try:
        data = raw if isinstance(raw, dict) else extract_json(raw)
        return ExtractedForm(**data)
    except (ValidationError, KeyError, TypeError) as e:
        raw_str = json.dumps(raw) if isinstance(raw, dict) else str(raw)
        raise LLMParseError(
            f"Failed to parse extracted form: {e}",
            raw_response=raw_str,
        ) from e
