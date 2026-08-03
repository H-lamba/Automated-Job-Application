"""
core/exceptions.py — Domain-specific exceptions for the Career Agent.

Every exception carries a human-readable message and, where applicable,
structured metadata so callers can make decisions without parsing strings.
"""

from __future__ import annotations

# ──────────────────────────────────────────────────────────────────────────────
# Base
# ──────────────────────────────────────────────────────────────────────────────


class CareerAgentError(Exception):
    """Root exception for all Career Agent errors."""

    def __init__(self, message: str, *, context: dict | None = None) -> None:
        super().__init__(message)
        self.context: dict = context or {}


# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────


class ConfigurationError(CareerAgentError):
    """Raised when required configuration is missing or invalid."""


# ──────────────────────────────────────────────────────────────────────────────
# LLM
# ──────────────────────────────────────────────────────────────────────────────


class LLMError(CareerAgentError):
    """Base class for LLM-related errors."""


class LLMTimeoutError(LLMError):
    """Raised when an LLM request exceeds the configured timeout."""


class LLMParseError(LLMError):
    """Raised when the LLM response cannot be parsed into the expected format."""

    def __init__(self, message: str, *, raw_response: str = "", **kwargs) -> None:
        super().__init__(message, **kwargs)
        self.raw_response = raw_response


class LLMConnectionError(LLMError):
    """Raised when the Ollama server cannot be reached."""


# ──────────────────────────────────────────────────────────────────────────────
# Discovery
# ──────────────────────────────────────────────────────────────────────────────


class DiscoveryError(CareerAgentError):
    """Base class for job discovery errors."""


class JobSourceError(DiscoveryError):
    """Raised when a job source fails to return data."""

    def __init__(self, message: str, *, source_name: str = "", **kwargs) -> None:
        super().__init__(message, **kwargs)
        self.source_name = source_name


class NormalizationError(DiscoveryError):
    """Raised when a raw job cannot be normalised into a JobListing."""


# ──────────────────────────────────────────────────────────────────────────────
# Application
# ──────────────────────────────────────────────────────────────────────────────


class ApplicationError(CareerAgentError):
    """Base class for application-flow errors."""


class ATSNotSupportedError(ApplicationError):
    """Raised when the detected ATS has no matching adapter."""

    def __init__(self, message: str, *, ats_type: str = "", url: str = "", **kwargs) -> None:
        super().__init__(message, **kwargs)
        self.ats_type = ats_type
        self.url = url


class ApplicationFailedError(ApplicationError):
    """Raised when an application cannot be completed."""

    def __init__(self, message: str, *, job_id: str = "", step: str = "", **kwargs) -> None:
        super().__init__(message, **kwargs)
        self.job_id = job_id
        self.step = step


class DailyLimitReachedError(ApplicationError):
    """Raised when the daily application limit has been reached."""


# ──────────────────────────────────────────────────────────────────────────────
# Browser / Vision
# ──────────────────────────────────────────────────────────────────────────────


class BrowserError(CareerAgentError):
    """Base class for browser automation errors."""


class PageLoadError(BrowserError):
    """Raised when a page fails to load within the configured timeout."""


class ElementNotFoundError(BrowserError):
    """Raised when a required page element cannot be located."""


class VisionParseError(CareerAgentError):
    """Raised when the vision module cannot interpret a screenshot."""


# ──────────────────────────────────────────────────────────────────────────────
# Documents
# ──────────────────────────────────────────────────────────────────────────────


class DocumentError(CareerAgentError):
    """Base class for document management errors."""


class DocumentNotFoundError(DocumentError):
    """Raised when the requested document file does not exist."""

    def __init__(self, message: str, *, document_type: str = "", **kwargs) -> None:
        super().__init__(message, **kwargs)
        self.document_type = document_type


# ──────────────────────────────────────────────────────────────────────────────
# Memory
# ──────────────────────────────────────────────────────────────────────────────


class MemoryError(CareerAgentError):
    """Raised when the memory layer encounters an unexpected error."""


# ──────────────────────────────────────────────────────────────────────────────
# Profile
# ──────────────────────────────────────────────────────────────────────────────


class ProfileError(CareerAgentError):
    """Raised when the user profile is missing or malformed."""
