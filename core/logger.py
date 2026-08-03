"""
core/logger.py — Centralised Loguru configuration.

Call `setup_logging(settings)` once at application startup.
Everywhere else, import `logger` from this module:

    from core.logger import logger
    logger.info("Something happened")
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger  # re-exported for convenience

if TYPE_CHECKING:
    from core.config import Settings

# Remove the default Loguru handler so we control everything.
logger.remove()

__all__ = ["logger", "setup_logging"]


def setup_logging(settings: Settings) -> None:
    """
    Configure Loguru sinks based on application settings.

    Two sinks are created:
    1. Console — coloured, human-readable (always active)
    2. File    — rotating JSON (machine-readable, for analytics)
    """
    logs_dir = Path(settings.storage.logs_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)

    # ── Console sink ─────────────────────────────────────────────────────────
    console_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>"
    )
    logger.add(
        sys.stderr,
        format=console_format,
        level=settings.logging.level,
        colorize=True,
        backtrace=True,
        diagnose=True,
    )

    # ── File sink — plain text (human readable) ───────────────────────────────
    logger.add(
        logs_dir / "career_agent_{time:YYYY-MM-DD}.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
        level=settings.logging.level,
        rotation=settings.logging.rotation,
        retention=settings.logging.retention,
        compression="gz",
        encoding="utf-8",
        backtrace=True,
        diagnose=False,  # Don't expose local vars in file logs
    )

    # ── File sink — structured JSON (machine readable) ────────────────────────
    if settings.logging.json_logs:
        logger.add(
            logs_dir / "career_agent_structured_{time:YYYY-MM-DD}.jsonl",
            format="{message}",
            level="DEBUG",
            rotation=settings.logging.rotation,
            retention=settings.logging.retention,
            compression="gz",
            encoding="utf-8",
            serialize=True,  # Loguru's built-in JSON serialisation
        )

    logger.info("Logging initialised — level={}", settings.logging.level)
