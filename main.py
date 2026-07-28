"""
main.py — FastAPI application entry point.

Startup sequence:
1. Load settings from config.yaml
2. Configure logging
3. Validate profile exists
4. Initialise database (create tables)
5. Check Ollama connectivity
6. Start APScheduler
7. Register API routes
8. Serve

Shutdown sequence:
1. Stop scheduler
2. Close database connections
"""

from __future__ import annotations

from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.routes import applications, documents, jobs
from api.schemas import HealthResponse, StatsResponse
from core.config import get_settings
from core.database import close_db, init_db
from core.logger import logger, setup_logging
from memory.memory_manager import MemoryManager
from scheduler.job_scheduler import create_scheduler, get_scheduler

# ──────────────────────────────────────────────────────────────────────────────
# Application lifespan
# ──────────────────────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage startup and shutdown lifecycle."""

    # ── Startup ──────────────────────────────────────────────────────────────
    settings = get_settings()
    setup_logging(settings)

    logger.info("=" * 60)
    logger.info("  Autonomous Career Agent — starting up")
    logger.info("  Reasoning model : {}", settings.llm.reasoning_model)
    logger.info("  Vision model    : {} ({})", settings.llm.vision_model, settings.llm.vision_backend)
    logger.info("  Dry run         : {}", settings.application.dry_run)
    logger.info("=" * 60)

    # Initialise database
    await init_db(settings.storage.database_url)

    # Check Ollama
    from llm.client import OllamaClient
    llm_client = OllamaClient.from_settings(settings)
    ollama_ok = await llm_client.health_check()
    if not ollama_ok:
        logger.warning(
            "⚠️  Ollama is not reachable or model '{}' is not available. "
            "Discovery scoring will be skipped until Ollama is running.",
            settings.llm.reasoning_model,
        )

    # Check profile
    try:
        from profile.loader import load_profile
        profile = load_profile(settings.profile.path)
        app.state.profile = profile
    except Exception as e:
        logger.warning("⚠️  Profile not loaded: {}. Fill in profile/profile.yaml", e)
        app.state.profile = None

    # Store shared instances in app state
    app.state.settings = settings
    app.state.llm_client = llm_client

    # Start scheduler
    scheduler = create_scheduler(settings)
    scheduler.start()
    logger.info("Scheduler started")

    logger.info("✓ Career Agent ready — API docs at http://localhost:8000/docs")

    yield  # ← Server runs here

    # ── Shutdown ─────────────────────────────────────────────────────────────
    logger.info("Career Agent shutting down...")

    sched = get_scheduler()
    if sched and sched.running:
        sched.shutdown(wait=False)

    await close_db()
    logger.info("Career Agent stopped")


# ──────────────────────────────────────────────────────────────────────────────
# FastAPI application
# ──────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Autonomous Career Agent",
    description=(
        "AI-powered job discovery and application system. "
        "Discovers jobs from Greenhouse, Lever, and Ashby; "
        "scores them against your profile using Ollama LLMs; "
        "and automates the application process."
    ),
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# CORS — allow localhost for dashboard development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:8080", "http://127.0.0.1:*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ──────────────────────────────────────────────────────────────────
app.include_router(jobs.router, prefix="/api/v1")
app.include_router(applications.router, prefix="/api/v1")
app.include_router(documents.router, prefix="/api/v1")


# ──────────────────────────────────────────────────────────────────────────────
# Root endpoints
# ──────────────────────────────────────────────────────────────────────────────


@app.get("/", include_in_schema=False)
async def root():
    return {"name": "Autonomous Career Agent", "version": "0.1.0", "docs": "/docs"}


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    """Check system health: Ollama, database, profile."""
    settings = get_settings()

    # Check Ollama
    from llm.client import OllamaClient
    llm = OllamaClient.from_settings(settings)
    ollama_ok = await llm.health_check()

    # Check database
    db_ok = True
    try:
        from core.database import get_session
        async with get_session(settings.storage.database_url) as db:
            from sqlalchemy import text
            await db.execute(text("SELECT 1"))
    except Exception:
        db_ok = False

    # Check profile
    from pathlib import Path
    profile_ok = Path(settings.profile.path).exists()

    overall = "healthy" if (ollama_ok and db_ok and profile_ok) else "degraded"

    return HealthResponse(
        status=overall,
        ollama_connected=ollama_ok,
        database_connected=db_ok,
        reasoning_model=settings.llm.reasoning_model,
        vision_model=settings.llm.vision_model,
        profile_loaded=profile_ok,
    )


@app.get("/stats", response_model=StatsResponse, tags=["System"])
async def stats():
    """Return aggregate statistics about all job applications."""
    settings = get_settings()
    from core.database import get_session
    from sqlalchemy.ext.asyncio import AsyncSession
    async with get_session(settings.storage.database_url) as db:
        memory = MemoryManager(db=db, chroma_path=settings.storage.chroma_path)
        application_stats = await memory.get_application_stats()
    return StatsResponse(**application_stats)


# ──────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
